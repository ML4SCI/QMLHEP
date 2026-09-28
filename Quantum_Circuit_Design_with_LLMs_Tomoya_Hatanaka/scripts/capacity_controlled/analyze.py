#!/usr/bin/env python
"""Analyze the capacity-controlled T1 pilot: sanitized machine-readable results,
statistics, and figures — all from the durable store (never hand-copied).

Writes into outputs/capacity_controlled_t1_v1/:
  experiment_manifest.json, candidate_results.csv, run_summary.csv/json,
  baseline_summary.csv, statistical_analysis.json, figure_data.json, report.md,
  and 13 figures (PNG + SVG). The store itself is never published.

Usage:
  python scripts/capacity_controlled/analyze.py \
      --run-dir runs/capacity_controlled_t1_v1 --tag pilot
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy import stats  # noqa: E402

from llm_vqc.evaluation.model import HybridQNNModel  # noqa: E402
from llm_vqc.evaluation.seeds import train_seed_for_circuit  # noqa: E402
from llm_vqc.evaluation.store import ResultStore  # noqa: E402
from llm_vqc.experiments.capacity_controlled import space as S  # noqa: E402
from llm_vqc.ir.budget import ProposalOutcome  # noqa: E402
from llm_vqc.ir.schema import CircuitIR  # noqa: E402
from llm_vqc.tasks.t1_gaussian import T1GaussianPeakTask  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "capacity_controlled_t1_v1"
TASK = "T1"
ARMS = ("controlled_random", "controlled_evolutionary", "controlled_greedy")
ARM_LABEL = {"controlled_random": "Random", "controlled_evolutionary": "Evolutionary",
             "controlled_greedy": "Greedy"}
ARM_COLOR = {"controlled_random": "#3767b3", "controlled_evolutionary": "#c8781e",
             "controlled_greedy": "#8a5fb0"}
CONSUMING = (ProposalOutcome.VALID, ProposalOutcome.DUPLICATE, ProposalOutcome.FAILED)
BOOT = 2000
CHECKPOINTS = (10, 25)


def _save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=150, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def cliffs_delta(a, b):
    gt = sum(1 for x, y in itertools.product(a, b) if x > y)
    lt = sum(1 for x, y in itertools.product(a, b) if x < y)
    return (gt - lt) / (len(a) * len(b))


def boot_ci(values, fn=np.median, n=BOOT, seed=0):
    rng = np.random.default_rng(seed)
    vals = np.asarray(values, dtype=float)
    stat = [fn(rng.choice(vals, size=len(vals), replace=True)) for _ in range(n)]
    return float(np.percentile(stat, 2.5)), float(np.percentile(stat, 97.5))


def total_params_for(ir: CircuitIR) -> int:
    m = HybridQNNModel(ir, raw_feature_dim=S.RAW_FEATURE_DIM, head_out_dim=1)
    return int(sum(p.numel() for p in m.parameters()))


def anytime_curve(store, run_id, seed):
    best, curve = None, []
    for e in store.iter_proposal_events(run_id):
        if e.outcome in CONSUMING and e.structural_hash:
            ts = train_seed_for_circuit(seed, e.structural_hash)
            c = store.get_cached(TASK, e.structural_hash, ts)
            if c and c.val_metric_value is not None and (best is None or c.val_metric_value < best):
                best = c.val_metric_value
        curve.append(best)
    return curve


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=str(ROOT / "runs" / "capacity_controlled_t1_v1"))
    ap.add_argument("--tag", default="pilot")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    OUT.mkdir(parents=True, exist_ok=True)

    summary = json.loads((run_dir / f"{args.tag}_summary.json").read_text())
    seeds = summary["seeds"]
    budget = summary["budget"]
    store = ResultStore(run_dir / "results.sqlite")

    # ---- candidate_results.csv: every evaluated candidate + full capacity ----
    cand_rows = []
    capacity_set = set()
    for e in store.all_evaluations(TASK):
        if not e.circuit_canonical_json or not e.circuit_cost:
            continue
        ir = CircuitIR.model_validate_json(e.circuit_canonical_json)
        m = HybridQNNModel(ir, raw_feature_dim=S.RAW_FEATURE_DIM, head_out_dim=1)
        named = dict(m.named_parameters())
        embed = int(sum(p.numel() for n, p in named.items() if n.startswith("embed.")))
        head = int(sum(p.numel() for n, p in named.items() if n.startswith("head.")))
        quantum = int(sum(p.numel() for n, p in named.items() if n.startswith("q_layer.")))
        total = embed + quantum + head
        capacity_set.add(total)
        c = e.circuit_cost
        cand_rows.append({
            "structural_hash": (e.structural_hash or "")[:12],
            "encoding_type": ir.encoding.type, "n_qubits": c.n_qubits,
            "encoding_inputs": c.input_count, "n_measurements": c.n_qubits,
            "depth": c.depth, "gate_count": c.gate_count,
            "two_qubit_gate_count": c.two_qubit_gate_count,
            "embed_params": embed, "quantum_params": quantum, "head_params": head,
            "total_trainable_params": total,
            "val_rmse": e.val_metric_value,
            "wall_clock_seconds": round(e.wall_clock_seconds, 4),
            "train_seed": e.train_seed,
        })

    # FAIL LOUD if capacity is not uniform
    if capacity_set != {S.TOTAL_TRAINABLE_PARAMS}:
        raise RuntimeError(f"CAPACITY NOT UNIFORM across candidates: {sorted(capacity_set)} "
                           f"(expected only {S.TOTAL_TRAINABLE_PARAMS})")
    print(f"[verify] all {len(cand_rows)} evaluated candidates have exactly "
          f"{S.TOTAL_TRAINABLE_PARAMS} trainable params")

    cfields = list(cand_rows[0].keys())
    with (OUT / "candidate_results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cfields); w.writeheader(); w.writerows(cand_rows)

    # ---- run_summary + anytime curves + selected capacity --------------------
    run_rows, curves = [], {}
    selected_capacity = {}
    for r in summary["runs"]:
        rid = r["run_id"]; seed = r["seed"]
        curves[rid] = anytime_curve(store, rid, seed)
        h = r["selected_structural_hash"]
        cap = None
        if h is not None:
            ts = train_seed_for_circuit(seed, h)
            c = store.get_cached(TASK, h, ts)
            if c and c.circuit_canonical_json:
                ir = CircuitIR.model_validate_json(c.circuit_canonical_json)
                cap = {"depth": c.circuit_cost.depth, "gate_count": c.circuit_cost.gate_count,
                       "two_qubit_gate_count": c.circuit_cost.two_qubit_gate_count,
                       "total_trainable_params": total_params_for(ir)}
                selected_capacity[rid] = cap
        run_rows.append({
            "run_id": rid, "arm": r["arm"], "seed": seed,
            "num_valid": r["ledger_summary"]["num_valid"],
            "num_invalid": r["ledger_summary"]["num_invalid"],
            "num_duplicate": r["ledger_summary"]["num_duplicate"],
            "num_failed": r["ledger_summary"]["num_failed"],
            "consumed_budget": r["ledger_summary"]["consumed_budget"],
            "selected_val_rmse": r["selected_val_metric_value"],
            "test_rmse": r["test_metric_value"],
            "selected_total_params": cap["total_trainable_params"] if cap else None,
            "selected_depth": cap["depth"] if cap else None,
            "selected_two_qubit": cap["two_qubit_gate_count"] if cap else None,
        })
    rfields = list(run_rows[0].keys())
    with (OUT / "run_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rfields); w.writeheader(); w.writerows(run_rows)
    (OUT / "run_summary.json").write_text(json.dumps(run_rows, indent=2))

    # ---- baseline_summary.csv ------------------------------------------------
    bl = summary["baselines"]
    bfields = ["baseline", "kind", "seed", "total_params", "val_rmse", "test_rmse",
               "wall_clock_seconds"]
    with (OUT / "baseline_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=bfields, extrasaction="ignore")
        w.writeheader(); w.writerows(bl)

    # ---- statistical analysis ------------------------------------------------
    val_by_arm = {a: [r["selected_val_rmse"] for r in run_rows if r["arm"] == a] for a in ARMS}
    test_by_arm = {a: [r["test_rmse"] for r in run_rows if r["arm"] == a and r["test_rmse"] is not None]
                   for a in ARMS}

    def arm_stats(vmap):
        out = {}
        for a in ARMS:
            v = [x for x in vmap[a] if x is not None]
            if not v:
                out[a] = {"n": 0}; continue
            lo, hi = boot_ci(v)
            out[a] = {"n": len(v), "values": v, "median": float(np.median(v)),
                      "iqr": float(np.percentile(v, 75) - np.percentile(v, 25)),
                      "mean": float(np.mean(v)), "boot_ci95_median": [lo, hi]}
        return out

    def cross_arm(vmap):
        complete = [a for a in ARMS if len([x for x in vmap[a] if x is not None]) == len(seeds)]
        res = {}
        if len(complete) >= 2:
            groups = [[x for x in vmap[a] if x is not None] for a in complete]
            H, p = stats.kruskal(*groups)
            res["kruskal_wallis"] = {"arms": complete, "H": float(H), "p_value": float(p)}
            pairwise, raw = [], []
            for a, b in itertools.combinations(complete, 2):
                u, pu = stats.mannwhitneyu([x for x in vmap[a]], [x for x in vmap[b]],
                                           alternative="two-sided")
                d = cliffs_delta([x for x in vmap[a]], [x for x in vmap[b]])
                raw.append(pu); pairwise.append({"a": a, "b": b, "U": float(u),
                                                 "p_raw": float(pu), "cliffs_delta": float(d)})
            order = np.argsort(raw); m = len(raw); holm = [0.0] * m; run = 0.0
            for rank, idx in enumerate(order):
                run = max(run, (m - rank) * raw[idx]); holm[idx] = min(1.0, run)
            for e, ph in zip(pairwise, holm, strict=True):
                e["p_holm"] = float(ph)
            res["pairwise"] = pairwise
        return res

    # baselines aggregate
    def bl_agg(name):
        v = [x["val_rmse"] for x in bl if x["baseline"] == name and x["val_rmse"] is not None]
        t = [x["test_rmse"] for x in bl if x["baseline"] == name and x["test_rmse"] is not None]
        return {"val_median": float(np.median(v)) if v else None,
                "test_median": float(np.median(t)) if t else None,
                "test_values": t}

    baseline_names = ["C1_bottleneck", "C2_param_matched_mlp", "Q1_fixed_hea", "Q2_fixed_random"]
    analysis = {
        "note": f"Capacity-controlled T1 v1 pilot; budget={budget}, seeds={seeds}, n={len(seeds)}. "
                "Descriptive/underpowered pilot. Lower RMSE better. Every candidate has exactly "
                f"{S.TOTAL_TRAINABLE_PARAMS} trainable params.",
        "validation_rmse": {"per_arm": arm_stats(val_by_arm), "cross_arm": cross_arm(val_by_arm)},
        "test_rmse": {"per_arm": arm_stats(test_by_arm), "cross_arm": cross_arm(test_by_arm)},
        "baselines": {n: bl_agg(n) for n in baseline_names},
        "anytime_medians": {},
        "capacity_verification": {"unique_total_param_counts": sorted(capacity_set),
                                  "expected": S.TOTAL_TRAINABLE_PARAMS,
                                  "n_candidates": len(cand_rows)},
    }
    for ck in CHECKPOINTS:
        analysis["anytime_medians"][ck] = {}
        for a in ARMS:
            vals = [curves[f"{a}_s{s}"][ck - 1] for s in seeds
                    if len(curves[f"{a}_s{s}"]) >= ck and curves[f"{a}_s{s}"][ck - 1] is not None]
            if vals:
                analysis["anytime_medians"][ck][a] = {"median": float(np.median(vals)), "values": vals}
    # arm-vs-baseline test comparison
    best_arm_test = {a: (min(test_by_arm[a]) if test_by_arm[a] else None) for a in ARMS}
    analysis["arm_vs_baseline_test"] = {
        "best_arm_test_rmse": best_arm_test,
        "C1_bottleneck_test_median": bl_agg("C1_bottleneck")["test_median"],
        "Q1_fixed_hea_test_median": bl_agg("Q1_fixed_hea")["test_median"],
        "Q2_fixed_random_test_median": bl_agg("Q2_fixed_random")["test_median"],
    }
    (OUT / "statistical_analysis.json").write_text(json.dumps(analysis, indent=2))

    # ---- figure_data.json: histories + pred/true for reproducible figures ----
    task = T1GaussianPeakTask()
    tv = task.build(seed=0); test = task.build_test(seed=0)
    # training histories for the selected circuit of each run
    histories = {}
    for r in summary["runs"]:
        h = r["selected_structural_hash"]
        if h is None:
            continue
        ts = train_seed_for_circuit(r["seed"], h)
        c = store.get_cached(TASK, h, ts)
        if c:
            histories[r["run_id"]] = {"train_loss": c.train_loss_history,
                                      "val_rmse": c.val_metric_history}
    # best overall run by test RMSE -> pred vs true on TEST
    best_run = min((r for r in summary["runs"] if r["test_metric_value"] is not None),
                   key=lambda r: r["test_metric_value"])
    h = best_run["selected_structural_hash"]; ts = train_seed_for_circuit(best_run["seed"], h)
    c = store.get_cached(TASK, h, ts); w = store.get_trained_weights(TASK, h, ts)
    ir = CircuitIR.model_validate_json(c.circuit_canonical_json)
    model = HybridQNNModel(ir, raw_feature_dim=S.RAW_FEATURE_DIM, head_out_dim=1)
    sd = {k: torch.tensor(v, dtype=torch.float64).reshape(model.state_dict()[k].shape)
          for k, v in w["classical_state"].items()}
    model.load_state_dict(sd); model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(test.features, dtype=torch.float64)).numpy().reshape(-1)
    true = np.asarray(test.targets)
    figdata = {
        "best_run": {"run_id": best_run["run_id"], "arm": best_run["arm"],
                     "test_rmse": best_run["test_metric_value"],
                     "true": true.tolist(), "pred": pred.tolist()},
        "histories": histories, "anytime_curves": curves,
        "candidate_scatter": [{"depth": x["depth"], "two_qubit": x["two_qubit_gate_count"],
                               "total_params": x["total_trainable_params"],
                               "val_rmse": x["val_rmse"], "runtime": x["wall_clock_seconds"]}
                              for x in cand_rows if x["val_rmse"] is not None],
    }
    (OUT / "figure_data.json").write_text(json.dumps(figdata, indent=2))
    store.close()

    # ---- figures -------------------------------------------------------------
    _make_figures(run_rows, bl, analysis, figdata, seeds, budget, capacity_set, cand_rows)

    # ---- manifest ------------------------------------------------------------
    manifest = {
        "package": "capacity_controlled_t1_v1",
        "generated_at_utc": summary.get("git_sha"),
        "git_sha": summary.get("git_sha"),
        "tag": args.tag, "budget": budget, "seeds": seeds, "n_seeds": len(seeds),
        "arms": list(ARMS),
        "files": sorted(p.name for p in OUT.iterdir() if p.is_file()),
        "reproduction": {
            "run": f"python scripts/capacity_controlled/run_experiment.py --budget {budget} "
                   f"--seeds {','.join(map(str, seeds))} --tag {args.tag}",
            "analyze": f"python scripts/capacity_controlled/analyze.py --tag {args.tag}",
            "legacy_audit": "python scripts/capacity_controlled/audit_legacy_capacity.py "
                            "--store <legacy runs/pilot_t1/results.sqlite>",
        },
        "integrity": "no LLM, no amplitude encoding, validation-only feedback, "
                     "protected test once, uniform 105-param capacity verified.",
    }
    (OUT / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote machine-readable results + 13 figures to {OUT}")


def _make_figures(run_rows, bl, analysis, figdata, seeds, budget, capacity_set, cand_rows):
    curves = figdata["anytime_curves"]
    val_by_arm = {a: [r["selected_val_rmse"] for r in run_rows if r["arm"] == a] for a in ARMS}
    test_by_arm = {a: [r["test_rmse"] for r in run_rows if r["arm"] == a] for a in ARMS}

    def _bars(vmap, title, fname, ylabel):
        fig, ax = plt.subplots(figsize=(7.5, 4.6))
        for i, a in enumerate(ARMS):
            v = [x for x in vmap[a] if x is not None]
            ax.bar(i, np.median(v), 0.6, color=ARM_COLOR[a], alpha=0.85)
            ax.scatter([i] * len(v), v, color="black", s=20, zorder=3)
        ax.set_xticks(range(len(ARMS))); ax.set_xticklabels([ARM_LABEL[a] for a in ARMS])
        ax.set_ylabel(ylabel); ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.25)
        _save(fig, fname)

    # 1, 2
    _bars(val_by_arm, f"Validation RMSE by arm (capacity-controlled, n={len(seeds)})\n"
          "bars=median, dots=seeds; every candidate = 105 params",
          "fig01_validation_rmse_by_arm", "validation RMSE (lower better)")
    _bars(test_by_arm, f"Protected test RMSE by arm (capacity-controlled, n={len(seeds)})\n"
          "bars=median, dots=seeds", "fig02_test_rmse_by_arm", "protected test RMSE (lower better)")

    # 3 anytime
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for a in ARMS:
        for s in seeds:
            c = curves[f"{a}_s{s}"]
            ax.plot(range(1, len(c) + 1), c, color=ARM_COLOR[a], alpha=0.8, lw=1.4, marker=".",
                    ms=3, label=ARM_LABEL[a] if s == seeds[0] else None)
    ax.set_yscale("log"); ax.set_xlabel(f"candidates evaluated (budget B={budget})")
    ax.set_ylabel("best-so-far validation RMSE (log)")
    ax.set_title("Anytime search behaviour (capacity-controlled)", fontsize=10)
    ax.legend(fontsize=9); ax.grid(alpha=0.25, which="both")
    _save(fig, "fig03_anytime_curves")

    # 4 training curves for selected circuits (one line per run, val RMSE)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for rid, hist in figdata["histories"].items():
        arm = rid.rsplit("_s", 1)[0]
        ax.plot(range(1, len(hist["val_rmse"]) + 1), hist["val_rmse"],
                color=ARM_COLOR.get(arm, "gray"), alpha=0.7, lw=1.3)
    ax.set_xlabel("epoch"); ax.set_ylabel("validation RMSE")
    ax.set_title("Training curves of every run's selected circuit (val RMSE, 20 epochs)", fontsize=10)
    ax.grid(alpha=0.25)
    from matplotlib.lines import Line2D
    ax.legend([Line2D([0], [0], color=ARM_COLOR[a]) for a in ARMS],
              [ARM_LABEL[a] for a in ARMS], fontsize=9)
    _save(fig, "fig04_training_curves_selected")

    # 5 pred vs true (test), 6 residuals
    br = figdata["best_run"]; true = np.array(br["true"]); pred = np.array(br["pred"])
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.scatter(true, pred, s=10, alpha=0.5, color="#4a9e5c", edgecolor="none")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1.3)
    ax.set_xlabel("true μ"); ax.set_ylabel("predicted μ̂"); ax.set_aspect("equal")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title(f"Predicted vs true μ — protected test\nbest run {br['run_id']} "
                 f"({ARM_LABEL[br['arm']]}), RMSE={br['test_rmse']:.5f}, n={len(true)}", fontsize=9)
    ax.grid(alpha=0.25)
    _save(fig, "fig05_pred_vs_true_test")

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.scatter(true, pred - true, s=10, alpha=0.5, color="#c8781e", edgecolor="none")
    ax.axhline(0, color="grey", ls="--", lw=1.2)
    ax.set_xlabel("true μ"); ax.set_ylabel("residual (μ̂ − μ)")
    ax.set_title(f"Protected-test residuals — best run {br['run_id']}", fontsize=10)
    ax.grid(alpha=0.25)
    _save(fig, "fig06_test_residuals")

    # 7,8,9 performance vs depth / two-qubit / total params
    sc = figdata["candidate_scatter"]
    def _scatter(key, xlabel, fname, title):
        fig, ax = plt.subplots(figsize=(7.5, 4.4))
        xs = [c[key] for c in sc]; ys = [c["val_rmse"] for c in sc]
        ax.scatter(xs, ys, s=14, alpha=0.4, color="#3767b3", edgecolor="none")
        ax.set_xlabel(xlabel); ax.set_ylabel("validation RMSE (lower better)")
        ax.set_yscale("log"); ax.set_title(title, fontsize=10); ax.grid(alpha=0.25, which="both")
        _save(fig, fname)
    _scatter("depth", "circuit depth", "fig07_perf_vs_depth",
             "Validation RMSE vs circuit depth (all controlled candidates)")
    _scatter("two_qubit", "two-qubit gate count", "fig08_perf_vs_two_qubit",
             "Validation RMSE vs two-qubit gate count")

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    xs = [c["total_params"] for c in sc]; ys = [c["val_rmse"] for c in sc]
    ax.scatter(xs, ys, s=16, alpha=0.5, color="#8a5fb0", edgecolor="none")
    ax.set_xlabel("total trainable parameters"); ax.set_ylabel("validation RMSE (lower better)")
    ax.set_yscale("log")
    ax.set_title("Validation RMSE vs total trainable params\n(vertical line: capacity is fixed — "
                 "the whole point)", fontsize=10)
    ax.axvline(S.TOTAL_TRAINABLE_PARAMS, color="crimson", ls="--", lw=1.5)
    ax.grid(alpha=0.25, which="both")
    _save(fig, "fig09_perf_vs_total_params")

    # 10 capacity verification
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    totals = [c["total_trainable_params"] for c in cand_rows]
    ax.plot(range(len(totals)), totals, ".", color="#4a9e5c", ms=5)
    ax.axhline(S.TOTAL_TRAINABLE_PARAMS, color="crimson", ls="--", lw=1.5)
    ax.set_ylim(S.TOTAL_TRAINABLE_PARAMS - 20, S.TOTAL_TRAINABLE_PARAMS + 20)
    ax.set_xlabel("evaluated candidate index")
    ax.set_ylabel("total trainable params")
    ax.set_title(f"CAPACITY VERIFICATION: all {len(totals)} candidates have exactly "
                 f"{S.TOTAL_TRAINABLE_PARAMS} params\n(unique values observed: "
                 f"{sorted(capacity_set)})", fontsize=10)
    _save(fig, "fig10_capacity_verification")

    # 11 comparison with baselines (test RMSE)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    labels, meds, cols, allv = [], [], [], []
    for a in ARMS:
        v = [x for x in test_by_arm[a] if x is not None]
        labels.append(ARM_LABEL[a]); meds.append(np.median(v)); cols.append(ARM_COLOR[a]); allv.append(v)
    for name, disp, col in (("C1_bottleneck", "C1 bottleneck", "#999999"),
                            ("C2_param_matched_mlp", "C2 MLP", "#666666"),
                            ("Q1_fixed_hea", "Q1 fixed HEA", "#2a7f7f"),
                            ("Q2_fixed_random", "Q2 fixed rand", "#7f2a5a")):
        t = [x["test_rmse"] for x in bl if x["baseline"] == name and x["test_rmse"] is not None]
        labels.append(disp); meds.append(np.median(t)); cols.append(col); allv.append(t)
    xpos = range(len(labels))
    ax.bar(xpos, meds, 0.6, color=cols, alpha=0.85)
    for i, v in enumerate(allv):
        ax.scatter([i] * len(v), v, color="black", s=16, zorder=3)
    ax.set_xticks(list(xpos)); ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("protected test RMSE (lower better)")
    ax.set_title(f"Search arms vs classical & fixed-circuit baselines (test RMSE, n={len(seeds)})\n"
                 "search arms & Q1/Q2 = 105 params; C1=93, C2=107", fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    _save(fig, "fig11_arms_vs_baselines")

    # 12 invalid/duplicate/failed rates
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    x = np.arange(len(ARMS)); w = 0.25
    for j, (key, lab) in enumerate((("num_invalid", "invalid"), ("num_duplicate", "duplicate"),
                                    ("num_failed", "failed"))):
        rates = []
        for a in ARMS:
            tot = sum(r["consumed_budget"] + r["num_invalid"] for r in run_rows if r["arm"] == a)
            cnt = sum(r[key] for r in run_rows if r["arm"] == a)
            rates.append(cnt / max(1, tot))
        ax.bar(x + (j - 1) * w, rates, w, label=lab)
    ax.set_xticks(x); ax.set_xticklabels([ARM_LABEL[a] for a in ARMS])
    ax.set_ylabel("fraction of proposals"); ax.legend(fontsize=9)
    ax.set_title("Invalid / duplicate / failed proposal rates by arm", fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    _save(fig, "fig12_proposal_rates")

    # 13 runtime per evaluated candidate
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    rts = [c["wall_clock_seconds"] for c in cand_rows if c["wall_clock_seconds"]]
    ax.hist(rts, bins=30, color="#3767b3", alpha=0.8)
    ax.set_xlabel("training wall-clock seconds per candidate")
    ax.set_ylabel("count")
    ax.set_title(f"Runtime per evaluated candidate (n={len(rts)}, "
                 f"median={np.median(rts):.2f}s)", fontsize=10)
    ax.grid(alpha=0.25)
    _save(fig, "fig13_runtime_per_candidate")


if __name__ == "__main__":
    main()
