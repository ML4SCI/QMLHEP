#!/usr/bin/env python
"""Analyze the v2 matrix: machine-readable summaries, equivalence statistics, and
16 figures — all from the durable per-split stores and the run summary. Raw
stores are never published.

Usage: python scripts/capacity_controlled/analyze_v2.py --tag pilot
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
OUT = ROOT / "outputs" / "capacity_controlled_t1_v2"
TASK = "T1"
ARMS = ("controlled_random", "controlled_evolutionary", "controlled_greedy")
LABEL = {"controlled_random": "Random", "controlled_evolutionary": "Evolutionary",
         "controlled_greedy": "Greedy"}
COLOR = {"controlled_random": "#3767b3", "controlled_evolutionary": "#c8781e",
         "controlled_greedy": "#8a5fb0"}
CONSUMING = (ProposalOutcome.VALID, ProposalOutcome.DUPLICATE, ProposalOutcome.FAILED)
DELTA = 0.002
DELTAS = (0.001, 0.002, 0.003)


def _save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=150, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def cliffs_delta(a, b):
    gt = sum(1 for x, y in itertools.product(a, b) if x > y)
    lt = sum(1 for x, y in itertools.product(a, b) if x < y)
    return (gt - lt) / (len(a) * len(b)) if a and b else 0.0


def tost(d, delta, alpha=0.05):
    d = np.asarray(d, float); n = len(d)
    mean = float(np.mean(d)); se = float(np.std(d, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    df = n - 1
    if se == 0:
        eq = abs(mean) < delta
        return {"mean_diff": mean, "se": 0.0, "ci90": [mean, mean], "tost_p": 0.0 if eq else 1.0,
                "equivalent": bool(eq)}
    p_upper = float(stats.t.cdf((mean - delta) / se, df))
    p_lower = float(stats.t.sf((mean + delta) / se, df))
    tcrit = float(stats.t.ppf(0.95, df))
    ci = [mean - tcrit * se, mean + tcrit * se]
    tost_p = max(p_upper, p_lower)
    return {"mean_diff": mean, "se": se, "ci90": ci, "tost_p": tost_p,
            "equivalent": bool(ci[0] > -delta and ci[1] < delta)}


def blocked_bootstrap_median_diff(cells_a, cells_b, split_of, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    keys = list(cells_a.keys())
    by_split = {}
    for k in keys:
        by_split.setdefault(split_of[k], []).append(k)
    meds = []
    for _ in range(n):
        resampled = []
        for _s, ks in by_split.items():
            idx = rng.integers(0, len(ks), size=len(ks))
            resampled += [ks[i] for i in idx]
        diffs = [cells_a[k] - cells_b[k] for k in resampled]
        meds.append(float(np.median(diffs)))
    return [float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))]


def anytime_curves(store, run_id, seed):
    """Return best-so-far val vs (proposals, unique trainings, cumulative walltime)."""
    best = None
    seen = set()
    prop_x, uniq_x, time_x, ys = [], [], [], []
    cum_time = 0.0
    n_prop = 0
    n_uniq = 0
    for e in store.iter_proposal_events(run_id):
        if e.outcome in CONSUMING:
            n_prop += 1
            if e.structural_hash:
                ts = train_seed_for_circuit(seed, e.structural_hash)
                c = store.get_cached(TASK, e.structural_hash, ts)
                if e.structural_hash not in seen:
                    seen.add(e.structural_hash); n_uniq += 1
                    if c is not None:
                        cum_time += float(c.wall_clock_seconds or 0.0)
                if c and c.val_metric_value is not None and (best is None or c.val_metric_value < best):
                    best = c.val_metric_value
            prop_x.append(n_prop); uniq_x.append(n_uniq); time_x.append(cum_time); ys.append(best)
    return {"proposals": prop_x, "unique": uniq_x, "walltime": time_x, "best_val": ys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="pilot")
    ap.add_argument("--run-dir", default=str(ROOT / "runs" / "capacity_controlled_t1_v2"))
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    OUT.mkdir(parents=True, exist_ok=True)
    summary = json.loads((run_dir / f"{args.tag}_summary.json").read_text())
    splits, seeds = summary["splits"], summary["seeds"]

    stores = {s: ResultStore(run_dir / f"split_{s}" / "results.sqlite") for s in splits}

    # ---- candidate_results.csv + capacity verification ----
    cand_rows, capset = [], set()
    for s in splits:
        for e in stores[s].all_evaluations(TASK):
            if not e.circuit_canonical_json or not e.circuit_cost:
                continue
            ir = CircuitIR.model_validate_json(e.circuit_canonical_json)
            m = HybridQNNModel(ir, raw_feature_dim=S.RAW_FEATURE_DIM, head_out_dim=1)
            named = dict(m.named_parameters())
            emb = int(sum(p.numel() for n, p in named.items() if n.startswith("embed.")))
            hd = int(sum(p.numel() for n, p in named.items() if n.startswith("head.")))
            qp = int(sum(p.numel() for n, p in named.items() if n.startswith("q_layer.")))
            total = emb + qp + hd; capset.add(total)
            c = e.circuit_cost
            cand_rows.append({"split": s, "structural_hash": (e.structural_hash or "")[:12],
                              "encoding_type": ir.encoding.type, "n_qubits": c.n_qubits,
                              "depth": c.depth, "gate_count": c.gate_count,
                              "two_qubit_gate_count": c.two_qubit_gate_count,
                              "embed_params": emb, "quantum_params": qp, "head_params": hd,
                              "total_trainable_params": total, "val_rmse": e.val_metric_value,
                              "wall_clock_seconds": round(e.wall_clock_seconds, 4)})
    if capset != {S.TOTAL_TRAINABLE_PARAMS}:
        raise RuntimeError(f"CAPACITY NOT UNIFORM: {sorted(capset)} != {{{S.TOTAL_TRAINABLE_PARAMS}}}")
    print(f"[verify] all {len(cand_rows)} controlled candidates have exactly "
          f"{S.TOTAL_TRAINABLE_PARAMS} params")
    with (OUT / "candidate_results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cand_rows[0].keys())); w.writeheader(); w.writerows(cand_rows)

    # ---- run_summary + resource_accounting + anytime ----
    run_rows, resource_rows, curves = [], [], {}
    test_cell = {a: {} for a in ARMS}   # (split,seed) -> test rmse
    val_cell = {a: {} for a in ARMS}
    split_of = {}
    for r in summary["runs"]:
        a, s, seed = r["arm"], r["split"], r["seed"]
        cell = (s, seed); split_of[cell] = s
        test_cell[a][cell] = r["test_metric_value"]; val_cell[a][cell] = r["selected_val_metric_value"]
        led = r["ledger_summary"]
        run_rows.append({"run_id": r["run_id"], "arm": a, "split": s, "seed": seed,
                         "consumed_budget": led["consumed_budget"], "num_valid": led["num_valid"],
                         "num_invalid": led["num_invalid"], "num_duplicate": led["num_duplicate"],
                         "num_failed": led["num_failed"], "num_unique": led["num_unique"],
                         "selected_val_rmse": r["selected_val_metric_value"],
                         "test_rmse": r["test_metric_value"]})
        cur = anytime_curves(stores[s], r["run_id"], seed)
        curves[r["run_id"]] = cur
        resource_rows.append({"run_id": r["run_id"], "arm": a, "split": s, "seed": seed,
                              "proposals_consumed": led["consumed_budget"],
                              "unique_trainings": led["num_unique"],
                              "duplicates": led["num_duplicate"],
                              "invalid": led["num_invalid"], "failed": led["num_failed"],
                              "cumulative_walltime_s": round(cur["walltime"][-1], 3) if cur["walltime"] else 0.0})
    with (OUT / "run_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(run_rows[0].keys())); w.writeheader(); w.writerows(run_rows)
    (OUT / "run_summary.json").write_text(json.dumps(run_rows, indent=2))
    with (OUT / "resource_accounting.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(resource_rows[0].keys())); w.writeheader(); w.writerows(resource_rows)

    # ---- split_summary.csv ----
    split_rows = []
    for s in splits:
        for a in ARMS:
            tv = [test_cell[a][(s, sd)] for sd in seeds if test_cell[a].get((s, sd)) is not None]
            vv = [val_cell[a][(s, sd)] for sd in seeds if val_cell[a].get((s, sd)) is not None]
            split_rows.append({"split": s, "arm": a, "n": len(tv),
                               "test_median": float(np.median(tv)), "test_iqr": float(np.percentile(tv, 75) - np.percentile(tv, 25)),
                               "val_median": float(np.median(vv))})
    with (OUT / "split_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(split_rows[0].keys())); w.writeheader(); w.writerows(split_rows)

    # ---- aux: classical / analytical / quantum ablation summaries ----
    aux = summary["aux"]
    classical_rows, analytical_rows, ablation_rows = [], [], []
    for s in splits:
        for a in aux[str(s)]["analytical"]:
            analytical_rows.append(a)
        for seed in seeds:
            for name, v in aux[str(s)]["classical"][str(seed)].items():
                classical_rows.append({"split": s, "seed": seed, "model": name,
                                       "val_rmse": v["val_rmse"], "test_rmse": v["test_rmse"],
                                       "total_params": v["total_params"]})
            for name, v in aux[str(s)]["ablations"][str(seed)].items():
                ablation_rows.append({"split": s, "seed": seed, "condition": name,
                                      "val_rmse": v.get("val_rmse"), "test_rmse": v.get("test_rmse"),
                                      "total_params": v.get("total_params"),
                                      "trainable_params": v.get("trainable_params"),
                                      "frozen_quantum_params": v.get("frozen_quantum_params"),
                                      "two_qubit_gate_count": v.get("two_qubit_gate_count")})
    for rows, fn in ((classical_rows, "classical_search_summary.csv"),
                     (analytical_rows, "analytical_baseline_summary.csv"),
                     (ablation_rows, "quantum_ablation_summary.csv")):
        with (OUT / fn).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    # ---- statistics ----
    def pooled(cellmap, a):
        return [cellmap[a][(s, sd)] for s in splits for sd in seeds if cellmap[a].get((s, sd)) is not None]

    def boot_ci(vals, seed=0, n=2000):
        rng = np.random.default_rng(seed); v = np.asarray(vals, float)
        m = [np.median(rng.choice(v, len(v), replace=True)) for _ in range(n)]
        return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]

    stat = {"note": f"v2: {len(splits)} splits x {len(seeds)} seeds; lower RMSE better; "
                    f"every controlled candidate = {S.TOTAL_TRAINABLE_PARAMS} params.",
            "per_arm_test": {}, "per_arm_val": {}}
    for a in ARMS:
        tv = pooled(test_cell, a); vv = pooled(val_cell, a)
        stat["per_arm_test"][a] = {"n": len(tv), "median": float(np.median(tv)),
                                   "iqr": float(np.percentile(tv, 75) - np.percentile(tv, 25)),
                                   "mean": float(np.mean(tv)), "sd": float(np.std(tv, ddof=1)),
                                   "boot_ci95_median": boot_ci(tv)}
        stat["per_arm_val"][a] = {"n": len(vv), "median": float(np.median(vv)),
                                  "iqr": float(np.percentile(vv, 75) - np.percentile(vv, 25))}
    # KW + MWU + Cliff on pooled test
    groups = [pooled(test_cell, a) for a in ARMS]
    H, p = stats.kruskal(*groups)
    stat["kruskal_wallis_test"] = {"H": float(H), "p_value": float(p)}
    pw, raw = [], []
    for a, b in itertools.combinations(ARMS, 2):
        u, pu = stats.mannwhitneyu(pooled(test_cell, a), pooled(test_cell, b), alternative="two-sided")
        pw.append({"a": a, "b": b, "U": float(u), "p_raw": float(pu),
                   "cliffs_delta": cliffs_delta(pooled(test_cell, a), pooled(test_cell, b))}); raw.append(pu)
    order = np.argsort(raw); m = len(raw); holm = [0.0] * m; run = 0.0
    for rank, idx in enumerate(order):
        run = max(run, (m - rank) * raw[idx]); holm[idx] = min(1.0, run)
    for e, ph in zip(pw, holm, strict=True):
        e["p_holm"] = float(ph)
    stat["pairwise_test"] = pw

    # ---- equivalence analysis (paired TOST + blocked bootstrap) ----
    equiv = {"margin_primary": DELTA, "pairs": {}}
    for a, b in itertools.combinations(ARMS, 2):
        common = [c for c in test_cell[a] if c in test_cell[b]
                  and test_cell[a][c] is not None and test_cell[b][c] is not None]
        d = [test_cell[a][c] - test_cell[b][c] for c in common]
        pair = {"n_cells": len(common), "paired_diff_sd": float(np.std(d, ddof=1)),
                "blocked_bootstrap_ci95_median_diff":
                    blocked_bootstrap_median_diff({c: test_cell[a][c] for c in common},
                                                  {c: test_cell[b][c] for c in common}, split_of),
                "tost_by_delta": {f"{dl}": tost(d, dl) for dl in DELTAS}}
        equiv["pairs"][f"{a}__vs__{b}"] = pair
    all_equiv = all(equiv["pairs"][k]["tost_by_delta"][f"{DELTA}"]["equivalent"] for k in equiv["pairs"])
    equiv["all_pairs_equivalent_at_primary_delta"] = bool(all_equiv)
    stat["equivalence_summary"] = {"all_pairs_equivalent_at_0.002": bool(all_equiv)}
    (OUT / "equivalence_analysis.json").write_text(json.dumps(equiv, indent=2))

    # baselines aggregate for stat + gates
    ana_med = {}
    for name in ("A1_argmax", "A2_quadratic_peak", "A3_weighted_centroid", "A4_gaussian_nlls"):
        v = [r["test_rmse"] for r in analytical_rows if r["estimator"] == name]
        ana_med[name] = float(np.median(v)) if v else None
    abl_med = {}
    for name in ("Q0_no_quantum", "QF_frozen_random", "QP_product_state", "QE_fixed_entangled", "QR_fixed_random"):
        v = [r["test_rmse"] for r in ablation_rows if r["condition"] == name and r["test_rmse"] is not None]
        abl_med[name] = float(np.median(v)) if v else None
    c3v = [r["test_rmse"] for r in classical_rows if r["model"] == "C3_classical_search" and r["test_rmse"] is not None]
    best_arm_test = min(stat["per_arm_test"][a]["median"] for a in ARMS)
    stat["baselines_test_median"] = {**ana_med, **abl_med,
                                     "C1_bottleneck": float(np.median([r["test_rmse"] for r in classical_rows if r["model"] == "C1_bottleneck"])),
                                     "C2_param_matched_mlp": float(np.median([r["test_rmse"] for r in classical_rows if r["model"] == "C2_param_matched_mlp"])),
                                     "C3_classical_search": float(np.median(c3v)),
                                     "QS_searched_best_arm": best_arm_test}
    (OUT / "statistical_analysis.json").write_text(json.dumps(stat, indent=2))

    # ---- figure_data (pred/true best run on test) ----
    task = T1GaussianPeakTask()
    best_run = min((r for r in summary["runs"] if r["test_metric_value"] is not None),
                   key=lambda r: r["test_metric_value"])
    bs = best_run["split"]; test = task.build_test(seed=bs); tvb = task.build(seed=bs)
    h = best_run["selected_structural_hash"]; ts = train_seed_for_circuit(best_run["seed"], h)
    c = stores[bs].get_cached(TASK, h, ts); w = stores[bs].get_trained_weights(TASK, h, ts)
    ir = CircuitIR.model_validate_json(c.circuit_canonical_json)
    model = HybridQNNModel(ir, raw_feature_dim=21, head_out_dim=1)
    model.load_state_dict({k: torch.tensor(v, dtype=torch.float64).reshape(model.state_dict()[k].shape)
                           for k, v in w["classical_state"].items()})
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(test.features, dtype=torch.float64)).numpy().reshape(-1)
    figdata = {"best_run": {"run_id": best_run["run_id"], "arm": best_run["arm"],
                            "test_rmse": best_run["test_metric_value"],
                            "true": np.asarray(test.targets).tolist(), "pred": pred.tolist()},
               "curves": curves}
    (OUT / "figure_data.json").write_text(json.dumps(figdata, indent=2))
    for st in stores.values():
        st.close()

    _figures(splits, seeds, test_cell, val_cell, split_rows, curves, equiv, stat,
             ana_med, abl_med, classical_rows, ablation_rows, analytical_rows, cand_rows,
             run_rows, figdata, capset)
    _manifests(args.tag, summary, splits, seeds)
    print(f"wrote v2 machine-readable results + figures to {OUT}")


def _figures(splits, seeds, test_cell, val_cell, split_rows, curves, equiv, stat,
             ana_med, abl_med, classical_rows, ablation_rows, analytical_rows, cand_rows,
             run_rows, figdata, capset):
    def pooled(cm, a):
        return [cm[a][(s, sd)] for s in splits for sd in seeds if cm[a].get((s, sd)) is not None]

    # 1 & 2: test/val by arm and split
    for cm, title, fname, ylab in ((test_cell, "Protected-test RMSE by arm & split", "fig01_test_rmse_by_arm_split", "test RMSE"),
                                   (val_cell, "Validation RMSE by arm & split", "fig02_val_rmse_by_arm_split", "val RMSE")):
        fig, ax = plt.subplots(figsize=(9, 4.6))
        x = np.arange(len(ARMS)); wd = 0.22
        for j, s in enumerate(splits):
            for i, a in enumerate(ARMS):
                v = [cm[a][(s, sd)] for sd in seeds if cm[a].get((s, sd)) is not None]
                ax.bar(x[i] + (j - 1) * wd, np.median(v), wd, color=COLOR[a],
                       alpha=0.4 + 0.2 * j, edgecolor="black", lw=0.4,
                       label=f"split {s}" if i == 0 else None)
                ax.scatter([x[i] + (j - 1) * wd] * len(v), v, color="black", s=8, zorder=3)
        ax.set_xticks(x); ax.set_xticklabels([LABEL[a] for a in ARMS])
        ax.set_ylabel(ylab); ax.set_title(f"{title} (n={len(seeds)}/split, bars=median)", fontsize=10)
        ax.legend(fontsize=8, title="shade=split"); ax.grid(axis="y", alpha=0.25)
        _save(fig, fname)

    # 3: equivalence intervals
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    labels, means, los, his = [], [], [], []
    for k, pr in equiv["pairs"].items():
        t = pr["tost_by_delta"][f"{DELTA}"]
        labels.append(k.replace("controlled_", "").replace("__vs__", " vs "))
        means.append(t["mean_diff"]); los.append(t["ci90"][0]); his.append(t["ci90"][1])
    y = np.arange(len(labels))
    ax.errorbar(means, y, xerr=[np.array(means) - np.array(los), np.array(his) - np.array(means)],
                fmt="o", color="#3767b3", capsize=4)
    ax.axvline(0, color="grey", lw=1)
    ax.axvline(-DELTA, color="crimson", ls="--", lw=1.4); ax.axvline(DELTA, color="crimson", ls="--", lw=1.4)
    ax.axvspan(-DELTA, DELTA, color="green", alpha=0.08)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("mean paired test-RMSE difference (90% CI)")
    ax.set_title(f"Equivalence: 90% CIs vs margin ±{DELTA} (green=equivalence region)", fontsize=10)
    _save(fig, "fig03_equivalence_intervals")

    # 4,5,6: anytime by proposals / unique / walltime
    for axis, fname, xlab in (("proposals", "fig04_anytime_proposals", "candidate proposals"),
                              ("unique", "fig05_anytime_unique_trainings", "unique circuit trainings"),
                              ("walltime", "fig06_anytime_walltime", "cumulative wall-clock (s)")):
        fig, ax = plt.subplots(figsize=(8.5, 4.6))
        for a in ARMS:
            for rid, cur in curves.items():
                if not rid.startswith(a):
                    continue
                xs, ys = cur[axis], cur["best_val"]
                xs = [x for x, y in zip(xs, ys) if y is not None]; ys = [y for y in ys if y is not None]
                if xs:
                    ax.plot(xs, ys, color=COLOR[a], alpha=0.3, lw=0.8)
        for a in ARMS:
            ax.plot([], [], color=COLOR[a], label=LABEL[a])
        ax.set_yscale("log"); ax.set_xlabel(xlab); ax.set_ylabel("best-so-far val RMSE (log)")
        ax.set_title(f"Anytime search: best val vs {xlab}", fontsize=10)
        ax.legend(fontsize=9); ax.grid(alpha=0.25, which="both")
        _save(fig, fname)

    # 7: duplicate/invalid/failure rates
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    x = np.arange(len(ARMS)); wd = 0.25
    for j, (key, lab) in enumerate((("num_invalid", "invalid"), ("num_duplicate", "duplicate"), ("num_failed", "failed"))):
        rates = []
        for a in ARMS:
            tot = sum(r["consumed_budget"] + r["num_invalid"] for r in run_rows if r["arm"] == a)
            cnt = sum(r[key] for r in run_rows if r["arm"] == a)
            rates.append(cnt / max(1, tot))
        ax.bar(x + (j - 1) * wd, rates, wd, label=lab)
    ax.set_xticks(x); ax.set_xticklabels([LABEL[a] for a in ARMS]); ax.legend(fontsize=9)
    ax.set_ylabel("fraction of proposals"); ax.set_title("Invalid / duplicate / failed rates by arm", fontsize=10)
    ax.grid(axis="y", alpha=0.25); _save(fig, "fig07_proposal_rates")

    # 8: runtime distribution
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    rts = [c["wall_clock_seconds"] for c in cand_rows if c["wall_clock_seconds"]]
    ax.hist(rts, bins=40, color="#3767b3", alpha=0.8)
    ax.set_xlabel("training wall-clock s per candidate"); ax.set_ylabel("count")
    ax.set_title(f"Runtime per candidate (n={len(rts)}, median={np.median(rts):.2f}s)", fontsize=10)
    ax.grid(alpha=0.25); _save(fig, "fig08_runtime_distribution")

    # 9: searched VQC vs analytical
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    names = [("QS Random", pooled(test_cell, "controlled_random")),
             ("QS Evolutionary", pooled(test_cell, "controlled_evolutionary")),
             ("QS Greedy", pooled(test_cell, "controlled_greedy"))]
    for nm in ("A1_argmax", "A2_quadratic_peak", "A3_weighted_centroid", "A4_gaussian_nlls"):
        names.append((nm.replace("_", " "), [r["test_rmse"] for r in analytical_rows if r["estimator"] == nm]))
    _grouped_bar(ax, names, "Searched VQC vs analytical estimators (protected-test RMSE)")
    _save(fig, "fig09_vqc_vs_analytical")

    # 10: searched VQC vs classical MLP search
    fig, ax = plt.subplots(figsize=(8, 4.6))
    names = [("QS Random", pooled(test_cell, "controlled_random")),
             ("QS Evolutionary", pooled(test_cell, "controlled_evolutionary")),
             ("QS Greedy", pooled(test_cell, "controlled_greedy")),
             ("C1 bottleneck", [r["test_rmse"] for r in classical_rows if r["model"] == "C1_bottleneck"]),
             ("C2 MLP", [r["test_rmse"] for r in classical_rows if r["model"] == "C2_param_matched_mlp"]),
             ("C3 MLP search", [r["test_rmse"] for r in classical_rows if r["model"] == "C3_classical_search"])]
    _grouped_bar(ax, names, "Searched VQC vs classical (incl. fair C3 MLP search)")
    _save(fig, "fig10_vqc_vs_classical_search")

    # 11: quantum ablation comparison
    fig, ax = plt.subplots(figsize=(9, 4.6))
    names = [("Q0 no-quantum", [r["test_rmse"] for r in ablation_rows if r["condition"] == "Q0_no_quantum"]),
             ("QF frozen-q", [r["test_rmse"] for r in ablation_rows if r["condition"] == "QF_frozen_random"]),
             ("QP product", [r["test_rmse"] for r in ablation_rows if r["condition"] == "QP_product_state"]),
             ("QR fixed-rand", [r["test_rmse"] for r in ablation_rows if r["condition"] == "QR_fixed_random"]),
             ("QE fixed-HEA", [r["test_rmse"] for r in ablation_rows if r["condition"] == "QE_fixed_entangled"]),
             ("QS searched(best)", pooled(test_cell, min(ARMS, key=lambda a: np.median(pooled(test_cell, a)))))]
    _grouped_bar(ax, names, "Quantum ablations: is quantum / training / entanglement / search useful?")
    _save(fig, "fig11_quantum_ablations")

    # 12,13: pred vs true + residuals
    br = figdata["best_run"]; true = np.array(br["true"]); pred = np.array(br["pred"])
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.scatter(true, pred, s=8, alpha=0.4, color="#4a9e5c"); ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set_xlabel("true μ"); ax.set_ylabel("pred μ̂"); ax.set_aspect("equal"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title(f"Pred vs true (test) — best run {br['run_id']}\nRMSE={br['test_rmse']:.5f}", fontsize=9)
    ax.grid(alpha=0.25); _save(fig, "fig12_pred_vs_true")
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.scatter(true, pred - true, s=8, alpha=0.4, color="#c8781e"); ax.axhline(0, color="grey", ls="--")
    ax.set_xlabel("true μ"); ax.set_ylabel("residual"); ax.set_title("Protected-test residuals (best run)", fontsize=10)
    ax.grid(alpha=0.25); _save(fig, "fig13_residuals")

    # 14: capacity verification
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    totals = [c["total_trainable_params"] for c in cand_rows]
    ax.plot(range(len(totals)), totals, ".", color="#4a9e5c", ms=3)
    ax.axhline(105, color="crimson", ls="--"); ax.set_ylim(85, 125)
    ax.set_xlabel("candidate index"); ax.set_ylabel("total trainable params")
    ax.set_title(f"Capacity verification: all {len(totals)} candidates = 105 (unique: {sorted(capset)})", fontsize=10)
    _save(fig, "fig14_capacity_verification")

    # 15: split-to-split variability
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for a in ARMS:
        meds = [np.median([test_cell[a][(s, sd)] for sd in seeds if test_cell[a].get((s, sd)) is not None]) for s in splits]
        ax.plot(splits, meds, "-o", color=COLOR[a], label=LABEL[a])
    ax.set_xlabel("data split seed"); ax.set_ylabel("median test RMSE"); ax.set_xticks(splits)
    ax.set_title("Split-to-split variability of arm medians", fontsize=10); ax.legend(fontsize=9); ax.grid(alpha=0.25)
    _save(fig, "fig15_split_variability")

    # 16: legacy v1 vs controlled v2
    fig, ax = plt.subplots(figsize=(8, 4.4))
    v1 = {"Random": 0.01772, "Evolutionary": 0.01690, "Greedy": 0.01736}
    v2 = {LABEL[a]: np.median(pooled(test_cell, a)) for a in ARMS}
    x = np.arange(3); labels = ["Random", "Evolutionary", "Greedy"]
    ax.bar(x - 0.2, [v1[l] for l in labels], 0.4, label="v1 (1 split, n=5)", color="#999999")
    ax.bar(x + 0.2, [v2[l] for l in labels], 0.4, label="v2 (3 splits, n=10)", color="#3767b3")
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel("median test RMSE")
    ax.set_title("Legacy v1 vs controlled v2 (protected-test medians)", fontsize=10); ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25); _save(fig, "fig16_v1_vs_v2")


def _grouped_bar(ax, named_values, title):
    labels = [n for n, _ in named_values]
    meds = [np.median(v) if v else np.nan for _, v in named_values]
    x = np.arange(len(labels))
    ax.bar(x, meds, 0.62, color="#3767b3", alpha=0.8)
    for i, (_, v) in enumerate(named_values):
        if v:
            ax.scatter([i] * len(v), v, color="black", s=10, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("protected-test RMSE (lower better)"); ax.set_title(title, fontsize=10)
    ax.grid(axis="y", alpha=0.25)


def _manifests(tag, summary, splits, seeds):
    files = sorted(p.name for p in OUT.iterdir() if p.is_file())
    manifest = {"package": "capacity_controlled_t1_v2", "tag": tag, "git_sha": summary.get("git_sha"),
                "splits": splits, "seeds": seeds, "budget": summary["budget"], "arms": ARMS,
                "files": files,
                "reproduction": {
                    "run": f"python scripts/capacity_controlled/run_experiment_v2.py --splits {','.join(map(str,splits))} --seeds {','.join(map(str,seeds))} --budget {summary['budget']} --tag {tag}",
                    "analyze": f"python scripts/capacity_controlled/analyze_v2.py --tag {tag}",
                    "power": "python scripts/capacity_controlled/power_analysis.py"},
                "integrity": "no LLM, no amplitude in primary, validation-only selection, protected test once, uniform 105-param capacity verified, equivalence pre-registered."}
    (OUT / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2))
    (OUT / "artifact_manifest.json").write_text(json.dumps(
        {"figures": [f for f in files if f.startswith("fig") and f.endswith(".png")],
         "tables": [f for f in files if f.endswith(".csv")],
         "analyses": [f for f in files if f.endswith(".json")]}, indent=2))
    cfg = {"space_name": "T1_capacity_controlled_v1 (inherited)", "version": "v2",
           "splits": splits, "seeds": seeds, "budget": summary["budget"],
           "total_trainable_params": S.TOTAL_TRAINABLE_PARAMS, "arms": list(ARMS),
           "equivalence_margin": DELTA, "git_sha": summary.get("git_sha")}
    (OUT / "experiment_config.json").write_text(json.dumps(cfg, indent=2))


if __name__ == "__main__":
    main()
