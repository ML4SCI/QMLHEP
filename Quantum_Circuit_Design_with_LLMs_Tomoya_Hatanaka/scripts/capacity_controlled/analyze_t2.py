#!/usr/bin/env python
"""Analyze the T2-v1 matrix: machine-readable summaries, equivalence + paired
ablation statistics, selection-gain, classification diagnostics, power, and 18
figures. All from the durable per-split stores + run summary. No raw stores are
published.

Usage: python scripts/capacity_controlled/analyze_t2.py --tag pilot
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
from scipy import stats  # noqa: E402

from llm_vqc.evaluation.model import HybridQNNModel  # noqa: E402
from llm_vqc.evaluation.seeds import train_seed_for_circuit  # noqa: E402
from llm_vqc.evaluation.store import ResultStore  # noqa: E402
from llm_vqc.experiments.capacity_controlled import space as S  # noqa: E402
from llm_vqc.experiments.capacity_controlled import t2_task as T2  # noqa: E402
from llm_vqc.ir.budget import ProposalOutcome  # noqa: E402
from llm_vqc.ir.schema import CircuitIR  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "capacity_controlled_t2_v1"
TASK = "T2"
ARMS = ("controlled_random", "controlled_evolutionary", "controlled_greedy")
LABEL = {"controlled_random": "Random", "controlled_evolutionary": "Evolutionary",
         "controlled_greedy": "Greedy"}
COLOR = {"controlled_random": "#3767b3", "controlled_evolutionary": "#c8781e",
         "controlled_greedy": "#8a5fb0"}
CONSUMING = (ProposalOutcome.VALID, ProposalOutcome.DUPLICATE, ProposalOutcome.FAILED)
DELTA = T2.EQUIVALENCE_MARGIN         # 0.03
DELTAS = (0.02, 0.03, 0.05)
CAP = T2.TOTAL_TRAINABLE_PARAMS       # 61
PRIMARY = "balanced_accuracy_error"


def _save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=150, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def cliffs_delta(a, b):
    gt = sum(1 for x, y in itertools.product(a, b) if x > y)
    lt = sum(1 for x, y in itertools.product(a, b) if x < y)
    return (gt - lt) / (len(a) * len(b)) if a and b else 0.0


def cohen_dz(d):
    d = np.asarray(d, float)
    sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd > 0 else 0.0


def tost(d, delta, alpha=0.05):
    d = np.asarray(d, float); n = len(d)
    mean = float(d.mean()); se = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    if se == 0:
        eq = abs(mean) < delta
        return {"mean_diff": mean, "ci90": [mean, mean], "tost_p": 0.0 if eq else 1.0, "equivalent": bool(eq)}
    df = n - 1
    p = max(float(stats.t.cdf((mean - delta) / se, df)), float(stats.t.sf((mean + delta) / se, df)))
    tcrit = float(stats.t.ppf(0.95, df))
    ci = [mean - tcrit * se, mean + tcrit * se]
    return {"mean_diff": mean, "ci90": ci, "tost_p": p, "equivalent": bool(ci[0] > -delta and ci[1] < delta)}


def blocked_bootstrap(cells_a, cells_b, split_of, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    by = {}
    for k in cells_a:
        by.setdefault(split_of[k], []).append(k)
    meds = []
    for _ in range(n):
        rs = []
        for ks in by.values():
            idx = rng.integers(0, len(ks), len(ks)); rs += [ks[i] for i in idx]
        meds.append(float(np.median([cells_a[k] - cells_b[k] for k in rs])))
    return [float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))]


def splitmean_tost(cells_a, cells_b, splits, seeds, delta):
    means = [np.mean([cells_a[(s, sd)] - cells_b[(s, sd)] for sd in seeds
                      if (s, sd) in cells_a and (s, sd) in cells_b]) for s in splits]
    return tost([m for m in means if not np.isnan(m)], delta)


def anytime(store, run_id, seed):
    best, seen, cum, npr, nuq = None, set(), 0.0, 0, 0
    px, ux, tx, ys, order = [], [], [], [], []
    for e in store.iter_proposal_events(run_id):
        if e.outcome in CONSUMING:
            npr += 1
            if e.structural_hash:
                ts = train_seed_for_circuit(seed, e.structural_hash)
                c = store.get_cached(TASK, e.structural_hash, ts)
                if e.structural_hash not in seen:
                    seen.add(e.structural_hash); nuq += 1
                    if c is not None:
                        cum += float(c.wall_clock_seconds or 0.0)
                if c and c.val_metric_value is not None and (best is None or c.val_metric_value < best):
                    best = c.val_metric_value
                order.append((e.structural_hash, best))
            px.append(npr); ux.append(nuq); tx.append(cum); ys.append(best)
    return {"proposals": px, "unique": ux, "walltime": tx, "best_val": ys, "order": order}


def best_of_B(store, run_id, seed, test, B):
    """Selected circuit's protected-test primary metric using best-of-first-B val."""
    best_v, best_h = None, None
    consumed = 0
    for e in store.iter_proposal_events(run_id):
        if e.outcome in CONSUMING:
            consumed += 1
            if e.structural_hash:
                ts = train_seed_for_circuit(seed, e.structural_hash)
                c = store.get_cached(TASK, e.structural_hash, ts)
                if c and c.val_metric_value is not None and (best_v is None or c.val_metric_value < best_v):
                    best_v, best_h = c.val_metric_value, e.structural_hash
            if consumed >= B:
                break
    if best_h is None:
        return None
    ts = train_seed_for_circuit(seed, best_h)
    c = store.get_cached(TASK, best_h, ts); w = store.get_trained_weights(TASK, best_h, ts)
    ir = CircuitIR.model_validate_json(c.circuit_canonical_json)
    md = T2.evaluate_all_test_metrics(ir, w["classical_state"], test)
    return md[PRIMARY]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="pilot")
    ap.add_argument("--run-dir", default=str(ROOT / "runs" / "capacity_controlled_t2_v1"))
    args = ap.parse_args()
    run_dir = Path(args.run_dir); OUT.mkdir(parents=True, exist_ok=True)
    summary = json.loads((run_dir / f"{args.tag}_summary.json").read_text())
    splits, seeds, abl_seeds = summary["splits"], summary["seeds"], summary["abl_seeds"]
    stores = {s: ResultStore(run_dir / f"split_{s}" / "results.sqlite") for s in splits}
    tests = {s: T2.build_t2_test(s) for s in splits}

    # candidate_results + capacity verify
    cand, capset = [], set()
    for s in splits:
        for e in stores[s].all_evaluations(TASK):
            if not e.circuit_canonical_json or not e.circuit_cost:
                continue
            ir = CircuitIR.model_validate_json(e.circuit_canonical_json)
            m = HybridQNNModel(ir, raw_feature_dim=T2.RAW_FEATURE_DIM, head_out_dim=1)
            total = int(sum(p.numel() for p in m.parameters())); capset.add(total)
            c = e.circuit_cost
            cand.append({"split": s, "structural_hash": (e.structural_hash or "")[:12],
                         "depth": c.depth, "gate_count": c.gate_count,
                         "two_qubit_gate_count": c.two_qubit_gate_count,
                         "total_trainable_params": total, "val_logloss": e.val_metric_value,
                         "wall_clock_seconds": round(e.wall_clock_seconds, 4)})
    if capset != {CAP}:
        raise RuntimeError(f"T2 CAPACITY NOT UNIFORM: {sorted(capset)} != {{{CAP}}}")
    print(f"[verify] all {len(cand)} candidates have exactly {CAP} params")
    _csv(OUT / "candidate_results.csv", cand)

    # run_summary + resource + anytime + test cells
    run_rows, res_rows, curves = [], [], {}
    test_cell = {a: {} for a in ARMS}; val_cell = {a: {} for a in ARMS}; split_of = {}
    conf_by_arm = {a: {"tn": 0, "fp": 0, "fn": 0, "tp": 0} for a in ARMS}
    for r in summary["runs"]:
        a, s, seed = r["arm"], r["split"], r["seed"]; cell = (s, seed); split_of[cell] = s
        t = r["test"]
        test_cell[a][cell] = t[PRIMARY] if t else None
        val_cell[a][cell] = r["selected_val_logloss"]
        led = r["ledger_summary"]
        if t:
            for k in conf_by_arm[a]:
                conf_by_arm[a][k] += t["confusion"][k]
        run_rows.append({"run_id": r["run_id"], "arm": a, "split": s, "seed": seed,
                         "consumed_budget": led["consumed_budget"], "num_invalid": led["num_invalid"],
                         "num_duplicate": led["num_duplicate"], "num_failed": led["num_failed"],
                         "num_unique": led["num_unique"], "selected_val_logloss": r["selected_val_logloss"],
                         "test_balacc_error": t[PRIMARY] if t else None,
                         "test_auc": t["auc"] if t else None, "test_accuracy": t["accuracy"] if t else None,
                         "test_logloss": t["logloss"] if t else None, "test_brier": t["brier"] if t else None})
        cur = anytime(stores[s], r["run_id"], seed); curves[r["run_id"]] = cur
        res_rows.append({"run_id": r["run_id"], "arm": a, "split": s, "seed": seed,
                         "proposals_consumed": led["consumed_budget"], "unique_trainings": led["num_unique"],
                         "duplicates": led["num_duplicate"], "invalid": led["num_invalid"],
                         "failed": led["num_failed"],
                         "cumulative_walltime_s": round(cur["walltime"][-1], 3) if cur["walltime"] else 0.0})
    _csv(OUT / "run_summary.csv", run_rows); (OUT / "run_summary.json").write_text(json.dumps(run_rows, indent=2))
    _csv(OUT / "resource_accounting.csv", res_rows)

    # split_summary
    split_rows = []
    for s in splits:
        for a in ARMS:
            tv = [test_cell[a][(s, sd)] for sd in seeds if test_cell[a].get((s, sd)) is not None]
            split_rows.append({"split": s, "arm": a, "n": len(tv), "test_bae_median": float(np.median(tv)),
                               "test_bae_iqr": float(np.percentile(tv, 75) - np.percentile(tv, 25))})
    _csv(OUT / "split_summary.csv", split_rows)

    # aux: classical, paired ablations
    aux = summary["aux"]
    cls_rows, c6_rows, ft_rows, pe_rows = [], [], [], []
    for s in splits:
        for seed in seeds:
            for b in aux[str(s)]["classical"][str(seed)]:
                base = {"split": s, "seed": seed, "baseline": b["baseline"],
                        "total_params": b.get("total_params"),
                        "balanced_accuracy_error": b["balanced_accuracy_error"],
                        "accuracy": b["accuracy"], "auc": b["auc"], "logloss": b["logloss"]}
                cls_rows.append(base)
                if b["baseline"] == "C6_fair_nas":
                    c6_rows.append({"split": s, "seed": seed, "selected": json.dumps(b["selected_config"]),
                                    "total_params": b["total_params"], "val_bae": b.get("val_bae"),
                                    "test_bae": b["balanced_accuracy_error"], "n_candidates": b.get("n_candidates")})
        for row in aux[str(s)]["freeze_train"]:
            ft_rows.append({**row, "diff_frozen_minus_trainable": row["frozen_bae"] - row["trainable_bae"]})
        for row in aux[str(s)]["product_entangled"]:
            pe_rows.append({**row, "diff_product_minus_entangled": row["product_bae"] - row["entangled_bae"]})
    _csv(OUT / "classical_baseline_summary.csv", cls_rows)
    _csv(OUT / "classical_search_summary.csv", c6_rows)
    _csv(OUT / "paired_freeze_train_results.csv", ft_rows)
    _csv(OUT / "paired_entanglement_results.csv", pe_rows)

    # combined quantum-ablation summary (per-condition test bal-acc-error aggregates)
    def _cond_stats(vals):
        v = [x for x in vals if x is not None]
        return {"n": len(v), "median": float(np.median(v)), "mean": float(np.mean(v)),
                "sd": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0}
    qa_rows = [
        {"condition": "QT_trainable", "metric": "balanced_accuracy_error",
         **_cond_stats([r["trainable_bae"] for r in ft_rows])},
        {"condition": "QF_frozen", "metric": "balanced_accuracy_error",
         **_cond_stats([r["frozen_bae"] for r in ft_rows])},
        {"condition": "QE_entangled", "metric": "balanced_accuracy_error",
         **_cond_stats([r["entangled_bae"] for r in pe_rows])},
        {"condition": "QP_product", "metric": "balanced_accuracy_error",
         **_cond_stats([r["product_bae"] for r in pe_rows])},
        {"condition": "QS_searched_best_arm", "metric": "balanced_accuracy_error",
         **_cond_stats([test_cell[min(ARMS, key=lambda a: np.median([x for x in test_cell[a].values() if x is not None]))][c]
                        for c in test_cell["controlled_random"]])},
    ]
    _csv(OUT / "quantum_ablation_summary.csv", qa_rows)

    # selection gain
    sel_rows = []
    fixed_dist = [r["trainable_bae"] for r in ft_rows if r["trainable_bae"] is not None]
    for r in summary["runs"]:
        if r["arm"] != "controlled_random":
            continue
        s, seed = r["split"], r["seed"]
        for B in (1, 5, 10, 25):
            v = best_of_B(stores[s], r["run_id"], seed, tests[s], B)
            sel_rows.append({"split": s, "seed": seed, "B": B, "selected_test_bae": v})
    _csv(OUT / "selection_gain_summary.csv", sel_rows)
    for st in stores.values():
        st.close()

    # ---- statistics ----
    def pooled(cm, a):
        return [cm[a][(s, sd)] for s in splits for sd in seeds if cm[a].get((s, sd)) is not None]

    def bootci(v, seed=0, n=2000):
        rng = np.random.default_rng(seed); v = np.asarray(v, float)
        return [float(np.percentile([np.median(rng.choice(v, len(v), True)) for _ in range(n)], p)) for p in (2.5, 97.5)]

    stat = {"note": f"T2-v1 {len(splits)} splits x {len(seeds)} seeds; primary test metric "
                    f"{PRIMARY} (lower better); selection on val logloss; every candidate {CAP} params.",
            "per_arm_test": {}, "per_arm_val_logloss": {}}
    for a in ARMS:
        tv = pooled(test_cell, a); vv = pooled(val_cell, a)
        stat["per_arm_test"][a] = {"n": len(tv), "median": float(np.median(tv)),
                                   "iqr": float(np.percentile(tv, 75) - np.percentile(tv, 25)),
                                   "mean": float(np.mean(tv)), "sd": float(np.std(tv, ddof=1)),
                                   "boot_ci95_median": bootci(tv)}
        stat["per_arm_val_logloss"][a] = {"median": float(np.median(vv))}
    groups = [pooled(test_cell, a) for a in ARMS]
    H, p = stats.kruskal(*groups)
    stat["kruskal_wallis_test"] = {"H": float(H), "p_value": float(p)}
    pw, raw = [], []
    for a, b in itertools.combinations(ARMS, 2):
        u, pu = stats.mannwhitneyu(pooled(test_cell, a), pooled(test_cell, b), alternative="two-sided")
        pw.append({"a": a, "b": b, "p_raw": float(pu), "cliffs_delta": cliffs_delta(pooled(test_cell, a), pooled(test_cell, b))}); raw.append(pu)
    order = np.argsort(raw); m = len(raw); holm = [0.0] * m; run = 0.0
    for rank, idx in enumerate(order):
        run = max(run, (m - rank) * raw[idx]); holm[idx] = min(1.0, run)
    for e, ph in zip(pw, holm, strict=True):
        e["p_holm"] = float(ph)
    stat["pairwise_test"] = pw

    # ablations paired stats
    ft_diff = [r["diff_frozen_minus_trainable"] for r in ft_rows if r["trainable_bae"] is not None and r["frozen_bae"] is not None]
    pe_diff = [r["diff_product_minus_entangled"] for r in pe_rows if r["product_bae"] is not None and r["entangled_bae"] is not None]
    def paired_block(d, label):
        d = [x for x in d if x is not None]
        w = stats.wilcoxon(d) if len(d) > 0 and any(x != 0 for x in d) else None
        return {"n": len(d), "mean_diff": float(np.mean(d)), "median_diff": float(np.median(d)),
                "ci95_mean": [float(np.mean(d) - 1.96 * np.std(d, ddof=1) / np.sqrt(len(d))),
                              float(np.mean(d) + 1.96 * np.std(d, ddof=1) / np.sqrt(len(d)))],
                "cohen_dz": cohen_dz(d), "wilcoxon_p": float(w.pvalue) if w else None,
                "frac_positive": float(np.mean(np.asarray(d) > 0)),
                "interpretation": label}
    stat["ablation_freeze_minus_train"] = paired_block(ft_diff, "positive => trainable better than frozen")
    stat["ablation_product_minus_entangled"] = paired_block(pe_diff, "positive => entangled better than product")

    # baselines aggregate
    def base_med(name):
        v = [r["balanced_accuracy_error"] for r in cls_rows if r["baseline"] == name]
        return float(np.median(v)) if v else None
    stat["baselines_test_bae_median"] = {n: base_med(n) for n in
        ("C0_trivial", "C1_logreg", "C2_linear_svm", "C3_rbf_svm", "C4_trees", "C5_fixed_mlp", "C6_fair_nas")}
    stat["baselines_test_bae_median"]["QS_searched_best_arm"] = min(stat["per_arm_test"][a]["median"] for a in ARMS)
    # selection gain summary
    def sel_med(B):
        v = [r["selected_test_bae"] for r in sel_rows if r["B"] == B and r["selected_test_bae"] is not None]
        return float(np.median(v)) if v else None
    stat["selection_gain"] = {"fixed_arch_expected_test_bae": float(np.median(fixed_dist)) if fixed_dist else None,
                              "best_of_B_median": {str(B): sel_med(B) for B in (1, 5, 10, 25)}}
    (OUT / "statistical_analysis.json").write_text(json.dumps(stat, indent=2))

    # ---- equivalence ----
    equiv = {"primary_metric": PRIMARY, "margin_primary": DELTA, "pairs": {}}
    for a, b in itertools.combinations(ARMS, 2):
        common = [c for c in test_cell[a] if c in test_cell[b] and test_cell[a][c] is not None and test_cell[b][c] is not None]
        d = [test_cell[a][c] - test_cell[b][c] for c in common]
        equiv["pairs"][f"{a}__vs__{b}"] = {
            "n_cells": len(common), "paired_diff_sd": float(np.std(d, ddof=1)),
            "tost_by_delta": {f"{dl}": tost(d, dl) for dl in DELTAS},
            "blocked_bootstrap_ci95_median_diff": blocked_bootstrap({c: test_cell[a][c] for c in common},
                                                                    {c: test_cell[b][c] for c in common}, split_of),
            "splitmean_cluster_tost_0.03": splitmean_tost(test_cell[a], test_cell[b], splits, seeds, DELTA)}
    equiv["all_pairs_equivalent_at_primary"] = bool(all(
        equiv["pairs"][k]["tost_by_delta"][f"{DELTA}"]["equivalent"] for k in equiv["pairs"]))
    (OUT / "equivalence_analysis.json").write_text(json.dumps(equiv, indent=2))

    # ---- power (observed SD) ----
    obs_sd = float(np.median([equiv["pairs"][k]["paired_diff_sd"] for k in equiv["pairs"]]))
    _power(obs_sd, DELTA, len(splits) * len(seeds))

    # ---- figures ----
    _figures(summary, splits, seeds, test_cell, val_cell, split_rows, curves, equiv, stat,
             cls_rows, ft_rows, pe_rows, sel_rows, fixed_dist, cand, run_rows, conf_by_arm,
             run_dir, stores_reopen=run_dir)
    _manifests(args.tag, summary, splits, seeds)
    print(f"wrote T2-v1 machine-readable results + 18 figures to {OUT}")


def _csv(path, rows):
    if not rows:
        path.write_text(""); return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore"); w.writeheader(); w.writerows(rows)


def _power(sd, delta, n, out=OUT):
    rng = np.random.default_rng(1); res = {}
    for true in (0.0, 0.01, 0.02, 0.03):
        eq = det = 0
        for _ in range(10000):
            d = rng.normal(true, sd, n)
            m = d.mean(); se = d.std(ddof=1) / np.sqrt(n); df = n - 1
            if max(stats.t.cdf((m - delta) / se, df), stats.t.sf((m + delta) / se, df)) < 0.05:
                eq += 1
            if stats.ttest_1samp(d, 0.0).pvalue < 0.05:
                det += 1
        res[f"{true:.2f}"] = {"P_equivalence": eq / 10000, "P_detect": det / 10000}
    (out / "power_analysis.json").write_text(json.dumps(
        {"note": "post-run power using observed paired-diff SD", "observed_paired_diff_sd": sd,
         "n_cells": n, "margin": delta, "scenarios": res}, indent=2))


def _figures(summary, splits, seeds, test_cell, val_cell, split_rows, curves, equiv, stat,
             cls_rows, ft_rows, pe_rows, sel_rows, fixed_dist, cand, run_rows, conf_by_arm, run_dir, stores_reopen):
    def pooled(cm, a):
        return [cm[a][(s, sd)] for s in splits for sd in seeds if cm[a].get((s, sd)) is not None]

    # 1 & 2 by arm & split
    for cm, title, fn, yl in ((test_cell, "Protected-test balanced-accuracy error", "fig01_test_bae_by_arm_split", "test bal-acc error"),
                              (val_cell, "Validation log-loss (selection metric)", "fig02_val_logloss_by_arm_split", "val log-loss")):
        fig, ax = plt.subplots(figsize=(9, 4.6)); x = np.arange(len(ARMS)); wd = 0.15
        for j, s in enumerate(splits):
            for i, a in enumerate(ARMS):
                v = [cm[a][(s, sd)] for sd in seeds if cm[a].get((s, sd)) is not None]
                ax.bar(x[i] + (j - 2) * wd, np.median(v), wd, color=COLOR[a], alpha=0.35 + 0.12 * j,
                       edgecolor="black", lw=0.3, label=f"split {s}" if i == 0 else None)
        ax.set_xticks(x); ax.set_xticklabels([LABEL[a] for a in ARMS]); ax.set_ylabel(yl)
        ax.set_title(f"{title} by arm & split (median, n={len(seeds)}/split)", fontsize=10)
        ax.legend(fontsize=7, ncol=5); ax.grid(axis="y", alpha=0.25); _save(fig, fn)

    # 3 equivalence intervals
    fig, ax = plt.subplots(figsize=(8.5, 4.0)); labs, mn, lo, hi = [], [], [], []
    for k, pr in equiv["pairs"].items():
        t = pr["tost_by_delta"][f"{DELTA}"]; labs.append(k.replace("controlled_", "").replace("__vs__", " vs "))
        mn.append(t["mean_diff"]); lo.append(t["ci90"][0]); hi.append(t["ci90"][1])
    y = np.arange(len(labs))
    ax.errorbar(mn, y, xerr=[np.array(mn) - np.array(lo), np.array(hi) - np.array(mn)], fmt="o", color="#3767b3", capsize=4)
    ax.axvline(0, color="grey"); ax.axvline(-DELTA, color="crimson", ls="--"); ax.axvline(DELTA, color="crimson", ls="--")
    ax.axvspan(-DELTA, DELTA, color="green", alpha=0.08); ax.set_yticks(y); ax.set_yticklabels(labs, fontsize=9)
    ax.set_xlabel("mean paired test bal-acc-error difference (90% CI)")
    ax.set_title(f"Equivalence: 90% CIs vs margin ±{DELTA}", fontsize=10); _save(fig, "fig03_equivalence_intervals")

    # 4,5,6 anytime
    for axis, fn, xl in (("proposals", "fig04_anytime_proposals", "proposals"),
                         ("unique", "fig05_anytime_unique_trainings", "unique trainings"),
                         ("walltime", "fig06_anytime_walltime", "cumulative wall-clock (s)")):
        fig, ax = plt.subplots(figsize=(8.5, 4.6))
        for a in ARMS:
            for rid, cur in curves.items():
                if not rid.startswith(a):
                    continue
                xs = [x for x, y in zip(cur[axis], cur["best_val"]) if y is not None]
                ys = [y for y in cur["best_val"] if y is not None]
                if xs:
                    ax.plot(xs, ys, color=COLOR[a], alpha=0.25, lw=0.7)
        for a in ARMS:
            ax.plot([], [], color=COLOR[a], label=LABEL[a])
        ax.set_xlabel(xl); ax.set_ylabel("best-so-far val log-loss"); ax.legend(fontsize=9)
        ax.set_title(f"Anytime: best val log-loss vs {xl}", fontsize=10); ax.grid(alpha=0.25); _save(fig, fn)

    # 7 classical baselines
    fig, ax = plt.subplots(figsize=(9, 4.6))
    names = [("QS best arm", pooled(test_cell, min(ARMS, key=lambda a: np.median(pooled(test_cell, a)))))]
    for n in ("C0_trivial", "C1_logreg", "C2_linear_svm", "C3_rbf_svm", "C4_trees", "C5_fixed_mlp", "C6_fair_nas"):
        names.append((n.replace("_", " "), [r["balanced_accuracy_error"] for r in cls_rows if r["baseline"] == n]))
    _grouped(ax, names, "Searched VQC vs classical baselines (test bal-acc error)"); _save(fig, "fig07_classical_baselines")

    # 8 fair classical search vs VQC search
    fig, ax = plt.subplots(figsize=(7, 4.4))
    names = [(LABEL[a], pooled(test_cell, a)) for a in ARMS]
    names.append(("C6 fair NAS", [r["balanced_accuracy_error"] for r in cls_rows if r["baseline"] == "C6_fair_nas"]))
    _grouped(ax, names, "VQC search arms vs fair classical NAS (C6)"); _save(fig, "fig08_vqc_vs_fair_classical")

    # 9 frozen vs trainable paired diffs
    _paired_hist("fig09_freeze_vs_train", [r["diff_frozen_minus_trainable"] for r in ft_rows if r["trainable_bae"] is not None and r["frozen_bae"] is not None],
                 "frozen − trainable bal-acc-error (paired)\npositive ⇒ training quantum angles helps")
    # 10 product vs entangled paired diffs
    _paired_hist("fig10_product_vs_entangled", [r["diff_product_minus_entangled"] for r in pe_rows if r["product_bae"] is not None and r["entangled_bae"] is not None],
                 "product − entangled bal-acc-error (paired)\npositive ⇒ entanglement helps")

    # 11 selection gain
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    Bs = (1, 5, 10, 25); meds = []
    for B in Bs:
        v = [r["selected_test_bae"] for r in sel_rows if r["B"] == B and r["selected_test_bae"] is not None]
        meds.append(np.median(v)); ax.scatter([B] * len(v), v, color="#3767b3", alpha=0.3, s=12)
    ax.plot(Bs, meds, "-o", color="#3767b3", label="best-of-B (median)")
    if fixed_dist:
        ax.axhline(np.median(fixed_dist), color="crimson", ls="--", label="expected fixed arch (median)")
    ax.set_xlabel("selection budget B"); ax.set_ylabel("protected-test bal-acc error")
    ax.set_title("Validation-selection gain: best-of-B vs fixed architecture", fontsize=10)
    ax.legend(fontsize=9); ax.grid(alpha=0.25); _save(fig, "fig11_selection_gain")

    # 12 capacity verification
    fig, ax = plt.subplots(figsize=(7.5, 4.0)); tot = [c["total_trainable_params"] for c in cand]
    ax.plot(range(len(tot)), tot, ".", color="#4a9e5c", ms=3); ax.axhline(CAP, color="crimson", ls="--")
    ax.set_ylim(CAP - 20, CAP + 20); ax.set_xlabel("candidate index"); ax.set_ylabel("total trainable params")
    ax.set_title(f"Capacity verification: all {len(tot)} candidates = {CAP} (unique {sorted(set(tot))})", fontsize=10)
    _save(fig, "fig12_capacity_verification")

    # 13 rates
    fig, ax = plt.subplots(figsize=(7.5, 4.4)); x = np.arange(len(ARMS)); wd = 0.25
    for j, (key, lab) in enumerate((("num_invalid", "invalid"), ("num_duplicate", "duplicate"), ("num_failed", "failed"))):
        rates = [sum(r[key] for r in run_rows if r["arm"] == a) / max(1, sum(r["consumed_budget"] + r["num_invalid"] for r in run_rows if r["arm"] == a)) for a in ARMS]
        ax.bar(x + (j - 1) * wd, rates, wd, label=lab)
    ax.set_xticks(x); ax.set_xticklabels([LABEL[a] for a in ARMS]); ax.legend(fontsize=9)
    ax.set_ylabel("fraction of proposals"); ax.set_title("Invalid / duplicate / failed rates by arm", fontsize=10)
    ax.grid(axis="y", alpha=0.25); _save(fig, "fig13_proposal_rates")

    # 14 runtime
    fig, ax = plt.subplots(figsize=(7.5, 4.0)); rt = [c["wall_clock_seconds"] for c in cand if c["wall_clock_seconds"]]
    ax.hist(rt, bins=40, color="#3767b3", alpha=0.8); ax.set_xlabel("training s per candidate"); ax.set_ylabel("count")
    ax.set_title(f"Runtime per candidate (n={len(rt)}, median={np.median(rt):.2f}s)", fontsize=10); _save(fig, "fig14_runtime")

    # 15 confusion matrices (aggregate per arm)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    for ax, a in zip(axes, ARMS):
        c = conf_by_arm[a]; mat = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
        ax.imshow(mat, cmap="Blues"); ax.set_title(LABEL[a], fontsize=10)
        for (i, j), v in np.ndenumerate(mat):
            ax.text(j, i, str(v), ha="center", va="center", fontsize=11)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["pred 3", "pred 8"]); ax.set_yticks([0, 1]); ax.set_yticklabels(["true 3", "true 8"])
    fig.suptitle("Aggregate protected-test confusion matrices (all splits/seeds)", fontsize=11)
    fig.tight_layout(); _save(fig, "fig15_confusion_matrices")

    # 16 ROC + 17 calibration for the best run
    best = min((r for r in summary["runs"] if r["test"] is not None), key=lambda r: r["test"][PRIMARY])
    bs, bseed = best["split"], best["seed"]
    st = ResultStore(run_dir / f"split_{bs}" / "results.sqlite")
    h = best["selected_structural_hash"]; ts = train_seed_for_circuit(bseed, h)
    c = st.get_cached(TASK, h, ts); w = st.get_trained_weights(TASK, h, ts); st.close()
    ir = CircuitIR.model_validate_json(c.circuit_canonical_json)
    md = T2.evaluate_all_test_metrics(ir, w["classical_state"], T2.build_t2_test(bs), with_curves=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(md["roc"]["fpr"], md["roc"]["tpr"], "-o", color="#3767b3", ms=3); ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title(f"ROC — best run {best['run_id']} (AUC={md['auc']:.3f})", fontsize=9)
    ax.grid(alpha=0.25); _save(fig, "fig16_roc")
    fig, ax = plt.subplots(figsize=(5, 5))
    cal = md["calibration"]
    ax.plot([c["conf"] for c in cal], [c["acc"] for c in cal], "-o", color="#c8781e"); ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set_xlabel("mean predicted prob"); ax.set_ylabel("empirical accuracy"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title(f"Calibration — best run {best['run_id']} (Brier={md['brier']:.3f})", fontsize=9); ax.grid(alpha=0.25); _save(fig, "fig17_calibration")

    # 18 split variability
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for a in ARMS:
        meds = [np.median([test_cell[a][(s, sd)] for sd in seeds if test_cell[a].get((s, sd)) is not None]) for s in splits]
        ax.plot(splits, meds, "-o", color=COLOR[a], label=LABEL[a])
    ax.set_xlabel("data split seed"); ax.set_ylabel("median test bal-acc error"); ax.set_xticks(splits)
    ax.set_title("Split-to-split variability", fontsize=10); ax.legend(fontsize=9); ax.grid(alpha=0.25); _save(fig, "fig18_split_variability")


def _grouped(ax, named, title):
    labs = [n for n, _ in named]; meds = [np.median(v) if v else np.nan for _, v in named]
    x = np.arange(len(labs)); ax.bar(x, meds, 0.62, color="#3767b3", alpha=0.8)
    for i, (_, v) in enumerate(named):
        if v:
            ax.scatter([i] * len(v), v, color="black", s=8, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(labs, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("test bal-acc error (lower better)"); ax.set_title(title, fontsize=10); ax.grid(axis="y", alpha=0.25)


def _paired_hist(fn, diffs, title):
    diffs = [d for d in diffs if d is not None]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.hist(diffs, bins=30, color="#3767b3", alpha=0.8)
    ax.axvline(0, color="crimson", ls="--", lw=1.5)
    ax.axvline(np.median(diffs), color="black", lw=1.5, label=f"median={np.median(diffs):+.4f}")
    ax.set_xlabel("paired difference"); ax.set_ylabel("count")
    ax.set_title(f"{title}\nn={len(diffs)}, frac>0={np.mean(np.asarray(diffs) > 0):.2f}", fontsize=9)
    ax.legend(fontsize=9); _save(fig, fn)


def _manifests(tag, summary, splits, seeds):
    files = sorted(p.name for p in OUT.iterdir() if p.is_file())
    (OUT / "experiment_manifest.json").write_text(json.dumps(
        {"package": "capacity_controlled_t2_v1", "tag": tag, "git_sha": summary.get("git_sha"),
         "splits": splits, "seeds": seeds, "budget": summary["budget"], "arms": ARMS, "files": files,
         "reproduction": {"run": f"python scripts/capacity_controlled/run_experiment_t2.py --splits {','.join(map(str, splits))} --seeds {','.join(map(str, seeds))} --budget {summary['budget']} --tag {tag}",
                          "analyze": f"python scripts/capacity_controlled/analyze_t2.py --tag {tag}"},
         "integrity": "no LLM, no amplitude in primary, selection on val logloss, protected test once, uniform 61-param capacity verified, equivalence pre-registered."}, indent=2))
    (OUT / "artifact_manifest.json").write_text(json.dumps(
        {"figures": [f for f in files if f.startswith("fig") and f.endswith(".png")],
         "tables": [f for f in files if f.endswith(".csv")], "analyses": [f for f in files if f.endswith(".json")]}, indent=2))
    (OUT / "experiment_config.json").write_text(json.dumps(
        {"space": "T2_capacity_controlled_v1", "total_trainable_params": CAP, "splits": splits, "seeds": seeds,
         "budget": summary["budget"], "arms": list(ARMS), "selection_metric": "logloss",
         "primary_test_metric": PRIMARY, "equivalence_margin": DELTA, "git_sha": summary.get("git_sha")}, indent=2))


if __name__ == "__main__":
    main()
