#!/usr/bin/env python
"""Stage 8 statistical analysis of the pilot experiment
(`scripts/pilot_experiment.py`'s stored results).

Reads ONLY from the stored SQLite result store (never re-derives numbers
by hand) -- reproducible-from-raw-results, per the work brief's "Do not
manually copy values into figures. Generate figures and tables from
stored raw results through reproducible scripts."

Computes, per LLM-VQC_MASTER_PLAN.md Section 6.3:
- anytime best-so-far validation-metric curves at checkpoints 10 and 25
  (25 is this pilot's full budget; the master plan's 10/25/60 checkpoint
  set is for the B=60 main matrix, not this B=25 pilot -- 60 does not
  exist here, so it is correctly omitted, not silently dropped);
- per (arm, checkpoint) median/IQR across the 3 repetition seeds;
- Kruskal-Wallis across arms at each checkpoint, then pairwise
  Mann-Whitney U with Holm correction if Kruskal-Wallis is significant;
- Cliff's delta effect sizes for every pairwise comparison;
- duplicate/invalid/failed rates per arm;
- final (budget=25) selected validation and TEST metrics per run.

With n=3 seeds per arm, this is manifestly an underpowered pilot for
confirmatory inference -- results are reported as descriptive statistics
and effect sizes, and the script explicitly avoids declaring
significance conclusions that the sample size cannot support (Section
6.3: "With n = 5-10 per cell, only report pairwise conclusions when
effect sizes are large -- otherwise report estimates with CIs and say
so"; n=3 here is smaller still).
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy import stats

from llm_vqc.evaluation.seeds import train_seed_for_circuit  # noqa: E402
from llm_vqc.evaluation.store import ResultStore  # noqa: E402
from llm_vqc.ir.budget import ProposalOutcome  # noqa: E402

RUN_DIR = Path(__file__).resolve().parent.parent / "runs" / "pilot_t1"
DB_PATH = RUN_DIR / "results.sqlite"
SUMMARY_PATH = RUN_DIR / "pilot_summary.json"
ANALYSIS_PATH = RUN_DIR / "pilot_analysis.json"

CHECKPOINTS = (10, 25)
LOWER_IS_BETTER = True  # T1 metric is RMSE


def cliffs_delta(a: list[float], b: list[float]) -> float:
    """Cliff's delta effect size: fraction of pairs where a > b minus the
    fraction where a < b, in [-1, 1]. No scipy builtin exists for this."""
    greater = sum(1 for x, y in itertools.product(a, b) if x > y)
    less = sum(1 for x, y in itertools.product(a, b) if x < y)
    return (greater - less) / (len(a) * len(b))


def anytime_best_so_far(
    store: ResultStore, run_id: str, task_name: str, run_seed: int
) -> list[float | None]:
    """Best validation metric seen after each of the first `max(CHECKPOINTS)`
    proposals in `run_id`, reconstructed purely from stored proposal
    events + the evaluations cache -- never recomputed by hand."""
    best = None
    curve: list[float | None] = []
    budget_consuming = (ProposalOutcome.VALID, ProposalOutcome.DUPLICATE, ProposalOutcome.FAILED)
    for record in store.iter_proposal_events(run_id):
        if record.outcome in budget_consuming:
            if record.structural_hash is not None:
                train_seed = train_seed_for_circuit(run_seed, record.structural_hash)
                cached = store.get_cached(task_name, record.structural_hash, train_seed)
                if cached is not None and cached.val_metric_value is not None:
                    value = cached.val_metric_value
                    if best is None or (value < best if LOWER_IS_BETTER else value > best):
                        best = value
        curve.append(best)
        if len(curve) >= max(CHECKPOINTS):
            break
    return curve


def main() -> None:
    summary = json.loads(SUMMARY_PATH.read_text())
    arms = summary["arms"]
    seeds = summary["seeds"]
    task_name = summary["task"]

    store = ResultStore(DB_PATH)

    curves: dict[str, dict[int, list[float]]] = {arm: {} for arm in arms}
    ledger_summaries: dict[str, list[dict]] = {arm: [] for arm in arms}

    for arm in arms:
        for seed in seeds:
            run_id = f"{arm}_s{seed}"
            curve = anytime_best_so_far(store, run_id, task_name, seed)
            for checkpoint in CHECKPOINTS:
                if checkpoint <= len(curve) and curve[checkpoint - 1] is not None:
                    curves[arm].setdefault(checkpoint, []).append(curve[checkpoint - 1])
            ledger_summaries[arm].append(store.count_proposal_events(run_id))

    print("=" * 70)
    print("STAGE 8 PILOT ANALYSIS")
    print("n=3 seeds per arm: DESCRIPTIVE / EXPLORATORY ONLY, not confirmatory.")
    print("=" * 70)

    analysis: dict = {"checkpoints": {}}

    for checkpoint in CHECKPOINTS:
        print(f"\n--- checkpoint {checkpoint} (best-so-far validation RMSE, lower=better) ---")
        per_arm_values = {}
        for arm in arms:
            values = curves[arm].get(checkpoint, [])
            per_arm_values[arm] = values
            if values:
                median = float(np.median(values))
                iqr = float(np.percentile(values, 75) - np.percentile(values, 25))
                print(
                    f"  {arm:14s} n={len(values)} median={median:.5f} "
                    f"IQR={iqr:.5f} values={values}"
                )
            else:
                print(f"  {arm:14s} n=0 (no run reached this checkpoint)")

        checkpoint_result: dict = {"per_arm_values": per_arm_values}

        complete_arms = [a for a in arms if len(per_arm_values.get(a, [])) == len(seeds)]
        if len(complete_arms) >= 2:
            groups = [per_arm_values[a] for a in complete_arms]
            kw_stat, kw_p = stats.kruskal(*groups)
            print(f"  Kruskal-Wallis across {complete_arms}: H={kw_stat:.4f} p={kw_p:.4f}")
            checkpoint_result["kruskal_wallis"] = {
                "statistic": float(kw_stat),
                "p_value": float(kw_p),
            }

            pairwise = []
            pairs = list(itertools.combinations(complete_arms, 2))
            raw_p = []
            for a, b in pairs:
                u_stat, u_p = stats.mannwhitneyu(
                    per_arm_values[a], per_arm_values[b], alternative="two-sided"
                )
                delta = cliffs_delta(per_arm_values[a], per_arm_values[b])
                raw_p.append(u_p)
                pairwise.append(
                    {
                        "a": a,
                        "b": b,
                        "u_statistic": float(u_stat),
                        "p_raw": float(u_p),
                        "cliffs_delta": float(delta),
                    }
                )

            # Holm-Bonferroni correction across the pairwise family.
            order = np.argsort(raw_p)
            m = len(raw_p)
            holm_p = [0.0] * m
            running_max = 0.0
            for rank, idx in enumerate(order):
                adjusted = (m - rank) * raw_p[idx]
                running_max = max(running_max, adjusted)
                holm_p[idx] = min(1.0, running_max)
            for entry, p_holm in zip(pairwise, holm_p, strict=True):
                entry["p_holm"] = float(p_holm)
                print(
                    f"    {entry['a']} vs {entry['b']}: U={entry['u_statistic']:.2f} "
                    f"p_raw={entry['p_raw']:.4f} p_holm={entry['p_holm']:.4f} "
                    f"cliffs_delta={entry['cliffs_delta']:.3f}"
                )
            checkpoint_result["pairwise"] = pairwise
        else:
            print("  (fewer than 2 arms have all 3 seeds at this checkpoint -- no cross-arm stats)")

        analysis["checkpoints"][checkpoint] = checkpoint_result

    print("\n--- Duplicate / invalid / failed rates per arm (aggregated over seeds) ---")
    rate_summary = {}
    for arm in arms:
        totals = {
            "num_valid": 0,
            "num_invalid": 0,
            "num_duplicate": 0,
            "num_failed": 0,
            "num_proposed": 0,
        }
        for seed in seeds:
            run_id = f"{arm}_s{seed}"
            events = list(store.iter_proposal_events(run_id))
            totals["num_proposed"] += len(events)
            for e in events:
                totals[f"num_{e.outcome.value}"] += 1
        rate_summary[arm] = totals
        n = totals["num_proposed"] or 1
        print(
            f"  {arm:14s} proposed={totals['num_proposed']} "
            f"invalid_rate={totals['num_invalid']/n:.2%} "
            f"duplicate_rate={totals['num_duplicate']/n:.2%} "
            f"failed_rate={totals['num_failed']/n:.2%}"
        )
    analysis["duplicate_invalid_failed_rates"] = rate_summary

    print("\n--- Final (budget=25) selected validation vs test metric per run ---")
    for run in summary["runs"]:
        print(
            f"  {run['run_id']:18s} val={run['selected_val_metric_value']} "
            f"test={run['test_metric_value']}"
        )
    analysis["final_runs"] = summary["runs"]

    ANALYSIS_PATH.write_text(json.dumps(analysis, indent=2))
    print(f"\nWrote {ANALYSIS_PATH}")
    store.close()


if __name__ == "__main__":
    main()
