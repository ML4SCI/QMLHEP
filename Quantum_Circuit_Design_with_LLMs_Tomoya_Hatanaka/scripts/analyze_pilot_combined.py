#!/usr/bin/env python
"""Combined analysis of the full Stage 8 pilot: the non-LLM arms
(`scripts/pilot_experiment.py` -> pilot_summary.json) plus the real-API
LLM arms (`scripts/pilot_llm_experiment.py` -> pilot_llm_summary.json),
all sharing one SQLite store and identical conditions (T1, frozen split
seed 0, B=25, seeds {0,1,2}, 20-epoch training).

Reads ONLY stored results (SQLite + the two summary JSONs) — no
hand-typed values. n=3 seeds per arm: descriptive statistics and effect
sizes only; Kruskal-Wallis / Mann-Whitney U + Holm are reported as
exploratory, never as confirmatory significance claims.

Outputs (to outputs/pilot_t1/): combined per-arm val/test bars, anytime
best-so-far curves (median across seeds), duplicate/invalid rates, LLM
token/latency usage, pilot_combined.csv, pilot_combined_analysis.json.
"""

from __future__ import annotations

import csv
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

from llm_vqc.evaluation.seeds import train_seed_for_circuit  # noqa: E402
from llm_vqc.evaluation.store import ResultStore  # noqa: E402
from llm_vqc.ir.budget import ProposalOutcome  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_DIR = _REPO_ROOT / "runs" / "pilot_t1"
OUT_DIR = _REPO_ROOT / "outputs" / "pilot_t1"
DB_PATH = RUN_DIR / "results.sqlite"

SEEDS = (0, 1, 2)
BUDGET = 25
CHECKPOINTS = (10, 25)
NOTE = "n=3 seeds/arm -- descriptive + effect sizes only, exploratory stats, NOT confirmatory"

# arm label -> run_id prefix (run ids are f"{prefix}_s{seed}")
ARMS = {
    "random": "random",
    "evolutionary": "evolutionary",
    "greedy": "greedy",
    "llm_iter (closed)": "llm_iter_closed",
    "llm_iter (open)": "llm_iter_open",
    "llm_evo": "llm_evo",
}


def cliffs_delta(a, b):
    greater = sum(1 for x, y in itertools.product(a, b) if x > y)
    less = sum(1 for x, y in itertools.product(a, b) if x < y)
    return (greater - less) / (len(a) * len(b))


def anytime_curve(store, run_id, run_seed):
    best = None
    curve = []
    consuming = (ProposalOutcome.VALID, ProposalOutcome.DUPLICATE, ProposalOutcome.FAILED)
    for record in store.iter_proposal_events(run_id):
        if record.outcome in consuming and record.structural_hash is not None:
            ts = train_seed_for_circuit(run_seed, record.structural_hash)
            cached = store.get_cached("T1", record.structural_hash, ts)
            if cached is not None and cached.val_metric_value is not None:
                v = cached.val_metric_value
                if best is None or v < best:
                    best = v
        curve.append(best)
    return curve


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    nonllm = json.loads((RUN_DIR / "pilot_summary.json").read_text())
    llm = json.loads((RUN_DIR / "pilot_llm_summary.json").read_text())
    runs_by_id = {r["run_id"]: r for r in nonllm["runs"] + llm["runs"] if "ledger_summary" in r}
    store = ResultStore(DB_PATH)

    rows = []
    curves = {}  # label -> list of per-seed curves
    for label, prefix in ARMS.items():
        curves[label] = []
        for seed in SEEDS:
            run_id = f"{prefix}_s{seed}"
            r = runs_by_id.get(run_id)
            if r is None:
                continue
            ls = r["ledger_summary"]
            rows.append({
                "arm": label, "seed": seed,
                "consumed_budget": ls["consumed_budget"],
                "num_valid": ls["num_valid"], "num_invalid": ls["num_invalid"],
                "num_duplicate": ls["num_duplicate"], "num_failed": ls["num_failed"],
                "val_rmse": r.get("selected_val_metric_value"),
                "test_rmse": r.get("test_metric_value"),
                "structural_hash": r.get("selected_structural_hash"),
                "llm_calls": r.get("llm_call_count"),
                "input_tokens": r.get("input_tokens"),
                "output_tokens": r.get("output_tokens"),
                "mean_latency_s": r.get("mean_latency_seconds"),
                "stop_reason": r.get("stop_reason"),
            })
            curves[label].append(anytime_curve(store, run_id, seed))

    csv_path = OUT_DIR / "pilot_combined.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {csv_path}")

    labels = list(ARMS.keys())

    def per_arm(key):
        return {
            lab: [r[key] for r in rows if r["arm"] == lab and r[key] is not None]
            for lab in labels
        }

    val_by_arm = per_arm("val_rmse")
    test_by_arm = per_arm("test_rmse")

    # --- val + test bars (median with per-seed dots) ------------------------
    for name, data, color, title in (
        ("pilot_val_performance", val_by_arm, "steelblue", "Selected-circuit validation RMSE"),
        ("pilot_test_performance", test_by_arm, "darkorange", "Protected final-test RMSE"),
    ):
        fig, ax = plt.subplots(figsize=(8.5, 4.5))
        medians = [float(np.median(data[lab])) if data[lab] else 0.0 for lab in labels]
        ax.bar(labels, medians, color=color, alpha=0.75)
        for i, lab in enumerate(labels):
            ax.scatter([i] * len(data[lab]), data[lab], color="black", zorder=3, s=18)
        ax.set_ylabel("RMSE (lower is better)")
        ax.set_title(f"Pilot (T1, B=25): {title}\nbars=median, dots=individual seeds ({NOTE})")
        plt.xticks(rotation=20, ha="right")
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"{name}.png", dpi=150)
        fig.savefig(OUT_DIR / f"{name}.svg")
        plt.close(fig)

    # --- anytime curves (median across seeds) --------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for lab in labels:
        seed_curves = [c for c in curves[lab] if c]
        if not seed_curves:
            continue
        min_len = min(len(c) for c in seed_curves)
        aligned = np.array([
            [v if v is not None else np.nan for v in c[:min_len]] for c in seed_curves
        ])
        median_curve = np.nanmedian(aligned, axis=0)
        ax.plot(range(1, min_len + 1), median_curve, marker=".", label=lab)
    ax.set_xlabel("Proposals made (invalid ones excluded from budget but shown in sequence)")
    ax.set_ylabel("Best-so-far validation RMSE (median across 3 seeds)")
    ax.set_title(f"Pilot anytime curves ({NOTE})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "pilot_anytime_curves.png", dpi=150)
    fig.savefig(OUT_DIR / "pilot_anytime_curves.svg")
    plt.close(fig)

    # --- duplicate / invalid rates -------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    x = np.arange(len(labels))
    width = 0.35
    inv, dup = [], []
    for lab in labels:
        arm_rows = [r for r in rows if r["arm"] == lab]
        proposed = sum(
            r["num_valid"] + r["num_invalid"] + r["num_duplicate"] + r["num_failed"]
            for r in arm_rows
        ) or 1
        inv.append(sum(r["num_invalid"] for r in arm_rows) / proposed)
        dup.append(sum(r["num_duplicate"] for r in arm_rows) / proposed)
    ax.bar(x - width / 2, inv, width, label="invalid rate")
    ax.bar(x + width / 2, dup, width, label="duplicate rate")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Fraction of all proposals (3 seeds pooled)")
    ax.set_title(f"Pilot invalid/duplicate proposal rates ({NOTE})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "pilot_proposal_rates.png", dpi=150)
    fig.savefig(OUT_DIR / "pilot_proposal_rates.svg")
    plt.close(fig)

    # --- LLM usage ------------------------------------------------------------
    llm_rows = [r for r in rows if r["llm_calls"]]
    if llm_rows:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
        run_labels = [f"{r['arm']} s{r['seed']}" for r in llm_rows]
        ax1.bar(run_labels, [r["mean_latency_s"] for r in llm_rows], color="seagreen")
        ax1.set_ylabel("Mean call latency (s)")
        ax1.tick_params(axis="x", rotation=45)
        ax2.bar(run_labels, [r["input_tokens"] for r in llm_rows], label="input tokens", alpha=0.7)
        ax2.bar(run_labels, [r["output_tokens"] for r in llm_rows],
                bottom=[r["input_tokens"] for r in llm_rows], label="output tokens", alpha=0.7)
        ax2.set_ylabel("Tokens per run (stacked in+out)")
        ax2.tick_params(axis="x", rotation=45)
        ax2.legend(fontsize=8)
        fig.suptitle(f"Real OpenAI usage per LLM run ({NOTE})")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "pilot_llm_usage.png", dpi=150)
        fig.savefig(OUT_DIR / "pilot_llm_usage.svg")
        plt.close(fig)

    # --- exploratory stats ------------------------------------------------------
    analysis = {"note": NOTE, "checkpoints": {}}
    for checkpoint in CHECKPOINTS:
        per_arm_cp = {}
        for lab in labels:
            vals = []
            for c in curves[lab]:
                consuming_vals = [v for v in c if v is not None]
                # checkpoint over budget-consuming evaluations: index on the
                # curve position where `checkpoint` consumed evals are reached
                # (invalid proposals do not consume budget). Approximate by
                # taking the best-so-far after `checkpoint` consuming events.
                if len(consuming_vals) >= 1:
                    vals.append(
                        min(consuming_vals[: checkpoint]) if consuming_vals else None
                    )
            per_arm_cp[lab] = [v for v in vals if v is not None]
        entry = {
            lab: {
                "n": len(v),
                "median": float(np.median(v)) if v else None,
                "values": v,
            }
            for lab, v in per_arm_cp.items()
        }
        complete = [lab for lab in labels if len(per_arm_cp[lab]) == len(SEEDS)]
        if len(complete) >= 2:
            h, p = stats.kruskal(*[per_arm_cp[lab] for lab in complete])
            entry["kruskal_wallis"] = {"H": float(h), "p": float(p)}
            pairs = []
            raw_p = []
            for a, b in itertools.combinations(complete, 2):
                u, up = stats.mannwhitneyu(per_arm_cp[a], per_arm_cp[b], alternative="two-sided")
                raw_p.append(up)
                pairs.append({"a": a, "b": b, "U": float(u), "p_raw": float(up),
                              "cliffs_delta": float(cliffs_delta(per_arm_cp[a], per_arm_cp[b]))})
            order = np.argsort(raw_p)
            running = 0.0
            for rank, idx in enumerate(order):
                running = max(running, (len(raw_p) - rank) * raw_p[idx])
                pairs[idx]["p_holm"] = min(1.0, float(running))
            entry["pairwise"] = pairs
        analysis["checkpoints"][checkpoint] = entry

    (OUT_DIR / "pilot_combined_analysis.json").write_text(json.dumps(analysis, indent=2))
    store.close()

    # --- verify plots ----------------------------------------------------------
    for png in OUT_DIR.glob("pilot_*.png"):
        img = plt.imread(png)
        assert img.size > 0
        print(f"RENDER_CHECK {png.name}: shape={img.shape} size={png.stat().st_size}")
    print(f"Wrote {OUT_DIR / 'pilot_combined_analysis.json'}")


if __name__ == "__main__":
    main()
