#!/usr/bin/env python
"""Analysis of the (interrupted) Groq free-tier pilot, read entirely from
the durable SQLite store -- works whether or not the run's summary JSON
was written. Incomplete cells are reported as unfinished, never filled in.

Selection rule (the declared one): best validation metric among the run's
own evaluated candidates. Protected final test is (re)computed once per
completed run from stored weights -- deterministic, no API. n<=3 seeds,
budget B=10: descriptive only.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from llm_vqc.evaluation.final_test import evaluate_on_test  # noqa: E402
from llm_vqc.evaluation.seeds import train_seed_for_circuit  # noqa: E402
from llm_vqc.evaluation.store import ResultStore  # noqa: E402
from llm_vqc.ir.budget import ProposalOutcome  # noqa: E402
from llm_vqc.ir.schema import CircuitIR  # noqa: E402
from llm_vqc.tasks.t1_gaussian import T1GaussianPeakTask  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "groq_pilot_t1"
STORE = ROOT / "runs" / "groq_pilot_t1" / "results.sqlite"
BUDGET = 10
CONDS = ("random", "evolutionary", "greedy", "llm_open", "llm_closed", "llm_evo")
SEEDS = (0, 1, 2)
NOTE = "Groq gpt-oss-20b pilot, B=10, n<=3 seeds -- descriptive only, INTERRUPTED matrix"
CONSUMING = (ProposalOutcome.VALID, ProposalOutcome.DUPLICATE, ProposalOutcome.FAILED)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    store = ResultStore(STORE)
    task = T1GaussianPeakTask()
    train_val = task.build(seed=0)
    test_split = task.build_test(seed=0)

    rows, curves = [], {}
    for cond in CONDS:
        for seed in SEEDS:
            rid = f"groq_gpt_oss_20b_{cond}_s{seed}"
            events = list(store.iter_proposal_events(rid))
            if not events:
                rows.append({"arm": cond, "seed": seed, "status": "not_started"})
                continue
            n_inv = sum(1 for e in events if e.outcome == ProposalOutcome.INVALID)
            n_dup = sum(1 for e in events if e.outcome == ProposalOutcome.DUPLICATE)
            n_fail = sum(1 for e in events if e.outcome == ProposalOutcome.FAILED)
            consumed = sum(1 for e in events if e.outcome in CONSUMING)
            best_v, best_h, curve = None, None, []
            for e in events:
                if e.outcome in CONSUMING and e.structural_hash:
                    ts = train_seed_for_circuit(seed, e.structural_hash)
                    c = store.get_cached("T1", e.structural_hash, ts)
                    if c is not None and c.val_metric_value is not None:
                        if best_v is None or c.val_metric_value < best_v:
                            best_v, best_h = c.val_metric_value, e.structural_hash
                curve.append(best_v)
            curves[rid] = curve
            row = {"arm": cond, "seed": seed,
                   "status": "complete" if consumed >= BUDGET else "interrupted",
                   "consumed_budget": consumed, "n_proposals": len(events),
                   "invalid": n_inv, "duplicate": n_dup, "failed": n_fail,
                   "val_rmse": best_v, "hash": (best_h or "")[:12]}
            if best_h is not None and consumed >= BUDGET:
                ts = train_seed_for_circuit(seed, best_h)
                w = store.get_trained_weights("T1", best_h, ts)
                c = store.get_cached("T1", best_h, ts)
                if w and c and c.circuit_canonical_json:
                    ir = CircuitIR.model_validate_json(c.circuit_canonical_json)
                    tr = evaluate_on_test(ir, w["classical_state"], test_split,
                                          train_val.spec, ts)
                    row["test_rmse"] = tr.test_metric_value
                    if c.circuit_cost:
                        row.update({"depth": c.circuit_cost.depth,
                                    "gates": c.circuit_cost.gate_count,
                                    "two_qubit": c.circuit_cost.two_qubit_gate_count,
                                    "params": c.circuit_cost.parameter_count})
            calls = list(store.iter_llm_calls(rid))
            if calls:
                row["llm_calls"] = len(calls)
                row["in_tok"] = sum(x.input_tokens for x in calls)
                row["out_tok"] = sum(x.output_tokens for x in calls)
                row["mean_latency_s"] = round(
                    sum(x.latency_seconds for x in calls) / len(calls), 2)
            rows.append(row)

    fields = ["arm", "seed", "status", "consumed_budget", "n_proposals", "invalid",
              "duplicate", "failed", "val_rmse", "test_rmse", "hash", "depth", "gates",
              "two_qubit", "params", "llm_calls", "in_tok", "out_tok", "mean_latency_s"]
    with (OUT / "groq_pilot_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    (OUT / "groq_pilot_summary.json").write_text(json.dumps({"note": NOTE, "rows": rows}, indent=2))

    done = [r for r in rows if r.get("status") == "complete" and r.get("val_rmse") is not None]

    def bars(key, fname, color, title):
        arms = sorted({r["arm"] for r in done})
        vals = {a: [r[key] for r in done if r["arm"] == a and r.get(key) is not None]
                for a in arms}
        arms = [a for a in arms if vals[a]]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(arms, [float(np.median(vals[a])) for a in arms], color=color, alpha=0.75)
        for i, a in enumerate(arms):
            ax.scatter([i] * len(vals[a]), vals[a], color="black", s=16, zorder=3)
        ax.set_ylabel("RMSE (lower is better)")
        ax.set_title(f"{title}\nbars=median, dots=seeds ({NOTE})", fontsize=9)
        plt.xticks(rotation=15, ha="right")
        fig.tight_layout()
        fig.savefig(OUT / f"{fname}.png", dpi=150)
        fig.savefig(OUT / f"{fname}.svg")
        plt.close(fig)

    bars("val_rmse", "groq_val_performance", "steelblue", "Validation RMSE by arm")
    bars("test_rmse", "groq_test_performance", "darkorange", "Protected final-test RMSE by arm")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for rid, curve in curves.items():
        if curve:
            lab = rid.replace("groq_gpt_oss_20b_", "")
            ax.plot(range(1, len(curve) + 1), curve, marker=".", label=lab)
    ax.set_xlabel("Proposals made")
    ax.set_ylabel("Best-so-far validation RMSE (lower is better)")
    ax.set_title(f"Anytime curves per run ({NOTE})", fontsize=9)
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(OUT / "groq_anytime_curves.png", dpi=150)
    fig.savefig(OUT / "groq_anytime_curves.svg")
    plt.close(fig)

    started = [r for r in rows if r.get("status") != "not_started"]
    fig, ax = plt.subplots(figsize=(8, 4))
    arms = sorted({r["arm"] for r in started})
    inv = [sum(r["invalid"] for r in started if r["arm"] == a)
           / max(1, sum(r["n_proposals"] for r in started if r["arm"] == a)) for a in arms]
    dup = [sum(r["duplicate"] for r in started if r["arm"] == a)
           / max(1, sum(r["n_proposals"] for r in started if r["arm"] == a)) for a in arms]
    x = np.arange(len(arms))
    ax.bar(x - 0.2, inv, 0.4, label="invalid rate")
    ax.bar(x + 0.2, dup, 0.4, label="duplicate rate")
    ax.set_xticks(x)
    ax.set_xticklabels(arms, rotation=15, ha="right")
    ax.set_ylabel("Fraction of proposals")
    ax.set_title(f"Invalid/duplicate proposal rates ({NOTE})", fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "groq_proposal_rates.png", dpi=150)
    fig.savefig(OUT / "groq_proposal_rates.svg")
    plt.close(fig)

    llm_rows = [r for r in rows if r.get("llm_calls")]
    if llm_rows:
        labels = [f"{r['arm']} s{r['seed']}" for r in llm_rows]
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
        a1.bar(labels, [r["in_tok"] for r in llm_rows], label="input", alpha=0.7)
        a1.bar(labels, [r["out_tok"] for r in llm_rows],
               bottom=[r["in_tok"] for r in llm_rows], label="output", alpha=0.7)
        a1.set_ylabel("Tokens per run")
        a1.legend(fontsize=8)
        a1.tick_params(axis="x", rotation=30)
        a2.bar(labels, [r["mean_latency_s"] for r in llm_rows], color="seagreen")
        a2.set_ylabel("Mean call latency (s)")
        a2.tick_params(axis="x", rotation=30)
        fig.suptitle(f"Groq LLM usage ({NOTE})", fontsize=9)
        fig.tight_layout()
        fig.savefig(OUT / "groq_llm_usage.png", dpi=150)
        fig.savefig(OUT / "groq_llm_usage.svg")
        plt.close(fig)

    store.close()
    for p in OUT.glob("groq_*.png"):
        img = plt.imread(p)
        assert img.size > 0
        print(f"RENDER_CHECK {p.name} shape={img.shape} size={p.stat().st_size}")
    print(f"Wrote CSV/JSON + plots to {OUT}")


if __name__ == "__main__":
    main()
