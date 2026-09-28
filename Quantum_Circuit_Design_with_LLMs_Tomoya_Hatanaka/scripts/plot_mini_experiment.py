#!/usr/bin/env python
"""Generate plots + CSV/Markdown summary for the mini real-API experiment
(`scripts/mini_llm_experiment.py`), reading only from the stored SQLite
results / JSON summary -- no hand-typed values.

n=1 seed, tiny budget: every plot is explicitly labeled descriptive-only,
no significance markers are drawn.
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

from llm_vqc.evaluation.seeds import train_seed_for_circuit  # noqa: E402
from llm_vqc.evaluation.store import ResultStore  # noqa: E402
from llm_vqc.ir.budget import ProposalOutcome  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_DIR = _REPO_ROOT / "runs" / "mini_llm_experiment"  # raw SQLite/JSON results
OUTPUT_DIR = _REPO_ROOT / "outputs" / "mini_llm_experiment"  # generated figures/report
DB_PATH = RUN_DIR / "results.sqlite"
SUMMARY_PATH = RUN_DIR / "mini_summary.json"
CSV_PATH = OUTPUT_DIR / "mini_results.csv"
REPORT_PATH = OUTPUT_DIR / "mini_report.md"

ARM_LABELS = {
    "random": "random",
    "evolutionary": "evolutionary",
    "greedy": "greedy",
    "llm_iter_open": "llm_iter (open-loop)",
    "llm_iter_closed": "llm_iter (closed-loop)",
}
DESCRIPTIVE_NOTE = "n=1 seed -- descriptive only, no significance testing"


def anytime_curve(
    store: ResultStore, run_id: str, task_name: str, run_seed: int
) -> list[float | None]:
    best = None
    curve: list[float | None] = []
    consuming = (ProposalOutcome.VALID, ProposalOutcome.DUPLICATE, ProposalOutcome.FAILED)
    for record in store.iter_proposal_events(run_id):
        if record.outcome in consuming and record.structural_hash is not None:
            ts = train_seed_for_circuit(run_seed, record.structural_hash)
            cached = store.get_cached(task_name, record.structural_hash, ts)
            if cached is not None and cached.val_metric_value is not None:
                v = cached.val_metric_value
                if best is None or v < best:
                    best = v
        curve.append(best)
    return curve


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = json.loads(SUMMARY_PATH.read_text())
    store = ResultStore(DB_PATH)
    run_ids = [r["run_id"] for r in summary["runs"] if "ledger_summary" in r]

    rows = []
    for run in summary["runs"]:
        run_id = run.get("run_id")
        if "ledger_summary" not in run:
            continue
        ls = run["ledger_summary"]
        rows.append(
            {
                "arm": run_id,
                "consumed_budget": ls["consumed_budget"],
                "num_valid": ls["num_valid"],
                "num_invalid": ls["num_invalid"],
                "num_duplicate": ls["num_duplicate"],
                "num_failed": ls["num_failed"],
                "selected_val_metric": run.get("selected_val_metric_value"),
                "test_metric": run.get("test_metric_value"),
                "structural_hash": run.get("selected_structural_hash"),
                "stop_reason": run.get("stop_reason"),
            }
        )

    with CSV_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {CSV_PATH}")

    arms = [r["arm"] for r in rows]
    labels = [ARM_LABELS.get(a, a) for a in arms]

    # --- 1. validation performance by arm ---------------------------------
    val_values = [r["selected_val_metric"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, [v if v is not None else 0 for v in val_values], color="steelblue")
    ax.set_ylabel("Validation RMSE (lower is better)")
    ax.set_title(f"Mini experiment: selected-circuit validation RMSE by arm\n({DESCRIPTIVE_NOTE})")
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "mini_val_performance.png", dpi=150)
    fig.savefig(OUTPUT_DIR / "mini_val_performance.svg")
    plt.close(fig)

    # --- 2. final-test performance by arm ----------------------------------
    test_values = [r["test_metric"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, [v if v is not None else 0 for v in test_values], color="darkorange")
    ax.set_ylabel("Protected final-test RMSE (lower is better)")
    ax.set_title(f"Mini experiment: protected final-test RMSE by arm\n({DESCRIPTIVE_NOTE})")
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "mini_test_performance.png", dpi=150)
    fig.savefig(OUTPUT_DIR / "mini_test_performance.svg")
    plt.close(fig)

    # --- 3. anytime best-so-far validation curve ---------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    for run_id in run_ids:
        curve = anytime_curve(store, run_id, "T1", 0)
        xs = list(range(1, len(curve) + 1))
        ax.plot(xs, curve, marker="o", label=ARM_LABELS.get(run_id, run_id))
    ax.set_xlabel("Evaluated candidates (this run)")
    ax.set_ylabel("Best-so-far validation RMSE")
    ax.set_title(f"Mini experiment: anytime best-so-far validation RMSE\n({DESCRIPTIVE_NOTE})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "mini_anytime_curve.png", dpi=150)
    fig.savefig(OUTPUT_DIR / "mini_anytime_curve.svg")
    plt.close(fig)

    # --- 4. invalid/duplicate rates by arm ----------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.35
    x = range(len(arms))
    invalid_rate = [
        r["num_invalid"] / max(1, r["consumed_budget"] + r["num_invalid"]) for r in rows
    ]
    duplicate_rate = [r["num_duplicate"] / max(1, r["consumed_budget"]) for r in rows]
    ax.bar([i - width / 2 for i in x], invalid_rate, width, label="invalid rate")
    ax.bar([i + width / 2 for i in x], duplicate_rate, width, label="duplicate rate")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Rate (fraction of proposals)")
    ax.set_title(f"Mini experiment: invalid/duplicate proposal rates by arm\n({DESCRIPTIVE_NOTE})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "mini_proposal_rates.png", dpi=150)
    fig.savefig(OUTPUT_DIR / "mini_proposal_rates.svg")
    plt.close(fig)

    # --- 5. LLM call latency and token usage --------------------------------
    llm_calls = summary.get("llm_calls", [])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    call_labels = [f"{c['run_id']}" for c in llm_calls]
    ax1.bar(call_labels, [c["latency_seconds"] for c in llm_calls], color="seagreen")
    ax1.set_ylabel("Latency (seconds)")
    ax1.set_title("Real OpenAI call latency")
    ax1.tick_params(axis="x", rotation=45)
    ax2.bar(
        call_labels,
        [c["input_tokens"] for c in llm_calls],
        label="input tokens",
        alpha=0.7,
    )
    ax2.bar(
        call_labels,
        [c["output_tokens"] for c in llm_calls],
        bottom=[c["input_tokens"] for c in llm_calls],
        label="output tokens",
        alpha=0.7,
    )
    ax2.set_ylabel("Tokens")
    ax2.set_title("Real OpenAI token usage (stacked in+out)")
    ax2.tick_params(axis="x", rotation=45)
    ax2.legend(fontsize=8)
    fig.suptitle(f"Mini experiment: real LLM call latency/tokens ({DESCRIPTIVE_NOTE})")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "mini_llm_usage.png", dpi=150)
    fig.savefig(OUTPUT_DIR / "mini_llm_usage.svg")
    plt.close(fig)

    store.close()

    # --- verify every plot file exists and has nonzero size -----------------
    plot_files = [
        "mini_val_performance.png", "mini_val_performance.svg",
        "mini_test_performance.png", "mini_test_performance.svg",
        "mini_anytime_curve.png", "mini_anytime_curve.svg",
        "mini_proposal_rates.png", "mini_proposal_rates.svg",
        "mini_llm_usage.png", "mini_llm_usage.svg",
    ]
    for name in plot_files:
        path = OUTPUT_DIR / name
        size = path.stat().st_size if path.exists() else 0
        print(f"CHECK {name}: exists={path.exists()} size={size}")
        assert path.exists() and size > 0, f"{name} missing or empty"

    # --- render-verify each PNG programmatically (re-decode via matplotlib) -
    for name in [n for n in plot_files if n.endswith(".png")]:
        img = plt.imread(OUTPUT_DIR / name)
        print(f"RENDER_CHECK {name}: shape={img.shape}")
        assert img.size > 0

    # --- Markdown report -----------------------------------------------------
    md = [
        "# Mini LAQS-Bench experiment (real OpenAI API, time-boxed)",
        "",
        f"- Provider: {summary['provider']}",
        f"- Model: {summary['model']}",
        f"- Real calls made: {summary['real_calls_made']} / {summary['max_real_calls']}",
        f"- Elapsed: {summary['elapsed_seconds']:.1f} s",
        f"- Budget per arm: {summary['budget_per_arm']}",
        f"- Arms: {', '.join(summary['arms'])}",
        f"- Omitted: {', '.join(summary['omitted_arms'])}",
        "",
        "## Per-arm results (n=1 seed, descriptive only)",
        "",
        "| arm | val RMSE | test RMSE | consumed budget | invalid | duplicate | failed |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        md.append(
            f"| {ARM_LABELS.get(r['arm'], r['arm'])} | {r['selected_val_metric']} | "
            f"{r['test_metric']} | {r['consumed_budget']} | {r['num_invalid']} | "
            f"{r['num_duplicate']} | {r['num_failed']} |"
        )
    md += [
        "",
        "## Plots",
        "",
        "- mini_val_performance.png/.svg",
        "- mini_test_performance.png/.svg",
        "- mini_anytime_curve.png/.svg",
        "- mini_proposal_rates.png/.svg",
        "- mini_llm_usage.png/.svg",
        "",
        "## Limitations",
        "",
        "n=1 seed, budget=2 candidates per arm, 1 training epoch, tiny time-boxed "
        "run (5-minute wall-clock cap). NOT publication evidence. No inferential "
        "statistics computed or reported. Cost per real API call is not reported "
        "(OpenAI's chat completions API does not return a dollar cost); token "
        "counts and latency are the authoritative usage record.",
        "",
        "## Reproduction (excluding the API key)",
        "",
        "```",
        "OPENAI_API_KEY=\"$(pbpaste | tr -d '\\r\\n')\" \\",
        "  .venv/bin/python scripts/mini_llm_experiment.py",
        ".venv/bin/python scripts/plot_mini_experiment.py",
        "```",
    ]
    REPORT_PATH.write_text("\n".join(md))
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
