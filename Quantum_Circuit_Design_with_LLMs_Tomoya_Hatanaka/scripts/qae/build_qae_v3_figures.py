"""Build the QAE v3 figures for the deck and paper from committed artifacts.

Inputs:  outputs/qae_tfim_neutral_v3/{selected_results.csv,anytime_mean.csv,
         candidate_results.csv,proposal_quality.json}
Outputs: fig01_selected_test_fidelity, fig02_anytime, fig03_paired_seeds,
         fig04_proposal_quality (png+svg each).
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path("outputs/qae_tfim_neutral_v3")

METHOD_ORDER = ["Random", "Greedy", "LLM-Open", "LLM-Closed"]
COLORS = {
    "Random": "#9aa0a6",
    "Greedy": "#f9ab00",
    "LLM-Open": "#1a73e8",
    "LLM-Closed": "#d93025",
}
DESCRIPTIONS = {
    "Random": "Random\nno semantics\nno feedback",
    "Greedy": "Greedy\nno semantics\nfeedback",
    "LLM-Open": "LLM-Open\nsemantics\nno feedback",
    "LLM-Closed": "LLM-Closed\nsemantics\nfeedback",
}
LEGEND = {
    "Random": "Random (no semantics, no feedback)",
    "Greedy": "Greedy (no semantics, feedback)",
    "LLM-Open": "LLM-Open (semantics, no feedback)",
    "LLM-Closed": "LLM-Closed (semantics, feedback)",
}


def fig_selected(selected: pd.DataFrame) -> None:
    stats = (
        selected.groupby("method")["test_fid"]
        .agg(["mean", "std", "count"])
        .reindex(METHOD_ORDER)
    )
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    x = np.arange(len(stats))
    err = stats["std"] / np.sqrt(stats["count"])
    bars = ax.bar(x, stats["mean"], yerr=err, capsize=4,
                  color=[COLORS[m] for m in stats.index],
                  edgecolor="black", linewidth=0.6)
    for bar, mean in zip(bars, stats["mean"], strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, mean + 0.004, f"{mean:.4f}",
                ha="center", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([DESCRIPTIONS[m] for m in stats.index], fontsize=9)
    ax.set_ylabel("Protected-test trash fidelity (higher is better)")
    ax.set_ylim(min(0.86, stats["mean"].min() - 0.03), 1.0)
    ax.set_title("Selected-candidate test fidelity, mean over 12 paired seeds "
                 "(error bars: s.e.m.)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(BASE / "fig01_selected_test_fidelity.png", dpi=200)
    fig.savefig(BASE / "fig01_selected_test_fidelity.svg")
    plt.close(fig)


def fig_anytime(anytime: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for method in METHOD_ORDER:
        group = anytime[anytime["method"] == method]
        if group.empty:
            continue
        ax.plot(group["candidates_used"], group["best_so_far_test_fid"],
                marker="o", markersize=4,
                label=LEGEND[method],
                color=COLORS[method])
    ax.set_xlabel("Candidate evaluations used (budget B = 8)")
    ax.set_ylabel("Best-so-far test fidelity of\nvalidation-selected candidate")
    ax.set_title("Anytime / sample-efficiency curve, mean over 12 seeds "
                 "(higher is better)")
    ax.legend(fontsize=8.5, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(BASE / "fig02_anytime.png", dpi=200)
    fig.savefig(BASE / "fig02_anytime.svg")
    plt.close(fig)


def fig_paired(selected: pd.DataFrame) -> None:
    """The four edges of the 2x2: semantics effect (top row) and feedback
    effect (bottom row)."""
    pivot = selected.pivot(index="seed", columns="method", values="test_fid")
    panels = [
        ("Random", "LLM-Open", "Add semantics (no feedback)"),
        ("Greedy", "LLM-Closed", "Add semantics (with feedback)"),
        ("Random", "Greedy", "Add feedback (no semantics)"),
        ("LLM-Open", "LLM-Closed", "Add feedback (with semantics)"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(11.6, 3.8), sharey=True)
    for ax, (left, right, subtitle) in zip(axes, panels, strict=True):
        for seed in pivot.index:
            improved = pivot.loc[seed, right] > pivot.loc[seed, left]
            ax.plot([0, 1], [pivot.loc[seed, left], pivot.loc[seed, right]],
                    marker="o", markersize=3.5,
                    color="#1a73e8" if improved else "#d93025",
                    alpha=0.7, linewidth=1.1)
        wins = int((pivot[right] > pivot[left]).sum())
        ax.set_xticks([0, 1])
        ax.set_xticklabels([left, right], fontsize=9)
        ax.set_title(f"{subtitle}\n{right} better in {wins}/12 seeds", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Protected-test trash fidelity\n(higher is better)")
    fig.suptitle("Per-seed paired differences along the 2×2 design "
                 "(blue line = right-hand method better)", fontsize=11)
    fig.tight_layout()
    fig.savefig(BASE / "fig03_paired_seeds.png", dpi=200)
    fig.savefig(BASE / "fig03_paired_seeds.svg")
    plt.close(fig)


def fig_quality(candidates: pd.DataFrame, quality: dict) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    methods = METHOD_ORDER
    total = [quality[m]["evaluations"] for m in methods]
    fallback = [quality[m]["fallback_evaluations"] for m in methods]
    x = np.arange(len(methods))
    ax.bar(x, total, color="#dadce0", edgecolor="black", linewidth=0.6,
           label="evaluations (LLM/heuristic proposal)")
    ax.bar(x, fallback, color="#d93025", edgecolor="black", linewidth=0.6,
           label="replaced by flagged random fallback")
    for xi, (t, f) in enumerate(zip(total, fallback, strict=True)):
        ax.text(xi, t + 1.5, f"{f}/{t}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=9)
    ax.set_ylabel("Candidate evaluations (12 seeds × B=8)")
    ax.set_title("Invalid/duplicate proposals replaced by fallbacks")
    ax.legend(fontsize=8, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(BASE / "fig04_proposal_quality.png", dpi=200)
    fig.savefig(BASE / "fig04_proposal_quality.svg")
    plt.close(fig)


def main() -> None:
    selected = pd.read_csv(BASE / "selected_results.csv")
    anytime = pd.read_csv(BASE / "anytime_mean.csv")
    candidates = pd.read_csv(BASE / "candidate_results.csv")
    quality = json.loads((BASE / "proposal_quality.json").read_text())
    fig_selected(selected)
    fig_anytime(anytime)
    fig_paired(selected)
    fig_quality(candidates, quality)
    print("figures written to", BASE)


if __name__ == "__main__":
    main()
