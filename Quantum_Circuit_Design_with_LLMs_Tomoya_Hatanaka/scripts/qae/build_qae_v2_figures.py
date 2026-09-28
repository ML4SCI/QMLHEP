"""Build the QAE v2 figures for the deck and paper from committed artifacts.

Inputs:  outputs/qae_tfim_api_v2/{selected_results.csv,anytime_mean.csv}
Outputs: outputs/qae_tfim_api_v2/fig01_selected_test_fidelity.{png,svg}
         outputs/qae_tfim_api_v2/fig02_anytime.{png,svg}
         outputs/qae_tfim_api_v2/fig03_paired_seeds.{png,svg}
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path("outputs/qae_tfim_api_v2")

METHOD_ORDER = [
    "Random",
    "Evolutionary",
    "RY-random",
    "Reference-QAE",
    "LLM-API-open",
    "LLM-API-closed",
]
COLORS = {
    "Random": "#9aa0a6",
    "Evolutionary": "#f9ab00",
    "RY-random": "#34a853",
    "Reference-QAE": "#7b1fa2",
    "LLM-API-open": "#1a73e8",
    "LLM-API-closed": "#d93025",
    "LLM-chat-frozen": "#80b3f0",
}
LABELS = {
    "Random": "Random",
    "Evolutionary": "Evolutionary",
    "RY-random": "RY-only\nrandom",
    "Reference-QAE": "Hand-designed\n(no search)",
    "LLM-API-open": "LLM API\nopen-loop",
    "LLM-API-closed": "LLM API\nclosed-loop",
    "LLM-chat-frozen": "LLM chat pilot (frozen)",
}
LONG_LABELS = {
    "RY-random": "RY-only random (physics-informed control)",
    "LLM-API-open": "LLM API open-loop",
    "LLM-API-closed": "LLM API closed-loop",
}


def fig_selected(selected: pd.DataFrame) -> None:
    stats = (
        selected[selected["method"].isin(METHOD_ORDER)]
        .groupby("method")["test_fid"]
        .agg(["mean", "std", "count"])
        .reindex(METHOD_ORDER)
    )
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    x = np.arange(len(stats))
    err = stats["std"] / np.sqrt(stats["count"])
    bars = ax.bar(
        x,
        stats["mean"],
        yerr=err,
        capsize=4,
        color=[COLORS[m] for m in stats.index],
        edgecolor="black",
        linewidth=0.6,
    )
    for bar, mean in zip(bars, stats["mean"], strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            mean + 0.004,
            f"{mean:.4f}",
            ha="center",
            fontsize=9,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[m] for m in stats.index], fontsize=8.5)
    ax.set_ylabel("Protected-test trash fidelity (higher is better)")
    ax.set_ylim(0.88, 1.0)
    ax.set_title(
        "Selected-candidate test fidelity, mean over 12 seeds (error bars: s.e.m.)"
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(BASE / "fig01_selected_test_fidelity.png", dpi=200)
    fig.savefig(BASE / "fig01_selected_test_fidelity.svg")
    plt.close(fig)


def fig_anytime(anytime: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for method in METHOD_ORDER:
        if method == "Reference-QAE":
            continue
        group = anytime[anytime["method"] == method]
        if group.empty:
            continue
        ax.plot(
            group["candidates_used"],
            group["best_so_far_test_fid"],
            marker="o",
            markersize=4,
            label=LONG_LABELS.get(method, method),
            color=COLORS[method],
        )
    ax.set_xlabel("Candidate evaluations used (budget B = 8)")
    ax.set_ylabel("Best-so-far test fidelity of\nvalidation-selected candidate")
    ax.set_title("Anytime curve, mean over 12 seeds (higher is better)")
    ax.legend(fontsize=8, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(BASE / "fig02_anytime.png", dpi=200)
    fig.savefig(BASE / "fig02_anytime.svg")
    plt.close(fig)


def fig_paired(selected: pd.DataFrame) -> None:
    pivot = selected.pivot(index="seed", columns="method", values="test_fid")
    pairs = [("Random", "LLM-API-open"), ("RY-random", "LLM-API-open")]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.0), sharey=True)
    for ax, (left, right) in zip(axes, pairs, strict=True):
        for seed in pivot.index:
            improved = pivot.loc[seed, right] > pivot.loc[seed, left]
            ax.plot(
                [0, 1],
                [pivot.loc[seed, left], pivot.loc[seed, right]],
                marker="o",
                markersize=4,
                color="#1a73e8" if improved else "#d93025",
                alpha=0.7,
                linewidth=1.2,
            )
        wins = int((pivot[right] > pivot[left]).sum())
        ax.set_xticks([0, 1])
        ax.set_xticklabels(
            [LONG_LABELS.get(left, left), LONG_LABELS.get(right, right)],
            fontsize=8,
        )
        ax.set_title(f"LLM better in {wins}/12 seeds", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Protected-test trash fidelity\n(higher is better)")
    fig.suptitle("Paired per-seed comparison (same seeds, same budget)", fontsize=11)
    fig.tight_layout()
    fig.savefig(BASE / "fig03_paired_seeds.png", dpi=200)
    fig.savefig(BASE / "fig03_paired_seeds.svg")
    plt.close(fig)


def main() -> None:
    selected = pd.read_csv(BASE / "selected_results.csv")
    anytime = pd.read_csv(BASE / "anytime_mean.csv")
    fig_selected(selected)
    fig_anytime(anytime)
    fig_paired(selected)
    print("figures written to", BASE)


if __name__ == "__main__":
    main()
