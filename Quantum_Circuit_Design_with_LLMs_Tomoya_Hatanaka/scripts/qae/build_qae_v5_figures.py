"""Build the QAE v5 figures for the deck and paper from committed artifacts.

Inputs:  outputs/qae_tfim_neutral_v5/{selected_results.csv,anytime_mean.csv,
         candidate_results.csv,refinement_gains.json,proposal_quality.json}
Outputs: fig01_selected_test_fidelity (seed-level dots + mean + 95% CI),
         fig02_anytime, fig03_refinement_gains, fig04_proposal_quality.

Axis language follows the v5 protocol: "held-out test trash fidelity
F_trash = P(q2q3=00)", higher is better; error bars are bootstrap 95% CIs.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path("outputs/qae_tfim_neutral_v5")

METHOD_ORDER = ["Random", "Greedy", "LLM-Open", "LLM-Closed"]
COLORS = {
    "Random": "#9aa0a6",
    "Greedy": "#f9ab00",
    "LLM-Open": "#1a73e8",
    "LLM-Closed": "#d93025",
}
SUBTITLES = {
    "Random": "no semantics\nbreadth-only",
    "Greedy": "no semantics\n4 starts + 4 local",
    "LLM-Open": "semantics\nopen-loop batch",
    "LLM-Closed": "semantics\n4 starts + 4 redesigns",
}
LEGEND = {
    "Random": "Random (no semantics, breadth-only)",
    "Greedy": "Greedy (no semantics, 4 warm starts + 4 one-change refinements)",
    "LLM-Open": "LLM-Open (semantics, open-loop batch of 8)",
    "LLM-Closed": "LLM-Closed (semantics, 4 warm starts + 4 free-form redesigns)",
}
YLABEL = "Held-out test trash fidelity  $F_{trash}=P(q_2q_3{=}00)$\n(higher is better)"


def _boot_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    boots = np.array([
        rng.choice(values, size=len(values), replace=True).mean() for _ in range(10_000)
    ])
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def fig_selected(selected: pd.DataFrame) -> None:
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    jitter = np.random.default_rng(1)
    ymin = 1.0
    for i, method in enumerate(METHOD_ORDER):
        vals = selected[selected["method"] == method]["test_fid"].to_numpy()
        ymin = min(ymin, vals.min())
        x = i + jitter.uniform(-0.13, 0.13, size=len(vals))
        ax.scatter(x, vals, s=26, color=COLORS[method], alpha=0.65,
                   edgecolor="black", linewidth=0.4, zorder=3,
                   label=None)
        mean = vals.mean()
        lo, hi = _boot_ci(vals, rng)
        ax.errorbar([i + 0.30], [mean], yerr=[[mean - lo], [hi - mean]],
                    fmt="D", markersize=6, color=COLORS[method],
                    ecolor="black", elinewidth=1.4, capsize=5, zorder=4)
        ax.text(i + 0.30, hi + 0.006, f"{mean:.4f}", ha="center", fontsize=9.5)
    ax.set_xticks(range(len(METHOD_ORDER)))
    ax.set_xticklabels(
        [f"{m}\n{SUBTITLES[m]}" for m in METHOD_ORDER], fontsize=8.5
    )
    ax.set_ylabel(YLABEL, fontsize=9.5)
    ax.set_ylim(max(0.0, ymin - 0.03), 1.005)
    ax.set_title(
        "Held-out test, all 12 paired seeds (dots); diamonds: mean ± bootstrap 95% CI",
        fontsize=10.5,
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(BASE / "fig01_selected_test_fidelity.png", dpi=200)
    fig.savefig(BASE / "fig01_selected_test_fidelity.svg")
    plt.close(fig)


def fig_anytime(anytime: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    for method in METHOD_ORDER:
        group = anytime[anytime["method"] == method]
        if group.empty:
            continue
        ax.plot(group["candidates_used"], group["best_so_far_test_fid"],
                marker="o", markersize=4, label=LEGEND[method],
                color=COLORS[method])
    ax.axvline(4.5, color="#5f6368", linestyle="--", linewidth=1)
    ax.text(4.5, ax.get_ylim()[0] + 0.01, " explore | refine (adaptive arms)",
            fontsize=8, color="#5f6368", va="bottom")
    ax.set_xlabel("Candidate evaluations used (budget B = 8)")
    ax.set_ylabel("Best-so-far held-out test $F_{trash}$ of\nvalidation-selected candidate")
    ax.set_title("Anytime / sample-efficiency curve, mean over 12 seeds "
                 "(higher is better)")
    ax.legend(fontsize=8, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(BASE / "fig02_anytime.png", dpi=200)
    fig.savefig(BASE / "fig02_anytime.svg")
    plt.close(fig)


def fig_refinement(frame: pd.DataFrame, gains: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.0), sharey=True)
    for ax, method in zip(axes, ("Greedy", "LLM-Closed"), strict=True):
        for _seed, group in frame[frame["method"] == method].groupby("seed"):
            group = group.sort_values("order")
            warm = group[group["order"] <= 4]
            warm_best = warm.loc[warm["val_loss"].idxmin(), "test_fid"]
            final_best = group.loc[group["val_loss"].idxmin(), "test_fid"]
            improved = final_best > warm_best
            ax.plot([0, 1], [warm_best, final_best], marker="o", markersize=4,
                    color="#1a73e8" if improved else "#9aa0a6",
                    alpha=0.75, linewidth=1.2)
        g = gains[method]
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["best of 4\nwarm starts", "final selected\n(after refinement)"],
                           fontsize=9)
        ax.set_title(
            f"{method}: refinement gain {g['mean_test_gain']:+.4f}\n"
            f"(95% CI [{g['test_gain_bootstrap_95ci'][0]:+.4f}, "
            f"{g['test_gain_bootstrap_95ci'][1]:+.4f}], "
            f"{g['seeds_improved']}/12 improved)", fontsize=9.5)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Held-out test $F_{trash}$ (higher is better)")
    fig.suptitle("Did the 4 refinement evaluations add value after the warm start? "
                 "(blue = improved, grey = unchanged/worse)", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(BASE / "fig03_refinement_gains.png", dpi=200)
    fig.savefig(BASE / "fig03_refinement_gains.svg")
    plt.close(fig)


def fig_quality(quality: dict) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    methods = METHOD_ORDER
    total = [quality[m]["evaluations"] for m in methods]
    fallback = [quality[m]["fallback_evaluations"] for m in methods]
    x = np.arange(len(methods))
    ax.bar(x, total, color="#dadce0", edgecolor="black", linewidth=0.6,
           label="candidate evaluations (12 seeds × B=8)")
    ax.bar(x, fallback, color="#d93025", edgecolor="black", linewidth=0.6,
           label="replaced by flagged random fallback")
    for xi, (t, f) in enumerate(zip(total, fallback, strict=True)):
        ax.text(xi, t + 1.5, f"{f}/{t}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=9)
    ax.set_ylabel("Evaluations")
    ax.set_title("Duplicate/invalid proposals replaced by fallbacks")
    ax.legend(fontsize=8, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(BASE / "fig04_proposal_quality.png", dpi=200)
    fig.savefig(BASE / "fig04_proposal_quality.svg")
    plt.close(fig)


def fig_redesign(frame: pd.DataFrame) -> None:
    """v5 diagnostic: edit distance of each refinement proposal from its
    incumbent, colored by acceptance; Greedy's one-change band for contrast."""
    refine = frame[(frame["method"] == "LLM-Closed")
                   & (frame["phase"] == "refine") & (~frame["fallback"])].copy()
    # acceptance = val_loss beats all earlier candidates in the seed
    accepted_idx = []
    for _seed, group in frame[frame["method"] == "LLM-Closed"].groupby("seed"):
        best = float("inf")
        for idx, row in group.sort_values("order").iterrows():
            if row["phase"] == "refine" and row["val_loss"] < best and idx in refine.index:
                accepted_idx.append(idx)
            best = min(best, row["val_loss"])
    refine["accepted"] = refine.index.isin(accepted_idx)
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    jitter = np.random.default_rng(2)
    for accepted, color, label in ((True, "#1a73e8", "accepted (new incumbent)"),
                                   (False, "#9aa0a6", "rejected (incumbent kept)")):
        sub = refine[refine["accepted"] == accepted]
        ax.scatter(sub["edit_distance"] * 16,
                   jitter.uniform(-0.3, 0.3, size=len(sub)),
                   s=42, color=color, alpha=0.75, edgecolor="black",
                   linewidth=0.4, label=f"{label} (n={len(sub)})")
    ax.axvspan(0, 2, color="#f9ab00", alpha=0.18)
    ax.text(1.0, 0.42, "Greedy's one-change\nneighbourhood (<=2 slots)",
            fontsize=8, color="#B7791F", ha="center")
    ax.set_xlabel("Gate slots changed from the incumbent (of 16)")
    ax.set_yticks([])
    ax.set_xlim(-0.5, 16.5)
    ax.set_title("v5 LLM-Closed redesign proposals: how far from the incumbent, "
                 "and were they accepted?", fontsize=10.5)
    ax.legend(fontsize=8.5, loc="upper right")
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(BASE / "fig05_redesign_locality.png", dpi=200)
    fig.savefig(BASE / "fig05_redesign_locality.svg")
    plt.close(fig)


def main() -> None:
    selected = pd.read_csv(BASE / "selected_results.csv")
    anytime = pd.read_csv(BASE / "anytime_mean.csv")
    frame = pd.read_csv(BASE / "candidate_results.csv")
    gains = json.loads((BASE / "refinement_gains.json").read_text())
    quality = json.loads((BASE / "proposal_quality.json").read_text())
    fig_selected(selected)
    fig_anytime(anytime)
    fig_refinement(frame, gains)
    fig_quality(quality)
    fig_redesign(frame)
    print("figures written to", BASE)


if __name__ == "__main__":
    main()
