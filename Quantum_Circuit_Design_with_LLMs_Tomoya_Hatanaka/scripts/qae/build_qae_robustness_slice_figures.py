"""Figures for the "tested one-factor slices" deck.

Every figure shows only conditions that were actually run. Groups are
always ordered Random -> Greedy -> Low -> High (model tiers), and within
a tier the Open and Closed search workflows are two horizontally offset
markers. Slices where the Low tier was not run show Random, Greedy and
High only; nothing is invented for Low.

Encoding (shared by every figure):
  Random  grey circle      Greedy  orange square
  Open    blue triangle    Closed  red diamond
  High tier = filled marker, Low tier = hollow marker
  small dots = the 12 paired seeds, large marker = mean,
  vertical bar = bootstrap 95% confidence interval; higher is better.

Inputs : outputs/qae_robustness/<condition>/{selected_results.csv,paired_stats.json,
         proposal_quality.json}
Outputs: outputs/qae_robustness/figures/slice_*.{png,svg}, baseline_groups.{png,svg}
         outputs/qae_robustness/slices.json (what each figure contains)
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("outputs/qae_robustness")
FIGURES = ROOT / "figures"

TIER = {"gpt-5.4-mini-2026-03-17": "High", "gpt-4.1-mini-2025-04-14": "Low"}
COLORS = {"Random": "#9aa0a6", "Greedy": "#f9ab00", "Open": "#1a73e8", "Closed": "#d93025"}
MARKERS = {"Random": "o", "Greedy": "s", "Open": "^", "Closed": "D"}
OFFSET = 0.2  # Open / Closed inside a tier group
YLABEL = "Held-out test trash fidelity  $F_{trash}$\n(higher is better)"
DLABEL = "paired difference\n(0 = no effect)"

# (slice name, [(condition key, panel title)], fixed-settings sentence)
SLICES = {
    "budget": ([("budget_b4", "B = 4"), ("reference", "B = 8\n(previous baseline)"),
                ("budget_b16", "B = 16")],
               "fixed: 4 qubits · Ising · High model"),
    "qubits": ([("reference", "4 qubits\n(previous baseline)"), ("qubits_n6", "6 qubits"),
                ("qubits_n8", "8 qubits")],
               "fixed: B = 8 · Ising · High model"),
    "hamiltonian": ([("reference", "Ising chain\n(previous baseline)"),
                     ("hamiltonian_xxz", "XXZ chain")],
                    "fixed: 4 qubits · B = 8 · High model"),
}


def _boot_ci(values, rng):
    boots = np.array([rng.choice(values, size=len(values), replace=True).mean()
                      for _ in range(10_000)])
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def selected(key: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / key / "selected_results.csv")


def values(key: str, method: str) -> np.ndarray:
    frame = selected(key)
    return frame[frame["method"] == method].sort_values("seed")["test_fid"].to_numpy()


def stats(key: str) -> dict:
    return json.loads((ROOT / key / "paired_stats.json").read_text())


def validity(key: str, method: str) -> float:
    quality = json.loads((ROOT / key / "proposal_quality.json").read_text())[method]
    return 1 - quality["fallback_evaluations"] / quality["evaluations"]


def _draw_point(ax, x, vals, workflow, filled, rng, jitter, label_mean=True):
    color, marker = COLORS[workflow], MARKERS[workflow]
    xs = x + jitter.uniform(-0.11, 0.11, size=len(vals))
    ax.scatter(xs, vals, s=13, color=color, alpha=0.35 if filled else 0.55,
               facecolors=color if filled else "none", edgecolors=color,
               linewidths=0.5, zorder=1)
    mean = vals.mean()
    low, high = _boot_ci(vals, rng)
    ax.errorbar(x, mean, yerr=[[mean - low], [high - mean]], fmt=marker, color=color,
                markersize=8, capsize=4, lw=1.8, zorder=3,
                markerfacecolor=color if filled else "white",
                markeredgecolor="black" if filled else color,
                markeredgewidth=0.6 if filled else 1.5)
    if label_mean:
        ax.annotate(f"{mean:.3f}", (x, high), textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=8.5, color=color, fontweight="bold")
    return mean


def draw_condition(ax, key: str, tiers: list[tuple[str, str]], rng, jitter,
                   label_mean=True) -> list[str]:
    """Groups: Random, Greedy, then one group per (tier, condition key) in the
    order given (Low first, then High). Returns the x tick labels."""
    labels = ["Random", "Greedy"]
    _draw_point(ax, 0, values(key, "Random"), "Random", True, rng, jitter, label_mean)
    _draw_point(ax, 1, values(key, "Greedy"), "Greedy", True, rng, jitter, label_mean)
    for g, (tier, tier_key) in enumerate(tiers, start=2):
        filled = tier == "High"
        _draw_point(ax, g - OFFSET, values(tier_key, "LLM-Open"), "Open", filled, rng,
                    jitter, label_mean)
        _draw_point(ax, g + OFFSET, values(tier_key, "LLM-Closed"), "Closed", filled, rng,
                    jitter, label_mean)
        labels.append(f"{tier}\n(Open · Closed)")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_xlim(-0.6, len(labels) - 0.4)
    ax.grid(axis="y", alpha=0.22)
    return labels


def draw_contrasts(ax, key: str, x: float, tier_label: str | None = None) -> None:
    for dx, contrast, workflow in ((-0.16, "LLM-Open_minus_Random", "Open"),
                                   (0.16, "LLM-Closed_minus_LLM-Open", "Closed")):
        entry = stats(key)[contrast]
        mean, (lo, hi) = entry["mean_paired_gain"], entry["bootstrap_95ci"]
        significant = entry["wilcoxon_exact_two_sided_p"] < 0.05
        color = COLORS[workflow]
        ax.errorbar(x + dx, mean, yerr=[[mean - lo], [hi - mean]], fmt="o", color=color,
                    markersize=6, capsize=3, lw=1.6, zorder=3,
                    markerfacecolor=color if significant else "white",
                    markeredgecolor=color, markeredgewidth=1.4)
        ax.annotate(f"{mean:+.3f}", (x + dx, hi), textcoords="offset points",
                    xytext=(0, 3), ha="center", fontsize=7.5, color=color)


def legend_handles(include_low: bool):
    handles = [
        plt.Line2D([], [], color=COLORS["Random"], marker="o", linestyle="", markersize=7,
                   markeredgecolor="black", markeredgewidth=0.6, label="Random"),
        plt.Line2D([], [], color=COLORS["Greedy"], marker="s", linestyle="", markersize=7,
                   markeredgecolor="black", markeredgewidth=0.6, label="Greedy"),
        plt.Line2D([], [], color=COLORS["Open"], marker="^", linestyle="", markersize=7,
                   markeredgecolor="black", markeredgewidth=0.6,
                   label="LLM-Open (High tier = filled)"),
        plt.Line2D([], [], color=COLORS["Closed"], marker="D", linestyle="", markersize=7,
                   markeredgecolor="black", markeredgewidth=0.6,
                   label="LLM-Closed (High tier = filled)"),
    ]
    if include_low:
        handles += [
            plt.Line2D([], [], color=COLORS["Open"], marker="^", linestyle="", markersize=7,
                       markerfacecolor="white", markeredgecolor=COLORS["Open"],
                       markeredgewidth=1.5, label="LLM-Open (Low tier = hollow)"),
            plt.Line2D([], [], color=COLORS["Closed"], marker="D", linestyle="", markersize=7,
                       markerfacecolor="white", markeredgecolor=COLORS["Closed"],
                       markeredgewidth=1.5, label="LLM-Closed (Low tier = hollow)"),
        ]
    handles += [
        plt.Line2D([], [], color="#444", marker="o", linestyle="", markersize=6,
                   markerfacecolor="#444", label="contrast: filled = p < 0.05"),
        plt.Line2D([], [], color="#444", marker="o", linestyle="", markersize=6,
                   markerfacecolor="white", markeredgewidth=1.3,
                   label="contrast: hollow = not detected"),
    ]
    return handles


def _save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(FIGURES / f"{name}.{suffix}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIGURES / f"{name}.png")


def fig_slice(name: str) -> dict:
    """Aligned small multiples for a High-only slice, plus a contrast row."""
    panels, fixed = SLICES[name]
    rng, jitter = np.random.default_rng(0), np.random.default_rng(1)
    n = len(panels)
    fig, axes = plt.subplots(2, n, figsize=(3.0 * n + 0.5, 5.0), sharey="row",
                             gridspec_kw={"height_ratios": [2.4, 1]})
    axes = np.atleast_2d(axes)
    ymin = 1.0
    for c, (key, title) in enumerate(panels):
        ax = axes[0, c]
        draw_condition(ax, key, [("High", key)], rng, jitter)
        for method in ("Random", "Greedy", "LLM-Open", "LLM-Closed"):
            ymin = min(ymin, values(key, method).min())
        ax.set_title(title, fontsize=11, fontweight="bold",
                     color="#202124" if key == "reference" else "#444")
        if key == "reference":
            for spine in ax.spines.values():
                spine.set_edgecolor("#202124")
                spine.set_linewidth(1.8)
        bx = axes[1, c]
        draw_contrasts(bx, key, 0)
        bx.axhline(0, color="black", lw=1.0)
        bx.set_xticks([-0.16, 0.16])
        bx.set_xticklabels(["Open −\nRandom", "Closed −\nOpen"], fontsize=9)
        bx.set_xlim(-0.6, 0.6)
        bx.grid(axis="y", alpha=0.22)
        bx.tick_params(axis="y", labelsize=8)
        if key == "reference":
            for spine in bx.spines.values():
                spine.set_edgecolor("#202124")
                spine.set_linewidth(1.8)
    axes[0, 0].set_ylim(max(0.0, ymin - 0.05), 1.03)
    axes[0, 0].set_ylabel(YLABEL, fontsize=9.5)
    axes[1, 0].set_ylabel(DLABEL, fontsize=9)
    axes[1, 0].margins(y=0.35)
    fig.legend(handles=legend_handles(include_low=False), loc="lower center", ncol=2,
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.09))
    fig.text(0.5, 1.005, fixed, ha="center", fontsize=10.5, color="#444",
             fontweight="bold")
    fig.tight_layout(h_pad=1.4)
    _save(fig, f"slice_{name}")
    return {"panels": [key for key, _ in panels], "fixed": fixed,
            "groups": ["Random", "Greedy", "High(Open, Closed)"]}


def fig_model() -> dict:
    """The one slice with both tiers: Random -> Greedy -> Low -> High, plus a
    proposal-validity panel."""
    rng, jitter = np.random.default_rng(0), np.random.default_rng(1)
    fig, (ax, vx) = plt.subplots(1, 2, figsize=(8.8, 4.9),
                                 gridspec_kw={"width_ratios": [2.6, 1]})
    draw_condition(ax, "reference", [("Low", "model_alt"), ("High", "reference")], rng, jitter)
    ymin = min(values(k, m).min() for k in ("reference", "model_alt")
               for m in ("Random", "Greedy", "LLM-Open", "LLM-Closed"))
    ax.set_ylim(ymin - 0.04, 1.03)
    ax.set_ylabel(YLABEL, fontsize=9.5)
    ax.set_title("fixed: 4 qubits · Ising · B = 8", fontsize=10.5, fontweight="bold",
                 color="#444")
    ax.axvspan(2.5, 3.5, color="#000000", alpha=0.045, zorder=0)
    ax.text(3.0, ymin - 0.025, "previous baseline", ha="center", fontsize=8,
            color="#202124", fontweight="bold")

    tiers = [("Low", "model_alt"), ("High", "reference")]
    xs = np.arange(len(tiers))
    width = 0.36
    for dx, method, workflow in ((-width / 2, "LLM-Open", "Open"),
                                 (width / 2, "LLM-Closed", "Closed")):
        vals = [validity(key, method) for _tier, key in tiers]
        bars = vx.bar(xs + dx, vals, width=width, color=COLORS[workflow],
                      edgecolor="white", linewidth=0.8, label=f"LLM-{workflow}")
        for bar, value in zip(bars, vals, strict=True):
            vx.annotate(f"{value * 100:.0f}%", (bar.get_x() + bar.get_width() / 2, value),
                        textcoords="offset points", xytext=(0, 3), ha="center",
                        fontsize=8.5, fontweight="bold", color=COLORS[workflow])
    vx.set_xticks(xs)
    vx.set_xticklabels([tier for tier, _ in tiers], fontsize=9.5)
    vx.set_ylim(0, 1.22)
    vx.set_yticks(np.arange(0, 1.01, 0.25))
    vx.set_ylabel("Share of evaluated LLM proposals\nthat met the gate-count contract",
                  fontsize=8.5)
    vx.set_title("Proposal validity", fontsize=10.5, fontweight="bold", color="#444")
    vx.grid(axis="y", alpha=0.22)
    vx.legend(fontsize=8, loc="upper center", ncol=2, frameon=False)
    fig.legend(handles=legend_handles(include_low=True)[:6], loc="lower center", ncol=3,
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    _save(fig, "slice_model")
    return {"panels": ["reference", "model_alt"], "fixed": "4 qubits · Ising · B = 8",
            "groups": ["Random", "Greedy", "Low(Open, Closed)", "High(Open, Closed)"]}


def fig_baseline() -> dict:
    rng, jitter = np.random.default_rng(0), np.random.default_rng(1)
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    draw_condition(ax, "reference", [("High", "reference")], rng, jitter)
    ymin = min(values("reference", m).min()
               for m in ("Random", "Greedy", "LLM-Open", "LLM-Closed"))
    ax.set_ylim(ymin - 0.04, 1.02)
    ax.set_ylabel(YLABEL, fontsize=9.5)
    ax.set_title("Previous baseline: 4 qubits · Ising chain · B = 8 · High model · "
                 "12 paired seeds", fontsize=10, fontweight="bold")
    ax.legend(handles=legend_handles(include_low=False)[:4], fontsize=8, loc="lower right",
              framealpha=0.95)
    fig.tight_layout()
    _save(fig, "baseline_groups")
    return {"panels": ["reference"], "groups": ["Random", "Greedy", "High(Open, Closed)"]}


def main() -> None:
    manifest = {"tiers": TIER, "group_order": ["Random", "Greedy", "Low", "High"],
                "figures": {}}
    manifest["figures"]["baseline_groups"] = fig_baseline()
    for name in SLICES:
        manifest["figures"][f"slice_{name}"] = fig_slice(name)
    manifest["figures"]["slice_model"] = fig_model()
    (ROOT / "slices.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest["figures"], indent=1))


if __name__ == "__main__":
    main()
