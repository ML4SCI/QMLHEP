"""Build the QAE robustness figures and cross-condition tables from committed
artifacts.

Inputs:  outputs/qae_robustness/<condition>/{selected_results.csv,
         paired_stats.json,proposal_quality.json,api_usage.json}
Outputs: outputs/qae_robustness/figures/*.{png,svg}
         outputs/qae_robustness/{summary.json,robustness_table.csv,
         contrast_table.csv}

Visual language is the previous study's: fixed method colours, every seed
point drawn, bootstrap 95% CI error bars, "higher is better" axis labels.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from llm_vqc.experiments.qae_robustness import study

ROOT = Path("outputs/qae_robustness")
FIGURES = ROOT / "figures"

METHOD_ORDER = ["Random", "Greedy", "LLM-Open", "LLM-Closed"]
COLORS = {
    "Random": "#9aa0a6",
    "Greedy": "#f9ab00",
    "LLM-Open": "#1a73e8",
    "LLM-Closed": "#d93025",
}
MARKERS = {"Random": "o", "Greedy": "s", "LLM-Open": "^", "LLM-Closed": "D"}
LEGEND = {
    "Random": "Random (no semantics, B independent draws)",
    "Greedy": "Greedy (no semantics, B/2 starts + B/2 one-change steps)",
    "LLM-Open": "LLM-Open (semantics, open-loop batch of B)",
    "LLM-Closed": "LLM-Closed (semantics, B/2 starts + B/2 redesigns)",
}
YLABEL = ("Held-out test trash fidelity  $F_{trash}$\n"
          "(higher is better)")
DLABEL = "Paired difference in test fidelity\n(0 = no effect)"
CONTRASTS = [
    ("LLM-Closed_minus_LLM-Open", "Closed - Open", "#d93025"),
    ("LLM-Open_minus_Random", "Open - Random", "#1a73e8"),
    ("LLM-Closed_minus_Random", "Closed - Random", "#8430ce"),
    ("Greedy_minus_Random", "Greedy - Random", "#f9ab00"),
]
PREVIOUS = "previous condition"


def _boot_ci(values, rng):
    boots = np.array([rng.choice(values, size=len(values), replace=True).mean()
                      for _ in range(10_000)])
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def _selected(key: str) -> pd.DataFrame | None:
    path = ROOT / key / "selected_results.csv"
    return pd.read_csv(path) if path.exists() else None


def _stats(key: str) -> dict:
    path = ROOT / key / "paired_stats.json"
    return json.loads(path.read_text()) if path.exists() else {}


def _save(fig, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(FIGURES / f"{name}.{suffix}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIGURES / f"{name}.png")


def _method_legend(ax, ncol=1, loc="lower left", fontsize=7.2):
    handles = [
        plt.Line2D([], [], color=COLORS[m], marker=MARKERS[m], linestyle="-",
                   markersize=5.5, label=LEGEND[m])
        for m in METHOD_ORDER
    ]
    ax.legend(handles=handles, fontsize=fontsize, ncol=ncol, loc=loc,
              framealpha=0.92, borderpad=0.5, labelspacing=0.35,
              handletextpad=0.5, edgecolor="#dddddd")


# ------------------------------------------------------- single condition ---

def fig_single_condition(key: str, name: str, title: str) -> None:
    selected = _selected(key)
    if selected is None:
        return
    rng = np.random.default_rng(0)
    jitter = np.random.default_rng(1)
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    lows = []
    for i, method in enumerate(METHOD_ORDER):
        values = selected[selected["method"] == method]["test_fid"].to_numpy()
        if not len(values):
            continue
        lows.append(values.min())
        x = i + jitter.uniform(-0.13, 0.13, size=len(values))
        ax.scatter(x, values, s=26, color=COLORS[method], alpha=0.55,
                   edgecolors="white", linewidths=0.6, zorder=2)
        mean = values.mean()
        low, high = _boot_ci(values, rng)
        ax.errorbar(i, mean, yerr=[[mean - low], [high - mean]], fmt=MARKERS[method],
                    color=COLORS[method], markersize=9, capsize=6, lw=2.2, zorder=3,
                    markeredgecolor="black", markeredgewidth=0.7)
        ax.annotate(f"{mean:.4f}", (i, high), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=9.5, fontweight="bold",
                    color=COLORS[method])
    ax.set_xticks(range(len(METHOD_ORDER)))
    ax.set_xticklabels(METHOD_ORDER, fontsize=10)
    ax.set_ylabel(YLABEL, fontsize=10)
    ax.set_title(title, fontsize=11.5, fontweight="bold")
    ax.set_ylim(min(lows) - 0.03, 1.005)
    ax.grid(axis="y", alpha=0.25)
    ax.text(0.995, 0.02, "12 paired seeds; dots = seeds, bars = mean with bootstrap 95% CI",
            transform=ax.transAxes, ha="right", fontsize=8, color="#444")
    _save(fig, name)


# ------------------------------------------------------------ factor grid ---

def fig_factor(keys: list[str], levels: list[str], name: str, title: str,
               xlabel: str, previous_index: int, suptitle: bool = False) -> None:
    present = [(k, lab) for k, lab in zip(keys, levels, strict=True) if _selected(k) is not None]
    if len(present) < 2:
        print("skipping", name, "- not enough completed conditions")
        return
    rng = np.random.default_rng(0)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13.2, 4.25),
                                 gridspec_kw={"width_ratios": [1.15, 1]})
    xs = np.arange(len(present))

    for method in METHOD_ORDER:
        means, los, his = [], [], []
        for j, (key, _label) in enumerate(present):
            values = _selected(key)
            values = values[values["method"] == method]["test_fid"].to_numpy()
            means.append(values.mean())
            low, high = _boot_ci(values, rng)
            los.append(means[-1] - low)
            his.append(high - means[-1])
            jitter = np.random.default_rng(100 + j).uniform(-0.06, 0.06, len(values))
            ax.scatter(j + jitter, values, s=13, color=COLORS[method], alpha=0.28,
                       edgecolors="none", zorder=1)
        ax.errorbar(xs, means, yerr=[los, his], color=COLORS[method],
                    marker=MARKERS[method], markersize=8, capsize=5, lw=2.0, zorder=3,
                    markeredgecolor="black", markeredgewidth=0.6)
    ax.axvspan(previous_index - 0.32, previous_index + 0.32, color="#000000", alpha=0.055,
               zorder=0)
    ax.annotate(PREVIOUS, (previous_index, 1.045), ha="center", fontsize=8.5,
                color="#333", fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels([lab for _k, lab in present], fontsize=10)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(YLABEL, fontsize=9.5)
    ax.set_xlim(-0.45, len(present) - 0.55)
    ax.set_ylim(0, 1.10)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.grid(axis="y", alpha=0.25)
    ax.set_title("Selected-architecture quality", fontsize=10.5, fontweight="bold")
    _method_legend(ax)

    for offset, (contrast, label, color) in zip(
            np.linspace(-0.18, 0.18, len(CONTRASTS)), CONTRASTS, strict=True):
        means, los, his = [], [], []
        for key, _label in present:
            entry = _stats(key).get(contrast, {})
            mean = entry.get("mean_paired_gain", np.nan)
            ci = entry.get("bootstrap_95ci", [np.nan, np.nan])
            means.append(mean)
            los.append(mean - ci[0])
            his.append(ci[1] - mean)
        bx.errorbar(xs + offset, means, yerr=[los, his], fmt="o", color=color,
                    markersize=6, capsize=4, lw=1.8, label=label,
                    markeredgecolor="black", markeredgewidth=0.5)
    bx.axhline(0, color="black", lw=1.0)
    bx.axvspan(previous_index - 0.32, previous_index + 0.32, color="#000000", alpha=0.055,
               zorder=0)
    bx.set_xticks(xs)
    bx.set_xticklabels([lab for _k, lab in present], fontsize=10)
    bx.set_xlabel(xlabel, fontsize=10)
    bx.set_ylabel(DLABEL, fontsize=9)
    bx.set_xlim(-0.45, len(present) - 0.55)
    bx.grid(axis="y", alpha=0.25)
    bx.set_title("Paired contrasts (bootstrap 95% CI)", fontsize=10.5, fontweight="bold")
    bx.margins(y=0.30)
    bx.legend(fontsize=7.6, ncol=2, loc="upper center", framealpha=0.92,
              columnspacing=1.0, handletextpad=0.4, edgecolor="#dddddd")

    if suptitle:
        fig.suptitle(title, fontsize=12.5, fontweight="bold", y=1.015)
    _save(fig, name)


# ----------------------------------------------------------- two-cell A/B ---

def fig_ab(key_a: str, key_b: str, label_a: str, label_b: str,
           name: str, title: str, note: str, suptitle: bool = False) -> None:
    if _selected(key_a) is None or _selected(key_b) is None:
        print("skipping", name, "- missing condition")
        return
    rng = np.random.default_rng(0)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13.2, 4.25),
                                 gridspec_kw={"width_ratios": [1.25, 1]})
    width = 0.36
    lows = []
    for i, method in enumerate(METHOD_ORDER):
        for side, key in enumerate((key_a, key_b)):
            values = _selected(key)
            values = values[values["method"] == method]["test_fid"].to_numpy()
            lows.append(values.min())
            centre = i + (side - 0.5) * width * 1.35
            jitter = np.random.default_rng(7 + side).uniform(-0.055, 0.055, len(values))
            ax.scatter(centre + jitter, values, s=18,
                       color=COLORS[method], alpha=0.30 if side == 0 else 0.55,
                       edgecolors="none", zorder=1)
            mean = values.mean()
            low, high = _boot_ci(values, rng)
            ax.errorbar(centre, mean, yerr=[[mean - low], [high - mean]],
                        fmt=MARKERS[method], color=COLORS[method],
                        markersize=8, capsize=5, lw=2.0, zorder=3,
                        markerfacecolor=COLORS[method] if side else "white",
                        markeredgecolor=COLORS[method], markeredgewidth=1.4)
    ax.set_xticks(range(len(METHOD_ORDER)))
    ax.set_xticklabels(METHOD_ORDER, fontsize=10)
    ax.set_ylabel(YLABEL, fontsize=9.5)
    ax.set_ylim(min(min(lows) - 0.04, 0.55), 1.02)
    ax.grid(axis="y", alpha=0.25)
    ax.set_title(f"open marker = {label_a}      filled marker = {label_b}",
                 fontsize=10.5, fontweight="bold")
    _method_legend(ax, loc="lower right")

    ys = np.arange(len(CONTRASTS))[::-1]
    for y, (contrast, _label, color) in zip(ys, CONTRASTS, strict=True):
        for side, key in enumerate((key_a, key_b)):
            entry = _stats(key).get(contrast, {})
            mean = entry.get("mean_paired_gain", np.nan)
            ci = entry.get("bootstrap_95ci", [np.nan, np.nan])
            offset = 0.17 * (side - 0.5) * 2
            bx.errorbar(mean, y + offset, xerr=[[mean - ci[0]], [ci[1] - mean]],
                        fmt="o" if side else "o", color=color, markersize=7,
                        capsize=4, lw=1.8,
                        markerfacecolor=color if side else "white",
                        markeredgecolor=color, markeredgewidth=1.5)
    bx.axvline(0, color="black", lw=1.0)
    bx.set_yticks(ys)
    bx.set_yticklabels([label for _c, label, _col in CONTRASTS], fontsize=9.5)
    bx.set_xlabel(DLABEL, fontsize=9)
    bx.set_ylim(-0.6, len(CONTRASTS) - 0.4)
    bx.grid(axis="x", alpha=0.25)
    bx.set_title("Paired contrasts (bootstrap 95% CI)", fontsize=10.5, fontweight="bold")

    if suptitle:
        fig.suptitle(title, fontsize=12.5, fontweight="bold", y=1.015)
        fig.text(0.5, -0.035, note, ha="center", fontsize=8.5, color="#444")
    _save(fig, name)


# ------------------------------------------------------------ forest plot ---

def fig_summary(order: list[tuple[str, str]], name: str) -> None:
    present = [(k, lab) for k, lab in order if _stats(k)]
    if len(present) < 2:
        return
    fig, axes = plt.subplots(1, len(CONTRASTS), figsize=(13.2, 3.65), sharey=True)
    ys = np.arange(len(present))[::-1]
    for ax, (contrast, label, color) in zip(axes, CONTRASTS, strict=True):
        reference = _stats("reference").get(contrast, {}).get("mean_paired_gain")
        if reference is not None:
            ax.axvline(reference, color="#666", lw=1.1, ls="--", zorder=1)
        for y, (key, _lab) in zip(ys, present, strict=True):
            entry = _stats(key).get(contrast, {})
            mean = entry.get("mean_paired_gain", np.nan)
            ci = entry.get("bootstrap_95ci", [np.nan, np.nan])
            significant = (entry.get("wilcoxon_exact_two_sided_p", 1.0) or 1.0) < 0.05
            ax.errorbar(mean, y, xerr=[[mean - ci[0]], [ci[1] - mean]], fmt="o",
                        color=color, markersize=8 if key != "reference" else 10,
                        capsize=4, lw=1.9, zorder=3,
                        markerfacecolor=color if significant else "white",
                        markeredgecolor=color, markeredgewidth=1.6)
        ax.axvline(0, color="black", lw=1.0, zorder=2)
        ax.set_title(label, fontsize=11, fontweight="bold", color=color)
        ax.grid(axis="x", alpha=0.25)
        ax.set_xlabel("paired difference", fontsize=9)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([lab for _k, lab in present], fontsize=9.5)
    axes[0].set_ylim(-0.7, len(present) - 0.3)
    _save(fig, name)


def fig_validity(order: list[tuple[str, str]], name: str) -> None:
    """Resource-contract compliance, split by the arm that made the call.

    The two semantic arms differ in how many candidates one call must
    produce (B for the open batch, B/2 for the warm batch, 1 per redesign),
    so this panel is what separates "the model reasoned worse" from "the
    model could not satisfy the contract in one response".
    """
    rows = []
    for key, label in order:
        path = ROOT / key / "proposal_quality.json"
        if not path.exists():
            continue
        quality = json.loads(path.read_text())
        entry = {"label": label}
        for method in ("LLM-Open", "LLM-Closed"):
            stats = quality.get(method, {})
            evaluations = stats.get("evaluations", 0)
            fallbacks = stats.get("fallback_evaluations", 0)
            entry[method] = (1 - fallbacks / evaluations) if evaluations else None
        rows.append(entry)
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(8.6, 3.9))
    xs = np.arange(len(rows))
    width = 0.38
    for offset, method in ((-width / 2, "LLM-Open"), (width / 2, "LLM-Closed")):
        values = [r[method] if r[method] is not None else 0 for r in rows]
        bars = ax.bar(xs + offset, values, width=width, color=COLORS[method],
                      label=method, edgecolor="white", linewidth=0.8)
        for bar, value in zip(bars, values, strict=True):
            ax.annotate(f"{value * 100:.0f}", (bar.get_x() + bar.get_width() / 2, value),
                        textcoords="offset points", xytext=(0, 3), ha="center",
                        fontsize=8.5, fontweight="bold", color=COLORS[method])
    ax.axhline(rows[0]["LLM-Open"], color="#888", lw=1.0, ls="--", alpha=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([r["label"] for r in rows], fontsize=8.5, rotation=20, ha="right")
    ax.set_ylabel("Share of evaluated candidates that were\nvalid model proposals "
                  "(higher is better)", fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8.5, loc="lower left", ncol=2, framealpha=0.95)
    ax.set_title("Resource-contract compliance by arm\n"
                 "(dashed line = previous condition's open batch)",
                 fontsize=10.5, fontweight="bold")
    _save(fig, name)


def main() -> None:
    summary = study.summarize_all(ROOT)
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    table = study.summary_table(summary)
    table.to_csv(ROOT / "robustness_table.csv", index=False)

    contrast_rows = []
    for key, entry in summary["conditions"].items():
        if entry.get("status") == "missing":
            continue
        for name, contrast in entry["contrasts"].items():
            if contrast is None:
                continue
            contrast_rows.append({
                "condition": key, "label": entry["label"], "contrast": name,
                "mean_paired_gain": contrast["mean_paired_gain"],
                "ci_lo": contrast["bootstrap_95ci"][0],
                "ci_hi": contrast["bootstrap_95ci"][1],
                "cohens_dz": contrast["cohens_dz"],
                "wins": contrast["wins_b"], "n": contrast["n"],
                "wilcoxon_p": contrast["wilcoxon_exact_two_sided_p"],
                "sign_test_p": contrast["sign_test_two_sided_p"],
            })
    pd.DataFrame(contrast_rows).to_csv(ROOT / "contrast_table.csv", index=False)

    fig_single_condition(
        "reference", "fig01_previous_baseline",
        "Previous QAE baseline: 4 qubits, transverse-field Ising chain, B = 8",
    )
    fig_factor(["budget_b4", "reference", "budget_b16"], ["B = 4", "B = 8", "B = 16"],
               "fig02_budget", "Effect of the candidate-evaluation budget",
               "Candidate-evaluation budget B", previous_index=1)
    fig_factor(["reference", "qubits_n6", "qubits_n8"],
               ["4 qubits", "6 qubits", "8 qubits"],
               "fig03_qubits", "Effect of the qubit count",
               "Qubits (3 rotations + 1 CNOT per qubit, all methods)",
               previous_index=0)
    fig_ab("reference", "hamiltonian_xxz",
           "transverse-field Ising", "XXZ Heisenberg",
           "fig04_hamiltonian", "Effect of the Hamiltonian family (4 qubits, B = 8)",
           "Same circuit space, trainer, prompts and seeds; only the Hamiltonian "
           "facts in the task description change.")
    fig_ab("reference", "model_alt",
           "reference model", "alternative model",
           "fig05_model", "Effect of the underlying language model (4 qubits, B = 8)",
           "Identical prompt bytes, temperature, schema, retries and token limit; "
           "only the model identifier changes.")
    order = [
        ("reference", "Previous baseline"),
        ("budget_b4", "Budget B = 4"),
        ("budget_b16", "Budget B = 16"),
        ("qubits_n6", "6 qubits"),
        ("qubits_n8", "8 qubits"),
        ("hamiltonian_xxz", "XXZ Hamiltonian"),
        ("model_alt", "Alternative model"),
    ]
    fig_summary(order, "fig06_summary_forest")
    fig_validity(order, "fig07_proposal_validity")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
