"""Figures for the minimum-budget-to-target study.

Validation only. Every panel states the metric, the target, the condition,
the configured budget B, the seed count, and whether the cell was measured
in this study or re-analysed from the historical logs. Colours and markers
are the previous deck's, unchanged.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

DECK_RC = {
    "font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13,
    "xtick.labelsize": 12.5, "ytick.labelsize": 12.5, "legend.fontsize": 11.5,
}

METHOD_ORDER = ["Random", "Greedy", "LLM-Open", "LLM-Closed"]
COLORS = {"Random": "#9aa0a6", "Greedy": "#f9ab00",
          "LLM-Open": "#1a73e8", "LLM-Closed": "#d93025"}
MARKERS = {"Random": "o", "Greedy": "s", "LLM-Open": "^", "LLM-Closed": "D"}
LEGEND = {
    "Random": "Random (no semantics, B independent draws)",
    "Greedy": "Greedy (no semantics, B/2 starts + B/2 one-change steps)",
    "LLM-Open": "LLM-Open (semantics, open-loop batch of B)",
    "LLM-Closed": "LLM-Closed (semantics, B/2 starts + B/2 redesigns)",
}
REQUIRED = 10
N_SEEDS = 12
TARGETS = (0.95, 0.99)

# Which cells this study executed, and which are re-analysed history.
NEW_CELLS = {"target_tfim_b6", "target_xxz_b10"}
# Audience-facing names: the deck says "Ising chain", never the internal key.
TITLES = {
    "budget_b4": "4-qubit Ising chain, reference model, B = 4",
    "target_tfim_b6": "4-qubit Ising chain, reference model, B = 6",
    "reference": "4-qubit Ising chain, reference model, B = 8",
    "budget_b16": "4-qubit Ising chain, reference model, B = 16",
    "hamiltonian_xxz": "4-qubit XXZ chain, reference model, B = 8",
    "target_xxz_b10": "4-qubit XXZ chain, reference model, B = 10",
    "qubits_n6": "6-qubit Ising chain, reference model, B = 8",
    "qubits_n8": "8-qubit Ising chain, reference model, B = 8",
    "model_alt": "4-qubit Ising chain, alternative model, B = 8",
}
TFIM_LADDER = [("budget_b4", 4), ("target_tfim_b6", 6),
               ("reference", 8), ("budget_b16", 16)]
XXZ_LADDER = [("hamiltonian_xxz", 8), ("target_xxz_b10", 10)]
YLABEL = ("Best-so-far validation trash fidelity\n"
          "$F_{val}$ (higher is better)")
FOOTER = ("Metric: validation trash fidelity (never test). "
          f"{N_SEEDS} paired seeds (0-11). "
          f"Pass = at least {REQUIRED}/{N_SEEDS} seeds at or above the target.")


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def provenance(key: str) -> str:
    return ("measured in this study" if key in NEW_CELLS
            else "re-analysed historical log")


def _finish(fig, ax_or_axes, path: Path, footer: str = FOOTER) -> None:
    fig.text(0.5, 0.005, footer, ha="center", fontsize=7.6, color="#444")
    fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def curve_figure(curves: list[dict], key: str, n_warm: int, out: Path) -> None:
    """Attainment count and mean best-so-far against candidates evaluated."""
    rows = [r for r in curves if r["condition"] == key]
    if not rows:
        return
    budget = int(rows[0]["configured_budget"])
    methods = [m for m in METHOD_ORDER if any(r["method"] == m for r in rows)]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.1))

    for ax, (field, label, target) in zip(axes, [
        ("success_095", f"Seeds reaching $F_{{val}} \\geq 0.95$ (of {N_SEEDS})", 0.95),
        ("success_099", f"Seeds reaching $F_{{val}} \\geq 0.99$ (of {N_SEEDS})", 0.99),
    ], strict=True):
        for method in methods:
            series = sorted((r for r in rows if r["method"] == method),
                            key=lambda r: int(r["candidates_used"]))
            xs = [int(r["candidates_used"]) for r in series]
            ys = [int(r[field]) for r in series]
            ax.plot(xs, ys, color=COLORS[method], marker=MARKERS[method],
                    markersize=6, lw=2.0, markeredgecolor="black",
                    markeredgewidth=0.5, label=LEGEND[method])
        ax.axhline(REQUIRED, color="black", lw=1.1, ls="--")
        ax.text(budget, REQUIRED + 0.25, f"pass line {REQUIRED}/{N_SEEDS}",
                ha="right", fontsize=8, fontweight="bold")
        ax.axvline(n_warm + 0.5, color="#666", lw=1.0, ls=":")
        ax.text(n_warm + 0.62, 0.4, "exploration | refinement",
                fontsize=7.6, color="#555", rotation=90, va="bottom")
        ax.set_xlim(0.6, budget + 0.4)
        ax.set_ylim(-0.4, N_SEEDS + 0.6)
        ax.set_xticks(range(1, budget + 1))
        ax.set_yticks(range(0, N_SEEDS + 1, 2))
        ax.set_xlabel("Candidates trained and evaluated per seed")
        ax.set_ylabel(label, fontsize=9)
        ax.set_title(f"Target $F_{{val}} \\geq {target}$", fontsize=10)
        ax.grid(alpha=0.25)

    handles = [Line2D([], [], color=COLORS[m], marker=MARKERS[m], lw=2,
                      markeredgecolor="black", markeredgewidth=0.5,
                      label=LEGEND[m]) for m in methods]
    fig.legend(handles=handles, loc="lower center", ncol=min(2, len(methods)),
               bbox_to_anchor=(0.5, -0.16), frameon=True, fontsize=8.4,
               edgecolor="#dddddd")
    fig.suptitle(f"{TITLES.get(key, key)}  ({provenance(key)})", fontsize=11.5)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    _finish(fig, axes, out / f"attainment_{key}")


def mean_curve_figure(curves: list[dict], key: str, n_warm: int, out: Path) -> None:
    rows = [r for r in curves if r["condition"] == key]
    if not rows:
        return
    budget = int(rows[0]["configured_budget"])
    methods = [m for m in METHOD_ORDER if any(r["method"] == m for r in rows)]
    fig, ax = plt.subplots(figsize=(6.4, 4.1))
    for method in methods:
        series = sorted((r for r in rows if r["method"] == method),
                        key=lambda r: int(r["candidates_used"]))
        xs = [int(r["candidates_used"]) for r in series]
        ax.plot(xs, [float(r["mean_best_validation"]) for r in series],
                color=COLORS[method], marker=MARKERS[method], markersize=6,
                lw=2.0, markeredgecolor="black", markeredgewidth=0.5,
                label=LEGEND[method])
        ax.fill_between(xs, [float(r["min_best_validation"]) for r in series],
                        [float(r["max_best_validation"]) for r in series],
                        color=COLORS[method], alpha=0.10, lw=0)
    for target, style in zip(TARGETS, ("--", ":"), strict=False):
        ax.axhline(target, color="black", lw=1.0, ls=style)
        ax.text(budget, target + 0.002, f"target {target}", ha="right", fontsize=8)
    ax.axvline(n_warm + 0.5, color="#666", lw=1.0, ls=":")
    ax.set_xticks(range(1, budget + 1))
    ax.set_xlabel("Candidates trained and evaluated per seed")
    ax.set_ylabel(YLABEL, fontsize=9)
    ax.set_title(f"{TITLES.get(key, key)}\n({provenance(key)}; band = min-max over "
                 f"{N_SEEDS} seeds)", fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.6, loc="lower right", edgecolor="#dddddd")
    fig.tight_layout()
    _finish(fig, ax, out / f"mean_best_{key}")


def ladder_figure(endpoints: list[dict], ladder: list[tuple[str, int]],
                  name: str, title: str, out: Path,
                  methods: list[str] | None = None,
                  unverified: list[int] = ()) -> None:
    """Seeds reaching 0.95 against configured budget, one line per method.

    Each point is an INDEPENDENTLY executed run at that configured budget.
    The points are joined only to make the ladder readable; no monotonicity
    is claimed, and unverified budgets are drawn as an explicit gap.
    """
    lookup = {(r["condition"], r["method"]): r for r in endpoints
              if float(r["target"]) == 0.95}
    methods = methods or METHOD_ORDER
    plt.rcParams.update(DECK_RC)
    fig, ax = plt.subplots(figsize=(9.2, 4.3))
    for method in methods:
        xs, ys, provs = [], [], []
        for key, budget in ladder:
            row = lookup.get((key, method))
            if row is None:
                continue
            xs.append(budget)
            ys.append(int(row["successes"]))
            provs.append(key in NEW_CELLS)
        if not xs:
            continue
        ax.plot(xs, ys, color=COLORS[method], lw=1.8, alpha=0.75, zorder=2)
        for x, y, is_new in zip(xs, ys, provs, strict=False):
            ax.plot([x], [y], marker=MARKERS[method], markersize=11 if is_new else 8,
                    color=COLORS[method], markeredgecolor="black",
                    markeredgewidth=1.6 if is_new else 0.6, zorder=3)
        ax.plot([], [], color=COLORS[method], marker=MARKERS[method], lw=1.8,
                markeredgecolor="black", markeredgewidth=0.6, label=LEGEND[method])
    ax.axhline(REQUIRED, color="black", lw=1.2, ls="--")
    ax.text(ax.get_xlim()[0], REQUIRED + 0.3, f"pass line {REQUIRED}/{N_SEEDS}",
            ha="left", fontsize=12.5, fontweight="bold")
    for budget in unverified:
        ax.axvline(budget, color="#bbbbbb", lw=8, alpha=0.35, zorder=0)
        ax.text(budget, N_SEEDS + 0.35, "not run", ha="center", fontsize=11,
                color="#666")
    ax.set_xticks([b for _, b in ladder] + list(unverified))
    ax.set_ylim(-0.5, N_SEEDS + 1.1)
    ax.set_yticks(range(0, N_SEEDS + 1, 2))
    ax.set_xlabel("Configured budget B (candidates per seed)\n"
                  "each point is a separate run; thick outline = measured now",
                  fontsize=13)
    ax.set_ylabel(f"Seeds reaching $F_{{val}} \\geq 0.95$\n(of {N_SEEDS})",
                  fontsize=13)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=11.5, loc="upper center", bbox_to_anchor=(0.5, -0.27),
              ncol=2, edgecolor="#dddddd", frameon=True)
    fig.tight_layout()
    _finish(fig, ax, out / name,
            FOOTER + " Runs at different B are separate policies and are never concatenated.")


def margin_figure(audit: Path, pairs: list[tuple[str, int, str]], name: str,
                  title: str, method: str, out: Path) -> None:
    """Per-seed final best validation against the target, for one method at
    two budgets.

    A count of seeds above a threshold says nothing about how far above it
    they are. When a whole distribution sits within a few thousandths of the
    target, the count is a knife-edge statistic and a tiny shift flips
    several seeds; this panel makes that visible instead of hiding it behind
    the count.
    """
    plt.rcParams.update(DECK_RC)
    series = []
    for key, budget, label in pairs:
        path = audit / "data" / key / "candidate_results.csv"
        if not path.exists():
            continue
        best: dict[int, float] = {}
        for row in read_rows(path):
            if row["method"] != method:
                continue
            seed = int(row["seed"])
            best[seed] = max(best.get(seed, 0.0), float(row["val_fid"]))
        series.append((label, budget, [best[s] for s in sorted(best)],
                       key in NEW_CELLS))
    if len(series) < 2:
        return
    fig, ax = plt.subplots(figsize=(9.2, 4.3))
    width = 0.36
    for i, (label, _budget, values, is_new) in enumerate(series):
        xs = [s + (i - 0.5) * width for s in range(len(values))]
        ax.bar(xs, values, width=width * 0.9, color=COLORS[method],
               alpha=1.0 if is_new else 0.42, edgecolor="black",
               linewidth=1.3 if is_new else 0.5,
               label=f"{label} ({'measured now' if is_new else 'earlier run'})")
    ax.axhline(0.95, color="black", lw=1.4, ls="--")
    ax.text(-0.55, 0.9505, "target 0.95", fontsize=12.5, fontweight="bold",
            va="bottom")
    low = min(min(v) for _, _, v, _ in series)
    ax.set_ylim(min(0.93, low - 0.005), 0.985)
    ax.set_xticks(range(len(series[0][2])))
    ax.set_xlabel("Seed", fontsize=13)
    ax.set_ylabel("Final best-so-far\nvalidation trash fidelity", fontsize=9.5)
    ax.set_title(title, fontsize=10.5)
    ax.grid(alpha=0.25, axis="y")
    ax.legend(fontsize=8.2, loc="lower right", edgecolor="#dddddd", ncol=2)
    fig.tight_layout()
    _finish(fig, ax, out / name,
            "Every seed of both runs lies within a few thousandths of the "
            "target, so the pass count is highly sensitive to small shifts.")


def per_seed_figure(hits: list[dict], key: str, out: Path) -> None:
    """Per-seed first-hit candidate index at 0.95; censored seeds marked."""
    rows = [r for r in hits if r["condition"] == key and float(r["target"]) == 0.95]
    if not rows:
        return
    budget = int(rows[0]["configured_budget"])
    methods = [m for m in METHOD_ORDER if any(r["method"] == m for r in rows)]
    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    width = 0.8 / len(methods)
    for i, method in enumerate(methods):
        by_seed = {int(r["seed"]): r for r in rows if r["method"] == method}
        for seed in range(N_SEEDS):
            row = by_seed.get(seed)
            x = seed + (i - (len(methods) - 1) / 2) * width
            if row is None:
                continue
            if row["right_censored"] in ("True", True, "true"):
                ax.plot([x], [budget + 0.7], marker="x", markersize=7,
                        color=COLORS[method], markeredgewidth=1.6)
            else:
                ax.bar(x, int(row["first_hit_in_this_trace"]), width=width * 0.92,
                       color=COLORS[method], edgecolor="black", linewidth=0.4)
    ax.axhline(budget + 0.7, color="#999", lw=0.8, ls=":")
    ax.text(N_SEEDS - 0.5, budget + 0.95, "x = never reached within B",
            ha="right", fontsize=7.8, color="#555")
    ax.set_xticks(range(N_SEEDS))
    ax.set_xlabel("Seed")
    ax.set_ylabel("First candidate index\nreaching $F_{val} \\geq 0.95$", fontsize=9)
    ax.set_ylim(0, budget + 1.6)
    ax.set_title(f"{TITLES.get(key, key)}  ({provenance(key)})", fontsize=10.5)
    handles = [Line2D([], [], color=COLORS[m], lw=6, label=LEGEND[m])
               for m in methods]
    ax.legend(handles=handles, fontsize=7.2, loc="upper left", ncol=2,
              edgecolor="#dddddd")
    ax.grid(alpha=0.22, axis="y")
    fig.tight_layout()
    _finish(fig, ax, out / f"first_hit_{key}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True,
                        help="directory produced by analyze_budget_targets.py")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = args.out or (args.audit / "figures")
    out.mkdir(parents=True, exist_ok=True)

    curves = read_rows(args.audit / "validation_curves.csv")
    endpoints = read_rows(args.audit / "endpoint_attainment.csv")
    hits = read_rows(args.audit / "per_seed_first_hits.csv")

    budgets = {r["condition"]: int(r["configured_budget"]) for r in curves}
    for key, budget in budgets.items():
        curve_figure(curves, key, budget // 2, out)
        mean_curve_figure(curves, key, budget // 2, out)
        per_seed_figure(hits, key, out)

    ladder_figure(
        endpoints, [c for c in TFIM_LADDER if c[0] in budgets], "ladder_tfim",
        "Ising chain: seeds reaching $F_{val} \\geq 0.95$ at each configured budget",
        out, unverified=[12, 14])
    margin_figure(
        args.audit,
        [("hamiltonian_xxz", 8, "B = 8"), ("target_xxz_b10", 10, "B = 10")],
        "margin_xxz",
        "XXZ chain, closed loop: how far each seed lands from the target",
        "LLM-Closed", out)
    margin_figure(
        args.audit,
        [("reference", 8, "B = 8"), ("target_tfim_b6", 6, "B = 6")],
        "margin_tfim_open",
        "Ising chain, open loop: how far each seed lands from the target",
        "LLM-Open", out)
    ladder_figure(
        endpoints, [c for c in XXZ_LADDER if c[0] in budgets], "ladder_xxz",
        "XXZ chain: closed-loop boundary probe (the only method run at B = 10)",
        out, methods=["LLM-Closed"])

    index = sorted(p.name for p in out.glob("*.png"))
    (out / "figures.json").write_text(json.dumps({
        "figures": index,
        "metric": "validation trash fidelity",
        "targets": list(TARGETS),
        "n_seeds": N_SEEDS,
        "pass_rule": f"{REQUIRED} of {N_SEEDS} paired seeds",
        "measured_in_this_study": sorted(NEW_CELLS),
    }, indent=2) + "\n")
    print(f"Wrote {len(index)} figures to {out}")


if __name__ == "__main__":
    main()
