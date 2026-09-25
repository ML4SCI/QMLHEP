#!/usr/bin/env python
"""Regenerate every figure in `docs/presentation/202607_methodology_results/`.

Reproducible and offline: no API calls, no access to the raw SQLite store.
The two data sources are:

1. The deterministic task/model/training code (`llm_vqc.*`) — used to draw
   the data-generation figures, the concrete selected circuits, the training
   curves, and predicted-vs-true, all from a documented fixed seed.
2. `data/pilot_nonllm.json` — the *committed*, sanitized numeric extract of
   the B=25 non-LLM pilot (medians, per-run validation/test RMSE, best-so-far
   anytime curves, selected-circuit IRs). It was produced once from the
   durable store `runs/pilot_t1/results.sqlite`; every number in it is
   re-derivable by `scripts/analyze_pilot.py`.

Run:  .venv/bin/python scripts/build_methodology_figures.py

Every figure is written as both PNG (for slide tools) and SVG (for editing).
The script asserts that locally re-trained numbers match the stored pilot
numbers, so a regression in the training code cannot silently change a figure.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pennylane as qml  # noqa: E402
import torch  # noqa: E402

from llm_vqc.evaluation.model import HybridQNNModel  # noqa: E402
from llm_vqc.evaluation.training import TrainingConfig, train_model  # noqa: E402
from llm_vqc.ir.compiler_pennylane import to_qnode  # noqa: E402
from llm_vqc.ir.expand import build_program  # noqa: E402
from llm_vqc.ir.schema import CircuitIR  # noqa: E402
from llm_vqc.tasks import t1_gaussian as t1  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "docs" / "presentation" / "202607_methodology_results"
FIG = PKG / "figures"
DATA = PKG / "data"
SEED = 0  # the pilot's frozen data-split seed, documented everywhere in the deck

# Brand-neutral, colour-blind-safe palette (consistent across all figures).
C_TRAIN, C_VAL, C_TEST = "#3767b3", "#c8781e", "#4a9e5c"
C_RANDOM, C_EVO, C_GREEDY = "#3767b3", "#c8781e", "#8a5fb0"
ARM_COLOR = {"random": C_RANDOM, "evolutionary": C_EVO, "greedy": C_GREEDY}
ARM_LABEL = {"random": "Random", "evolutionary": "Evolutionary", "greedy": "Greedy"}


def _save(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=150, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}.png / .svg")


# ---------------------------------------------------------------------------
# 1. Task and data figures (generated from the actual T1 implementation)
# ---------------------------------------------------------------------------

def _raw_curves(mus, amps, sigmas, noise_seed=None):
    x = t1.X_GRID[None, :]
    y = (
        amps[:, None]
        / (sigmas[:, None] * math.sqrt(2 * math.pi))
        * np.exp(-((x - mus[:, None]) ** 2) / (2 * sigmas[:, None] ** 2))
    )
    if noise_seed is not None:
        rng = np.random.default_rng(noise_seed)
        y = y + rng.normal(0.0, 0.01, size=y.shape)
    return y


def fig_data_generation() -> None:
    # A handful of curves that hold mu fixed while amplitude/width vary, so the
    # reader sees exactly what is nuisance (A, sigma, noise) and what is the
    # target (the peak location mu).
    mu = 0.62
    configs = [(0.6, 0.03), (1.0, 0.05), (1.4, 0.08)]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    colors = ["#3767b3", "#c8781e", "#4a9e5c"]
    for (a, s), col in zip(configs, colors):
        y = _raw_curves(np.array([mu]), np.array([a]), np.array([s]), noise_seed=hash((a, s)) % 2**32)[0]
        ax.plot(t1.X_GRID, y, "-o", color=col, ms=4, lw=1.4,
                label=f"A={a}, σ={s}")
    ax.axvline(mu, color="crimson", ls="--", lw=1.6)
    ax.annotate("target μ = 0.62\n(peak position — the ONLY thing the model predicts)",
                xy=(mu, ax.get_ylim()[1] * 0.5), xytext=(mu + 0.06, ax.get_ylim()[1] * 0.7),
                color="crimson", fontsize=9,
                arrowprops=dict(arrowstyle="->", color="crimson"))
    ax.set_xlabel("x  (21 fixed grid points, linspace(0, 1, 21))")
    ax.set_ylabel("curve height  y(x)  = model input feature")
    ax.set_title("T1 data generation: same peak position μ, different nuisances (A, σ, noise)\n"
                 "y(x) = A/(σ√2π)·exp(−(x−μ)²/2σ²) + ε,  ε~N(0, 0.01²)", fontsize=10)
    ax.legend(title="nuisance variables", fontsize=9)
    ax.grid(alpha=0.25)
    _save(fig, "t1_data_generation")


def fig_example_curves() -> None:
    # Raw vs per-sample min-max normalised, several random samples, coloured
    # by their true mu — the exact transform the model actually sees.
    rng = np.random.default_rng(t1.data_split_seed(SEED, t1.SPEC.name))
    n = 8
    mus = rng.uniform(0, 1, n)
    amps = rng.uniform(0.5, 1.5, n)
    sigmas = rng.uniform(0.01, 0.1, n)
    noise = rng.normal(0, 0.01, size=(n, len(t1.X_GRID)))
    raw = _raw_curves(mus, amps, sigmas) + noise
    norm = (raw - raw.min(axis=1, keepdims=True)) / (
        raw.max(axis=1, keepdims=True) - raw.min(axis=1, keepdims=True) + 1e-12
    )
    order = np.argsort(mus)
    cmap = plt.cm.viridis
    fig, (axr, axn) = plt.subplots(1, 2, figsize=(11, 4.3), sharex=True)
    for i in order:
        col = cmap(mus[i])
        axr.plot(t1.X_GRID, raw[i], color=col, lw=1.3, alpha=0.9)
        axn.plot(t1.X_GRID, norm[i], color=col, lw=1.3, alpha=0.9)
        axr.axvline(mus[i], color=col, ls=":", lw=0.8, alpha=0.6)
    axr.set_title("Raw curves y(x)  (21 features)", fontsize=10)
    axn.set_title("Per-sample min-max normalised  x̂  (what the embed sees)", fontsize=10)
    for ax in (axr, axn):
        ax.set_xlabel("x")
        ax.grid(alpha=0.25)
    axr.set_ylabel("height")
    axn.set_ylabel("normalised height")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    fig.colorbar(sm, ax=[axr, axn], label="true peak position μ (target)", shrink=0.85)
    fig.suptitle("T1 example samples, coloured by their target μ  (fixed seed = data_split_seed(0, T1))",
                 fontsize=11)
    _save(fig, "t1_example_curves")


def fig_split_summary() -> None:
    tv = t1.T1GaussianPeakTask().build(seed=SEED)
    test = t1.T1GaussianPeakTask().build_test(seed=SEED)
    splits = [("train", tv.train.targets, t1.N_TRAIN, C_TRAIN),
              ("validation", tv.val.targets, t1.N_VAL, C_VAL),
              ("protected test", test.targets, t1.N_TEST, C_TEST)]
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(11, 4.3),
                                   gridspec_kw={"width_ratios": [1, 1.4]})
    # left: sizes + roles
    names = [s[0] for s in splits]
    sizes = [s[2] for s in splits]
    cols = [s[3] for s in splits]
    bars = axa.barh(names[::-1], sizes[::-1], color=cols[::-1])
    for b, sz in zip(bars, sizes[::-1]):
        axa.text(b.get_width() + 30, b.get_y() + b.get_height() / 2, str(sz),
                 va="center", fontsize=10)
    axa.set_xlabel("number of samples")
    axa.set_title("Frozen split  (data_split_seed = 0)\n"
                  "150 train · 250 val · 2000 test — disjoint by construction", fontsize=10)
    axa.set_xlim(0, t1.N_TEST * 1.15)
    # right: mu distributions per split (all U[0,1] by construction)
    for name, tg, _, col in splits:
        axb.hist(tg, bins=25, range=(0, 1), density=True, histtype="step",
                 lw=1.8, color=col, label=f"{name} (n={len(tg)})")
    axb.axhline(1.0, color="grey", ls="--", lw=1, alpha=0.7)
    axb.set_xlabel("target μ")
    axb.set_ylabel("density")
    axb.set_title("μ ~ U[0, 1] in every split\n(search never sees test; test scored once)", fontsize=10)
    axb.legend(fontsize=8)
    _save(fig, "t1_split_summary")


# ---------------------------------------------------------------------------
# 2. Model / parameter figures
# ---------------------------------------------------------------------------

def fig_parameter_taxonomy() -> None:
    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.axis("off")

    def box(x, y, w, h, text, fc, ec="#333", fontsize=9, tc="black"):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, lw=1.4))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fontsize, color=tc, wrap=True)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.6))

    y = 3.2
    # forward path boxes
    box(0.2, y, 1.7, 0.9, "21 normalised\ncurve values\nx̂", "#eef2f8")
    box(2.2, y, 1.9, 0.9, "trainable\nLinear embed\n(21 → q_enc)", "#cfe0f5")
    box(4.4, y, 1.7, 0.9, "sigmoid → [0, π]\nencoding angles\n(input-dependent)", "#f5e6cf")
    box(6.4, y, 1.9, 0.9, "parameterised\nVQC\n(structure fixed by search)", "#e6d6f0")
    box(8.6, y, 1.7, 0.9, "trainable\nLinear head\n(q_out → 1)", "#cfe0f5")
    box(10.5, y, 1.3, 0.9, "sigmoid\n→ μ̂", "#d7efdc")
    for x1, x2 in [(1.9, 2.2), (4.1, 4.4), (6.1, 6.4), (8.3, 8.6), (10.3, 10.5)]:
        arrow(x1, y + 0.45, x2, y + 0.45)

    # taxonomy legend below
    ax.text(6.0, 2.7, "Five parameter roles — kept strictly distinct", ha="center",
            fontsize=11, fontweight="bold")
    rows = [
        ("A. Architecture parameters", "#e6d6f0",
         "chosen by the SEARCH algorithm, not gradient descent: n_qubits, encoding type/gate,\n"
         "layer sequence, entangle pattern, measurement observable. Frozen before training."),
        ("B. Trainable quantum gate angles", "#e6d6f0",
         "the VQC's rotation angles (RX/RY/RZ/CRZ weights). Learned by AdamW. greedy_s1: 2 · random_s0: 5."),
        ("C. Trainable classical embed params", "#cfe0f5",
         "Linear(21 → q_enc) weight+bias. Learned by AdamW."),
        ("D. Trainable classical head params", "#cfe0f5",
         "Linear(q_out → 1) weight+bias. Learned by AdamW."),
        ("E. Input-dependent encoding angles", "#f5e6cf",
         "NOT trainable and NOT free: computed per-sample as sigmoid(embed(x̂))·π. Change with the input."),
    ]
    yy = 2.25
    for title, col, desc in rows:
        ax.add_patch(plt.Rectangle((0.2, yy - 0.16), 0.32, 0.32, facecolor=col, edgecolor="#333"))
        ax.text(0.65, yy, title, fontsize=9.5, fontweight="bold", va="center")
        ax.text(4.7, yy, desc, fontsize=8.3, va="center")
        yy -= 0.48
    ax.set_xlim(0, 12.2)
    ax.set_ylim(-0.1, 4.4)
    ax.set_title("HybridQNNModel forward path and the five parameter roles", fontsize=12)
    _save(fig, "model_parameter_taxonomy")


def _draw_circuit(ir_dict: dict, name: str, title: str) -> None:
    ir = CircuitIR.model_validate(ir_dict)
    program = build_program(ir)
    qnode = to_qnode(ir)
    inputs = np.full(program.num_inputs, 0.5)
    weights = np.linspace(0.1, 1.0, program.num_parameters) if program.num_parameters else np.array([])
    fig, ax = qml.draw_mpl(qnode, decimals=2, style="pennylane")(inputs, weights)
    fig.suptitle(title, fontsize=10)
    _save(fig, name)


def fig_circuits(bundle: dict) -> None:
    g = next(r for r in bundle["runs"] if r["run_id"] == "greedy_s1")
    _draw_circuit(
        g["circuit_ir"], "circuit_greedy_s1",
        "Selected circuit — greedy seed 1 (angle-RY encoding, 2 qubits, 2 trainable CRZ angles)\n"
        f"val RMSE={g['selected_val_rmse']:.5f}  test RMSE={g['test_rmse']:.5f}",
    )
    r0 = next(r for r in bundle["runs"] if r["run_id"] == "random_s0")
    _draw_circuit(
        r0["circuit_ir"], "circuit_random_s0",
        "Best overall selected circuit — random seed 0 (amplitude encoding, 9 qubits, 5 trainable angles)\n"
        f"val RMSE={r0['selected_val_rmse']:.5f}  test RMSE={r0['test_rmse']:.5f}",
    )


# ---------------------------------------------------------------------------
# 3. Training figures (re-trained deterministically; asserted against store)
# ---------------------------------------------------------------------------

def fig_training_and_predictions(bundle: dict) -> None:
    r0 = next(r for r in bundle["runs"] if r["run_id"] == "random_s0")
    ir = CircuitIR.model_validate(r0["circuit_ir"])
    tv = t1.T1GaussianPeakTask().build(seed=SEED)
    out = train_model(ir, tv, TrainingConfig(), train_seed=r0["train_seed"])
    assert out.success
    stored = r0["val_metric_history"][-1]
    assert abs(out.final_val_metric - stored) < 1e-9, (out.final_val_metric, stored)
    print(f"  [check] re-trained random_s0 final val RMSE={out.final_val_metric:.9f} "
          f"matches stored {stored:.9f}")

    epochs = np.arange(1, len(out.train_loss_history) + 1)
    fig, ax1 = plt.subplots(figsize=(8, 4.6))
    ax1.plot(epochs, out.train_loss_history, "-o", color=C_TRAIN, ms=4, label="train loss (MSE)")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("training MSE loss", color=C_TRAIN)
    ax1.tick_params(axis="y", labelcolor=C_TRAIN)
    ax1.set_yscale("log")
    for e in TrainingConfig().lr_decay_epochs:
        ax1.axvline(e, color="grey", ls=":", lw=1, alpha=0.6)
    ax2 = ax1.twinx()
    ax2.plot(epochs, out.val_metric_history, "-s", color=C_VAL, ms=4, label="validation RMSE")
    ax2.set_ylabel("validation RMSE", color=C_VAL)
    ax2.tick_params(axis="y", labelcolor=C_VAL)
    ax1.set_title("Training dynamics — random seed 0 selected circuit\n"
                  "AdamW, lr=0.05, 20 epochs, batch 16, LR decay×0.5 at epochs 7/13/17 (dotted)",
                  fontsize=10)
    lines = ax1.get_lines()[:1] + ax2.get_lines()[:1]
    ax1.legend(lines, [l.get_label() for l in lines], fontsize=9, loc="upper right")
    _save(fig, "training_curves_random_s0")

    # predicted vs true on validation, using the exact re-trained weights
    model = HybridQNNModel(ir, raw_feature_dim=tv.spec.raw_feature_dim,
                           head_out_dim=tv.spec.classical_head_out_dim)
    state = {k: torch.tensor(v, dtype=torch.float64).reshape(model.state_dict()[k].shape)
             for k, v in out.trained_classical_state.items()}
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(tv.val.features, dtype=torch.float64)).numpy().reshape(-1)
    true = np.asarray(tv.val.targets)
    fig, ax = plt.subplots(figsize=(5.4, 5.2))
    ax.scatter(true, pred, s=18, color=C_VAL, alpha=0.7, edgecolor="none")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1.4, label="perfect prediction")
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    ax.set_xlabel("true peak position μ")
    ax.set_ylabel("predicted μ̂")
    ax.set_aspect("equal")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"Predicted vs true μ — validation set (n={len(true)})\n"
                 f"random seed 0 selected circuit · RMSE={rmse:.5f}", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    _save(fig, "pred_vs_true_random_s0")


# ---------------------------------------------------------------------------
# 4. Search / selection / results figures (from committed pilot bundle)
# ---------------------------------------------------------------------------

def fig_anytime(bundle: dict) -> None:
    curves = bundle["anytime_curves"]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for arm in ("random", "evolutionary", "greedy"):
        for s in (0, 1, 2):
            c = curves[f"{arm}_s{s}"]
            ax.plot(range(1, len(c) + 1), c, color=ARM_COLOR[arm], alpha=0.8, lw=1.5,
                    marker=".", ms=4,
                    label=ARM_LABEL[arm] if s == 0 else None)
    ax.set_yscale("log")
    ax.set_xlabel("candidate proposals evaluated (budget B = 25)")
    ax.set_ylabel("best-so-far validation RMSE (log scale, lower is better)")
    ax.set_title("Anytime search behaviour — how each run's SELECTED circuit is reached\n"
                 "final point of each line = the circuit that run carries to protected test", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25, which="both")
    _save(fig, "pilot_anytime_curves")


def fig_val_vs_test(bundle: dict) -> None:
    runs = bundle["runs"]
    arms = ("random", "evolutionary", "greedy")
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(arms))
    width = 0.36
    for i, arm in enumerate(arms):
        vals = [r["selected_val_rmse"] for r in runs if r["arm"] == arm]
        tests = [r["test_rmse"] for r in runs if r["arm"] == arm]
        ax.bar(x[i] - width / 2, np.median(vals), width, color=C_VAL,
               alpha=0.85, label="median validation RMSE" if i == 0 else None)
        ax.bar(x[i] + width / 2, np.mean(tests), width, color=C_TEST,
               alpha=0.85, label="mean protected-test RMSE" if i == 0 else None)
        ax.scatter([x[i] - width / 2] * 3, vals, color="black", s=16, zorder=3)
        ax.scatter([x[i] + width / 2] * 3, tests, color="black", s=16, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([ARM_LABEL[a] for a in arms])
    ax.set_ylabel("RMSE (lower is better)")
    ax.set_title("B=25 non-LLM pilot: validation (selection) vs protected test (final)\n"
                 "bars = median val / mean test · dots = 3 seeds · n=3, descriptive only "
                 "(Kruskal–Wallis p=0.079)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    # annotate the exact deck numbers on the validation bars
    med = {a: float(np.median([r["selected_val_rmse"] for r in runs if r["arm"] == a])) for a in arms}
    for i, a in enumerate(arms):
        ax.text(x[i] - width / 2, med[a] + 0.0008, f"{med[a]:.5f}", ha="center", fontsize=8)
    _save(fig, "pilot_val_vs_test")


def fig_quarantine_diagram() -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    ax.axis("off")

    def box(x, y, w, h, text, fc, fontsize=9):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor="#333", lw=1.4))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)

    def arrow(x1, y1, x2, y2, color="#333", style="-|>"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, color=color, lw=1.6))

    # search loop (left, uses train+val only)
    ax.add_patch(plt.Rectangle((0.1, 0.4), 5.9, 3.4, facecolor="#eef4fb",
                               edgecolor="#3767b3", lw=1.6, ls="--"))
    ax.text(3.05, 3.55, "SEARCH LOOP  —  sees TRAIN + VALIDATION only", ha="center",
            fontsize=10, color="#3767b3", fontweight="bold")
    box(0.4, 2.4, 2.4, 0.8, "propose Circuit IR\n(arm)", "#cfe0f5")
    box(3.2, 2.4, 2.5, 0.8, "train on 150 train\n(AdamW, 20 ep)", "#cfe0f5")
    box(0.4, 1.1, 2.4, 0.8, "score on 250 val\n→ validation RMSE", "#cfe0f5")
    box(3.2, 1.1, 2.5, 0.8, "select best-val\ncircuit per run", "#f5e6cf")
    arrow(2.8, 2.8, 3.2, 2.8)
    arrow(4.4, 2.4, 1.6, 1.9)
    arrow(2.8, 1.5, 3.2, 1.5)
    arrow(3.2, 2.4, 1.6, 2.4, color="#3767b3", style="-|>")  # loop back
    ax.text(2.4, 2.15, "repeat ×25", fontsize=7.5, color="#3767b3")

    # quarantine boundary
    ax.plot([6.4, 6.4], [0.2, 4.0], color="crimson", lw=2.2, ls="--")
    ax.text(6.4, 4.05, "QUARANTINE BOUNDARY", ha="center", color="crimson",
            fontsize=9, fontweight="bold")

    # final test (right)
    box(6.9, 2.2, 3.3, 1.1, "PROTECTED FINAL TEST\n2000 held-out samples\nscored EXACTLY ONCE\n"
        "(reload trained weights — no retrain)", "#d7efdc", fontsize=8.5)
    arrow(5.7, 1.5, 6.9, 2.4, color="crimson")
    ax.text(6.05, 2.0, "selected\ncircuit only", fontsize=7.5, color="crimson", ha="center")
    ax.text(8.55, 1.35,
            "final_test.py is never imported by the search path;\n"
            "EvaluationResult has no test field —\nleakage is structurally impossible.",
            ha="center", va="center", fontsize=8, style="italic")
    ax.set_xlim(0, 10.4)
    ax.set_ylim(0, 4.3)
    ax.set_title("Search-time validation vs protected final test", fontsize=12)
    _save(fig, "validation_vs_test_quarantine")


def main() -> None:
    bundle = json.loads((DATA / "pilot_nonllm.json").read_text())
    print("Task/data figures:")
    fig_data_generation()
    fig_example_curves()
    fig_split_summary()
    print("Model / parameter figures:")
    fig_parameter_taxonomy()
    fig_circuits(bundle)
    print("Training figures:")
    fig_training_and_predictions(bundle)
    print("Search / results figures:")
    fig_anytime(bundle)
    fig_val_vs_test(bundle)
    fig_quarantine_diagram()
    print(f"\nAll figures written to {FIG}")


if __name__ == "__main__":
    main()
