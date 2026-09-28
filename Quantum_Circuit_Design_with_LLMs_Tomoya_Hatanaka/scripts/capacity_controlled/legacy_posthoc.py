#!/usr/bin/env python
"""Exploratory post-hoc: re-read the LEGACY unrestricted B=25 pilot through the
capacity lens. EXPLORATORY, NOT CONFIRMATORY. The legacy experiment is not
redefined or overwritten; this only correlates its already-published numbers
with model capacity to show how much of the legacy signal was capacity, not
search strategy.

Inputs (both committed, sanitized):
  outputs/capacity_controlled_t1_v1/legacy_capacity_audit.csv   (all 222 circuits: val + capacity)
  docs/presentation/202607_methodology_results/data/pilot_nonllm.json  (9 selected: val + test + IR)

Usage: python scripts/capacity_controlled/legacy_posthoc.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

from llm_vqc.evaluation.model import HybridQNNModel  # noqa: E402
from llm_vqc.ir.schema import CircuitIR  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "capacity_controlled_t1_v1"
AUDIT_CSV = OUT / "legacy_capacity_audit.csv"
PILOT_JSON = ROOT / "docs/presentation/202607_methodology_results/data/pilot_nonllm.json"


def main() -> None:
    rows = list(csv.DictReader(AUDIT_CSV.open()))
    total = np.array([int(r["total_trainable_params"]) for r in rows])
    val = np.array([float(r["val_rmse"]) if r["val_rmse"] else np.nan for r in rows])
    enc = np.array([r["encoding_type"] for r in rows])
    mask = ~np.isnan(val)
    total_v, val_v, enc_v = total[mask], val[mask], enc[mask]

    rho, p = stats.spearmanr(total_v, val_v)
    angle = val_v[enc_v == "angle"]
    amp = val_v[enc_v == "amplitude"]
    mwu_u, mwu_p = stats.mannwhitneyu(angle, amp, alternative="two-sided")

    # selected circuits: val, test, total params, encoding
    bundle = json.loads(PILOT_JSON.read_text())
    selected = []
    for r in bundle["runs"]:
        ir = CircuitIR.model_validate(r["circuit_ir"])
        m = HybridQNNModel(ir, raw_feature_dim=21, head_out_dim=1)
        selected.append({
            "run_id": r["run_id"], "arm": r["arm"],
            "encoding": ir.encoding.type,
            "total_params": int(sum(p.numel() for p in m.parameters())),
            "val": r["selected_val_rmse"], "test": r["test_rmse"]})
    sel_params = np.array([s["total_params"] for s in selected])
    sel_test = np.array([s["test"] for s in selected])
    rho_sel, p_sel = stats.spearmanr(sel_params, sel_test)
    amp_selected = [s for s in selected if s["encoding"] == "amplitude"]

    result = {
        "note": "EXPLORATORY post-hoc of the legacy unrestricted B=25 pilot. Not confirmatory; "
                "the legacy experiment is not redefined.",
        "all_evaluated_circuits": {
            "n": int(mask.sum()),
            "spearman_val_vs_total_params": {"rho": float(rho), "p_value": float(p)},
            "val_rmse_by_encoding": {
                "angle": {"n": int(len(angle)), "median": float(np.median(angle))},
                "amplitude": {"n": int(len(amp)), "median": float(np.median(amp))},
                "mannwhitney_p": float(mwu_p)},
        },
        "selected_circuits": {
            "n": len(selected),
            "spearman_test_vs_total_params": {"rho": float(rho_sel), "p_value": float(p_sel)},
            "amplitude_selected_fraction": len(amp_selected) / len(selected),
            "amplitude_selected_runs": [s["run_id"] for s in amp_selected],
            "detail": selected},
        "interpretation": (
            "In the legacy space, lower validation RMSE is associated with LARGER total "
            "capacity, and amplitude-encoding circuits (which carry the largest classical "
            "embeddings) dominate the best selections. This is the capacity confound the "
            "controlled experiment removes."),
    }
    (OUT / "legacy_posthoc.json").write_text(json.dumps(result, indent=2))

    # figure: legacy val vs total params (all), + selected test vs params
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    colors = {"angle": "#3767b3", "amplitude": "#c8781e"}
    for e in ("angle", "amplitude"):
        mm = enc_v == e
        ax1.scatter(total_v[mm], val_v[mm], s=18, alpha=0.6, color=colors[e], label=f"{e} encoding")
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlabel("total trainable params (log)"); ax1.set_ylabel("validation RMSE (log)")
    ax1.set_title(f"Legacy: lower RMSE ↔ larger capacity\nSpearman ρ={rho:.2f} (p={p:.1e}), "
                  f"n={int(mask.sum())}", fontsize=10)
    ax1.legend(fontsize=9); ax1.grid(alpha=0.25, which="both")

    for s in selected:
        ax2.scatter(s["total_params"], s["test"], s=60, color=colors[s["encoding"]],
                    edgecolor="black", zorder=3)
    ax2.set_xscale("log")
    ax2.set_xlabel("selected circuit total params (log)")
    ax2.set_ylabel("protected test RMSE")
    ax2.set_title(f"Legacy selected circuits: best test ↔ big embeddings\n"
                  f"Spearman ρ={rho_sel:.2f}; amplitude selected in "
                  f"{len(amp_selected)}/{len(selected)} runs", fontsize=10)
    ax2.grid(alpha=0.25, which="both")
    from matplotlib.lines import Line2D
    ax2.legend([Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[e],
                       markeredgecolor="black", markersize=8) for e in ("angle", "amplitude")],
               ["angle", "amplitude"], fontsize=9)
    fig.suptitle("EXPLORATORY post-hoc: how much of the legacy signal was capacity, not search",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "legacy_posthoc.png", dpi=150)
    fig.savefig(OUT / "legacy_posthoc.svg")
    plt.close(fig)

    print(f"all-circuits Spearman(val, params) rho={rho:.3f} p={p:.2e}")
    print(f"val by encoding: angle median={np.median(angle):.5f} amplitude median={np.median(amp):.5f} "
          f"(MWU p={mwu_p:.2e})")
    print(f"selected test vs params Spearman rho={rho_sel:.3f} p={p_sel:.3f}; "
          f"amplitude selected {len(amp_selected)}/{len(selected)}")
    print(f"wrote {OUT}/legacy_posthoc.(json|png|svg)")


if __name__ == "__main__":
    main()
