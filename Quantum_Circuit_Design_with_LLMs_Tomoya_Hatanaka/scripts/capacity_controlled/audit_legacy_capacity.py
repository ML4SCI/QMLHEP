#!/usr/bin/env python
"""Legacy capacity-confound audit (research cycle: capacity-controlled T1 v1).

Quantifies how *total trainable parameter count* — and its classical/quantum
breakdown — varies across the circuits evaluated in the legacy unrestricted
B=25 non-LLM pilot. This is the evidence that the legacy search space did NOT
isolate VQC architecture: encoding type and qubit count drove order-of-magnitude
swings in the classical embedding, so "which arm won" was confounded by model
capacity.

Reads the durable store read-only (the store itself is gitignored and never
published). Instantiates each evaluated circuit's actual `HybridQNNModel` and
counts real `torch` parameters — no analytic shortcut — then writes a sanitized
CSV + JSON + figure into the output package. Historical results are NOT modified.

Usage:
  python scripts/capacity_controlled/audit_legacy_capacity.py \
      --store /path/to/runs/pilot_t1/results.sqlite
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from llm_vqc.evaluation.model import HybridQNNModel  # noqa: E402
from llm_vqc.evaluation.store import ResultStore  # noqa: E402
from llm_vqc.ir.schema import CircuitIR  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "capacity_controlled_t1_v1"
RAW_FEATURE_DIM = 21  # T1
HEAD_OUT_DIM = 1


def param_breakdown(ir: CircuitIR) -> dict:
    """Exact trainable-parameter breakdown by instantiating the real model."""
    model = HybridQNNModel(ir, raw_feature_dim=RAW_FEATURE_DIM, head_out_dim=HEAD_OUT_DIM)
    named = dict(model.named_parameters())
    embed = int(sum(p.numel() for n, p in named.items() if n.startswith("embed.")))
    head = int(sum(p.numel() for n, p in named.items() if n.startswith("head.")))
    quantum = int(sum(p.numel() for n, p in named.items() if n.startswith("q_layer.")))
    total = int(sum(p.numel() for p in model.parameters()))
    return {"embed_params": embed, "quantum_params": quantum, "head_params": head,
            "total_trainable_params": total}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True, help="path to legacy runs/pilot_t1/results.sqlite")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    store = ResultStore(args.store)
    evals = store.all_evaluations("T1")
    store.close()

    rows = []
    for r in evals:
        if r.circuit_canonical_json is None or r.circuit_cost is None:
            continue
        ir = CircuitIR.model_validate_json(r.circuit_canonical_json)
        pb = param_breakdown(ir)
        cost = r.circuit_cost
        rows.append({
            "structural_hash": (r.structural_hash or "")[:12],
            "encoding_type": ir.encoding.type,
            "n_qubits": cost.n_qubits,
            "encoding_inputs": cost.input_count,
            "n_measurements": len(ir.measurements.wires) if ir.measurements.wires != "all"
                              else cost.n_qubits,
            "depth": cost.depth,
            "gate_count": cost.gate_count,
            "two_qubit_gate_count": cost.two_qubit_gate_count,
            **pb,
            "val_rmse": r.val_metric_value,
        })

    # sort by total params for readability
    rows.sort(key=lambda x: x["total_trainable_params"])
    fields = ["structural_hash", "encoding_type", "n_qubits", "encoding_inputs",
              "n_measurements", "depth", "gate_count", "two_qubit_gate_count",
              "embed_params", "quantum_params", "head_params", "total_trainable_params",
              "val_rmse"]
    with (OUT / "legacy_capacity_audit.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    totals = [x["total_trainable_params"] for x in rows]
    embeds = [x["embed_params"] for x in rows]
    quants = [x["quantum_params"] for x in rows]
    by_enc = {}
    for x in rows:
        by_enc.setdefault(x["encoding_type"], []).append(x["total_trainable_params"])
    summary = {
        "note": "Legacy unrestricted B=25 non-LLM pilot capacity audit; read-only, "
                "historical results unchanged. Total trainable params = embed + quantum + head.",
        "n_circuits": len(rows),
        "total_trainable_params": {
            "min": int(min(totals)), "max": int(max(totals)),
            "median": float(np.median(totals)),
            "ratio_max_over_min": float(max(totals) / min(totals)),
        },
        "embed_params": {"min": int(min(embeds)), "max": int(max(embeds))},
        "quantum_params": {"min": int(min(quants)), "max": int(max(quants))},
        "by_encoding_type": {k: {"n": len(v), "min_total": int(min(v)),
                                 "max_total": int(max(v)),
                                 "median_total": float(np.median(v))}
                             for k, v in by_enc.items()},
    }
    (OUT / "legacy_capacity_audit.json").write_text(json.dumps(summary, indent=2))

    # Figure: total trainable params per circuit, coloured by encoding, log scale
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    colors = {"angle": "#3767b3", "amplitude": "#c8781e"}
    for enc in ("angle", "amplitude"):
        xs = [i for i, x in enumerate(rows) if x["encoding_type"] == enc]
        ys = [rows[i]["total_trainable_params"] for i in xs]
        ax1.scatter(xs, ys, s=22, color=colors[enc], alpha=0.8, label=f"{enc} encoding")
    ax1.set_yscale("log")
    ax1.set_xlabel("evaluated circuit (legacy pilot, sorted by total params)")
    ax1.set_ylabel("total trainable parameters (log)")
    ax1.set_title(f"Legacy capacity varies {summary['total_trainable_params']['ratio_max_over_min']:.0f}× "
                  f"({min(totals)}–{max(totals)} params)", fontsize=10)
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.25, which="both")

    # stacked breakdown for the 9 selected-ish extremes (min & max few)
    ax2.scatter(quants, embeds, s=26, c=[colors[x["encoding_type"]] for x in rows], alpha=0.8)
    ax2.set_xlabel("quantum trainable params")
    ax2.set_ylabel("classical embed params (log)")
    ax2.set_yscale("log")
    ax2.set_title("Classical embed capacity is set by encoding, not by the VQC", fontsize=10)
    ax2.grid(alpha=0.25, which="both")
    fig.suptitle("Legacy unrestricted pilot: the search space changed model capacity, "
                 "not just architecture", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "legacy_capacity_audit.png", dpi=150)
    fig.savefig(OUT / "legacy_capacity_audit.svg")
    plt.close(fig)

    print(f"n_circuits={len(rows)}")
    print(f"total params: min={min(totals)} max={max(totals)} "
          f"ratio={max(totals)/min(totals):.1f}x")
    print(f"by encoding: {summary['by_encoding_type']}")
    print(f"wrote {OUT}/legacy_capacity_audit.(csv|json|png|svg)")


if __name__ == "__main__":
    main()
