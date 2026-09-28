"""Run one or more budget-target cells (protocol:
`docs/research/QAE_BUDGET_TARGET_PROTOCOL.md`).

Validation only: no candidate is evaluated on the test set, and no test
column is written. Every cell is resumable -- the LLM-Open pool and each
LLM-Closed seed store are reloaded from disk instead of being re-called.

Usage:
  python scripts/qae/run_budget_target.py --list
  python scripts/qae/run_budget_target.py --estimate
  python scripts/qae/run_budget_target.py --status
  python scripts/qae/run_budget_target.py --cells target_tfim_b6
  python scripts/qae/run_budget_target.py --cells all
  python scripts/qae/run_budget_target.py --cells target_tfim_b6 --seeds 0 1

Paid arms require `LLM_API_BUDGET_USD` to be set to an explicit nonzero
hard cap; the code refuses to call the API otherwise.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from llm_vqc.experiments.qae_budget_targets.conditions import (
    TARGET_CELLS,
    TARGET_CELLS_BY_KEY,
)
from llm_vqc.experiments.qae_budget_targets.runner import (
    PRICE_SOURCE,
    PRICE_USD_PER_1M,
    STUDY_ROOT,
    completed_seeds,
    run_cell,
)
from llm_vqc.experiments.qae_robustness.conditions import VERIFY_SEEDS
from llm_vqc.llm.budget import LLMApiBudget

# Measured on the committed reference logs (mean per call, 4 qubits):
# ~800 input and ~350 output tokens. Repairs observed at +3..+8% of calls;
# 15% is used as a deliberately conservative upper bound for the estimate.
REF_INPUT_TOKENS = 800
REF_OUTPUT_TOKENS = 350
REPAIR_OVERHEAD = 0.15


def estimate(keys: list[str]) -> dict:
    per_cell = {}
    totals = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "usd": 0.0}
    for key in keys:
        cell = TARGET_CELLS_BY_KEY[key]
        condition = cell.condition
        n_seeds = len(VERIFY_SEEDS)
        calls = 0
        if cell.runs_open:
            calls += 1
        if cell.runs_closed:
            calls += n_seeds * (1 + condition.budget - condition.n_warm)
        calls = int(round(calls * (1 + REPAIR_OVERHEAD)))
        input_tokens = calls * REF_INPUT_TOKENS
        output_tokens = calls * REF_OUTPUT_TOKENS
        price = PRICE_USD_PER_1M[condition.model]
        usd = round(input_tokens / 1e6 * price["input"]
                    + output_tokens / 1e6 * price["output"], 4)
        per_cell[key] = {
            "budget": condition.budget,
            "methods": list(cell.methods),
            "upper_bound_calls": calls,
            "upper_bound_input_tokens": input_tokens,
            "upper_bound_output_tokens": output_tokens,
            "upper_bound_usd_at_list_price": usd,
            "candidate_evaluations": condition.budget * n_seeds * len(cell.methods),
        }
        totals["calls"] += calls
        totals["input_tokens"] += input_tokens
        totals["output_tokens"] += output_tokens
        totals["usd"] = round(totals["usd"] + usd, 4)
    cap = LLMApiBudget.from_env()
    return {
        "price_source": PRICE_SOURCE,
        "per_cell": per_cell,
        "total": totals,
        "configured_cap_usd": None if cap is None else cap.cap_usd,
        "note": ("The enforced cap uses the code's conservative pre-call "
                 "estimates, which are larger than these list-price figures."),
    }


def status(root: Path) -> dict:
    out = {}
    for cell in TARGET_CELLS:
        done = completed_seeds(cell, root)
        destination = root / cell.key
        record = destination / "run_record.json"
        out[cell.key] = {
            "budget": cell.budget,
            "methods": list(cell.methods),
            "llm_closed_seeds_on_disk": done,
            "llm_open_pool_on_disk": (destination / "llm_open_pool.json").exists(),
            "completed": record.exists(),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", nargs="+", default=[],
                        help="cell keys, or 'all'")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="subset of seeds (default: the 12 paired seeds)")
    parser.add_argument("--root", type=Path, default=STUDY_ROOT)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--estimate", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--no-api", action="store_true",
                        help="non-API arms only; fails on a cell needing the API")
    parser.add_argument("--reuse-only", action="store_true",
                        help=("rebuild the tables and record from artefacts "
                              "already on disk; refuses to start if any paid "
                              "artefact is missing, and cannot call the API"))
    args = parser.parse_args()

    keys = [c.key for c in TARGET_CELLS] if args.cells == ["all"] else args.cells

    if args.list:
        print(json.dumps([c.describe() for c in TARGET_CELLS], indent=2))
    if args.estimate:
        print(json.dumps(estimate(keys or [c.key for c in TARGET_CELLS]), indent=2))
    if args.status:
        print(json.dumps(status(args.root), indent=2))
    if not keys:
        return

    if not args.no_api and not args.reuse_only:
        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit("OPENAI_API_KEY is not set; paid arms are blocked")
        if LLMApiBudget.from_env() is None:
            raise SystemExit(
                "LLM_API_BUDGET_USD is not set to an explicit nonzero cap; "
                "paid arms are blocked (this is the intended fail-closed path)"
            )
    seeds = tuple(args.seeds) if args.seeds else VERIFY_SEEDS
    for key in keys:
        run_cell(TARGET_CELLS_BY_KEY[key], with_api=not args.no_api,
                 reuse_only=args.reuse_only, seeds=seeds, root=args.root)


if __name__ == "__main__":
    main()
