"""Run the QAE-TFIM v5 incumbent-based free-form redesign benchmark.

Usage:
  python scripts/qae/run_qae_tfim_neutral_v5.py --smoke     # one cheap redesign parse check
  python scripts/qae/run_qae_tfim_neutral_v5.py --no-api    # Random + Greedy only
  python scripts/qae/run_qae_tfim_neutral_v5.py             # full run (needs LLM_API_BUDGET_USD)

Protocol: docs/research/QAE_PROTOCOL_V5.md (frozen before the run).
"""
import argparse
import json

import numpy as np

from llm_vqc.experiments.qae_tfim import neutral_v5


def smoke() -> None:
    from llm_vqc.llm.budget import LLMApiBudget

    budget = LLMApiBudget.from_env()
    if budget is None:
        raise SystemExit("LLM_API_BUDGET_USD not configured; smoke test refused")
    provider = neutral_v5._build_provider()
    rng = np.random.default_rng(0)
    incumbent = neutral_v5.sample_neutral_arch(rng)
    parsed, records = neutral_v5._complete_json(
        provider, budget, neutral_v5.redesign_prompt(incumbent, 0.9123),
        "smoke", neutral_v5.REFINE_CALL_COST_ESTIMATE_USD,
    )
    for record in records:
        print(f"model={record.model} in={record.input_tokens} out={record.output_tokens} "
              f"latency={record.latency_seconds:.1f}s errors={record.validation_errors}")
    if parsed is not None:
        entries = parsed.get("candidates", [])
        entry = entries[0] if entries else None
        architecture, errors = (
            neutral_v5.parse_candidate(entry) if entry is not None else (None, ["empty"])
        )
        print("strategy:", entry.get("strategy") if entry else None,
              "| rationale:", (entry.get("rationale") if entry else "")[:80])
        print("capacity-valid:", architecture is not None, errors[:1])
        if architecture is not None:
            print("edit distance from incumbent:",
                  round(neutral_v5._arch_distance(incumbent, architecture), 3))
    else:
        print("smoke call failed to produce JSON after retries")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-api", action="store_true")
    parser.add_argument("--no-closed-loop", action="store_true")
    parser.add_argument("--output", default="outputs/qae_tfim_neutral_v5")
    args = parser.parse_args()
    if args.smoke:
        smoke()
        return
    neutral_v5.main(
        args.output,
        with_api=not args.no_api,
        closed_loop=not args.no_closed_loop,
    )
    _ = json


if __name__ == "__main__":
    main()
