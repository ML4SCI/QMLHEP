"""Run the QAE-TFIM v3 neutral-space four-method benchmark.

Usage:
  python scripts/qae/run_qae_tfim_neutral_v3.py --smoke     # one cheap API parse check
  python scripts/qae/run_qae_tfim_neutral_v3.py --no-api    # Random + Greedy only
  python scripts/qae/run_qae_tfim_neutral_v3.py             # full run (needs LLM_API_BUDGET_USD)

Real API calls require OPENAI_API_KEY, OPENAI_MODEL, and an explicit nonzero
LLM_API_BUDGET_USD in the environment. Protocol: docs/research/QAE_PROTOCOL_V3.md.
"""
import argparse
import json

from llm_vqc.experiments.qae_tfim import neutral_v3


def smoke() -> None:
    from llm_vqc.llm.budget import LLMApiBudget

    budget = LLMApiBudget.from_env()
    if budget is None:
        raise SystemExit("LLM_API_BUDGET_USD not configured; smoke test refused")
    provider = neutral_v3._build_provider()
    parsed, records = neutral_v3._complete_json(
        provider,
        budget,
        neutral_v3.closed_loop_prompt([]),
        "smoke",
        neutral_v3.CLOSED_CALL_COST_ESTIMATE_USD,
    )
    for record in records:
        print(
            f"model={record.model} in={record.input_tokens} out={record.output_tokens} "
            f"latency={record.latency_seconds:.1f}s errors={record.validation_errors}"
        )
    if parsed is not None:
        entries = parsed.get("candidates", [])
        entry = entries[0] if entries else None
        architecture, errors = (
            neutral_v3.parse_candidate(entry) if entry is not None else (None, ["empty"])
        )
        print("parsed candidate:", json.dumps(entry))
        print("capacity-valid:", architecture is not None, errors)
    else:
        print("smoke call failed to produce JSON after retries")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-api", action="store_true")
    parser.add_argument("--no-closed-loop", action="store_true")
    parser.add_argument("--output", default="outputs/qae_tfim_neutral_v3")
    args = parser.parse_args()
    if args.smoke:
        smoke()
        return
    neutral_v3.main(
        args.output,
        with_api=not args.no_api,
        closed_loop=not args.no_closed_loop,
    )


if __name__ == "__main__":
    main()
