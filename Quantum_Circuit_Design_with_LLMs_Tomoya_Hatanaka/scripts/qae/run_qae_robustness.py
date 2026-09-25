"""Run the QAE robustness study (one factor at a time).

Protocol: docs/research/QAE_ROBUSTNESS_PROTOCOL.md (frozen before the run).

Usage:
  python scripts/qae/run_qae_robustness.py --list
  python scripts/qae/run_qae_robustness.py --estimate
  python scripts/qae/run_qae_robustness.py --conditions budget_b4 budget_b16
  python scripts/qae/run_qae_robustness.py --conditions all
  python scripts/qae/run_qae_robustness.py --conditions qubits_n6 --no-api
  python scripts/qae/run_qae_robustness.py --smoke model_alt

Every condition is resumable: the LLM-Open pool and each LLM-Closed seed
are stored on disk and reloaded instead of re-called.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from llm_vqc.experiments.qae_robustness import arms, study
from llm_vqc.experiments.qae_robustness import manifest as M
from llm_vqc.experiments.qae_robustness.conditions import (
    CONDITIONS,
    CONDITIONS_BY_KEY,
    REFERENCE,
    VERIFY_SEEDS,
    sample_neutral_arch,
    space_for,
)
from llm_vqc.experiments.qae_robustness.prompts import (
    max_output_tokens,
    redesign_prompt,
    warmstart_prompt,
)

# Measured on the reference run: ~774 input and ~347 output tokens per call
# at B=8/n=4. Scaled by the response size a condition actually requests.
REF_INPUT_TOKENS = 774
REF_OUTPUT_TOKENS = 347
# Published list prices are not read programmatically; this is a deliberately
# conservative upper bound used only for the pre-run estimate. The
# authoritative usage record remains the logged token counts.
CONSERVATIVE_USD_PER_1K_INPUT = 0.001
CONSERVATIVE_USD_PER_1K_OUTPUT = 0.008


def estimate(keys: list[str]) -> dict:
    total = {"calls": 0, "input_tokens": 0, "output_tokens": 0}
    per_condition = {}
    for key in keys:
        condition = CONDITIONS_BY_KEY[key]
        if condition.key == REFERENCE.key:
            per_condition[key] = {"calls": 0, "note": "reused from committed artifacts"}
            continue
        space = space_for(condition)
        scale = space.n_gates / 16
        n_seeds = len(VERIFY_SEEDS)
        calls = 1 + n_seeds * (1 + (condition.budget - condition.n_warm))
        # open batch (B candidates) + per seed: warm batch (B/2) + B/2 redesigns
        out_tokens = (
            REF_OUTPUT_TOKENS * condition.budget / 8 * scale
            + n_seeds * REF_OUTPUT_TOKENS * condition.n_warm / 8 * scale
            + n_seeds * (condition.budget - condition.n_warm)
            * REF_OUTPUT_TOKENS / 8 * scale
        )
        in_tokens = REF_INPUT_TOKENS * scale * calls
        entry = {
            "calls": calls,
            "input_tokens": int(in_tokens),
            "output_tokens": int(out_tokens),
            "usd_upper_bound": round(
                in_tokens / 1000 * CONSERVATIVE_USD_PER_1K_INPUT
                + out_tokens / 1000 * CONSERVATIVE_USD_PER_1K_OUTPUT, 4),
            "max_output_tokens_open": max_output_tokens(condition, condition.budget),
        }
        per_condition[key] = entry
        total["calls"] += calls
        total["input_tokens"] += int(in_tokens)
        total["output_tokens"] += int(out_tokens)
    total["usd_upper_bound"] = round(
        total["input_tokens"] / 1000 * CONSERVATIVE_USD_PER_1K_INPUT
        + total["output_tokens"] / 1000 * CONSERVATIVE_USD_PER_1K_OUTPUT, 4)
    return {"per_condition": per_condition, "total": total}


def smoke(key: str) -> None:
    """One cheap real call for a condition: does the frozen schema parse?"""
    from llm_vqc.experiments.qae_robustness.conditions import parse_candidate
    from llm_vqc.llm.budget import LLMApiBudget

    condition = CONDITIONS_BY_KEY[key]
    budget = LLMApiBudget.from_env()
    if budget is None:
        raise SystemExit("LLM_API_BUDGET_USD not configured; smoke test refused")
    space = space_for(condition)
    provider = arms.build_provider(condition, 1)
    incumbent = sample_neutral_arch(np.random.default_rng(0), space)
    parsed, records = arms.complete_json(
        provider, budget, redesign_prompt(condition, incumbent, 0.9123),
        f"smoke_{key}", arms.REFINE_CALL_COST_ESTIMATE_USD,
    )
    for record in records:
        print(f"model={record.model} in={record.input_tokens} "
              f"out={record.output_tokens} latency={record.latency_seconds:.1f}s "
              f"errors={record.validation_errors}")
    entry = (parsed or {}).get("candidates", [None])[0]
    architecture, errors = (parse_candidate(entry, space) if entry else (None, ["none"]))
    print(f"condition={key} model={condition.model} "
          f"max_output_tokens={provider.max_output_tokens} "
          f"capacity-valid={architecture is not None} {errors[:1]}")
    print("warm prompt bytes:", len(warmstart_prompt(condition)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", nargs="+", default=[])
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--estimate", action="store_true")
    parser.add_argument("--smoke", default="")
    parser.add_argument("--no-api", action="store_true")
    parser.add_argument("--root", default="outputs/qae_robustness")
    args = parser.parse_args()

    if args.list:
        for condition in CONDITIONS:
            print(f"{condition.key:18s} factor={condition.factor:12s} "
                  f"n={condition.n_qubits} family={condition.family} "
                  f"B={condition.budget} model={condition.model}")
        return
    if args.smoke:
        smoke(args.smoke)
        return

    keys = ([c.key for c in CONDITIONS]
            if args.conditions == ["all"] else args.conditions)
    if args.estimate:
        print(json.dumps(estimate(keys or [c.key for c in CONDITIONS]), indent=2))
        return
    if not keys:
        parser.error("pass --conditions, --list, --estimate or --smoke")

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifests.json").write_text(json.dumps(M.all_manifests(), indent=2))
    violations = {k: v for k, v in M.all_manifests()["violations"].items() if v}
    if violations:
        raise SystemExit(f"manifest verification failed: {violations}")

    study.main(keys, with_api=not args.no_api, root=root)


if __name__ == "__main__":
    main()
