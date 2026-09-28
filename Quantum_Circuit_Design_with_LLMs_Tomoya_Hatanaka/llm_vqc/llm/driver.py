"""`LLMDriver`: turns one provider call (or a bounded chain of repair
retries) into a parsed IR proposal, with full provenance.

**Malformed proposals are bounded, never free-repaired forever.**
`max_repair_attempts` is a hard cap (default 2, matching the master
plan's "invalid-IR rate < 20% after retry" Phase 5 gate language) --
after that many failed parses, `LLMDriver.propose()` returns
`proposal=None` and the caller (an LLM search arm) must hand the shared
harness *something* anyway (see `llm_vqc.search.arms.llm_iter_arm` /
`llm_evo_arm`, which return a deliberately-unparseable placeholder dict
in that case). That placeholder is then classified `INVALID` by the
*same* validator every other arm's proposals go through -- consistent,
budget-free-for-everyone accounting (governance decision G4-B), not a
special LLM penalty or a special LLM exemption.

**The repair loop targets JSON/schema problems only.** It re-prompts on
"the response wasn't valid JSON" or "the JSON doesn't match the
`CircuitIR` field types" -- genuine LLM-output problems worth a bounded
retry. It deliberately does NOT re-prompt on deeper cross-field semantic
issues (e.g. "a `star` pattern requires `center`"); those are
`llm_vqc.ir.validators.validate_proposal`'s job and flow through to the
harness's ordinary `INVALID` outcome, exactly as they would for any other
arm's proposal.

**Cost-cap enforcement happens before every single call**, including
repair retries -- `LLMApiBudget.check_can_afford` is checked immediately
before `provider.complete()`, so a cap cannot be exceeded even by a chain
of repair attempts.
"""

from __future__ import annotations

import json

from pydantic import ValidationError

from llm_vqc.ir.schema import CircuitIR
from llm_vqc.llm.budget import LLMApiBudget
from llm_vqc.llm.provider import LLMProvider
from llm_vqc.llm.records import LLMCallRecord, LLMProposalOutcome

_REPAIR_PROMPT_TEMPLATE = (
    "{original_user_prompt}\n\n"
    "Your previous response could not be used:\n"
    "  response: {raw_response!r}\n"
    "  problem: {errors}\n"
    "Respond again with ONLY a single JSON object matching the schema "
    "described above -- no prose, no markdown fences."
)


def _parse_and_schema_check(raw_text: str) -> tuple[dict | None, list[str]]:
    """Parse JSON and check it against `CircuitIR`'s field-level schema
    only (types, literal enums, numeric bounds) -- NOT the deeper
    cross-field semantic rules in `llm_vqc.ir.validators`, which are the
    harness's job."""
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        return None, [f"response is not valid JSON: {exc}"]
    if not isinstance(data, dict):
        return None, ["parsed JSON is not an object"]
    try:
        CircuitIR(**data)
    except ValidationError as exc:
        return None, [str(exc)]
    return data, []


class LLMDriver:
    def __init__(
        self,
        provider: LLMProvider,
        max_repair_attempts: int = 2,
        budget: LLMApiBudget | None = None,
        cost_estimate_per_call_usd: float = 0.0,
    ) -> None:
        self.provider = provider
        self.max_repair_attempts = max_repair_attempts
        self.budget = budget
        self.cost_estimate_per_call_usd = cost_estimate_per_call_usd

    def propose(
        self,
        proposal_id: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
    ) -> LLMProposalOutcome:
        records: list[LLMCallRecord] = []
        current_user_prompt = user_prompt

        for attempt in range(self.max_repair_attempts + 1):
            if self.budget is not None:
                self.budget.check_can_afford(self.cost_estimate_per_call_usd)

            response = self.provider.complete(system_prompt, current_user_prompt, temperature)

            if self.budget is not None:
                # If the provider didn't report a real cost (e.g. OpenAI's
                # chat completions API never returns a dollar figure), fall
                # back to charging the pre-call estimate against the cap --
                # conservative (never under-counts), never fabricates the
                # per-call LLMCallRecord's own reported cost field.
                actual_or_estimated = (
                    response.estimated_cost_usd
                    if response.estimated_cost_usd is not None
                    else self.cost_estimate_per_call_usd
                )
                self.budget.record_spend(actual_or_estimated)

            parsed, errors = _parse_and_schema_check(response.raw_text)
            records.append(
                LLMCallRecord(
                    proposal_id=proposal_id,
                    call_index=attempt,
                    system_prompt=system_prompt,
                    user_prompt=current_user_prompt,
                    model=response.model,
                    temperature=temperature,
                    raw_response=response.raw_text,
                    parsed_proposal=parsed,
                    validation_errors=errors,
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                    estimated_cost_usd=response.estimated_cost_usd,
                    latency_seconds=response.latency_seconds,
                )
            )

            if parsed is not None:
                return LLMProposalOutcome(
                    proposal=parsed, valid_schema=True, retry_count=attempt, call_records=records
                )

            current_user_prompt = _REPAIR_PROMPT_TEMPLATE.format(
                original_user_prompt=user_prompt,
                raw_response=response.raw_text,
                errors="; ".join(errors),
            )

        return LLMProposalOutcome(
            proposal=None,
            valid_schema=False,
            retry_count=self.max_repair_attempts,
            call_records=records,
        )
