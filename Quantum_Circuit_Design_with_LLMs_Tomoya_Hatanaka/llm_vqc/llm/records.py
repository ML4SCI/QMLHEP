"""Provenance schemas for LLM-driven proposals.

Every field the work brief requires is here: complete prompts, model
identity, temperature, raw response, parsed proposal, validation errors,
token usage, cost, latency. One `LLMCallRecord` per attempt (the first
try and every bounded repair retry each get their own record, never
overwritten), so a malformed-output repair loop leaves a full audit
trail rather than only the final outcome.
"""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class LLMCallRecord(BaseModel):
    proposal_id: str
    call_index: int  # 0 = first attempt, 1+ = repair retries
    system_prompt: str
    user_prompt: str
    model: str
    temperature: float
    raw_response: str
    parsed_proposal: dict | None
    validation_errors: list[str] = Field(default_factory=list)
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float | None = None  # None = not returned/knowable, never fabricated
    latency_seconds: float
    created_at: str = Field(default_factory=_now_iso)


class LLMProposalOutcome(BaseModel):
    """What `LLMDriver.propose()` returns: the final parsed proposal (or
    `None` if every repair attempt was exhausted without valid JSON/schema),
    plus the complete provenance trail."""

    proposal: dict | None
    valid_schema: bool
    retry_count: int
    call_records: list[LLMCallRecord]

    @property
    def total_cost_usd(self) -> float | None:
        """Sum of known per-call costs, or `None` if no call reported one
        (e.g. the OpenAI chat completions API never returns a dollar cost --
        only token usage; we do not fabricate a price)."""
        known = [
            r.estimated_cost_usd for r in self.call_records if r.estimated_cost_usd is not None
        ]
        return sum(known) if known else None

    @property
    def total_tokens(self) -> int:
        return sum(r.input_tokens + r.output_tokens for r in self.call_records)
