"""Hard cost-cap enforcement for external paid LLM API calls.

Controlling policy (verbatim, must not be weakened):

    External paid API calls are allowed only when an explicit nonzero
    hard cap is configured. Read the cap from `LLM_API_BUDGET_USD`. If the
    variable is absent, invalid, or zero, do not make paid API calls. If
    it is set, enforce it programmatically. Stop before issuing a request
    that could exceed the remaining cap. Do not treat a ChatGPT, Claude,
    or other UI subscription as programmatic API authorization.

`LLMApiBudget.from_env()` is the *only* sanctioned way to obtain a budget
object -- there is no constructor path that lets a caller supply a cap
that didn't come from `LLM_API_BUDGET_USD`, so "no cap configured" cannot
be silently bypassed by a caller passing some other number in.
"""

from __future__ import annotations

import os


class LLMBudgetExceededError(Exception):
    """Raised when a request's estimated cost would exceed the remaining cap."""


class LLMApiBudget:
    """Tracks spend against a hard USD cap read from `LLM_API_BUDGET_USD`."""

    def __init__(self, cap_usd: float) -> None:
        if cap_usd <= 0:
            raise ValueError("LLMApiBudget requires cap_usd > 0")
        self.cap_usd = cap_usd
        self.spent_usd = 0.0

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> LLMApiBudget | None:
        """Read `LLM_API_BUDGET_USD` and return a budget, or `None` if no
        paid API calls are authorized (absent, unparseable, or <= 0)."""
        source = env if env is not None else os.environ
        raw = source.get("LLM_API_BUDGET_USD")
        if raw is None or raw.strip() == "":
            return None
        try:
            value = float(raw)
        except ValueError:
            return None
        if value <= 0:
            return None
        return cls(cap_usd=value)

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self.cap_usd - self.spent_usd)

    def check_can_afford(self, estimated_cost_usd: float) -> None:
        """Raise `LLMBudgetExceededError` before issuing a request that
        could exceed the remaining cap. Call this BEFORE the request, not
        after -- the whole point is to never issue an over-cap request."""
        if estimated_cost_usd > self.remaining_usd:
            raise LLMBudgetExceededError(
                f"estimated cost ${estimated_cost_usd:.4f} exceeds remaining budget "
                f"${self.remaining_usd:.4f} of ${self.cap_usd:.4f} cap"
            )

    def record_spend(self, actual_cost_usd: float) -> None:
        self.spent_usd += actual_cost_usd
