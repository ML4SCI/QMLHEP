"""LLM provider abstraction, cost-cap enforcement, and provenance
recording for the `llm_iter` / `llm_evo` search arms (master plan Phase 5).

No paid API call is made anywhere in this package unless
`LLMApiBudget.from_env()` returns a budget (i.e. `LLM_API_BUDGET_USD` is
set to a valid positive number) -- see `llm_vqc.llm.budget` for the full
policy this enforces.
"""

from __future__ import annotations

from llm_vqc.llm.budget import LLMApiBudget, LLMBudgetExceededError
from llm_vqc.llm.driver import LLMDriver
from llm_vqc.llm.provider import LLMProvider, LLMResponse, MockLLMProvider
from llm_vqc.llm.records import LLMCallRecord, LLMProposalOutcome

__all__ = [
    "LLMApiBudget",
    "LLMBudgetExceededError",
    "LLMCallRecord",
    "LLMDriver",
    "LLMProposalOutcome",
    "LLMProvider",
    "LLMResponse",
    "MockLLMProvider",
]
