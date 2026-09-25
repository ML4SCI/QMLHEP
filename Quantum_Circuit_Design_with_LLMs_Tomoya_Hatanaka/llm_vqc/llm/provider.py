"""LLM provider abstraction: `LLMProvider` is the one interface both a
real (paid, network-calling) provider and the offline `MockLLMProvider`
implement, so `LLMDriver` and the LLM search arms never know or care
which one they are talking to.

**This module makes no network calls of any kind.** `MockLLMProvider` is
the only provider implemented and exercised in this work cycle, because
`LLM_API_BUDGET_USD` is unset in this environment (see
`llm_vqc.llm.budget`) -- per the controlling cost policy, that means no
paid API calls are authorized. A real provider (e.g. wrapping the OpenAI
or Anthropic SDK) is a small, mechanical addition on top of this
`Protocol` once a budget is configured and approved; it is intentionally
NOT implemented here so that this codebase cannot accidentally make a
paid call by import side effect alone.
"""

from __future__ import annotations

import hashlib
from typing import Protocol

import numpy as np
from pydantic import BaseModel


def _stable_seed(*parts: str) -> int:
    """Stable string(s) -> int seed, independent of PYTHONHASHSEED."""
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


class LLMResponse(BaseModel):
    """One completion from a provider, with full cost/latency provenance."""

    raw_text: str
    model: str
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float | None = None  # None = not returned/knowable, never fabricated
    latency_seconds: float


class LLMProvider(Protocol):
    model_name: str

    def complete(self, system_prompt: str, user_prompt: str, temperature: float) -> LLMResponse: ...


class MockLLMProvider:
    """Deterministic, offline, zero-cost stand-in for a real LLM provider.

    The response is a pure function of `(seed, system_prompt, user_prompt,
    temperature)` -- **not** of an internal call counter. This is
    deliberate: an internal mutable counter would reset to zero every time
    a process restarts with a fresh `MockLLMProvider` instance, silently
    breaking checkpoint/resume determinism for LLM arms (a resumed run
    would replay the *first* call's draw instead of continuing where it
    left off). Hashing the actual prompt text instead means the same
    `(arm state -> prompt)` mapping -- which the arm's checkpointed state
    reproduces exactly on resume -- always yields the same mock response,
    with no separate counter to lose.

    Given the same inputs, always returns the same response -- used to
    validate the entire LLM-arm pipeline (prompting, parsing, retry
    handling, provenance recording, cost accounting) with zero real spend
    and zero network access. `estimated_cost_usd` and `latency_seconds`
    are reported as `0.0`: this provider makes no real request, so there
    is no real cost or latency to report, and callers must never mistake
    these zeros for a real provider's pricing.
    """

    model_name = "mock-provider-v1"

    def __init__(self, seed: int, malformed_rate: float = 0.0) -> None:
        self.seed = seed
        self.malformed_rate = malformed_rate

    def complete(self, system_prompt: str, user_prompt: str, temperature: float) -> LLMResponse:
        rng = np.random.default_rng(
            _stable_seed(str(self.seed), system_prompt, user_prompt, str(temperature))
        )

        if rng.random() < self.malformed_rate:
            raw_text = "I propose a circuit with a superposition layer and some entanglement."
        else:
            raw_text = _mock_circuit_json(rng)

        return LLMResponse(
            raw_text=raw_text,
            model=self.model_name,
            input_tokens=len(system_prompt) + len(user_prompt),
            output_tokens=len(raw_text),
            estimated_cost_usd=0.0,
            latency_seconds=0.0,
        )


def _mock_circuit_json(rng: np.random.Generator) -> str:
    """A syntactically-plausible IR-shaped JSON string, drawn from the
    same grammar `llm_vqc.ir.sampler.sample_random_ir` uses -- the mock
    provider "guesses" circuits the way an LLM prompted with the grammar
    description might, without ever importing a real model."""
    from llm_vqc.ir.sampler import sample_random_ir

    ir = sample_random_ir(rng)
    return ir.model_dump_json()
