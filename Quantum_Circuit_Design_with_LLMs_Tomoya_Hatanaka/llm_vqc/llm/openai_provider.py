"""A real, network-calling OpenAI provider -- used only when a human has
explicitly configured `LLM_API_BUDGET_USD` for a specific, time-boxed run
(see `llm_vqc.llm.budget`). Never imported or constructed by any test in
this repository; never used unless the caller explicitly builds it with
a real API key.

Cost is intentionally reported as `None` (never fabricated): the OpenAI
chat completions API returns token usage but never a dollar figure.
Token counts are the authoritative usage record for this provider.
"""

from __future__ import annotations

import time

from openai import OpenAI

from llm_vqc.llm.provider import LLMProvider, LLMResponse


class OpenAIProvider:
    """Wraps one `openai.OpenAI` client. `complete()` makes one real,
    billed chat-completion request per call -- there is no local caching
    or mocking anywhere in this class.

    Rate-limit handling: `min_seconds_between_calls` paces requests
    client-side (e.g. 6.5 for a 10-requests/minute org limit), and 429
    responses are retried with a linear backoff up to
    `max_rate_limit_retries` times before the error propagates. 429'd
    attempts are not billed by OpenAI and count as part of the same
    logical `complete()` call (one `LLMResponse`, one call-cap unit)."""

    def __init__(
        self,
        api_key: str,
        model: str,
        max_output_tokens: int = 700,
        min_seconds_between_calls: float = 0.0,
        max_rate_limit_retries: int = 5,
    ) -> None:
        self.model_name = model
        self.max_output_tokens = max_output_tokens
        self.min_seconds_between_calls = min_seconds_between_calls
        self.max_rate_limit_retries = max_rate_limit_retries
        self._last_call_at = 0.0
        self._client = OpenAI(api_key=api_key)

    def complete(self, system_prompt: str, user_prompt: str, temperature: float) -> LLMResponse:
        from openai import RateLimitError

        start = time.monotonic()
        response = None
        for attempt in range(self.max_rate_limit_retries + 1):
            pace_wait = self.min_seconds_between_calls - (time.monotonic() - self._last_call_at)
            if pace_wait > 0:
                time.sleep(pace_wait)
            self._last_call_at = time.monotonic()
            try:
                response = self._client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temperature,
                    max_completion_tokens=self.max_output_tokens,
                )
                break
            except RateLimitError:
                if attempt >= self.max_rate_limit_retries:
                    raise
                time.sleep(7.0 * (attempt + 1))  # linear backoff past the RPM window
        assert response is not None
        latency = time.monotonic() - start

        raw_text = response.choices[0].message.content or ""
        usage = response.usage
        input_tokens = int(usage.prompt_tokens) if usage is not None else 0
        output_tokens = int(usage.completion_tokens) if usage is not None else 0

        return LLMResponse(
            raw_text=raw_text,
            model=response.model or self.model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=None,  # OpenAI never returns a dollar cost; not fabricated
            latency_seconds=latency,
        )


class CallBudgetExceededError(Exception):
    """Raised when a hard, run-specific cap on the TOTAL number of real
    provider calls (shared across every arm using this wrapper) would be
    exceeded. Distinct from `LLMApiBudget`'s dollar cap -- this is a call
    *count* cap, for this specific time-boxed mini experiment."""


class CallCountLimitedProvider:
    """Wraps any `LLMProvider` and enforces a hard cap on the total number
    of `complete()` calls made through this **shared** instance -- pass
    the SAME instance to every arm that should share the cap (e.g.
    `llm_iter` and `llm_evo` in one experiment), not one instance per arm.

    Nestable: wrapping one `CallCountLimitedProvider` in another gives a
    per-run cap inside a global cap (both enforced, innermost provider
    called only if every layer admits the call)."""

    def __init__(self, inner: LLMProvider, max_calls: int) -> None:
        self.model_name = inner.model_name
        self._inner = inner
        self.max_calls = max_calls
        self.calls_made = 0

    def complete(self, system_prompt: str, user_prompt: str, temperature: float) -> LLMResponse:
        if self.calls_made >= self.max_calls:
            raise CallBudgetExceededError(
                f"call {self.calls_made + 1} would exceed the hard cap of "
                f"{self.max_calls} real provider calls for this run"
            )
        self.calls_made += 1
        return self._inner.complete(system_prompt, user_prompt, temperature)
