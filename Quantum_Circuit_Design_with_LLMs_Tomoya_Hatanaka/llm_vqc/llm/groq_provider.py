"""Groq provider (free plan) for `openai/gpt-oss-20b`, via the OpenAI SDK
pointed at Groq's OpenAI-compatible endpoint.

Distinct provider/model condition from the OpenAI runs — never share a
result store, run_id namespace, or token accounting with them.

Uses Groq Strict Structured Outputs (`response_format.type="json_schema"`,
`strict=true`): the model can only emit JSON matching
`CIRCUIT_IR_STRICT_SCHEMA` (hand-adapted from the pydantic `CircuitIR`
for strict mode: every property required, `additionalProperties: false`
everywhere, nullable unions for logically-optional fields). Schema
validity is necessary but not sufficient — every proposal still passes
the repository's semantic IR validators downstream, unchanged.

`reasoning_effort="low"`, low temperature, small output cap; hidden
reasoning is never requested or stored (only the final message content).

`FreeTierController` enforces the free-plan envelope client-side:
cumulative token cap (with reserve below the documented daily limit),
request cap, sequential pacing under both RPM and TPM, one transport
retry on 429/5xx honoring retry-after. Crossing a cap raises
`FreeTierExhaustedError` BEFORE the request is issued, so the run
checkpoints cleanly (durable ledger + arm state) and can resume later
under the same provider/model only (enforced by the store's
config-compat hash, which includes provider and model).
"""

from __future__ import annotations

import time

from openai import OpenAI

from llm_vqc.llm.provider import LLMResponse

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = "openai/gpt-oss-20b"

# Groq's strict-mode validator mishandles a string|array anyOf for wires
# (observed live: it validated the model's legal string against the array
# branch and 400'd). Wires are therefore array-of-int ONLY in this schema.
# No expressiveness is lost: "all" is exactly the full index list, which
# the model can (and is prompted to) write out explicitly; the semantic
# validators accept explicit lists identically.
_WIRES = {
    "type": "array",
    "items": {"type": "integer"},
    "description": "Qubit indices as separate JSON integers, e.g. [0, 1, 2, 3]. "
                   'Never strings: ["0123"] and "all" are both invalid.',
}
_ROT_LAYER = {
    "type": "object", "additionalProperties": False,
    "required": ["type", "gates", "wires"],
    "properties": {
        "type": {"type": "string", "enum": ["rot"]},
        "gates": {"type": "array",
                  "items": {"type": "string", "enum": ["RX", "RY", "RZ", "H"]}},
        "wires": _WIRES,
    },
}
_ENT_LAYER = {
    "type": "object", "additionalProperties": False,
    "required": ["type", "pattern", "gate", "wires", "center", "pairs"],
    "properties": {
        "type": {"type": "string", "enum": ["entangle"]},
        "pattern": {"type": "string",
                    "enum": ["ring", "line", "star", "all_to_all", "pairs", "none"]},
        "gate": {"type": "string", "enum": ["CNOT", "CZ", "CRZ"]},
        "wires": _WIRES,
        "center": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
        "pairs": {"anyOf": [
            {"type": "array", "items": {"type": "array", "items": {"type": "integer"},
                                        "minItems": 2, "maxItems": 2}},
            {"type": "null"},
        ]},
    },
}
_REPEAT_LAYER = {
    "type": "object", "additionalProperties": False,
    "required": ["type", "times", "body"],
    "properties": {
        "type": {"type": "string", "enum": ["repeat"]},
        "times": {"type": "integer"},
        "body": {"type": "array", "items": {"anyOf": [_ROT_LAYER, _ENT_LAYER]}},
    },
}

#: Strict-mode JSON Schema for CircuitIR. `metadata` (a free-form string
#: map with a default) is omitted: strict mode cannot express arbitrary
#: keys, and the field is optional in the pydantic model.
CIRCUIT_IR_STRICT_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["n_qubits", "encoding", "layers", "measurements"],
    "properties": {
        "n_qubits": {"type": "integer"},
        "encoding": {
            "type": "object", "additionalProperties": False,
            "required": ["type", "gate", "wires", "reupload"],
            "properties": {
                "type": {"type": "string", "enum": ["angle", "amplitude"]},
                "gate": {"anyOf": [{"type": "string", "enum": ["RX", "RY", "RZ"]},
                                   {"type": "null"}]},
                "wires": _WIRES,
                "reupload": {"type": "integer"},
            },
        },
        "layers": {"type": "array",
                   "items": {"anyOf": [_ROT_LAYER, _ENT_LAYER, _REPEAT_LAYER]}},
        "measurements": {
            "type": "object", "additionalProperties": False,
            "required": ["observable", "wires"],
            "properties": {
                "observable": {"type": "string", "enum": ["X", "Y", "Z"]},
                "wires": _WIRES,
            },
        },
    },
}


class FreeTierExhaustedError(Exception):
    """An internal free-tier cap (tokens, requests, or reserve) would be
    crossed by the next request. Raised BEFORE issuing it."""


class FreeTierController:
    """Client-side free-plan envelope: cumulative token cap, request cap,
    RPM/TPM pacing. All checks happen before a request is issued."""

    def __init__(
        self,
        max_total_tokens: int = 180_000,
        max_requests: int = 150,
        min_seconds_between_calls: float = 8.0,  # <30 RPM and ~<8k TPM at ~1k tok/call
        expected_tokens_per_call: int = 1_000,
    ) -> None:
        self.max_total_tokens = max_total_tokens
        self.max_requests = max_requests
        self.min_seconds_between_calls = min_seconds_between_calls
        self.expected_tokens_per_call = expected_tokens_per_call
        self.total_tokens = 0
        self.requests_made = 0
        self._last_call_at = 0.0

    def check_before_request(self) -> None:
        if self.requests_made >= self.max_requests:
            raise FreeTierExhaustedError(
                f"request cap reached ({self.requests_made}/{self.max_requests})"
            )
        if self.total_tokens + self.expected_tokens_per_call > self.max_total_tokens:
            raise FreeTierExhaustedError(
                f"token cap would be crossed ({self.total_tokens} used, "
                f"+~{self.expected_tokens_per_call} expected > {self.max_total_tokens})"
            )

    def pace(self) -> None:
        wait = self.min_seconds_between_calls - (time.monotonic() - self._last_call_at)
        if wait > 0:
            time.sleep(wait)
        self._last_call_at = time.monotonic()

    def record(self, total_tokens: int) -> None:
        self.requests_made += 1
        self.total_tokens += total_tokens


class GroqProvider:
    """One real Groq chat-completion request per `complete()` call, with
    strict structured output constrained to `CIRCUIT_IR_STRICT_SCHEMA`."""

    def __init__(
        self,
        api_key: str,
        controller: FreeTierController,
        model: str = GROQ_MODEL,
        max_output_tokens: int = 512,
        temperature: float = 0.2,
        reasoning_effort: str = "low",
        max_transport_retries: int = 1,
    ) -> None:
        self.model_name = model
        self.controller = controller
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.max_transport_retries = max_transport_retries
        # max_retries=0: OUR controller owns retry/pacing, not the SDK.
        self._client = OpenAI(api_key=api_key, base_url=GROQ_BASE_URL, max_retries=0)

    def complete(self, system_prompt: str, user_prompt: str, temperature: float) -> LLMResponse:
        from openai import APIStatusError, BadRequestError, RateLimitError

        self.controller.check_before_request()
        start = time.monotonic()
        response = None
        for attempt in range(self.max_transport_retries + 1):
            self.controller.pace()
            try:
                response = self._client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.temperature,  # fixed low-temp decoding for this provider
                    max_completion_tokens=self.max_output_tokens,
                    reasoning_effort=self.reasoning_effort,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "circuit_ir",
                            "strict": True,
                            "schema": CIRCUIT_IR_STRICT_SCHEMA,
                        },
                    },
                )
                break
            except BadRequestError as exc:
                body = getattr(exc, "body", None) or {}
                err = body.get("error", body) if isinstance(body, dict) else {}
                if isinstance(err, dict) and err.get("code") == "json_validate_failed":
                    # Groq generated output that failed ITS schema check.
                    # This is malformed model output, not a transport
                    # failure: return the failed generation as the raw
                    # response so it flows through the repository's normal
                    # malformed/invalid-proposal accounting (a real,
                    # reportable property of this model -- not an abort).
                    latency = time.monotonic() - start
                    raw = str(err.get("failed_generation") or "")
                    approx_tokens = (len(system_prompt) + len(user_prompt) + len(raw)) // 4
                    self.controller.record(approx_tokens)  # usage not returned on 400
                    return LLMResponse(
                        raw_text=raw, model=self.model_name,
                        input_tokens=0, output_tokens=0,  # not reported by the API; not fabricated
                        estimated_cost_usd=None, latency_seconds=latency,
                    )
                raise
            except (RateLimitError, APIStatusError) as exc:
                status = getattr(exc, "status_code", None)
                transient = isinstance(exc, RateLimitError) or (status and status >= 500)
                if not transient or attempt >= self.max_transport_retries:
                    raise
                retry_after = 15.0
                headers = getattr(getattr(exc, "response", None), "headers", None)
                if headers is not None and headers.get("retry-after"):
                    try:
                        retry_after = float(headers["retry-after"])
                    except ValueError:
                        pass
                time.sleep(retry_after)
        assert response is not None
        latency = time.monotonic() - start

        raw_text = response.choices[0].message.content or ""
        usage = response.usage
        input_tokens = int(usage.prompt_tokens) if usage else 0
        output_tokens = int(usage.completion_tokens) if usage else 0
        total = int(usage.total_tokens) if usage else input_tokens + output_tokens
        self.controller.record(total)

        return LLMResponse(
            raw_text=raw_text,
            model=response.model or self.model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=None,  # free plan; no dollar figure returned or fabricated
            latency_seconds=latency,
        )
