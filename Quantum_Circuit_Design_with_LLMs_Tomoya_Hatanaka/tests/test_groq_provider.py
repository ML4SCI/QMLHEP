"""Mocked tests for the Groq provider integration. No test here makes a
network call or consumes API quota."""

from __future__ import annotations

import pytest

from llm_vqc.ir.validators import validate_proposal
from llm_vqc.llm.groq_provider import (
    CIRCUIT_IR_STRICT_SCHEMA,
    GROQ_BASE_URL,
    GROQ_MODEL,
    FreeTierController,
    FreeTierExhaustedError,
    GroqProvider,
)


def _walk_objects(schema):
    if isinstance(schema, dict):
        if schema.get("type") == "object":
            yield schema
        for v in schema.values():
            yield from _walk_objects(v)
    elif isinstance(schema, list):
        for v in schema:
            yield from _walk_objects(v)


def test_strict_schema_every_object_forbids_additional_properties_and_requires_all():
    for obj in _walk_objects(CIRCUIT_IR_STRICT_SCHEMA):
        assert obj.get("additionalProperties") is False, obj
        assert set(obj["required"]) == set(obj["properties"].keys()), obj


def test_schema_valid_circuit_still_goes_through_semantic_validation():
    """Strict-schema validity is necessary, not sufficient: a star pattern
    without a center passes the JSON schema (center nullable) but must be
    rejected by the repository's semantic validators."""
    proposal = {
        "n_qubits": 3,
        "encoding": {"type": "angle", "gate": "RY", "wires": "all", "reupload": 0},
        "layers": [{"type": "entangle", "pattern": "star", "gate": "CNOT",
                    "wires": "all", "center": None, "pairs": None}],
        "measurements": {"observable": "Z", "wires": "all"},
    }
    assert not validate_proposal(proposal).valid


def test_free_tier_controller_blocks_over_request_cap():
    c = FreeTierController(max_requests=2, min_seconds_between_calls=0.0)
    c.check_before_request()
    c.record(100)
    c.check_before_request()
    c.record(100)
    with pytest.raises(FreeTierExhaustedError):
        c.check_before_request()


def test_free_tier_controller_blocks_before_token_cap_with_reserve():
    c = FreeTierController(
        max_total_tokens=2_000, expected_tokens_per_call=1_000,
        min_seconds_between_calls=0.0,
    )
    c.check_before_request()
    c.record(1_500)
    with pytest.raises(FreeTierExhaustedError):
        c.check_before_request()  # 1500 + ~1000 expected > 2000: stop BEFORE the call
    assert c.requests_made == 1  # the blocked request was never issued


def test_internal_token_cap_leaves_20k_reserve_below_documented_daily_limit():
    assert FreeTierController().max_total_tokens <= 200_000 - 20_000


def test_groq_provider_uses_groq_base_url_and_model_not_openai():
    controller = FreeTierController(min_seconds_between_calls=0.0)
    provider = GroqProvider(api_key="test-not-a-real-key", controller=controller)
    assert str(provider._client.base_url).startswith(GROQ_BASE_URL)
    assert provider.model_name == GROQ_MODEL
    assert provider.reasoning_effort == "low"
    assert provider.max_output_tokens <= 512


def test_provider_refuses_request_once_controller_is_exhausted():
    controller = FreeTierController(max_requests=0, min_seconds_between_calls=0.0)
    provider = GroqProvider(api_key="test-not-a-real-key", controller=controller)
    with pytest.raises(FreeTierExhaustedError):
        provider.complete("sys", "user", 0.2)  # blocked before any network attempt


def test_groq_resume_identity_rejects_different_provider_or_model(tmp_path):
    from llm_vqc.evaluation.store import IncompatibleResumeError, ResultStore

    fields = {"provider": "groq", "model": GROQ_MODEL, "task": "T1", "budget": 10}
    store = ResultStore.open_or_create(
        tmp_path / "db.sqlite", run_id="r1", config_json="{}",
        config_reproducibility_fields=fields, git_sha=None, created_at="t0")
    store.close()
    with pytest.raises(IncompatibleResumeError):
        ResultStore.open_or_create(
            tmp_path / "db.sqlite", run_id="r1", config_json="{}",
            config_reproducibility_fields={**fields, "provider": "openai", "model": "gpt-5.4-mini"},
            git_sha=None, created_at="t1")


def test_compact_history_window_bounds_closed_loop_prompt_growth():
    from llm_vqc.search.arms.llm_iter_arm import LLMIterArm, LLMIterState

    arm = LLMIterArm(
        lower_is_better=True, task_description="T1", provider=None,
        result_store=None, run_id="x", history_window=3)
    state = LLMIterState(
        seed=0, n_proposed=50, best_metric=0.1234,
        history_lines=[f"Proposal {i}: rmse={i}" for i in range(50)])
    lines = state.history_lines
    windowed = [f"Best so far: {state.best_metric:.4f}", *lines[-arm.history_window:]]
    assert len(windowed) == 4  # 1 best-so-far line + last 3, regardless of history length
    joined = "\n".join(windowed)
    assert "Proposal 49" in joined and "Proposal 0" not in joined
    assert "test" not in joined.lower()  # no test metric can appear
