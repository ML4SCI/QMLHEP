"""Tests for the Stage 5 LLM infrastructure and search arms: cost-cap
enforcement, the offline mock provider, bounded repair retries, full
provenance recording, and the `llm_iter`/`llm_evo` arms driven through the
real `SearchRunner`.

Every test in this file uses `MockLLMProvider` only. No test in this
suite makes, or could make, a network call -- see
`test_llm_package_contains_no_network_or_sdk_imports`.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from llm_vqc.evaluation.store import ResultStore
from llm_vqc.evaluation.training import TrainingConfig
from llm_vqc.llm.budget import LLMApiBudget, LLMBudgetExceededError
from llm_vqc.llm.driver import LLMDriver
from llm_vqc.llm.provider import LLMResponse, MockLLMProvider
from llm_vqc.search.arms.llm_evo_arm import LLMEvoArm
from llm_vqc.search.arms.llm_iter_arm import LLMIterArm
from llm_vqc.search.runner import SearchRunner
from llm_vqc.tasks.t1_gaussian import T1GaussianPeakTask

TASK_DESCRIPTION = "T1 1D Gaussian-peak regression"
REPRO_FIELDS = {"task": "T1"}


@pytest.fixture(scope="module")
def t1_data():
    return T1GaussianPeakTask().build(seed=0)


def _training_config() -> TrainingConfig:
    return TrainingConfig(epochs=2)


def _store(tmp_path, run_id, name, budget):
    return ResultStore.open_or_create(
        tmp_path / name,
        run_id=run_id,
        config_json="{}",
        config_reproducibility_fields={**REPRO_FIELDS, "budget": budget},
        git_sha=None,
        created_at="t0",
    )


# --- LLMApiBudget: the hard cost-cap policy -------------------------------


def test_from_env_returns_none_when_unset():
    assert LLMApiBudget.from_env(env={}) is None


def test_from_env_returns_none_when_zero():
    assert LLMApiBudget.from_env(env={"LLM_API_BUDGET_USD": "0"}) is None


def test_from_env_returns_none_when_negative():
    assert LLMApiBudget.from_env(env={"LLM_API_BUDGET_USD": "-5"}) is None


def test_from_env_returns_none_when_unparseable():
    assert LLMApiBudget.from_env(env={"LLM_API_BUDGET_USD": "not-a-number"}) is None


def test_from_env_returns_budget_when_valid_positive():
    budget = LLMApiBudget.from_env(env={"LLM_API_BUDGET_USD": "10.50"})
    assert budget is not None
    assert budget.cap_usd == 10.50
    assert budget.remaining_usd == 10.50


def test_check_can_afford_raises_before_exceeding_cap():
    budget = LLMApiBudget(cap_usd=1.0)
    budget.record_spend(0.9)
    with pytest.raises(LLMBudgetExceededError):
        budget.check_can_afford(0.2)  # would bring total to 1.1 > 1.0 cap
    budget.check_can_afford(0.05)  # comfortably under the remaining ~0.1 is fine


def test_cannot_construct_budget_with_nonpositive_cap():
    with pytest.raises(ValueError):
        LLMApiBudget(cap_usd=0)
    with pytest.raises(ValueError):
        LLMApiBudget(cap_usd=-1)


# --- MockLLMProvider: deterministic, offline, zero-cost -------------------


def test_mock_provider_is_a_pure_function_of_prompt_text_not_a_counter():
    provider_a = MockLLMProvider(seed=0)
    provider_b = MockLLMProvider(seed=0)
    response_a = provider_a.complete("sys", "user1", 0.7)
    response_b = provider_b.complete("sys", "user1", 0.7)
    assert response_a.raw_text == response_b.raw_text

    # Same provider instance, called out of a fresh "process" (simulated by
    # a brand new provider object) with the same prompt still matches --
    # this is what makes checkpoint/resume determinism possible.
    provider_c = MockLLMProvider(seed=0)
    response_c = provider_c.complete("sys", "user1", 0.7)
    assert response_c.raw_text == response_a.raw_text


def test_mock_provider_different_prompts_give_different_responses():
    provider = MockLLMProvider(seed=0)
    r1 = provider.complete("sys", "user1", 0.7)
    r2 = provider.complete("sys", "user2", 0.7)
    assert r1.raw_text != r2.raw_text


def test_mock_provider_reports_zero_cost_and_latency():
    provider = MockLLMProvider(seed=0)
    response = provider.complete("sys", "user", 0.7)
    assert response.estimated_cost_usd == 0.0
    assert response.latency_seconds == 0.0


def test_llm_package_contains_no_network_or_sdk_imports():
    """No module in llm_vqc/llm may import a network/HTTP/SDK library
    EXCEPT `openai_provider.py`, whose entire purpose is to wrap the real
    OpenAI SDK for the human-approved, time-boxed real-API runs
    (`llm_vqc.llm.budget.LLMApiBudget` still gates whether it can ever be
    used with real spend). Every other module -- `MockLLMProvider`
    included -- must stay offline."""
    forbidden = {"requests", "httpx", "urllib", "urllib2", "socket", "openai", "anthropic"}
    sanctioned_exceptions = {"openai_provider.py", "groq_provider.py"}
    pkg = pathlib.Path("llm_vqc/llm")
    for path in pkg.glob("*.py"):
        if path.name in sanctioned_exceptions:
            continue
        tree = ast.parse(path.read_text())
        targets = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                targets.append((node.module or "").split(".")[0])
            elif isinstance(node, ast.Import):
                targets.extend(a.name.split(".")[0] for a in node.names)
        assert not (set(targets) & forbidden), (path.name, targets)


def test_llm_package_never_imports_final_test():
    pkg = pathlib.Path("llm_vqc/llm")
    for path in pkg.glob("*.py"):
        tree = ast.parse(path.read_text())
        targets = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                targets.append(node.module or "")
            elif isinstance(node, ast.Import):
                targets.extend(a.name for a in node.names)
        assert not any("final_test" in t for t in targets), (path.name, targets)


# --- LLMDriver: bounded repair retries ------------------------------------


def test_driver_succeeds_on_first_attempt_when_provider_is_well_formed():
    driver = LLMDriver(MockLLMProvider(seed=0, malformed_rate=0.0), max_repair_attempts=2)
    outcome = driver.propose("p1", "sys", "user", 0.7)
    assert outcome.proposal is not None
    assert outcome.valid_schema
    assert outcome.retry_count == 0
    assert len(outcome.call_records) == 1


def test_driver_gives_up_after_max_repair_attempts_never_unlimited():
    driver = LLMDriver(MockLLMProvider(seed=0, malformed_rate=1.0), max_repair_attempts=2)
    outcome = driver.propose("p1", "sys", "user", 0.7)
    assert outcome.proposal is None
    assert not outcome.valid_schema
    assert outcome.retry_count == 2
    assert len(outcome.call_records) == 3  # 1 first attempt + 2 repair retries, no more


def test_driver_charges_the_estimate_when_provider_reports_no_cost():
    """A real provider (e.g. OpenAI chat completions) reports cost=None.
    The driver must then charge the pre-call estimate against the dollar
    cap -- otherwise the cap would accumulate zero spend and be decorative."""

    class _NoCostProvider:
        model_name = "no-cost-test-provider"

        def complete(self, system_prompt, user_prompt, temperature):
            return LLMResponse(
                raw_text="not json",
                model=self.model_name,
                input_tokens=10,
                output_tokens=10,
                estimated_cost_usd=None,
                latency_seconds=0.01,
            )

    budget = LLMApiBudget(cap_usd=1.0)
    driver = LLMDriver(
        _NoCostProvider(), max_repair_attempts=1, budget=budget, cost_estimate_per_call_usd=0.25
    )
    driver.propose("p1", "sys", "user", 0.7)
    # 2 calls (first + one repair), 0.25 charged each despite cost=None.
    assert budget.spent_usd == pytest.approx(0.5)


def test_call_count_limited_provider_stops_at_exact_cap():
    from llm_vqc.llm.openai_provider import CallBudgetExceededError, CallCountLimitedProvider

    class _FakeInner:
        model_name = "fake"

        def complete(self, system_prompt, user_prompt, temperature):
            return LLMResponse(
                raw_text="{}", model="fake", input_tokens=1, output_tokens=1,
                latency_seconds=0.0,
            )

    provider = CallCountLimitedProvider(_FakeInner(), max_calls=2)
    provider.complete("s", "u", 0.1)
    provider.complete("s", "u", 0.1)
    assert provider.calls_made == 2
    with pytest.raises(CallBudgetExceededError):
        provider.complete("s", "u", 0.1)
    assert provider.calls_made == 2  # the rejected call was never made


def test_proposal_outcome_total_cost_is_none_when_no_call_reported_cost():
    from llm_vqc.llm.records import LLMCallRecord, LLMProposalOutcome

    record = LLMCallRecord(
        proposal_id="p1", call_index=0, system_prompt="s", user_prompt="u",
        model="m", temperature=0.7, raw_response="{}", parsed_proposal={},
        input_tokens=1, output_tokens=1, estimated_cost_usd=None, latency_seconds=0.0,
    )
    outcome = LLMProposalOutcome(
        proposal={}, valid_schema=True, retry_count=0, call_records=[record]
    )
    assert outcome.total_cost_usd is None  # never fabricated as 0.0


def test_driver_enforces_cost_cap_before_every_call_including_retries():
    class _ExpensiveMalformedProvider:
        model_name = "expensive-test-provider"

        def __init__(self):
            self.calls = 0

        def complete(self, system_prompt, user_prompt, temperature):
            self.calls += 1
            return LLMResponse(
                raw_text="not json",
                model=self.model_name,
                input_tokens=10,
                output_tokens=10,
                estimated_cost_usd=1.0,
                latency_seconds=0.01,
            )

    provider = _ExpensiveMalformedProvider()
    budget = LLMApiBudget(cap_usd=1.5)  # affords exactly 1 call, not 2
    driver = LLMDriver(
        provider, max_repair_attempts=3, budget=budget, cost_estimate_per_call_usd=1.0
    )

    from llm_vqc.llm.budget import LLMBudgetExceededError as _Exceeded

    with pytest.raises(_Exceeded):
        driver.propose("p1", "sys", "user", 0.7)
    assert provider.calls == 1  # stopped before the second call, not after


# --- LLM arms driven through the real SearchRunner ------------------------


def _llm_iter_arm(result_store, run_id, open_loop=False):
    return LLMIterArm(
        lower_is_better=True,
        task_description=TASK_DESCRIPTION,
        provider=MockLLMProvider(seed=0),
        result_store=result_store,
        run_id=run_id,
        budget=None,  # no LLM_API_BUDGET_USD configured -- no paid calls, ever
        open_loop=open_loop,
    )


def _llm_evo_arm(result_store, run_id):
    return LLMEvoArm(
        lower_is_better=True,
        task_description=TASK_DESCRIPTION,
        provider=MockLLMProvider(seed=0),
        result_store=result_store,
        run_id=run_id,
        budget=None,
    )


@pytest.mark.parametrize("open_loop", [False, True])
def test_llm_iter_completes_exact_budget_smoke_run(tmp_path, t1_data, open_loop):
    budget = 4
    store = _store(tmp_path, "run", f"llm_iter_{open_loop}.sqlite", budget)
    runner = SearchRunner(
        _llm_iter_arm(store, "run", open_loop=open_loop),
        "T1",
        t1_data,
        _training_config(),
        budget_limit=budget,
        run_seed=0,
        result_store=store,
        run_id="run",
    )
    result = runner.run()
    assert result.ledger_summary["consumed_budget"] == budget
    store.close()


def test_llm_evo_completes_exact_budget_smoke_run(tmp_path, t1_data):
    budget = 4
    store = _store(tmp_path, "run", "llm_evo.sqlite", budget)
    runner = SearchRunner(
        _llm_evo_arm(store, "run"),
        "T1",
        t1_data,
        _training_config(),
        budget_limit=budget,
        run_seed=0,
        result_store=store,
        run_id="run",
    )
    result = runner.run()
    assert result.ledger_summary["consumed_budget"] == budget
    store.close()


def test_llm_iter_is_deterministic_under_a_fixed_seed(tmp_path, t1_data):
    budget = 3
    results = []
    for i in range(2):
        store = _store(tmp_path, "run", f"det_{i}.sqlite", budget)
        runner = SearchRunner(
            _llm_iter_arm(store, "run"),
            "T1",
            t1_data,
            _training_config(),
            budget_limit=budget,
            run_seed=9,
            result_store=store,
            run_id="run",
        )
        results.append(runner.run())
        store.close()
    assert results[0].selected_structural_hash == results[1].selected_structural_hash
    assert results[0].ledger_summary == results[1].ledger_summary


def test_llm_iter_interrupted_run_resumes_to_same_result_as_unbroken(tmp_path, t1_data):
    budget = 4

    ref_store = _store(tmp_path, "ref", "iter_ref.sqlite", budget)
    ref_result = SearchRunner(
        _llm_iter_arm(ref_store, "ref"), "T1", t1_data, _training_config(),
        budget_limit=budget, run_seed=3, result_store=ref_store, run_id="ref",
    ).run()
    ref_store.close()

    partial_store = _store(tmp_path, "resume", "iter_resume.sqlite", 2)
    SearchRunner(
        _llm_iter_arm(partial_store, "resume"), "T1", t1_data, _training_config(),
        budget_limit=2, run_seed=3, result_store=partial_store, run_id="resume",
    ).run()
    partial_store.close()

    resumed_store = ResultStore.open_or_create(
        tmp_path / "iter_resume.sqlite",
        run_id="resume",
        config_json="{}",
        config_reproducibility_fields={**REPRO_FIELDS, "budget": budget},
        git_sha=None,
        created_at="t1",
        allow_incompatible=True,
    )
    resumed_result = SearchRunner(
        _llm_iter_arm(resumed_store, "resume"), "T1", t1_data, _training_config(),
        budget_limit=budget, run_seed=3, result_store=resumed_store, run_id="resume",
    ).run()
    resumed_store.close()

    assert resumed_result.selected_structural_hash == ref_result.selected_structural_hash
    assert resumed_result.ledger_summary == ref_result.ledger_summary


def test_llm_calls_are_fully_persisted_with_provenance(tmp_path, t1_data):
    budget = 3
    store = _store(tmp_path, "run", "provenance.sqlite", budget)
    runner = SearchRunner(
        _llm_iter_arm(store, "run"), "T1", t1_data, _training_config(),
        budget_limit=budget, run_seed=1, result_store=store, run_id="run",
    )
    runner.run()

    calls = list(store.iter_llm_calls("run"))
    assert len(calls) >= budget  # at least one LLM call per proposal
    for call in calls:
        assert call.system_prompt
        assert call.model
        assert call.raw_response
        assert call.input_tokens >= 0
        assert call.output_tokens >= 0
        assert call.estimated_cost_usd == 0.0  # mock provider: genuinely zero cost
    store.close()


def test_llm_arm_never_receives_a_budget_object_without_llm_api_budget_usd_set():
    """This work cycle's arms are always constructed with budget=None --
    asserting this directly documents that no paid call path is reachable
    without an explicit LLM_API_BUDGET_USD."""
    arm = _llm_iter_arm(result_store=None, run_id="x")  # result_store unused before propose()
    assert arm.driver.budget is None
