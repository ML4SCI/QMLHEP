"""Tests for the Stage 4 non-LLM search arms: `random`, `greedy`,
`evolutionary`. Exercises each arm through the real `SearchRunner` (not a
toy), on the real T1 task, matching Phase 4's acceptance criterion ("a
smoke grid on T1 completes unattended; best-so-far curves monotone").
"""

from __future__ import annotations

import numpy as np
import pytest

from llm_vqc.evaluation.store import ResultStore
from llm_vqc.evaluation.training import TrainingConfig
from llm_vqc.ir.validators import validate_proposal
from llm_vqc.search.arms.evolutionary_arm import EvolutionaryArm, EvolutionaryArmConfig
from llm_vqc.search.arms.greedy_arm import GreedyArm, minimal_ir
from llm_vqc.search.arms.mutation import crossover, mutate
from llm_vqc.search.arms.random_arm import RandomArm
from llm_vqc.search.runner import SearchRunner
from llm_vqc.tasks.t1_gaussian import T1GaussianPeakTask

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


ARM_FACTORIES = {
    "random": lambda: RandomArm(lower_is_better=True),
    "greedy": lambda: GreedyArm(lower_is_better=True, k=3),
    "evolutionary": lambda: EvolutionaryArm(
        lower_is_better=True, config=EvolutionaryArmConfig(mu=2, lambda_=2)
    ),
}


@pytest.mark.parametrize("arm_name", ["random", "greedy", "evolutionary"])
def test_arm_completes_a_smoke_run_at_exact_budget(tmp_path, t1_data, arm_name):
    budget = 6
    store = _store(tmp_path, f"run_{arm_name}", f"{arm_name}.sqlite", budget)
    runner = SearchRunner(
        ARM_FACTORIES[arm_name](),
        "T1",
        t1_data,
        _training_config(),
        budget_limit=budget,
        run_seed=0,
        result_store=store,
        run_id=f"run_{arm_name}",
    )
    result = runner.run()
    assert result.ledger_summary["consumed_budget"] == budget
    store.close()


@pytest.mark.parametrize("arm_name", ["random", "greedy", "evolutionary"])
def test_arm_is_deterministic_under_a_fixed_seed(tmp_path, t1_data, arm_name):
    budget = 4
    results = []
    for i in range(2):
        store = _store(tmp_path, "run", f"{arm_name}_{i}.sqlite", budget)
        runner = SearchRunner(
            ARM_FACTORIES[arm_name](),
            "T1",
            t1_data,
            _training_config(),
            budget_limit=budget,
            run_seed=11,
            result_store=store,
            run_id="run",
        )
        results.append(runner.run())
        store.close()
    assert results[0].selected_structural_hash == results[1].selected_structural_hash
    assert results[0].ledger_summary == results[1].ledger_summary


@pytest.mark.parametrize("arm_name", ["random", "greedy", "evolutionary"])
def test_arm_interrupted_run_resumes_to_same_result_as_unbroken(tmp_path, t1_data, arm_name):
    budget = 6

    ref_store = _store(tmp_path, "ref", f"{arm_name}_ref.sqlite", budget)
    ref_runner = SearchRunner(
        ARM_FACTORIES[arm_name](), "T1", t1_data, _training_config(),
        budget_limit=budget, run_seed=5, result_store=ref_store, run_id="ref",
    )
    ref_result = ref_runner.run()
    ref_store.close()

    partial_store = _store(tmp_path, "resume", f"{arm_name}_resume.sqlite", 3)
    partial_runner = SearchRunner(
        ARM_FACTORIES[arm_name](), "T1", t1_data, _training_config(),
        budget_limit=3, run_seed=5, result_store=partial_store, run_id="resume",
    )
    partial_runner.run()
    partial_store.close()

    resumed_store = ResultStore.open_or_create(
        tmp_path / f"{arm_name}_resume.sqlite",
        run_id="resume",
        config_json="{}",
        config_reproducibility_fields={**REPRO_FIELDS, "budget": budget},
        git_sha=None,
        created_at="t1",
        allow_incompatible=True,
    )
    resumed_runner = SearchRunner(
        ARM_FACTORIES[arm_name](), "T1", t1_data, _training_config(),
        budget_limit=budget, run_seed=5, result_store=resumed_store, run_id="resume",
    )
    resumed_result = resumed_runner.run()
    resumed_store.close()

    assert resumed_result.selected_structural_hash == ref_result.selected_structural_hash
    assert resumed_result.ledger_summary == ref_result.ledger_summary


# --- greedy-specific ------------------------------------------------------


def test_greedy_starts_from_the_fixed_minimal_circuit():
    arm = GreedyArm(lower_is_better=True)
    state = arm.initialize(seed=123)
    assert state.current_ir == minimal_ir(arm.start_n_qubits)
    assert state.round_index == 0


def test_greedy_advances_round_after_k_evaluations(tmp_path, t1_data):
    budget = 9  # exactly 3 rounds of k=3
    store = _store(tmp_path, "greedy_round", "greedy_round.sqlite", budget)
    runner = SearchRunner(
        GreedyArm(lower_is_better=True, k=3), "T1", t1_data, _training_config(),
        budget_limit=budget, run_seed=2, result_store=store, run_id="greedy_round",
    )
    runner.run()
    raw = store.load_run_state("greedy_round")
    state = GreedyArm(lower_is_better=True, k=3).deserialize_state(raw)
    assert state.round_index == 3  # advanced exactly once per full round of 3
    store.close()


# --- evolutionary-specific -------------------------------------------------


def test_evolutionary_population_size_matches_mu_after_generation_zero(tmp_path, t1_data):
    config = EvolutionaryArmConfig(mu=2, lambda_=2)
    budget = 2  # exactly the initial mu=2 population, generation 0 -> 1 transition only
    store = _store(tmp_path, "evo_pop", "evo_pop.sqlite", budget)
    runner = SearchRunner(
        EvolutionaryArm(lower_is_better=True, config=config), "T1", t1_data, _training_config(),
        budget_limit=budget, run_seed=4, result_store=store, run_id="evo_pop",
    )
    runner.run()
    raw = store.load_run_state("evo_pop")
    state = EvolutionaryArm(lower_is_better=True, config=config).deserialize_state(raw)
    assert state.generation == 1
    assert len(state.population) == config.mu
    assert len(state.pending) == config.lambda_
    store.close()


def test_mutation_operators_always_produce_valid_ir():
    rng = np.random.default_rng(0)
    base = minimal_ir(4)
    base = base.model_copy(update={"layers": [*base.layers]})
    for _ in range(50):
        mutated = mutate(base, rng, mutation_rate=1.0)
        assert validate_proposal(mutated).valid


def test_crossover_always_produces_valid_ir():
    rng = np.random.default_rng(0)
    from llm_vqc.ir.sampler import sample_random_ir

    for _ in range(20):
        parent_a = sample_random_ir(rng)
        parent_b = sample_random_ir(rng)
        child = crossover(parent_a, parent_b, rng)
        assert validate_proposal(child).valid
