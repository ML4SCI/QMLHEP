"""Tests for the shared search framework (Stage 3): `SearchArm`,
`SearchFeedback`, and `SearchRunner`.

Uses a minimal deterministic toy arm (uniform-random valid IR proposals,
same generator as the master-plan `random` arm) purely to exercise the
*shared runner's* invariants -- exact budget termination, checkpoint/
resume equivalence, determinism under a fixed seed, different behavior
under different seeds, and the structural no-test-access guarantee. The
real `random`/`evolutionary`/`greedy` arms are Stage 4.
"""

from __future__ import annotations

import ast
import subprocess
import sys

import numpy as np
import pytest
from pydantic import BaseModel

from llm_vqc.evaluation.seeds import derive_child_seed
from llm_vqc.evaluation.store import ResultStore
from llm_vqc.evaluation.training import TrainingConfig
from llm_vqc.ir.sampler import sample_random_ir
from llm_vqc.search.arm import SearchArm
from llm_vqc.search.comparison import is_strictly_better
from llm_vqc.search.feedback import SearchFeedback
from llm_vqc.search.runner import SearchRunner, SearchRunnerError
from llm_vqc.tasks.t1_gaussian import T1GaussianPeakTask

REPRO_FIELDS = {"task": "T1", "arm": "toy_random", "budget": 6}


class _ToyState(BaseModel):
    seed: int
    n_proposed: int = 0
    best_hash: str | None = None
    best_metric: float | None = None


class ToyRandomArm(SearchArm[_ToyState]):
    """Minimal arm: uniform-random valid IR, tracks best-so-far by
    validation metric. Deterministic given `state.seed` and
    `state.n_proposed` alone -- no arm-internal mutable RNG object, so
    resuming from a serialized `_ToyState` reproduces the same draws."""

    name = "toy_random"

    def initialize(self, seed: int) -> _ToyState:
        return _ToyState(seed=seed)

    def propose(self, state: _ToyState) -> dict:
        rng = np.random.default_rng(
            derive_child_seed(state.seed, "propose", str(state.n_proposed))
        )
        return sample_random_ir(rng).model_dump()

    def update_state(self, state: _ToyState, proposal: dict, feedback: SearchFeedback) -> _ToyState:
        best_hash, best_metric = state.best_hash, state.best_metric
        if feedback.val_metric_value is not None and is_strictly_better(
            feedback.val_metric_value, best_metric, lower_is_better=True
        ):
            best_hash, best_metric = feedback.structural_hash, feedback.val_metric_value
        return state.model_copy(
            update={
                "n_proposed": state.n_proposed + 1,
                "best_hash": best_hash,
                "best_metric": best_metric,
            }
        )

    def select_final(self, state: _ToyState) -> str | None:
        return state.best_hash

    def deserialize_state(self, raw_json: str) -> _ToyState:
        return _ToyState.model_validate_json(raw_json)


class _AlwaysInvalidArm(SearchArm[_ToyState]):
    """Pathological arm used only to test the runaway-loop safety cap."""

    name = "always_invalid"

    def initialize(self, seed: int) -> _ToyState:
        return _ToyState(seed=seed)

    def propose(self, state: _ToyState) -> dict:
        return {"n_qubits": 3, "encoding": {"type": "angle"}, "layers": [], "measurements": {}}

    def update_state(self, state: _ToyState, proposal: dict, feedback: SearchFeedback) -> _ToyState:
        return state.model_copy(update={"n_proposed": state.n_proposed + 1})

    def select_final(self, state: _ToyState) -> str | None:
        return None

    def deserialize_state(self, raw_json: str) -> _ToyState:
        return _ToyState.model_validate_json(raw_json)


@pytest.fixture(scope="module")
def t1_data():
    return T1GaussianPeakTask().build(seed=0)


def _training_config() -> TrainingConfig:
    return TrainingConfig(epochs=2)


def _store(tmp_path, run_id, budget=6, name="db.sqlite"):
    return ResultStore.open_or_create(
        tmp_path / name,
        run_id=run_id,
        config_json="{}",
        config_reproducibility_fields={**REPRO_FIELDS, "budget": budget},
        git_sha=None,
        created_at="t0",
    )


# --- Structural quarantine (the core guarantee) -------------------------


def test_search_package_never_imports_final_test():
    import pathlib

    pkg = pathlib.Path("llm_vqc/search")
    for path in pkg.glob("*.py"):
        tree = ast.parse(path.read_text())
        targets = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                targets.append(node.module or "")
            elif isinstance(node, ast.Import):
                targets.extend(a.name for a in node.names)
        assert not any("final_test" in t for t in targets), (path.name, targets)


def test_importing_search_runner_does_not_pull_in_final_test():
    code = (
        "import llm_vqc.search.runner; import sys; "
        "print('llm_vqc.evaluation.final_test' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.stdout.strip() == "False", out.stdout + out.stderr


def test_search_feedback_has_no_test_metric_field():
    field_names = set(SearchFeedback.model_fields.keys())
    assert not any("test" in name.lower() for name in field_names), field_names


# --- Exact budget termination --------------------------------------------


def test_runner_terminates_at_exact_budget(tmp_path, t1_data):
    budget = 4
    store = _store(tmp_path, "run1", budget=budget)
    runner = SearchRunner(
        ToyRandomArm(),
        "T1",
        t1_data,
        _training_config(),
        budget_limit=budget,
        run_seed=0,
        result_store=store,
        run_id="run1",
    )
    result = runner.run()
    assert result.ledger_summary["consumed_budget"] == budget
    store.close()


def test_runner_aborts_rather_than_looping_forever_on_a_degenerate_arm(tmp_path, t1_data):
    store = _store(tmp_path, "run_bad", budget=3)
    runner = SearchRunner(
        _AlwaysInvalidArm(),
        "T1",
        t1_data,
        _training_config(),
        budget_limit=3,
        run_seed=0,
        result_store=store,
        run_id="run_bad",
    )
    with pytest.raises(SearchRunnerError):
        runner.run()
    store.close()


# --- Determinism ----------------------------------------------------------


def test_same_seed_produces_identical_results(tmp_path, t1_data):
    budget = 3
    results = []
    for i in range(2):
        store = _store(tmp_path, "run_det", budget=budget, name=f"db{i}.sqlite")
        runner = SearchRunner(
            ToyRandomArm(),
            "T1",
            t1_data,
            _training_config(),
            budget_limit=budget,
            run_seed=42,
            result_store=store,
            run_id="run_det",
        )
        results.append(runner.run())
        store.close()

    assert results[0].selected_structural_hash == results[1].selected_structural_hash
    assert results[0].selected_val_metric_value == results[1].selected_val_metric_value
    assert results[0].ledger_summary == results[1].ledger_summary


def test_different_seeds_produce_different_proposal_sequences(tmp_path, t1_data):
    budget = 3
    store_a = _store(tmp_path, "run_a", budget=budget, name="a.sqlite")
    runner_a = SearchRunner(
        ToyRandomArm(), "T1", t1_data, _training_config(),
        budget_limit=budget, run_seed=1, result_store=store_a, run_id="run_a",
    )
    result_a = runner_a.run()
    store_a.close()

    store_b = _store(tmp_path, "run_b", budget=budget, name="b.sqlite")
    runner_b = SearchRunner(
        ToyRandomArm(), "T1", t1_data, _training_config(),
        budget_limit=budget, run_seed=2, result_store=store_b, run_id="run_b",
    )
    result_b = runner_b.run()
    store_b.close()

    assert result_a.selected_structural_hash != result_b.selected_structural_hash


# --- Checkpoint / resume equivalence --------------------------------------


def test_interrupted_run_resumes_to_the_same_final_result_as_unbroken_run(tmp_path, t1_data):
    budget = 4

    # Unbroken reference run.
    ref_store = _store(tmp_path, "run_ref", budget=budget, name="ref.sqlite")
    ref_runner = SearchRunner(
        ToyRandomArm(), "T1", t1_data, _training_config(),
        budget_limit=budget, run_seed=7, result_store=ref_store, run_id="run_ref",
    )
    ref_result = ref_runner.run()
    ref_store.close()

    # Interrupted run: run at budget=2 first (simulated crash), then resume
    # a fresh store handle at the full budget=4 and finish.
    resumable_store = _store(tmp_path, "run_resume", budget=2, name="resume.sqlite")
    partial_runner = SearchRunner(
        ToyRandomArm(), "T1", t1_data, _training_config(),
        budget_limit=2, run_seed=7, result_store=resumable_store, run_id="run_resume",
    )
    partial_runner.run()
    resumable_store.close()  # simulated crash / process exit

    resumed_store = ResultStore.open_or_create(
        tmp_path / "resume.sqlite",
        run_id="run_resume",
        config_json="{}",
        config_reproducibility_fields={**REPRO_FIELDS, "budget": budget},
        git_sha=None,
        created_at="t1",
        allow_incompatible=True,  # budget field intentionally differs from the partial run
    )
    resumed_runner = SearchRunner(
        ToyRandomArm(), "T1", t1_data, _training_config(),
        budget_limit=budget, run_seed=7, result_store=resumed_store, run_id="run_resume",
    )
    resumed_result = resumed_runner.run()
    resumed_store.close()

    assert resumed_result.selected_structural_hash == ref_result.selected_structural_hash
    assert resumed_result.ledger_summary == ref_result.ledger_summary


def test_arm_state_is_checkpointed_after_initialization_and_every_evaluation(tmp_path, t1_data):
    budget = 3
    store = _store(tmp_path, "run_ckpt", budget=budget)
    runner = SearchRunner(
        ToyRandomArm(), "T1", t1_data, _training_config(),
        budget_limit=budget, run_seed=3, result_store=store, run_id="run_ckpt",
    )
    runner.run()
    raw_state = store.load_run_state("run_ckpt")
    assert raw_state is not None
    state = ToyRandomArm().deserialize_state(raw_state)
    assert state.n_proposed == budget  # one update_state call per proposal, all consumed budget
    store.close()
