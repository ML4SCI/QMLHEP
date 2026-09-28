"""Tests for the search-facing evaluation harness, including the test-set
quarantine guarantee and budget accounting.
"""

from __future__ import annotations

import ast
import subprocess
import sys

import pytest

from llm_vqc.evaluation.harness import evaluate_candidate
from llm_vqc.evaluation.results import (
    CompilationOutcome,
    FailureCategory,
    TrainingOutcome,
    ValidationOutcome,
)
from llm_vqc.evaluation.training import TrainingConfig
from llm_vqc.ir.budget import BudgetLedger
from llm_vqc.tasks.t1_gaussian import T1GaussianPeakTask

VALID_PROPOSAL = {
    "n_qubits": 4,
    "encoding": {"type": "angle", "gate": "RY", "wires": "all"},
    "layers": [
        {"type": "rot", "gates": ["RY", "RZ"], "wires": "all"},
        {"type": "entangle", "pattern": "ring", "gate": "CNOT"},
    ],
    "measurements": {"observable": "Z", "wires": "all"},
}

INVALID_PROPOSAL = {
    "n_qubits": 3,
    "encoding": {"type": "angle"},  # missing required 'gate'
    "layers": [],
    "measurements": {"observable": "Z"},
}


@pytest.fixture(scope="module")
def t1_data():
    task = T1GaussianPeakTask()
    return task.build(seed=0)


# --- Structural test quarantine (the core guarantee) ------------------


def test_harness_module_has_no_import_referencing_final_test():
    with open("llm_vqc/evaluation/harness.py") as f:
        tree = ast.parse(f.read())
    import_targets = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            import_targets.append(node.module or "")
        elif isinstance(node, ast.Import):
            import_targets.extend(alias.name for alias in node.names)
    assert not any("final_test" in t for t in import_targets), import_targets


def test_importing_harness_does_not_transitively_import_final_test():
    code = (
        "import llm_vqc.evaluation.harness; import sys; "
        "print('llm_vqc.evaluation.final_test' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.stdout.strip() == "False", out.stdout + out.stderr


def test_evaluate_candidate_signature_has_no_test_data_parameter():
    import inspect

    from llm_vqc.evaluation.harness import evaluate_candidate as fn

    params = set(inspect.signature(fn).parameters.keys())
    assert not any("test" in p.lower() for p in params), params


# --- Functional behavior ------------------------------------------------


def test_valid_proposal_trains_successfully(t1_data):
    ledger = BudgetLedger()
    result = evaluate_candidate(
        VALID_PROPOSAL,
        task_name="T1",
        run_seed=0,
        train_val=t1_data,
        training_config=TrainingConfig(epochs=3),
        proposal_id="p1",
        ledger=ledger,
    )
    assert result.validation_outcome == ValidationOutcome.VALID
    assert result.compilation_outcome == CompilationOutcome.SUCCESS
    assert result.training_outcome == TrainingOutcome.SUCCESS
    assert result.val_metric_value is not None
    assert result.failure_category == FailureCategory.NONE


def test_invalid_proposal_never_reaches_compilation_or_training(t1_data):
    ledger = BudgetLedger()
    result = evaluate_candidate(
        INVALID_PROPOSAL,
        task_name="T1",
        run_seed=0,
        train_val=t1_data,
        training_config=TrainingConfig(epochs=3),
        proposal_id="p2",
        ledger=ledger,
    )
    assert result.validation_outcome == ValidationOutcome.INVALID
    assert result.compilation_outcome == CompilationOutcome.NOT_ATTEMPTED
    assert result.training_outcome == TrainingOutcome.NOT_ATTEMPTED
    assert result.failure_category == FailureCategory.INVALID_PROPOSAL
    assert len(result.validation_issues) > 0


def test_invalid_proposal_is_recorded_in_the_budget_ledger(t1_data):
    """An invalid proposal must not disappear from the budget record."""
    ledger = BudgetLedger()
    evaluate_candidate(
        INVALID_PROPOSAL,
        task_name="T1",
        run_seed=0,
        train_val=t1_data,
        training_config=TrainingConfig(epochs=3),
        proposal_id="p3",
        ledger=ledger,
    )
    assert ledger.num_proposed == 1
    assert ledger.num_invalid == 1
    assert ledger.num_valid == 0


def test_duplicate_detection_returns_cached_result_without_retraining(t1_data):
    class DictCache:
        def __init__(self):
            self.store = {}

        def get_cached(self, task_name, structural_hash, train_seed):
            return self.store.get((task_name, structural_hash, train_seed))

        def put_cached(self, task_name, structural_hash, train_seed, result, trained_weights):
            self.store[(task_name, structural_hash, train_seed)] = result

    cache = DictCache()
    ledger = BudgetLedger()
    config = TrainingConfig(epochs=3)

    r1 = evaluate_candidate(
        VALID_PROPOSAL, "T1", 0, t1_data, config, "p1", ledger, cache=cache
    )
    r2 = evaluate_candidate(
        VALID_PROPOSAL, "T1", 0, t1_data, config, "p2", ledger, cache=cache
    )

    assert r1.cache_hit is False
    assert r2.cache_hit is True
    assert r2.is_duplicate is True
    assert r2.val_metric_value == r1.val_metric_value
    assert r2.proposal_id == "p2"  # proposal identity updates even on cache hit
    assert ledger.num_duplicate == 1
    assert ledger.num_unique == 1


def test_invalid_ir_object_passed_directly_is_handled(t1_data):
    """validate_proposal also accepts an already-parsed CircuitIR; the
    harness must handle that path identically."""
    from llm_vqc.ir.schema import CircuitIR, EncodingSpec, MeasurementSpec

    # A minimal, structurally VALID but empty-layers IR (should train fine)
    ir = CircuitIR(
        n_qubits=3,
        encoding=EncodingSpec(type="angle", gate="RY", wires="all"),
        layers=[],
        measurements=MeasurementSpec(observable="Z", wires="all"),
    )
    ledger = BudgetLedger()
    result = evaluate_candidate(
        ir, "T1", 0, t1_data, TrainingConfig(epochs=2), "p1", ledger
    )
    assert result.validation_outcome == ValidationOutcome.VALID


def test_compilation_failure_is_reported_not_raised(t1_data, monkeypatch):
    """Force a compilation-stage exception (via monkeypatch, since a
    structurally valid IR from Phase 1's own validators essentially never
    fails to compile) and confirm the harness reports it as a structured
    failure rather than propagating the exception."""
    import llm_vqc.evaluation.harness as harness_module
    from llm_vqc.ir.compiler_pennylane import CompilerError

    def _boom(*args, **kwargs):
        raise CompilerError("synthetic compilation failure for testing")

    monkeypatch.setattr(harness_module, "circuit_cost_summary", _boom)

    ledger = BudgetLedger()
    result = evaluate_candidate(
        VALID_PROPOSAL, "T1", 0, t1_data, TrainingConfig(epochs=2), "p1", ledger
    )
    assert result.compilation_outcome == CompilationOutcome.FAILED
    assert result.failure_category == FailureCategory.COMPILATION_ERROR
    assert "synthetic compilation failure" in result.compilation_error
    assert result.training_outcome == TrainingOutcome.NOT_ATTEMPTED


def test_training_failure_is_reported_with_correct_failure_category(t1_data):
    ledger = BudgetLedger()
    # Absurd learning rate forces divergence deterministically.
    result = evaluate_candidate(
        VALID_PROPOSAL,
        "T1",
        0,
        t1_data,
        TrainingConfig(epochs=5, learning_rate=1e8),
        "p1",
        ledger,
    )
    assert result.compilation_outcome == CompilationOutcome.SUCCESS
    assert result.training_outcome == TrainingOutcome.FAILED
    assert result.failure_category == FailureCategory.TRAINING_DIVERGED
    assert result.val_metric_value is None
