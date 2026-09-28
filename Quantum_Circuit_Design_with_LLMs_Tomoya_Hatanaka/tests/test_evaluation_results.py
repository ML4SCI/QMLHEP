"""Tests for the evaluation-result schemas: serialization stability and
the structural guarantee that EvaluationResult never carries a test metric.
"""

from __future__ import annotations

import json

from llm_vqc.evaluation.results import (
    CompilationOutcome,
    EvaluationResult,
    FailureCategory,
    FinalTestResult,
    TrainingOutcome,
    ValidationOutcome,
)


def _sample_result() -> EvaluationResult:
    return EvaluationResult(
        proposal_id="p1",
        task_name="T1",
        run_seed=0,
        structural_hash="abc123",
        circuit_canonical_json="{}",
        validation_outcome=ValidationOutcome.VALID,
        compilation_outcome=CompilationOutcome.SUCCESS,
        training_outcome=TrainingOutcome.SUCCESS,
        val_metric_name="rmse",
        val_metric_value=0.05,
        val_metric_history=[0.1, 0.07, 0.05],
        train_loss_history=[0.2, 0.1, 0.05],
        epochs_completed=3,
        wall_clock_seconds=1.5,
        failure_category=FailureCategory.NONE,
        train_seed=999,
    )


def test_no_field_on_evaluation_result_mentions_test():
    """The structural guarantee: EvaluationResult has no field that could
    carry a test-set metric, by name."""
    field_names = set(EvaluationResult.model_fields.keys())
    assert not any("test" in name.lower() for name in field_names), field_names


def test_evaluation_result_round_trips_through_json():
    result = _sample_result()
    text = result.model_dump_json()
    restored = EvaluationResult.model_validate_json(text)
    assert restored == result


def test_evaluation_result_serialization_is_stable_across_calls():
    result = _sample_result()
    text1 = result.model_dump_json()
    text2 = result.model_dump_json()
    assert text1 == text2


def test_evaluation_result_is_valid_json():
    result = _sample_result()
    data = json.loads(result.model_dump_json())
    assert data["task_name"] == "T1"
    assert data["val_metric_value"] == 0.05


def test_invalid_proposal_result_has_no_metric_and_correct_failure_category():
    result = EvaluationResult(
        proposal_id="p2",
        task_name="T1",
        run_seed=0,
        structural_hash=None,
        circuit_canonical_json=None,
        validation_outcome=ValidationOutcome.INVALID,
        compilation_outcome=CompilationOutcome.NOT_ATTEMPTED,
        training_outcome=TrainingOutcome.NOT_ATTEMPTED,
        failure_category=FailureCategory.INVALID_PROPOSAL,
    )
    assert result.val_metric_value is None
    assert result.failure_category == FailureCategory.INVALID_PROPOSAL


def test_final_test_result_is_a_distinct_unrelated_class():
    """FinalTestResult must not be a subclass of EvaluationResult (a
    subclass relationship could make it easy to accidentally widen one
    schema into the other)."""
    assert not issubclass(FinalTestResult, EvaluationResult)
    assert not issubclass(EvaluationResult, FinalTestResult)


def test_final_test_result_round_trips():
    result = FinalTestResult(
        task_name="T1",
        structural_hash="abc123",
        train_seed=999,
        test_metric_name="rmse",
        test_metric_value=0.04,
        n_test_samples=2000,
    )
    restored = FinalTestResult.model_validate_json(result.model_dump_json())
    assert restored == result
