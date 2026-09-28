"""Tests for the SQLite result store: caching, resume, and failure
preservation."""

from __future__ import annotations

import pytest

from llm_vqc.evaluation.results import (
    CompilationOutcome,
    EvaluationResult,
    FailureCategory,
    TrainingOutcome,
    ValidationOutcome,
)
from llm_vqc.evaluation.store import IncompatibleResumeError, ResultStore


def _result(
    proposal_id="p1",
    val_metric=0.05,
    outcome=TrainingOutcome.SUCCESS,
    failure=FailureCategory.NONE,
):
    return EvaluationResult(
        proposal_id=proposal_id,
        task_name="T1",
        run_seed=0,
        structural_hash="h1",
        circuit_canonical_json="{}",
        validation_outcome=ValidationOutcome.VALID,
        compilation_outcome=CompilationOutcome.SUCCESS,
        training_outcome=outcome,
        val_metric_name="rmse",
        val_metric_value=val_metric if outcome == TrainingOutcome.SUCCESS else None,
        failure_category=failure,
    )


REPRO_FIELDS = {"task": "T1", "epochs": 5, "lr": 0.05}


def test_put_then_get_round_trips(tmp_path):
    db = tmp_path / "results.sqlite"
    store = ResultStore.open_or_create(
        db, run_id="r1", config_json="{}", config_reproducibility_fields=REPRO_FIELDS,
        git_sha=None, created_at="t0",
    )
    result = _result()
    store.put_cached("T1", "h1", 42, result, trained_weights={"a": [1, 2, 3]})

    cached = store.get_cached("T1", "h1", 42)
    assert cached is not None
    assert cached.val_metric_value == 0.05

    weights = store.get_trained_weights("T1", "h1", 42)
    assert weights == {"a": [1, 2, 3]}
    store.close()


def test_cache_miss_for_different_train_seed(tmp_path):
    db = tmp_path / "results.sqlite"
    store = ResultStore.open_or_create(
        db, run_id="r1", config_json="{}", config_reproducibility_fields=REPRO_FIELDS,
        git_sha=None, created_at="t0",
    )
    store.put_cached("T1", "h1", 42, _result(), None)
    assert store.get_cached("T1", "h1", 999) is None
    assert store.get_cached("T2", "h1", 42) is None
    store.close()


def test_resume_with_identical_config_preserves_prior_results(tmp_path):
    db = tmp_path / "results.sqlite"
    store1 = ResultStore.open_or_create(
        db, run_id="r1", config_json="{}", config_reproducibility_fields=REPRO_FIELDS,
        git_sha=None, created_at="t0",
    )
    store1.put_cached("T1", "h1", 42, _result(), None)
    store1.close()

    store2 = ResultStore.open_or_create(
        db, run_id="r1", config_json="{}", config_reproducibility_fields=REPRO_FIELDS,
        git_sha=None, created_at="t1",
    )
    assert store2.get_cached("T1", "h1", 42) is not None
    assert store2.count_evaluations() == 1
    store2.close()


def test_resume_with_incompatible_config_is_rejected(tmp_path):
    db = tmp_path / "results.sqlite"
    store1 = ResultStore.open_or_create(
        db, run_id="r1", config_json="{}", config_reproducibility_fields=REPRO_FIELDS,
        git_sha=None, created_at="t0",
    )
    store1.close()

    incompatible_fields = dict(REPRO_FIELDS, lr=0.5)
    with pytest.raises(IncompatibleResumeError):
        ResultStore.open_or_create(
            db, run_id="r1", config_json="{}", config_reproducibility_fields=incompatible_fields,
            git_sha=None, created_at="t1",
        )


def test_resume_incompatible_config_does_not_mutate_the_store(tmp_path):
    """A rejected resume attempt must not silently mix in the new
    (incompatible) config -- reopening with the ORIGINAL config afterward
    must still succeed."""
    db = tmp_path / "results.sqlite"
    store1 = ResultStore.open_or_create(
        db, run_id="r1", config_json="{}", config_reproducibility_fields=REPRO_FIELDS,
        git_sha=None, created_at="t0",
    )
    store1.put_cached("T1", "h1", 42, _result(), None)
    store1.close()

    incompatible_fields = dict(REPRO_FIELDS, lr=0.9)
    with pytest.raises(IncompatibleResumeError):
        ResultStore.open_or_create(
            db, run_id="r1", config_json="{}", config_reproducibility_fields=incompatible_fields,
            git_sha=None, created_at="t1",
        )

    # original config still works and prior data survived
    store3 = ResultStore.open_or_create(
        db, run_id="r1", config_json="{}", config_reproducibility_fields=REPRO_FIELDS,
        git_sha=None, created_at="t2",
    )
    assert store3.get_cached("T1", "h1", 42) is not None
    store3.close()


def test_allow_incompatible_explicitly_overrides(tmp_path):
    db = tmp_path / "results.sqlite"
    store1 = ResultStore.open_or_create(
        db, run_id="r1", config_json="{}", config_reproducibility_fields=REPRO_FIELDS,
        git_sha=None, created_at="t0",
    )
    store1.close()

    incompatible_fields = dict(REPRO_FIELDS, lr=0.9)
    store2 = ResultStore.open_or_create(
        db, run_id="r1", config_json="{}", config_reproducibility_fields=incompatible_fields,
        git_sha=None, created_at="t1", allow_incompatible=True,
    )
    store2.close()  # must not raise


def test_different_run_ids_in_the_same_db_are_independent(tmp_path):
    db = tmp_path / "results.sqlite"
    store = ResultStore.open_or_create(
        db, run_id="run_a", config_json="{}", config_reproducibility_fields=REPRO_FIELDS,
        git_sha=None, created_at="t0",
    )
    store.close()
    # A different run_id with a totally different config must succeed
    # even in the same physical database file.
    store2 = ResultStore.open_or_create(
        db,
        run_id="run_b",
        config_json="{}",
        config_reproducibility_fields={"totally": "different"},
        git_sha=None,
        created_at="t1",
    )
    store2.close()


def test_failed_training_runs_are_persisted_not_dropped(tmp_path):
    """A failed evaluation must be recorded like any other, not silently
    skipped."""
    db = tmp_path / "results.sqlite"
    store = ResultStore.open_or_create(
        db, run_id="r1", config_json="{}", config_reproducibility_fields=REPRO_FIELDS,
        git_sha=None, created_at="t0",
    )
    failed = _result(outcome=TrainingOutcome.FAILED, failure=FailureCategory.TRAINING_DIVERGED)
    store.put_cached("T1", "h_failed", 1, failed, trained_weights=None)

    cached = store.get_cached("T1", "h_failed", 1)
    assert cached is not None
    assert cached.training_outcome == TrainingOutcome.FAILED
    assert cached.failure_category == FailureCategory.TRAINING_DIVERGED
    store.close()


def test_partial_run_results_survive_an_abrupt_close_and_reopen(tmp_path):
    """Simulates a crash: write one row, close abruptly (no graceful
    shutdown call beyond close()), reopen a brand-new ResultStore
    instance directly against the same file, and confirm the row
    persisted (atomic commit-per-write, per put_cached's design)."""
    db = tmp_path / "results.sqlite"
    store1 = ResultStore.open_or_create(
        db, run_id="r1", config_json="{}", config_reproducibility_fields=REPRO_FIELDS,
        git_sha=None, created_at="t0",
    )
    store1.put_cached("T1", "h1", 1, _result(proposal_id="p1"), None)
    store1.close()

    # Reopen as a raw ResultStore (not via open_or_create) to simulate a
    # process that crashed before it could call open_or_create's
    # bookkeeping again.
    store2 = ResultStore(db)
    assert store2.count_evaluations() == 1
    assert store2.get_cached("T1", "h1", 1) is not None
    store2.close()


def test_all_evaluations_filters_by_task_name(tmp_path):
    db = tmp_path / "results.sqlite"
    store = ResultStore.open_or_create(
        db, run_id="r1", config_json="{}", config_reproducibility_fields=REPRO_FIELDS,
        git_sha=None, created_at="t0",
    )
    store.put_cached("T1", "h1", 1, _result(), None)

    t2_result = _result()
    t2_result = t2_result.model_copy(update={"task_name": "T2"})
    store.put_cached("T2", "h2", 1, t2_result, None)

    assert len(store.all_evaluations()) == 2
    assert len(store.all_evaluations(task_name="T1")) == 1
    assert len(store.all_evaluations(task_name="T2")) == 1
    store.close()


def test_result_store_context_manager_closes_cleanly(tmp_path):
    db = tmp_path / "results.sqlite"
    with ResultStore.open_or_create(
        db, run_id="r1", config_json="{}", config_reproducibility_fields=REPRO_FIELDS,
        git_sha=None, created_at="t0",
    ) as store:
        store.put_cached("T1", "h1", 1, _result(), None)
        assert store.count_evaluations() == 1
    # closed; reopening should still find the data
    store2 = ResultStore(db)
    assert store2.count_evaluations() == 1
    store2.close()
