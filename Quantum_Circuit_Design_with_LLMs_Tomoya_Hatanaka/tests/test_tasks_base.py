"""Tests for the shared task data abstractions (DataSplit, disjointness)."""

from __future__ import annotations

import numpy as np
import pytest

from llm_vqc.tasks.base import DataSplit, TaskDataError, assert_disjoint_splits


def _split(n: int, prefix: str) -> DataSplit:
    return DataSplit(
        features=np.random.default_rng(0).uniform(size=(n, 3)),
        targets=np.zeros(n),
        sample_ids=tuple(f"{prefix}-{i}" for i in range(n)),
    )


def test_datasplit_rejects_mismatched_features_and_targets_length():
    ids = tuple(str(i) for i in range(5))
    with pytest.raises(TaskDataError):
        DataSplit(features=np.zeros((5, 2)), targets=np.zeros(4), sample_ids=ids)


def test_datasplit_rejects_mismatched_sample_ids_length():
    ids = tuple(str(i) for i in range(4))
    with pytest.raises(TaskDataError):
        DataSplit(features=np.zeros((5, 2)), targets=np.zeros(5), sample_ids=ids)


def test_datasplit_rejects_duplicate_sample_ids_within_one_split():
    with pytest.raises(TaskDataError):
        DataSplit(
            features=np.zeros((3, 2)), targets=np.zeros(3), sample_ids=("a", "a", "b")
        )


def test_datasplit_rejects_empty_split():
    with pytest.raises(TaskDataError):
        DataSplit(features=np.zeros((0, 2)), targets=np.zeros(0), sample_ids=())


def test_assert_disjoint_splits_passes_for_genuinely_disjoint_splits():
    a = _split(3, "a")
    b = _split(3, "b")
    assert_disjoint_splits(a, b)  # must not raise


def test_assert_disjoint_splits_rejects_overlapping_splits():
    a = _split(3, "shared")
    b = _split(3, "shared")  # same id prefix -> same ids -> overlap
    with pytest.raises(TaskDataError):
        assert_disjoint_splits(a, b)


def test_assert_disjoint_splits_handles_more_than_two_splits():
    a, b, c = _split(2, "a"), _split(2, "b"), _split(2, "c")
    assert_disjoint_splits(a, b, c)
    with pytest.raises(TaskDataError):
        assert_disjoint_splits(a, b, a)
