"""Metric correctness against independently computed reference values."""

from __future__ import annotations

import math

import numpy as np
import pytest

from llm_vqc.evaluation.metrics import MetricError, auc, compute_metric, rmse


def test_rmse_matches_hand_computed_value():
    predictions = np.array([1.0, 2.0, 3.0])
    targets = np.array([1.0, 2.0, 5.0])
    # errors: 0, 0, -2 -> squared: 0, 0, 4 -> mean = 4/3 -> sqrt
    expected = math.sqrt(4.0 / 3.0)
    assert rmse(predictions, targets) == pytest.approx(expected)


def test_rmse_is_zero_for_perfect_predictions():
    x = np.array([0.1, 0.5, 0.9])
    assert rmse(x, x) == pytest.approx(0.0, abs=1e-12)


def test_rmse_is_symmetric_in_direction_of_error():
    a = rmse(np.array([1.0]), np.array([2.0]))
    b = rmse(np.array([2.0]), np.array([1.0]))
    assert a == pytest.approx(b)


def test_auc_matches_hand_computed_value_for_a_simple_case():
    # Perfect separation: all class-0 scores below all class-1 scores -> AUC = 1
    predictions = np.array([0.1, 0.2, 0.8, 0.9])
    targets = np.array([0, 0, 1, 1])
    assert auc(predictions, targets) == pytest.approx(1.0)


def test_auc_is_half_for_random_scoring_tied_predictions():
    predictions = np.array([0.5, 0.5, 0.5, 0.5])
    targets = np.array([0, 1, 0, 1])
    assert auc(predictions, targets) == pytest.approx(0.5)


def test_auc_raises_metric_error_for_single_class_targets():
    with pytest.raises(MetricError):
        auc(np.array([0.1, 0.9]), np.array([1, 1]))


def test_compute_metric_dispatches_correctly():
    assert compute_metric("rmse", np.array([1.0]), np.array([1.0])) == pytest.approx(0.0)
    assert compute_metric("auc", np.array([0.1, 0.9]), np.array([0, 1])) == pytest.approx(1.0)


def test_compute_metric_rejects_unknown_metric_name():
    with pytest.raises(MetricError):
        compute_metric("not_a_real_metric", np.array([1.0]), np.array([1.0]))
