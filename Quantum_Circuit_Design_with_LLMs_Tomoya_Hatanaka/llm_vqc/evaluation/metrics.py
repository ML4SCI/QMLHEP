"""Task metric computation, kept separate and independently testable.

Pure functions of (predictions, targets) — no model, no training state,
no task object — so "metric correctness against independently computed
values" (a required Phase 2 test) can check these directly against a
hand-computed reference without standing up a whole training run.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

_CLIP = 1e-7


class MetricError(Exception):
    """Raised when a metric cannot be computed (e.g. a degenerate batch)."""


def rmse(predictions: np.ndarray, targets: np.ndarray) -> float:
    predictions = np.asarray(predictions, dtype=np.float64).reshape(-1)
    targets = np.asarray(targets, dtype=np.float64).reshape(-1)
    return float(np.sqrt(np.mean((predictions - targets) ** 2)))


def auc(predictions: np.ndarray, targets: np.ndarray) -> float:
    predictions = np.asarray(predictions, dtype=np.float64).reshape(-1)
    targets = np.asarray(targets, dtype=np.float64).reshape(-1)
    if len(set(targets.tolist())) < 2:
        raise MetricError("AUC is undefined when targets contain a single class")
    return float(roc_auc_score(targets, predictions))


def logloss(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Binary cross-entropy (lower is better). Discriminative even when a task is
    linearly separable and AUC saturates at 1.0 — the T2-v1 selection metric."""
    p = np.clip(np.asarray(predictions, dtype=np.float64).reshape(-1), _CLIP, 1 - _CLIP)
    y = np.asarray(targets, dtype=np.float64).reshape(-1)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def balanced_accuracy_error(predictions: np.ndarray, targets: np.ndarray) -> float:
    """1 - balanced accuracy at threshold 0.5 (lower is better)."""
    p = np.asarray(predictions, dtype=np.float64).reshape(-1)
    y = np.asarray(targets, dtype=np.float64).reshape(-1)
    return float(1.0 - balanced_accuracy_score(y, (p >= 0.5).astype(np.float64)))


def classification_error(predictions: np.ndarray, targets: np.ndarray) -> float:
    """1 - accuracy at threshold 0.5 (lower is better)."""
    p = np.asarray(predictions, dtype=np.float64).reshape(-1)
    y = np.asarray(targets, dtype=np.float64).reshape(-1)
    return float(np.mean((p >= 0.5).astype(np.float64) != y))


def brier(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Mean squared error of probabilities (lower is better)."""
    p = np.asarray(predictions, dtype=np.float64).reshape(-1)
    y = np.asarray(targets, dtype=np.float64).reshape(-1)
    return float(np.mean((p - y) ** 2))


_METRIC_FUNCTIONS = {
    "rmse": rmse, "auc": auc, "logloss": logloss,
    "balanced_accuracy_error": balanced_accuracy_error,
    "classification_error": classification_error, "brier": brier,
}


def compute_metric(metric_name: str, predictions: np.ndarray, targets: np.ndarray) -> float:
    if metric_name not in _METRIC_FUNCTIONS:
        raise MetricError(f"unknown metric: {metric_name!r}")
    return _METRIC_FUNCTIONS[metric_name](predictions, targets)
