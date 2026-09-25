"""Task abstraction: data acquisition, splitting, and preprocessing.

This module is deliberately data-only. It never constructs, compiles, or
evaluates a circuit — a `Task` produces `TrainValData` (raw features,
targets, and fitted preprocessing), and `llm_vqc.evaluation` is
responsible for turning that plus a `CircuitIR` into a trained model and
a metric. This keeps task-specific logic out of the IR/compiler layer.

**Test quarantine, by construction, not by convention.** `TrainValData`
has no `test` attribute — it is a distinct type from whatever holds test
data, so a function typed to accept `TrainValData` cannot receive test
data even by mistake; there is no field to reach for. The test partition
is obtainable only through a task's separate `build_test(seed)` method,
which every task in this package defines but which nothing in
`llm_vqc.evaluation.harness` (the search-facing evaluator) ever calls or
imports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


class TaskDataError(Exception):
    """Raised when a task's data (splits, preprocessing) is malformed."""


@dataclass(frozen=True)
class DataSplit:
    """One partition (train, val, or test) of a task's raw data."""

    features: np.ndarray
    targets: np.ndarray
    sample_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        n = len(self.features)
        if len(self.targets) != n:
            raise TaskDataError(
                f"features has {n} rows but targets has {len(self.targets)}"
            )
        if len(self.sample_ids) != n:
            raise TaskDataError(
                f"features has {n} rows but sample_ids has {len(self.sample_ids)}"
            )
        if len(set(self.sample_ids)) != n:
            raise TaskDataError("sample_ids contains duplicates within a single split")
        if n == 0:
            raise TaskDataError("a DataSplit must contain at least one sample")

    def __len__(self) -> int:
        return len(self.features)


def assert_disjoint_splits(*splits: DataSplit) -> None:
    """Raise TaskDataError if any sample_id appears in more than one split.

    Every task's `build`/`build_test` implementation must pass its splits
    through this before returning them — it is the one shared guard
    against the "same sample silently appears in multiple partitions"
    failure mode, rather than each task re-implementing its own check.
    """
    seen: dict[str, int] = {}
    for split_index, split in enumerate(splits):
        for sample_id in split.sample_ids:
            if sample_id in seen:
                raise TaskDataError(
                    f"sample_id {sample_id!r} appears in both split "
                    f"{seen[sample_id]} and split {split_index} — splits must be disjoint"
                )
            seen[sample_id] = split_index


@dataclass(frozen=True)
class FittedPreprocessing:
    """Preprocessing parameters fit on TRAIN data only.

    `transform` must be a pure function of `params` and the input array —
    no re-fitting, no data-dependent branching on val/test statistics.
    Constructed only by a task's `_fit_preprocessing(train_split)`
    (private, train-only) and then reused unchanged for val and test.
    """

    kind: str
    params: dict[str, Any]
    # Callable[[np.ndarray, dict[str, Any]], np.ndarray]; stored so callers
    # never need to re-derive fit parameters from `kind` alone.
    _transform_fn: Any

    def transform(self, raw_features: np.ndarray) -> np.ndarray:
        return self._transform_fn(raw_features, self.params)


@dataclass(frozen=True)
class TaskSpec:
    """Static description of a task's interface contract with the evaluator.

    Deliberately does NOT pin a qubit count or circuit-input count: IR
    circuits proposed against a task may vary their own qubit count and
    encoding-wire count (that variability is the whole point of a search
    space), so a task cannot assume a fixed `num_circuit_inputs`. Instead
    every candidate circuit gets its own small trainable linear "embed"
    layer, sized dynamically as `(raw_feature_dim -> circuit.num_inputs)`
    by `llm_vqc.evaluation.model` — see that module's docstring for why
    this generalizes the master plan's explicit T1 "linear embed" wrapper
    to also cover T2 without hard-coupling either task to one circuit
    shape.
    """

    name: str
    description: str
    raw_feature_dim: int
    classical_head_out_dim: int
    metric_name: str
    lower_is_better: bool
    loss_name: str


@dataclass(frozen=True)
class TrainValData:
    """Everything the search-facing evaluator is allowed to see. No test data."""

    spec: TaskSpec
    train: DataSplit
    val: DataSplit
    split_seed: int
    preprocessing: FittedPreprocessing


class Task(Protocol):
    """A task builds TrainValData (search-visible) and, separately, test data."""

    spec: TaskSpec

    def build(self, seed: int) -> TrainValData: ...

    def build_test(self, seed: int) -> DataSplit: ...
