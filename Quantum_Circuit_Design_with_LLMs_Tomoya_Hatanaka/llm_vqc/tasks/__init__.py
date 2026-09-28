"""Task data layer: acquisition, splitting, preprocessing. See base.py."""

from llm_vqc.tasks.base import (
    DataSplit,
    FittedPreprocessing,
    Task,
    TaskDataError,
    TaskSpec,
    TrainValData,
    assert_disjoint_splits,
)
from llm_vqc.tasks.t1_gaussian import T1GaussianPeakTask
from llm_vqc.tasks.t2_digits import T2DigitsTask

TASK_REGISTRY: dict[str, type] = {
    "T1": T1GaussianPeakTask,
    "T2": T2DigitsTask,
}

__all__ = [
    "DataSplit",
    "FittedPreprocessing",
    "Task",
    "TaskDataError",
    "TaskSpec",
    "TrainValData",
    "assert_disjoint_splits",
    "T1GaussianPeakTask",
    "T2DigitsTask",
    "TASK_REGISTRY",
]
