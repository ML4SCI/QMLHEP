"""Fixed, reproducible task evaluation layer (Phase 2).

Every future search arm calls `harness.evaluate_candidate` and only that
function — see that module's docstring for the test-quarantine guarantee.
Final test scoring lives in `final_test`, deliberately never imported
here.
"""

from llm_vqc.evaluation.results import (
    CompilationOutcome,
    EvaluationResult,
    FailureCategory,
    FinalTestResult,
    TrainingOutcome,
    ValidationOutcome,
)
from llm_vqc.evaluation.training import TrainingConfig

__all__ = [
    "EvaluationResult",
    "FinalTestResult",
    "ValidationOutcome",
    "CompilationOutcome",
    "TrainingOutcome",
    "FailureCategory",
    "TrainingConfig",
]
