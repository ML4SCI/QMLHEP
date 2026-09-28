"""Evaluation result schemas.

Two schemas, intentionally **not** related by inheritance (a subclass
relationship would make it easy to accidentally widen one into the
other): `EvaluationResult` is what the search-facing evaluator
(`llm_vqc.evaluation.harness`) produces and is safe to hand to any search
algorithm or LLM agent — it can never contain a test metric, because the
class has no field for one. `FinalTestResult` is what the quarantined
final-test path (`llm_vqc.evaluation.final_test`) produces, and nothing
in this module or in `harness.py` constructs one.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field

from llm_vqc.ir.metrics import CircuitCostSummary
from llm_vqc.ir.validators import ValidationIssue


class ValidationOutcome(str, Enum):
    VALID = "valid"
    INVALID = "invalid"


class CompilationOutcome(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    NOT_ATTEMPTED = "not_attempted"


class TrainingOutcome(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    NOT_ATTEMPTED = "not_attempted"


class FailureCategory(str, Enum):
    NONE = "none"
    INVALID_PROPOSAL = "invalid_proposal"
    COMPILATION_ERROR = "compilation_error"
    TRAINING_DIVERGED = "training_diverged"
    TRAINING_ERROR = "training_error"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class EvaluationResult(BaseModel):
    """Search-visible result of evaluating one candidate proposal.

    Never carries a test metric — there is no field for one. Feed this
    object, and only this object, to any search algorithm or LLM agent.
    """

    # Proposal identity
    proposal_id: str
    task_name: str
    run_seed: int

    # Circuit identity (structural hash + full IR, for exact reproduction
    # and for dedup/caching — see llm_vqc.ir.canonicalize)
    structural_hash: str | None
    circuit_canonical_json: str | None

    # Outcomes, kept separate per stage
    validation_outcome: ValidationOutcome
    validation_issues: list[ValidationIssue] = Field(default_factory=list)
    compilation_outcome: CompilationOutcome
    compilation_error: str | None = None
    training_outcome: TrainingOutcome
    training_error: str | None = None

    # Search-visible metrics: VALIDATION ONLY.
    val_metric_name: str | None = None
    val_metric_value: float | None = None
    val_metric_history: list[float] = Field(default_factory=list)
    train_loss_history: list[float] = Field(default_factory=list)

    # Resource / budget usage
    epochs_completed: int = 0
    wall_clock_seconds: float = 0.0
    is_duplicate: bool = False
    cache_hit: bool = False

    # Circuit structural cost (depth, gate counts, param count — never a
    # performance metric)
    circuit_cost: CircuitCostSummary | None = None

    # Failure category (single top-level classification, for easy grouping)
    failure_category: FailureCategory = FailureCategory.NONE

    # Reproducibility metadata
    train_seed: int | None = None
    git_sha: str | None = None
    git_dirty: bool | None = None
    created_at: str = Field(default_factory=_now_iso)

    # Artifact references: pointers into the result store, not embedded
    # blobs — trained weights live in the store, keyed by
    # (structural_hash, train_seed); this field is that key, or None if
    # nothing was persisted (e.g. the proposal was invalid).
    trained_weights_ref: str | None = None


class FinalTestResult(BaseModel):
    """Result of evaluating an ALREADY-TRAINED circuit on the quarantined
    test partition. Constructed only by `llm_vqc.evaluation.final_test`.
    """

    task_name: str
    structural_hash: str
    train_seed: int
    test_metric_name: str
    test_metric_value: float
    n_test_samples: int
    created_at: str = Field(default_factory=_now_iso)
