"""Typed, serializable diagnostic result schema.

Like `llm_vqc.evaluation.results.EvaluationResult`, `DiagnosticResult`
carries NO task test metric and NO task test labels — a diagnostic is a
property of a circuit, computed with no task data at all (only a fixed
diagnostic input; see `llm_vqc.diagnostics.sampling`). This is enforced
structurally: there is no field on this class for a test metric, and the
diagnostics package never imports `llm_vqc.evaluation.final_test` (checked
by a test).

One `DiagnosticResult` holds the result of ONE diagnostic (expressibility,
entanglement, or gradient_variance) on ONE circuit under ONE config. The
`metrics` dict holds the diagnostic-specific numbers (documented per
diagnostic in the README); the surrounding fields are common provenance,
resource, and failure metadata shared by all three.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class DiagnosticFailureCategory(str, Enum):
    NONE = "none"
    INVALID_CIRCUIT = "invalid_circuit"
    NO_PARAMETERS = "no_parameters"  # not an error; a distinct, reportable outcome
    ALL_GRADIENTS_NON_FINITE = "all_gradients_non_finite"
    COMPUTATION_ERROR = "computation_error"


class DiagnosticResourceUsage(BaseModel):
    """What the diagnostic actually spent. Used for accounting and for
    later deciding which diagnostic operations consume scientific budget
    (see `llm_vqc/diagnostics/README.md` and the master plan)."""

    parameter_samples: int = 0
    states_generated: int = 0
    fidelity_evaluations: int = 0
    gradient_evaluations: int = 0
    backend_executions: int = 0
    failed_samples: int = 0
    completed_samples: int = 0
    wall_clock_seconds: float = 0.0
    cache_hit: bool = False


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class DiagnosticResult(BaseModel):
    """Search-visible result of one diagnostic on one circuit."""

    # Circuit identity
    structural_hash: str | None
    circuit_canonical_json: str | None

    # Diagnostic identity: name + version + the config it was computed
    # under + its config hash. Two results are the same measurement only
    # if ALL of these match (see store.py cache identity).
    diagnostic_name: str
    diagnostic_version: str
    config_hash: str
    config: dict[str, Any] = Field(default_factory=dict)

    # Sampling accounting
    samples_requested: int = 0
    samples_completed: int = 0

    # Seed + backend provenance
    run_seed: int | None = None
    base_seed: int | None = None
    backend: str | None = None
    diff_method: str | None = None

    # The actual metric(s) — diagnostic-specific keys, documented per
    # diagnostic. Never contains a task test metric.
    metrics: dict[str, float | None] = Field(default_factory=dict)
    # Dispersion / uncertainty measures, diagnostic-specific keys.
    uncertainty: dict[str, float | None] = Field(default_factory=dict)

    numerical_warnings: list[str] = Field(default_factory=list)
    failure_category: DiagnosticFailureCategory = DiagnosticFailureCategory.NONE
    error_message: str | None = None

    resource_usage: DiagnosticResourceUsage = Field(default_factory=DiagnosticResourceUsage)

    # Environment / git provenance
    git_sha: str | None = None
    git_dirty: bool | None = None
    created_at: str = Field(default_factory=_now_iso)
