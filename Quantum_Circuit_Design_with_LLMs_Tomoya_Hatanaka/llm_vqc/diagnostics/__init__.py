"""Circuit diagnostics (Phase 3): expressibility, Meyer-Wallach entangling
capability, and gradient variance.

Scientific measurements of a `CircuitIR`, computed identically for every
future LAQS-Bench search arm and with no access to task test data. See
`llm_vqc/diagnostics/README.md` for definitions, conventions, estimators,
and interpretation limits, and `compute_diagnostic` for the single entry
point.
"""

from llm_vqc.diagnostics.config import (
    DiagnosticConfig,
    EntanglementConfig,
    ExpressibilityConfig,
    GradientVarianceConfig,
)
from llm_vqc.diagnostics.harness import compute_diagnostic
from llm_vqc.diagnostics.results import (
    DiagnosticFailureCategory,
    DiagnosticResourceUsage,
    DiagnosticResult,
)

__all__ = [
    "DiagnosticConfig",
    "ExpressibilityConfig",
    "EntanglementConfig",
    "GradientVarianceConfig",
    "compute_diagnostic",
    "DiagnosticResult",
    "DiagnosticResourceUsage",
    "DiagnosticFailureCategory",
]
