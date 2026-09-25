"""Diagnostic configuration.

Every diagnostic in LAQS-Bench is a *scientific measurement* whose exact
protocol must be identical across every future search arm. That protocol
is captured here, in one frozen, hashable config object per diagnostic —
so two runs are comparable if and only if their diagnostic configs (and
diagnostic *versions*) match, and the store can refuse to silently mix
results computed under different protocols (see `llm_vqc.diagnostics.
store`).

The defaults reproduce Sim et al. 2019 (arXiv:1905.10876), the master
plan's cited reference for expressibility and entangling capability:
5,000 samples, 75 histogram bins. Unit-test and smoke configs use much
smaller counts (see `configs/` and the test suite); those are explicitly
*not* research-scale and must never have their outputs reported as
scientific findings.

**Sampling policy (shared by all three diagnostics), documented once here
and in `llm_vqc/diagnostics/README.md`:** every diagnostic samples the
circuit's *trainable weights* uniformly from `[0, 2*pi)` while holding the
data-encoding inputs at a FIXED diagnostic value (zeros for angle
encoding — which makes the encoding gates the identity, recovering
exactly Sim et al.'s pure-ansatz picture `U_variational(theta)|0...0>`;
a fixed seeded normalized vector for amplitude encoding). This treats the
circuit as a variational ansatz over its trainable parameters, which is
(a) faithful to Sim et al. (their sampled parameters are the trainable
rotation angles; their circuits have no separate data encoding), (b)
uniform across angle- and amplitude-encoded circuits, and (c) consistent
with the gradient-variance diagnostic, which must differentiate with
respect to the trainable weights. The alternative (also sampling the
encoding-input angles) is recorded in `DECISIONS.md` with the reason it
was not chosen.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# Bump this whenever the *meaning* of a diagnostic's output changes (a new
# estimator, a changed convention). Cache identity includes it, so a
# version bump correctly invalidates previously-stored results rather than
# silently serving stale numbers computed under the old definition.
EXPRESSIBILITY_VERSION = "1.0.0"
ENTANGLEMENT_VERSION = "1.0.0"
GRADIENT_VARIANCE_VERSION = "1.0.0"

TWO_PI = 6.283185307179586


class ExpressibilityConfig(BaseModel):
    """Sim et al. 2019 Eq. 17: Expr = D_KL( P_hat_PQC(F) || P_Haar(F) )."""

    model_config = {"frozen": True}

    diagnostic_name: Literal["expressibility"] = "expressibility"
    n_pairs: int = Field(default=5000, ge=2)
    n_bins: int = Field(default=75, ge=2)
    # A second bin count, evaluated on the SAME fidelities, reported only
    # as a robustness diagnostic (bin-count sensitivity) — never
    # substituted for the primary n_bins metric.
    robustness_n_bins: int = Field(default=100, ge=2)
    n_replicates: int = Field(default=5, ge=1)
    param_low: float = 0.0
    param_high: float = TWO_PI


class EntanglementConfig(BaseModel):
    """Sim et al. 2019 Eq. 21: Ent = mean over sampled theta of Q(|psi_theta>).

    Q is the Meyer-Wallach measure in the Brennen single-qubit-purity form
    (Eq. 19 <=> average single-qubit linear entropy).
    """

    model_config = {"frozen": True}

    diagnostic_name: Literal["entanglement"] = "entanglement"
    n_samples: int = Field(default=5000, ge=1)
    n_replicates: int = Field(default=5, ge=1)
    param_low: float = 0.0
    param_high: float = TWO_PI


class GradientVarianceConfig(BaseModel):
    """Barren-plateau-style trainability diagnostic.

    Variance, across random weight initializations, of the gradient of a
    FIXED cost observable (Pauli-Z on qubit 0 — the McClean et al. 2018
    convention) with respect to the trainable weights, at a fixed
    diagnostic input.
    """

    model_config = {"frozen": True}

    diagnostic_name: Literal["gradient_variance"] = "gradient_variance"
    n_inits: int = Field(default=200, ge=2)
    cost_observable: Literal["Z0"] = "Z0"
    param_low: float = 0.0
    param_high: float = TWO_PI
    n_bootstrap: int = Field(default=1000, ge=0)


class DiagnosticConfig(BaseModel):
    """Top-level config bundling all three diagnostics + shared backend settings."""

    model_config = {"frozen": True}

    backend: Literal["default.qubit"] = "default.qubit"
    diff_method: Literal["backprop"] = "backprop"
    dtype: Literal["float64"] = "float64"
    expressibility: ExpressibilityConfig = Field(default_factory=ExpressibilityConfig)
    entanglement: EntanglementConfig = Field(default_factory=EntanglementConfig)
    gradient_variance: GradientVarianceConfig = Field(default_factory=GradientVarianceConfig)
