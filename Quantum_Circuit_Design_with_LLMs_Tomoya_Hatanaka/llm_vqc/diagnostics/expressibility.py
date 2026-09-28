"""Expressibility diagnostic (Sim et al. 2019, arXiv:1905.10876, Eq. 17).

    Expr = D_KL( P_hat_PQC(F; theta) || P_Haar(F) )

where F = |<psi_theta | psi_phi>|^2 is the fidelity between two states
produced by two independently sampled trainable-weight vectors, P_hat_PQC
is the histogram of sampled fidelities, and P_Haar is the analytic Haar
fidelity distribution P_Haar(F) = (N-1)(1-F)^(N-2) with N = 2^n_qubits.

**Lower Expr = more expressible** (fidelity distribution closer to Haar).
The least-expressible case — a fixed circuit that always outputs the same
state — puts all fidelity mass at F=1 and attains the maximum
D_KL = (N-1) ln(n_bins) (Sim et al.); the idle / parameterless circuit is
exactly this case.

Conventions and numerical policy (also in `llm_vqc/diagnostics/README.md`):

- **Fidelity histogram:** `n_bins` equal-width bins over the fixed range
  [0, 1]. The bin count and range are FIXED across all circuits and all
  future search arms — never adapted per circuit (adaptive bins could
  make particular circuits look more expressive).
- **Haar reference:** computed analytically per bin, not sampled. For a
  bin [a, b], the exact Haar probability mass is the integral of the Haar
  PDF: (1-a)^(N-1) - (1-b)^(N-1). These are always strictly positive for
  a < b <= 1, so the KL reference never has a zero-probability bin.
- **KL zero-bin policy:** KL = sum_i P_i log(P_i / Q_i) with P = P_hat_PQC
  (empirical) and Q = P_Haar (analytic). Empty empirical bins (P_i = 0)
  contribute exactly 0 (the standard 0*log(0) = 0 convention); no epsilon
  smoothing is applied to the empirical distribution. This is well-defined
  because every Q_i > 0.
- **Robustness diagnostic:** the same fidelities are re-histogrammed at a
  second bin count (`robustness_n_bins`) and its KL is reported
  separately, clearly labeled as a bin-sensitivity robustness check — it
  never replaces the primary `n_bins` value.
- **Uncertainty:** repeated-seed dispersion. `n_replicates` fully
  independent seed-sets each produce one Expr estimate; the reported
  estimate is their mean and the reported uncertainty is their standard
  deviation (matching Sim et al.'s "error bars = std dev over five
  independent computations").
- **Known estimator bias:** KL / entropy estimators are biased at finite
  sample size, with the bias most pronounced at low sample counts (Sim
  et al. Appendix B). Small-sample smoke/test configs will therefore
  report inflated Expr relative to the true value; only research-scale
  counts (5000 pairs) should be interpreted quantitatively.
"""

from __future__ import annotations

import numpy as np

from llm_vqc.diagnostics.config import ExpressibilityConfig
from llm_vqc.diagnostics.sampling import (
    diagnostic_input,
    fidelity,
    sample_weight_vectors,
    statevector,
)
from llm_vqc.diagnostics.seeds import DiagnosticSeeds
from llm_vqc.ir.expand import CircuitProgram, build_program
from llm_vqc.ir.schema import CircuitIR


def haar_bin_probabilities(n_qubits: int, n_bins: int) -> np.ndarray:
    """Exact Haar fidelity probability mass per equal-width [0,1] bin.

    For bin [a, b], mass = integral_a^b (N-1)(1-F)^(N-2) dF
                         = (1-a)^(N-1) - (1-b)^(N-1),   N = 2^n_qubits.
    Always strictly positive, and sums to 1.
    """
    dim = 2**n_qubits
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    probs = (1.0 - edges[:-1]) ** (dim - 1) - (1.0 - edges[1:]) ** (dim - 1)
    return probs


def kl_divergence_to_haar(fidelities: np.ndarray, n_qubits: int, n_bins: int) -> float:
    """D_KL( P_hat_PQC(F) || P_Haar(F) ) at `n_bins` bins over [0, 1]."""
    counts, _ = np.histogram(fidelities, bins=n_bins, range=(0.0, 1.0))
    empirical = counts / counts.sum()
    haar = haar_bin_probabilities(n_qubits, n_bins)
    # Empty empirical bins (empirical == 0) contribute 0 under 0*log(0)=0;
    # haar is strictly positive everywhere, so no division-by-zero.
    mask = empirical > 0
    return float(np.sum(empirical[mask] * np.log(empirical[mask] / haar[mask])))


def _fidelities_for_replicate(
    ir: CircuitIR,
    program: CircuitProgram,
    config: ExpressibilityConfig,
    seeds: DiagnosticSeeds,
) -> np.ndarray:
    """Sample `n_pairs` fidelities for one replicate seed-set."""
    inputs = diagnostic_input(program, seeds.diagnostic_input)
    # 2 * n_pairs weight vectors -> n_pairs disjoint pairs.
    vectors = sample_weight_vectors(
        program, 2 * config.n_pairs, config.param_low, config.param_high, seeds.parameter_sampling
    )
    fidelities = np.empty(config.n_pairs, dtype=np.float64)
    for pair_index in range(config.n_pairs):
        state_a = statevector(ir, inputs, vectors[2 * pair_index], program)
        state_b = statevector(ir, inputs, vectors[2 * pair_index + 1], program)
        fidelities[pair_index] = fidelity(state_a, state_b)
    return fidelities


class ExpressibilityOutcome:
    """Plain container for the raw expressibility computation (pre-schema)."""

    def __init__(
        self,
        expr_mean: float,
        expr_std: float,
        per_replicate: list[float],
        robustness_expr_mean: float,
        robustness_expr_std: float,
        n_pairs_completed: int,
        mean_fidelity: float,
        upper_bound: float,
    ) -> None:
        self.expr_mean = expr_mean
        self.expr_std = expr_std
        self.per_replicate = per_replicate
        self.robustness_expr_mean = robustness_expr_mean
        self.robustness_expr_std = robustness_expr_std
        self.n_pairs_completed = n_pairs_completed
        self.mean_fidelity = mean_fidelity
        self.upper_bound = upper_bound


def compute_expressibility(
    ir: CircuitIR,
    config: ExpressibilityConfig,
    replicate_seed_sets: list[DiagnosticSeeds],
    program: CircuitProgram | None = None,
) -> ExpressibilityOutcome:
    """Compute expressibility (mean +/- std over independent replicates)."""
    program = program if program is not None else build_program(ir)
    dim = 2**ir.n_qubits

    primary_per_replicate: list[float] = []
    robustness_per_replicate: list[float] = []
    all_fidelities: list[np.ndarray] = []

    for seeds in replicate_seed_sets:
        fidelities = _fidelities_for_replicate(ir, program, config, seeds)
        all_fidelities.append(fidelities)
        primary_per_replicate.append(
            kl_divergence_to_haar(fidelities, ir.n_qubits, config.n_bins)
        )
        robustness_per_replicate.append(
            kl_divergence_to_haar(fidelities, ir.n_qubits, config.robustness_n_bins)
        )

    concatenated = np.concatenate(all_fidelities)
    upper_bound = (dim - 1) * np.log(config.n_bins)

    return ExpressibilityOutcome(
        expr_mean=float(np.mean(primary_per_replicate)),
        expr_std=float(np.std(primary_per_replicate)),
        per_replicate=primary_per_replicate,
        robustness_expr_mean=float(np.mean(robustness_per_replicate)),
        robustness_expr_std=float(np.std(robustness_per_replicate)),
        n_pairs_completed=int(concatenated.size),
        mean_fidelity=float(np.mean(concatenated)),
        upper_bound=float(upper_bound),
    )
