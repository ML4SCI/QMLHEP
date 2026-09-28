"""Expressibility diagnostic tests: analytic Haar reference, KL policy,
the idle-is-worst and HEA-depth-scan acceptance criteria, bin robustness."""

from __future__ import annotations

import numpy as np
import pytest

from llm_vqc.diagnostics.config import ExpressibilityConfig
from llm_vqc.diagnostics.expressibility import (
    compute_expressibility,
    haar_bin_probabilities,
    kl_divergence_to_haar,
)
from llm_vqc.diagnostics.seeds import diagnostic_base_seed, replicate_seeds
from llm_vqc.ir.canonicalize import structural_hash
from llm_vqc.ir.schema import (
    CircuitIR,
    EncodingSpec,
    EntangleLayer,
    MeasurementSpec,
    RepeatBlock,
    RotationLayer,
)

_MEAS = MeasurementSpec(observable="Z", wires="all")
_ENC = EncodingSpec(type="angle", gate="RY", wires="all")


def _expr(ir, n_pairs=400, n_rep=2):
    cfg = ExpressibilityConfig(n_pairs=n_pairs, n_replicates=n_rep)
    reps = replicate_seeds(diagnostic_base_seed(0, structural_hash(ir), "expressibility"), n_rep)
    return compute_expressibility(ir, cfg, reps)


def _hea(depth, n=3):
    block = [
        RotationLayer(gates=["RX", "RY", "RZ"], wires="all"),
        EntangleLayer(pattern="ring", gate="CNOT"),
    ]
    return CircuitIR(
        n_qubits=n, encoding=_ENC, layers=[RepeatBlock(times=depth, body=block)], measurements=_MEAS
    )


def test_haar_bin_probabilities_sum_to_one_and_are_positive():
    for n_qubits in (1, 2, 3, 4):
        for n_bins in (10, 75, 100):
            probs = haar_bin_probabilities(n_qubits, n_bins)
            assert probs.shape == (n_bins,)
            assert probs.sum() == pytest.approx(1.0, abs=1e-12)
            assert (probs > 0).all()


def test_kl_of_haar_samples_against_haar_is_near_zero():
    """Fidelities drawn from the analytic Haar distribution should give a
    small KL (finite-sample bias only), NOT a large value."""
    n_qubits = 3
    dim = 2**n_qubits
    rng = np.random.default_rng(0)
    # Sample fidelities from P_Haar(F) = (N-1)(1-F)^(N-2): if U ~ Uniform,
    # F = 1 - U^(1/(N-1)) has exactly that distribution.
    u = rng.uniform(0, 1, size=20000)
    fidelities = 1 - u ** (1 / (dim - 1))
    kl = kl_divergence_to_haar(fidelities, n_qubits, n_bins=75)
    assert kl < 0.05  # small finite-sample floor, not a large divergence


def test_kl_of_all_ones_fidelity_hits_analytic_upper_bound():
    """A fixed circuit (all fidelities == 1) attains the exact Sim et al.
    upper bound (N-1) ln(n_bins)."""
    n_qubits = 3
    dim = 2**n_qubits
    n_bins = 75
    fidelities = np.ones(1000)
    kl = kl_divergence_to_haar(fidelities, n_qubits, n_bins)
    assert kl == pytest.approx((dim - 1) * np.log(n_bins), rel=1e-9)


def test_empty_empirical_bins_do_not_break_kl():
    """Fidelities that occupy only a few bins (many empty empirical bins)
    must still give a finite KL (0*log0 = 0 policy)."""
    fidelities = np.full(500, 0.5)  # all in one bin
    kl = kl_divergence_to_haar(fidelities, n_qubits=2, n_bins=75)
    assert np.isfinite(kl)


def test_idle_circuit_is_least_expressible_at_the_upper_bound():
    idle = CircuitIR(n_qubits=3, encoding=_ENC, layers=[], measurements=_MEAS)
    outcome = _expr(idle)
    assert outcome.expr_mean == pytest.approx(outcome.upper_bound, abs=1e-6)
    assert outcome.mean_fidelity == pytest.approx(1.0, abs=1e-9)


def test_expressibility_improves_with_hea_depth():
    """Sim et al. ordering: deeper HEA is more expressible (LOWER KL),
    saturating. Idle is worse than any HEA depth."""
    idle_expr = _expr(CircuitIR(n_qubits=3, encoding=_ENC, layers=[], measurements=_MEAS)).expr_mean
    d1 = _expr(_hea(1), n_pairs=600).expr_mean
    d5 = _expr(_hea(5), n_pairs=600).expr_mean
    assert idle_expr > d1
    assert d1 > d5  # deeper => lower KL => more expressible


def test_expressibility_is_deterministic_given_seed():
    ir = _hea(2)
    a = _expr(ir)
    b = _expr(ir)
    assert a.per_replicate == b.per_replicate
    assert a.expr_mean == b.expr_mean


def test_robustness_bin_count_is_reported_separately_and_differs():
    outcome = _expr(_hea(2), n_pairs=600)
    # Primary and robustness use different bin counts on the same
    # fidelities; they should both be finite and generally not identical.
    assert np.isfinite(outcome.expr_mean)
    assert np.isfinite(outcome.robustness_expr_mean)
    assert outcome.expr_mean != outcome.robustness_expr_mean
