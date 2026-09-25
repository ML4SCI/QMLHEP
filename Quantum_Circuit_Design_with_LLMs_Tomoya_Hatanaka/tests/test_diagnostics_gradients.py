"""Gradient-variance diagnostic tests: autodiff-vs-finite-difference
correctness, determinism, zero-parameter and non-finite handling."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from llm_vqc.diagnostics.config import GradientVarianceConfig
from llm_vqc.diagnostics.gradients import (
    _cost_qnode,
    _gradient_at,
    compute_gradient_variance,
)
from llm_vqc.diagnostics.sampling import diagnostic_input
from llm_vqc.diagnostics.seeds import DiagnosticSeeds, diagnostic_base_seed
from llm_vqc.ir.canonicalize import structural_hash
from llm_vqc.ir.expand import build_program
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


def _hea(depth, n=3):
    block = [
        RotationLayer(gates=["RX", "RY", "RZ"], wires="all"),
        EntangleLayer(pattern="ring", gate="CNOT"),
    ]
    return CircuitIR(
        n_qubits=n, encoding=_ENC, layers=[RepeatBlock(times=depth, body=block)], measurements=_MEAS
    )


def _gv(ir, n_inits=40):
    cfg = GradientVarianceConfig(n_inits=n_inits, n_bootstrap=200)
    seeds = DiagnosticSeeds.from_base_seed(
        diagnostic_base_seed(0, structural_hash(ir), "gradient_variance")
    )
    return compute_gradient_variance(ir, cfg, seeds)


def test_autodiff_gradient_matches_finite_difference():
    ir = _hea(2)
    program = build_program(ir)
    qnode = _cost_qnode(ir, program)
    inp = diagnostic_input(program, 0)
    weights = np.random.default_rng(3).uniform(0, 2 * np.pi, program.num_parameters)

    autodiff = _gradient_at(qnode, inp, weights)

    eps = 1e-6
    fd = np.zeros_like(weights)
    for i in range(len(weights)):
        wp, wm = weights.copy(), weights.copy()
        wp[i] += eps
        wm[i] -= eps
        fp = float(qnode(torch.tensor(inp), torch.tensor(wp)))
        fm = float(qnode(torch.tensor(inp), torch.tensor(wm)))
        fd[i] = (fp - fm) / (2 * eps)

    assert np.max(np.abs(autodiff - fd)) < 1e-7


def test_gradient_variance_is_deterministic_given_seed():
    ir = _hea(2)
    a = _gv(ir)
    b = _gv(ir)
    assert a.aggregate_variance == b.aggregate_variance
    assert a.tracked_param_variance == b.tracked_param_variance
    assert a.mean_abs_gradient == b.mean_abs_gradient


def test_zero_parameter_circuit_is_reported_not_failed():
    ir = CircuitIR(
        n_qubits=2, encoding=_ENC,
        layers=[
            RotationLayer(gates=["H"], wires="all"),
            EntangleLayer(pattern="pairs", gate="CNOT", pairs=[(0, 1)]),
        ],
        measurements=_MEAS,
    )
    outcome = _gv(ir)
    assert outcome.n_parameters == 0
    assert outcome.failed is False
    assert outcome.aggregate_variance is None
    assert outcome.mean_abs_gradient is None


def test_raw_and_log_variance_are_both_reported():
    outcome = _gv(_hea(2))
    assert outcome.aggregate_variance is not None
    assert outcome.aggregate_log10_variance is not None
    # log10 of the raw value (unless raw is exactly zero -> -inf)
    if outcome.aggregate_variance > 0:
        assert outcome.aggregate_log10_variance == pytest.approx(
            np.log10(outcome.aggregate_variance)
        )


def test_bootstrap_ci_brackets_the_mean_abs_gradient():
    outcome = _gv(_hea(2))
    assert outcome.mean_abs_gradient_ci_low <= outcome.mean_abs_gradient
    assert outcome.mean_abs_gradient <= outcome.mean_abs_gradient_ci_high


def test_non_finite_gradients_are_excluded_and_counted(monkeypatch):
    """If some gradient evaluations return non-finite values they must be
    counted and excluded from the statistics, never silently averaged in."""
    ir = _hea(2)
    program = build_program(ir)
    cfg = GradientVarianceConfig(n_inits=10, n_bootstrap=0)
    base = diagnostic_base_seed(0, structural_hash(ir), "gradient_variance")
    seeds = DiagnosticSeeds.from_base_seed(base)

    import llm_vqc.diagnostics.gradients as grad_module

    real_grad = grad_module._gradient_at
    call_count = {"n": 0}

    def flaky_grad(qnode, inputs, weights):
        call_count["n"] += 1
        g = real_grad(qnode, inputs, weights)
        if call_count["n"] % 2 == 0:  # make every other init non-finite
            g = g.copy()
            g[0] = np.nan
        return g

    monkeypatch.setattr(grad_module, "_gradient_at", flaky_grad)
    outcome = compute_gradient_variance(ir, cfg, seeds, program)
    assert outcome.n_inits_non_finite == 5
    assert outcome.n_inits_completed == 5
    assert outcome.failed is False
    assert np.isfinite(outcome.aggregate_variance)


def test_all_non_finite_gradients_is_reported_as_failure(monkeypatch):
    ir = _hea(2)
    program = build_program(ir)
    cfg = GradientVarianceConfig(n_inits=6, n_bootstrap=0)
    base = diagnostic_base_seed(0, structural_hash(ir), "gradient_variance")
    seeds = DiagnosticSeeds.from_base_seed(base)

    import llm_vqc.diagnostics.gradients as grad_module

    def all_nan(qnode, inputs, weights):
        return np.full(program.num_parameters, np.nan)

    monkeypatch.setattr(grad_module, "_gradient_at", all_nan)
    outcome = compute_gradient_variance(ir, cfg, seeds, program)
    assert outcome.failed is True
    assert outcome.n_inits_non_finite == 6
    assert outcome.aggregate_variance is None
    assert "non-finite" in outcome.error_message
