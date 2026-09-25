"""Tests for diagnostic parameter/input sampling and statevector generation,
including cross-backend agreement and global-phase invariance."""

from __future__ import annotations

import numpy as np
import pytest

from llm_vqc.diagnostics.sampling import (
    diagnostic_input,
    fidelity,
    sample_weight_vectors,
    statevector,
    statevector_qiskit,
)
from llm_vqc.ir.expand import build_program
from llm_vqc.ir.schema import (
    CircuitIR,
    EncodingSpec,
    EntangleLayer,
    MeasurementSpec,
    RotationLayer,
)

_MEAS = MeasurementSpec(observable="Z", wires="all")


def _ir(n_qubits=3, gate="RY"):
    return CircuitIR(
        n_qubits=n_qubits,
        encoding=EncodingSpec(type="angle", gate=gate, wires="all"),
        layers=[
            RotationLayer(gates=["RY", "RZ"], wires="all"),
            EntangleLayer(pattern="ring", gate="CNOT"),
        ],
        measurements=_MEAS,
    )


def test_weight_sampling_is_deterministic_given_seed():
    program = build_program(_ir())
    a = sample_weight_vectors(program, 10, 0.0, 6.28, seed=5)
    b = sample_weight_vectors(program, 10, 0.0, 6.28, seed=5)
    assert np.array_equal(a, b)
    c = sample_weight_vectors(program, 10, 0.0, 6.28, seed=6)
    assert not np.array_equal(a, c)


def test_weight_sampling_respects_range_and_shape():
    program = build_program(_ir())
    vectors = sample_weight_vectors(program, 100, 1.0, 2.0, seed=0)
    assert vectors.shape == (100, program.num_parameters)
    assert vectors.min() >= 1.0 and vectors.max() < 2.0


def test_angle_encoding_diagnostic_input_is_zeros():
    program = build_program(_ir())
    inp = diagnostic_input(program, seed=0)
    assert np.array_equal(inp, np.zeros(program.num_inputs))


def test_amplitude_encoding_diagnostic_input_is_fixed_and_normalized():
    ir = CircuitIR(
        n_qubits=3,
        encoding=EncodingSpec(type="amplitude", wires="all"),
        layers=[RotationLayer(gates=["RX"], wires="all")],
        measurements=_MEAS,
    )
    program = build_program(ir)
    inp = diagnostic_input(program, seed=7)
    assert inp.shape == (program.num_inputs,)
    assert abs(np.linalg.norm(inp) - 1.0) < 1e-12
    # deterministic given seed
    assert np.array_equal(inp, diagnostic_input(program, seed=7))


def test_fidelity_is_global_phase_invariant():
    program = build_program(_ir())
    inp = diagnostic_input(program, 0)
    w = sample_weight_vectors(program, 1, 0.0, 6.28, seed=0)[0]
    state = statevector(_ir(), inp, w, program)
    phased = np.exp(1j * 1.2345) * state
    assert fidelity(state, phased) == pytest.approx(1.0, abs=1e-12)


def test_cross_backend_statevector_agreement_up_to_global_phase():
    ir = _ir()
    program = build_program(ir)
    inp = diagnostic_input(program, 0)
    for w in sample_weight_vectors(program, 5, 0.0, 6.28, seed=1):
        sv_pl = statevector(ir, inp, w, program)
        sv_qk = statevector_qiskit(ir, inp, w, program)
        assert fidelity(sv_pl, sv_qk) == pytest.approx(1.0, abs=1e-9)


def test_idle_circuit_prepares_computational_zero_state():
    ir = CircuitIR(
        n_qubits=2, encoding=EncodingSpec(type="angle", gate="RY", wires="all"),
        layers=[], measurements=_MEAS,
    )
    program = build_program(ir)
    sv = statevector(ir, diagnostic_input(program, 0), np.array([]), program)
    assert abs(sv[0] - 1.0) < 1e-12
    assert np.allclose(sv[1:], 0.0)


def test_bell_circuit_prepares_bell_state():
    ir = CircuitIR(
        n_qubits=2, encoding=EncodingSpec(type="angle", gate="RY", wires="all"),
        layers=[
            RotationLayer(gates=["H"], wires=[0]),
            EntangleLayer(pattern="pairs", gate="CNOT", pairs=[(0, 1)]),
        ],
        measurements=_MEAS,
    )
    program = build_program(ir)
    sv = statevector(ir, diagnostic_input(program, 0), np.array([]), program)
    expected = np.array([1, 0, 0, 1]) / np.sqrt(2)
    assert fidelity(sv, expected) == pytest.approx(1.0, abs=1e-12)
