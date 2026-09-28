"""Semantic (state/expectation-value) equivalence tests for the IR compilers.

These tests never rely on textual/structural equality of compiled
circuits (e.g. comparing `.draw()` output or QASM strings). Every
assertion here is either:

- a `Statevector.equiv()` comparison (Qiskit's own global-phase-aware
  equivalence check, independently trusted — not code written for this
  project) between an IR-compiled circuit and a hand-written "trusted
  reference" circuit using Qiskit's native API directly, or
- a numerical comparison of expectation values between the PennyLane and
  Qiskit compilations of the *same* IR + inputs + weights.
"""

from __future__ import annotations

import math

import numpy as np
import pennylane as qml
import pytest
import torch
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from llm_vqc.ir.compiler_pennylane import to_qnode
from llm_vqc.ir.compiler_qiskit import CompilerError, qiskit_observables, to_qiskit_bound
from llm_vqc.ir.schema import (
    CircuitIR,
    EncodingSpec,
    EntangleLayer,
    MeasurementSpec,
    RotationLayer,
)
from llm_vqc.ir.validators import validate_proposal


def _statevector_of(ir, inputs, weights) -> Statevector:
    circuit = to_qiskit_bound(ir, inputs=inputs, weights=weights)
    return Statevector(circuit)


# --- Single-qubit gates ------------------------------------------------


def test_single_qubit_rx_matches_hand_built_reference():
    ir = CircuitIR(
        n_qubits=2,
        # wire 1 is unused by this test but keeps n_qubits=2 valid (MIN_QUBITS)
        encoding=EncodingSpec(type="angle", gate="RY", wires=[1]),
        layers=[RotationLayer(gates=["RX"], wires=[0])],
        measurements=MeasurementSpec(observable="Z", wires="all"),
    )
    theta = 0.7
    compiled_sv = _statevector_of(ir, inputs=[0.0], weights=[theta])

    reference = QuantumCircuit(2)
    reference.rx(theta, 0)
    reference_sv = Statevector(reference)

    assert compiled_sv.equiv(reference_sv)


def test_single_qubit_h_matches_hand_built_reference():
    ir = CircuitIR(
        n_qubits=2,
        encoding=EncodingSpec(type="angle", gate="RY", wires=[1]),
        layers=[RotationLayer(gates=["H"], wires=[0])],
        measurements=MeasurementSpec(observable="Z", wires="all"),
    )
    compiled_sv = _statevector_of(ir, inputs=[0.0], weights=[])

    reference = QuantumCircuit(2)
    reference.h(0)
    reference_sv = Statevector(reference)

    assert compiled_sv.equiv(reference_sv)


# --- Controlled / multi-qubit gates -------------------------------------


def test_pairs_cnot_matches_hand_built_reference():
    ir = CircuitIR(
        n_qubits=3,
        encoding=EncodingSpec(type="angle", gate="RX", wires="all"),
        layers=[EntangleLayer(pattern="pairs", gate="CNOT", pairs=[(0, 1), (1, 2)])],
        measurements=MeasurementSpec(observable="Z", wires="all"),
    )
    inputs = [0.3, 0.5, 0.9]
    compiled_sv = _statevector_of(ir, inputs=inputs, weights=[])

    reference = QuantumCircuit(3)
    reference.rx(inputs[0], 0)
    reference.rx(inputs[1], 1)
    reference.rx(inputs[2], 2)
    reference.cx(0, 1)
    reference.cx(1, 2)
    reference_sv = Statevector(reference)

    assert compiled_sv.equiv(reference_sv)


def test_star_pattern_matches_hand_built_reference():
    ir = CircuitIR(
        n_qubits=4,
        encoding=EncodingSpec(type="angle", gate="RY", wires="all"),
        layers=[EntangleLayer(pattern="star", gate="CNOT", center=1)],
        measurements=MeasurementSpec(observable="Z", wires="all"),
    )
    inputs = [0.1, 0.2, 0.3, 0.4]
    compiled_sv = _statevector_of(ir, inputs=inputs, weights=[])

    reference = QuantumCircuit(4)
    for i, x in enumerate(inputs):
        reference.ry(x, i)
    reference.cx(1, 0)
    reference.cx(1, 2)
    reference.cx(1, 3)
    reference_sv = Statevector(reference)

    assert compiled_sv.equiv(reference_sv)


def test_crz_parameterized_entangle_matches_hand_built_reference():
    ir = CircuitIR(
        n_qubits=2,
        encoding=EncodingSpec(type="angle", gate="RY", wires="all"),
        layers=[EntangleLayer(pattern="pairs", gate="CRZ", pairs=[(0, 1)])],
        measurements=MeasurementSpec(observable="Z", wires="all"),
    )
    inputs = [0.2, 0.4]
    phi = 1.1
    compiled_sv = _statevector_of(ir, inputs=inputs, weights=[phi])

    reference = QuantumCircuit(2)
    reference.ry(inputs[0], 0)
    reference.ry(inputs[1], 1)
    reference.crz(phi, 0, 1)
    reference_sv = Statevector(reference)

    assert compiled_sv.equiv(reference_sv)


# --- Gate ordering sensitivity ------------------------------------------


def test_layer_order_changes_the_compiled_state_for_noncommuting_gates():
    """RX(pi/2) on wire 0 then CNOT(0,1) is not the same state as the reverse."""
    ir_forward = CircuitIR(
        n_qubits=2,
        encoding=EncodingSpec(type="angle", gate="RY", wires=[1]),
        layers=[
            RotationLayer(gates=["RX"], wires=[0]),
            EntangleLayer(pattern="pairs", gate="CNOT", pairs=[(0, 1)]),
        ],
        measurements=MeasurementSpec(observable="Z", wires="all"),
    )
    ir_reversed = CircuitIR(
        n_qubits=2,
        encoding=EncodingSpec(type="angle", gate="RY", wires=[1]),
        layers=[
            EntangleLayer(pattern="pairs", gate="CNOT", pairs=[(0, 1)]),
            RotationLayer(gates=["RX"], wires=[0]),
        ],
        measurements=MeasurementSpec(observable="Z", wires="all"),
    )
    theta = math.pi / 2
    sv_forward = _statevector_of(ir_forward, inputs=[0.3], weights=[theta])
    sv_reversed = _statevector_of(ir_reversed, inputs=[0.3], weights=[theta])

    assert not sv_forward.equiv(sv_reversed)


# --- Cross-backend consistency (PennyLane vs Qiskit) ---------------------


def test_pennylane_and_qiskit_agree_on_expectation_values():
    ir = CircuitIR(
        n_qubits=3,
        encoding=EncodingSpec(type="angle", gate="RY", wires="all"),
        layers=[
            RotationLayer(gates=["RY", "RZ"], wires="all"),
            EntangleLayer(pattern="ring", gate="CNOT"),
            RotationLayer(gates=["RX"], wires="all"),
        ],
        measurements=MeasurementSpec(observable="Z", wires="all"),
    )
    rng = np.random.default_rng(0)
    inputs = rng.uniform(-math.pi, math.pi, size=3)
    weights = rng.uniform(-math.pi, math.pi, size=9)  # RY,RZ*3 + RX*3 = 9 params

    qnode = to_qnode(ir)
    pennylane_expvals = [float(v) for v in qnode(inputs, weights)]

    qiskit_circuit = to_qiskit_bound(ir, inputs=list(inputs), weights=list(weights))
    sv = Statevector(qiskit_circuit)
    qiskit_expvals = [float(sv.expectation_value(op).real) for op in qiskit_observables(ir)]

    assert pennylane_expvals == pytest.approx(qiskit_expvals, abs=1e-8)


def test_pennylane_qnode_is_deterministic_across_repeated_calls():
    """Same IR + same config -> semantically equivalent output across runs."""
    ir = CircuitIR(
        n_qubits=2,
        encoding=EncodingSpec(type="angle", gate="RX", wires="all"),
        layers=[EntangleLayer(pattern="pairs", gate="CNOT", pairs=[(0, 1)])],
        measurements=MeasurementSpec(observable="Z", wires="all"),
    )
    qnode = to_qnode(ir)
    inputs = [0.4, 0.9]
    weights: list[float] = []
    out1 = [float(v) for v in qnode(inputs, weights)]
    out2 = [float(v) for v in qnode(inputs, weights)]
    assert out1 == out2


def test_qnode_correctly_broadcasts_over_a_batch_of_torch_inputs():
    """Regression test for a Phase 2 integration bug (see DECISIONS.md).

    `_apply_program` used to index encoding inputs as `inputs[i]`, which
    is correct for a single 1D sample but silently wrong for a batched
    2D torch tensor (batch dimension first): `inputs[i]` sliced batch row
    `i` instead of feature `i` across the whole batch, so every measured
    expectation value ended up a function of the wrong data. This test
    compiles a circuit, evaluates it on a batch, and separately evaluates
    it row-by-row (the trusted reference, exercising the already-verified
    single-sample path) — the batched call must match the row-by-row
    calls exactly, and different batch rows must generally produce
    different outputs (the bug produced identical/scrambled outputs
    across the batch).
    """
    ir = CircuitIR(
        n_qubits=3,
        encoding=EncodingSpec(type="angle", gate="RY", wires="all"),
        layers=[
            RotationLayer(gates=["RY", "RZ"], wires="all"),
            EntangleLayer(pattern="ring", gate="CNOT"),
        ],
        measurements=MeasurementSpec(observable="Z", wires="all"),
    )
    qnode = to_qnode(ir)
    weights = torch.tensor([0.3, -0.2, 0.5, 0.1, -0.4, 0.6], dtype=torch.float64)

    batch_inputs = torch.tensor(
        [[0.1, 0.2, 0.3], [0.9, 0.8, 0.7], [0.4, 0.4, 0.4]], dtype=torch.float64
    )
    batched_out = qnode(batch_inputs, weights)  # list of 3 tensors, each shape (3,)

    for row in range(3):
        reference_out = qnode(batch_inputs[row], weights)  # single-sample path (trusted)
        for measurement_index in range(3):
            assert float(batched_out[measurement_index][row]) == pytest.approx(
                float(reference_out[measurement_index]), abs=1e-10
            )

    # Different batch rows must give different outputs somewhere (proves
    # the batch axis is actually carrying per-row data, not collapsed).
    row0 = [float(batched_out[m][0]) for m in range(3)]
    row1 = [float(batched_out[m][1]) for m in range(3)]
    assert row0 != row1


def test_compiled_qnode_trains_on_toy_data():
    """Phase 1 acceptance criterion: "compiled QNode trains on toy data."

    Gradient-descends a compiled circuit toward a fixed target expectation
    value using PennyLane's own autograd interface (no evaluation harness,
    no task loader — that is Phase 2 — just proof that the compiled QNode
    is differentiable end-to-end and an optimizer can reduce a loss on
    it).
    """
    ir = CircuitIR(
        n_qubits=3,
        encoding=EncodingSpec(type="angle", gate="RY", wires="all"),
        layers=[
            RotationLayer(gates=["RY", "RZ"], wires="all"),
            EntangleLayer(pattern="ring", gate="CNOT"),
        ],
        measurements=MeasurementSpec(observable="Z", wires="all"),
    )
    program_num_parameters = 6  # RY, RZ * 3 wires
    qnode = to_qnode(ir)
    inputs = qml.numpy.array([0.1, 0.2, 0.3], requires_grad=False)
    weights = qml.numpy.array([0.5] * program_num_parameters, requires_grad=True)

    target = qml.numpy.array([-1.0, -1.0, -1.0])

    def cost(w):
        out = qnode(inputs, w)
        return sum((o - t) ** 2 for o, t in zip(out, target, strict=True))

    opt = qml.GradientDescentOptimizer(stepsize=0.3)
    initial_loss = float(cost(weights))
    w = weights
    for _ in range(15):
        w = opt.step(cost, w)
    final_loss = float(cost(w))

    assert final_loss < initial_loss, (
        f"gradient descent did not reduce loss: {initial_loss} -> {final_loss}"
    )


# --- Parameter binding validation ----------------------------------------


def test_parameter_binding_rejects_wrong_length_weights():
    ir = CircuitIR(
        n_qubits=2,
        encoding=EncodingSpec(type="angle", gate="RY", wires="all"),
        layers=[RotationLayer(gates=["RX"], wires="all")],
        measurements=MeasurementSpec(observable="Z", wires="all"),
    )
    with pytest.raises(CompilerError):
        to_qiskit_bound(ir, inputs=[0.1, 0.2], weights=[0.1])  # needs 2 weights, got 1


def test_parameter_binding_rejects_non_finite_values():
    ir = CircuitIR(
        n_qubits=2,
        encoding=EncodingSpec(type="angle", gate="RY", wires="all"),
        layers=[RotationLayer(gates=["RX"], wires="all")],
        measurements=MeasurementSpec(observable="Z", wires="all"),
    )
    with pytest.raises(CompilerError):
        to_qiskit_bound(ir, inputs=[0.1, 0.2], weights=[float("nan"), 0.1])
    with pytest.raises(CompilerError):
        to_qiskit_bound(ir, inputs=[0.1, float("inf")], weights=[0.1, 0.1])


def test_parameter_binding_accepts_correct_length():
    ir = CircuitIR(
        n_qubits=2,
        encoding=EncodingSpec(type="angle", gate="RY", wires="all"),
        layers=[RotationLayer(gates=["RX"], wires="all")],
        measurements=MeasurementSpec(observable="Z", wires="all"),
    )
    circuit = to_qiskit_bound(ir, inputs=[0.1, 0.2], weights=[0.3, 0.4])
    assert circuit.num_parameters == 0


# --- Invalid circuits (should never reach a compiler) --------------------


@pytest.mark.parametrize(
    "raw,expected_code_prefix",
    [
        (
            {
                "n_qubits": 2,
                "encoding": {"type": "angle", "gate": "RY", "wires": [5]},
                "layers": [],
                "measurements": {"observable": "Z", "wires": "all"},
            },
            "wires.out_of_bounds",
        ),
        (
            {
                "n_qubits": 2,
                "encoding": {"type": "angle", "gate": "RY", "wires": "all"},
                "layers": [{"type": "entangle", "pattern": "star", "gate": "CNOT"}],
                "measurements": {"observable": "Z", "wires": "all"},
            },
            "entangle.missing_center",
        ),
        (
            {
                "n_qubits": 3,
                "encoding": {"type": "angle", "gate": "RY", "wires": "all"},
                "layers": [
                    {
                        "type": "entangle",
                        "pattern": "pairs",
                        "gate": "CNOT",
                        "pairs": [[0, 0]],
                    }
                ],
                "measurements": {"observable": "Z", "wires": "all"},
            },
            "entangle.self_loop",
        ),
    ],
)
def test_invalid_circuits_are_rejected_with_expected_code(raw, expected_code_prefix):
    result = validate_proposal(raw)
    assert not result.valid
    codes = [issue.code for issue in result.issues]
    assert any(c.startswith(expected_code_prefix) for c in codes), codes
