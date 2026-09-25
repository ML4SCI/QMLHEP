"""Unit tests for phase-normalized Statevector state keys."""

from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.quantum_info import Clifford

from llm_vqc.circuit_explorer import state_key


def _single_gate_clifford(gate_name: str, num_qubits: int = 1) -> Clifford:
    circuit = QuantumCircuit(num_qubits)
    if gate_name == "H":
        circuit.h(0)
    elif gate_name == "X":
        circuit.x(0)
    elif gate_name == "Y":
        circuit.y(0)
    elif gate_name == "Z":
        circuit.z(0)
    elif gate_name == "CX":
        circuit.cx(0, 1)
    else:
        raise ValueError(f"Unsupported gate: {gate_name}")
    return Clifford(circuit)


def test_state_key_returns_bytes_and_is_deterministic() -> None:
    identity = Clifford(QuantumCircuit(1))
    key = state_key(identity)

    assert isinstance(key, bytes)
    assert key == state_key(identity)


def test_identity_and_z_share_key_on_one_qubit() -> None:
    identity = Clifford(QuantumCircuit(1))
    z_gate = _single_gate_clifford("Z")

    assert state_key(identity) == state_key(identity.compose(z_gate, front=False))


def test_x_and_y_share_key_on_one_qubit() -> None:
    identity = Clifford(QuantumCircuit(1))
    x_gate = _single_gate_clifford("X")
    y_gate = _single_gate_clifford("Y")

    x_state = identity.compose(x_gate, front=False)
    y_state = identity.compose(y_gate, front=False)

    assert state_key(x_state) == state_key(y_state)


def test_h_and_x_differ_on_one_qubit() -> None:
    identity = Clifford(QuantumCircuit(1))
    h_state = identity.compose(_single_gate_clifford("H"), front=False)
    x_state = identity.compose(_single_gate_clifford("X"), front=False)

    assert state_key(h_state) != state_key(x_state)


def test_distinct_two_qubit_states_have_different_keys() -> None:
    identity = Clifford(QuantumCircuit(2))
    h_state = identity.compose(_single_gate_clifford("H", num_qubits=2), front=False)
    cx_state = identity.compose(_single_gate_clifford("CX", num_qubits=2), front=False)

    assert state_key(h_state) != state_key(cx_state)
