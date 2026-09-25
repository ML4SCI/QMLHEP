"""Breadth-first exploration of Clifford circuit equivalence classes."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Clifford, StabilizerState, Statevector

MAX_DEPTH_LIMIT = 12
SINGLE_QUBIT_GATES = ("H", "X", "Y", "Z")
TWO_QUBIT_GATES = ("CX", "CY", "CZ")
SAMPLE_CIRCUIT_LIMIT = 10


class CircuitExplorerError(Exception):
    """Raised when circuit exploration inputs or limits are invalid."""


@dataclass(frozen=True)
class GateAction:
    """A gate applied to specific qubit indices."""

    name: str
    qubits: tuple[int, ...]


@dataclass
class CircuitRecord:
    """Minimum-depth circuit that first reaches a stabilizer state."""

    depth: int
    gates: tuple[GateAction, ...]
    clifford: Clifford


def _validate_inputs(num_qubits: int, max_depth: int) -> None:
    if num_qubits < 1:
        raise CircuitExplorerError("num_qubits must be at least 1.")
    if max_depth < 0:
        raise CircuitExplorerError("max_depth must be non-negative.")
    if max_depth > MAX_DEPTH_LIMIT:
        raise CircuitExplorerError(
            f"max_depth exceeds safety limit of {MAX_DEPTH_LIMIT}."
        )


def state_key(cliff: Clifford) -> bytes:
    """Hash the physical state U|0...0>, modulo global phase."""
    statevector = Statevector.from_int(0, 2**cliff.num_qubits).evolve(cliff)
    data = np.array(statevector.data, copy=True)

    magnitudes = np.abs(data)
    significant = magnitudes > 1e-5
    if np.any(significant):
        first_idx = int(np.argmax(significant))
        phase = data[first_idx] / magnitudes[first_idx]
        data *= np.conjugate(phase)

    # Round for numerical stability, then canonicalize negative zero: IEEE754
    # gives -0.0 and +0.0 distinct byte representations even though they are
    # numerically equal, which previously caused physically identical states
    # to hash to different keys. Adding 0.0 folds -0.0 into +0.0.
    real = np.round(data.real, 5) + 0.0
    imag = np.round(data.imag, 5) + 0.0
    data = real + 1j * imag
    return np.asarray(data, dtype=np.complex128).tobytes()


def format_gate(action: GateAction) -> str:
    """Render a gate action as a human-readable string."""
    if len(action.qubits) == 1:
        return f"{action.name}({action.qubits[0]})"
    control, target = action.qubits
    return f"{action.name}({control},{target})"


def format_circuit(gates: tuple[GateAction, ...]) -> str:
    """Render a gate sequence as a semicolon-separated string."""
    if not gates:
        return "I"
    return "; ".join(format_gate(gate) for gate in gates)


def _apply_gate_to_circuit(
    gate_name: str, qubits: tuple[int, ...], num_qubits: int
) -> Clifford:
    circuit = QuantumCircuit(num_qubits)
    if gate_name == "H":
        circuit.h(qubits[0])
    elif gate_name == "X":
        circuit.x(qubits[0])
    elif gate_name == "Y":
        circuit.y(qubits[0])
    elif gate_name == "Z":
        circuit.z(qubits[0])
    elif gate_name == "CX":
        circuit.cx(qubits[0], qubits[1])
    elif gate_name == "CY":
        circuit.cy(qubits[0], qubits[1])
    elif gate_name == "CZ":
        circuit.cz(qubits[0], qubits[1])
    else:
        raise CircuitExplorerError(f"Unsupported gate: {gate_name}")
    return Clifford(circuit)


def _enumerate_gate_actions(num_qubits: int) -> tuple[tuple[GateAction, Clifford], ...]:
    actions: list[tuple[GateAction, Clifford]] = []

    for gate_name in SINGLE_QUBIT_GATES:
        for qubit in range(num_qubits):
            qubits = (qubit,)
            action = GateAction(gate_name, qubits)
            cliff = _apply_gate_to_circuit(gate_name, qubits, num_qubits)
            actions.append((action, cliff))

    for gate_name in TWO_QUBIT_GATES:
        for control in range(num_qubits):
            for target in range(num_qubits):
                if control == target:
                    continue
                qubits = (control, target)
                action = GateAction(gate_name, qubits)
                cliff = _apply_gate_to_circuit(gate_name, qubits, num_qubits)
                actions.append((action, cliff))

    return tuple(actions)


def gates_to_quantum_circuit(
    gates: tuple[GateAction, ...], num_qubits: int
) -> QuantumCircuit:
    """Build a Qiskit circuit from an ordered gate sequence."""
    circuit = QuantumCircuit(num_qubits)
    for action in gates:
        if action.name == "H":
            circuit.h(action.qubits[0])
        elif action.name == "X":
            circuit.x(action.qubits[0])
        elif action.name == "Y":
            circuit.y(action.qubits[0])
        elif action.name == "Z":
            circuit.z(action.qubits[0])
        elif action.name == "CX":
            circuit.cx(action.qubits[0], action.qubits[1])
        elif action.name == "CY":
            circuit.cy(action.qubits[0], action.qubits[1])
        elif action.name == "CZ":
            circuit.cz(action.qubits[0], action.qubits[1])
        else:
            raise CircuitExplorerError(f"Unsupported gate: {action.name}")
    return circuit


def _gate_action_to_dict(action: GateAction) -> dict[str, Any]:
    return {"name": action.name, "qubits": list(action.qubits)}


def _select_best_circuit(
    visited: dict[bytes, CircuitRecord], max_depth: int
) -> dict[str, Any]:
    """Pick a representative minimum-depth circuit at the target depth."""
    candidates = [record for record in visited.values() if record.depth == max_depth]
    if not candidates:
        best_depth = max(record.depth for record in visited.values())
        candidates = [record for record in visited.values() if record.depth == best_depth]

    record = min(candidates, key=lambda item: format_circuit(item.gates))
    return {
        "depth": record.depth,
        "gate_count": record.depth,
        "circuit_str": format_circuit(record.gates),
        "gates": [_gate_action_to_dict(gate) for gate in record.gates],
    }


def _state_entropy(clifford: Clifford, num_qubits: int) -> float:
    """Shannon entropy (bits) of the measurement probability distribution."""
    statevector = Statevector.from_int(0, 2**num_qubits).evolve(clifford)
    probabilities = np.abs(statevector.data) ** 2
    probabilities = probabilities[probabilities > 1e-12]
    return float(-np.sum(probabilities * np.log2(probabilities)))


def _select_most_complex_state(
    visited: dict[bytes, CircuitRecord], num_qubits: int
) -> dict[str, Any]:
    """Pick the deepest minimum-depth circuit; tie-break by measurement entropy."""
    max_depth = max(record.depth for record in visited.values())
    candidates = [record for record in visited.values() if record.depth == max_depth]
    record = max(candidates, key=lambda item: _state_entropy(item.clifford, num_qubits))
    probabilities = _measurement_probabilities(record.clifford, num_qubits)
    return {
        "depth": record.depth,
        "gate_count": record.depth,
        "circuit_str": format_circuit(record.gates),
        "gates": [_gate_action_to_dict(gate) for gate in record.gates],
        "entropy_bits": round(_state_entropy(record.clifford, num_qubits), 4),
        "measurement_probabilities": probabilities,
    }


def _measurement_probabilities(clifford: Clifford, num_qubits: int) -> dict[str, float]:
    """Return computational-basis measurement probabilities for U|0...0>."""
    statevector = Statevector.from_int(0, 2**num_qubits).evolve(clifford)
    probabilities: dict[str, float] = {}
    for index, amplitude in enumerate(statevector.data):
        label = f"|{format(index, f'0{num_qubits}b')}>"
        probability = float(np.abs(amplitude) ** 2)
        if probability > 1e-12:
            probabilities[label] = round(probability, 6)
    return probabilities


def _compute_gate_distribution(
    visited: dict[bytes, CircuitRecord],
) -> dict[str, int]:
    """Count gate-type frequency across all minimum-depth (simplest) circuits."""
    counts = {gate: 0 for gate in SINGLE_QUBIT_GATES + TWO_QUBIT_GATES}
    for record in visited.values():
        for gate in record.gates:
            counts[gate.name] += 1
    return counts


def _build_sample_circuits(
    visited: dict[bytes, CircuitRecord], max_depth: int
) -> list[dict[str, Any]]:
    records = sorted(
        visited.values(), key=lambda record: (record.depth, format_circuit(record.gates))
    )
    samples: list[dict[str, Any]] = []
    seen_depths: set[int] = set()

    for record in records:
        if len(samples) >= SAMPLE_CIRCUIT_LIMIT:
            break
        if record.depth in seen_depths and record.depth not in {0, max_depth}:
            continue
        seen_depths.add(record.depth)
        stabilizer_labels = {
            "stabilizer": StabilizerState(record.clifford).clifford.to_labels(mode="S")
        }
        samples.append(
            {
                "depth": record.depth,
                "gate_count": record.depth,
                "circuit_str": format_circuit(record.gates),
                "stabilizer_labels": stabilizer_labels,
            }
        )

    seen_circuits = {sample["circuit_str"] for sample in samples}
    if len(samples) < SAMPLE_CIRCUIT_LIMIT:
        for record in records:
            if len(samples) >= SAMPLE_CIRCUIT_LIMIT:
                break
            circuit_str = format_circuit(record.gates)
            if circuit_str in seen_circuits:
                continue
            seen_circuits.add(circuit_str)
            samples.append(
                {
                    "depth": record.depth,
                    "gate_count": record.depth,
                    "circuit_str": circuit_str,
                    "stabilizer_labels": {
                        "stabilizer": StabilizerState(record.clifford).clifford.to_labels(mode="S")
                    },
                }
            )

    return samples


def explore_circuit_space(num_qubits: int, max_depth: int) -> dict[str, Any]:
    """
    Explore Clifford circuits up to max_depth using BFS with stabilizer pruning.

    Returns a JSON-serializable summary of unique stabilizer states reachable from
    |0...0>, including per-depth discovery counts and sample minimum-depth circuits.
    """
    _validate_inputs(num_qubits, max_depth)
    start_time = time.perf_counter()

    gate_actions = _enumerate_gate_actions(num_qubits)
    identity = Clifford(QuantumCircuit(num_qubits))
    initial_key = state_key(identity)

    visited: dict[bytes, CircuitRecord] = {
        initial_key: CircuitRecord(depth=0, gates=(), clifford=identity)
    }
    new_states_at_depth: dict[int, int] = {0: 1}
    explored_states_at_depth: dict[int, int] = {}
    exploration_trajectory: list[dict[str, Any]] = [
        {
            "step": 0,
            "depth": 0,
            "cumulative_unique_states": 1,
            "coverage_score": 1.0,
        }
    ]
    queue: deque[tuple[Clifford, tuple[GateAction, ...], int]] = deque(
        [(identity, (), 0)]
    )

    while queue:
        clifford, gates, depth = queue.popleft()
        if depth >= max_depth:
            continue

        # Get the last action to prune trivial self-inverses
        last_action = gates[-1] if gates else None

        for action, gate_clifford in gate_actions:
            # Skip if applying the exact same gate as the previous step
            if action == last_action:
                continue

            new_clifford = clifford.compose(gate_clifford, front=False)
            new_depth = depth + 1
            explored_states_at_depth[new_depth] = (
                explored_states_at_depth.get(new_depth, 0) + 1
            )
            key = state_key(new_clifford)
            if key in visited:
                continue

            new_gates = gates + (action,)
            visited[key] = CircuitRecord(
                depth=new_depth,
                gates=new_gates,
                clifford=new_clifford,
            )
            new_states_at_depth[new_depth] = new_states_at_depth.get(new_depth, 0) + 1
            exploration_trajectory.append(
                {
                    "step": len(exploration_trajectory),
                    "depth": new_depth,
                    "cumulative_unique_states": len(visited),
                    "coverage_score": float(len(visited)),
                }
            )

            if new_depth < max_depth:
                queue.append((new_clifford, new_gates, new_depth))

    states_at_exact_depth = new_states_at_depth.get(max_depth, 0)
    elapsed_seconds = time.perf_counter() - start_time

    return {
        "num_qubits": num_qubits,
        "max_depth": max_depth,
        "num_actions_per_step": len(gate_actions),
        "gate_set": list(SINGLE_QUBIT_GATES) + list(TWO_QUBIT_GATES),
        "total_unique_states": len(visited),
        "new_states_at_depth": {
            str(depth): count for depth, count in sorted(new_states_at_depth.items())
        },
        "explored_states_at_depth": {
            str(depth): count for depth, count in sorted(explored_states_at_depth.items())
        },
        "states_at_exact_depth_G": states_at_exact_depth,
        "exploration_trajectory": exploration_trajectory,
        "best_circuit": _select_best_circuit(visited, max_depth),
        "most_complex_state": _select_most_complex_state(visited, num_qubits),
        "gate_distribution": _compute_gate_distribution(visited),
        "sample_circuits": _build_sample_circuits(visited, max_depth),
        "elapsed_seconds": round(elapsed_seconds, 4),
    }
