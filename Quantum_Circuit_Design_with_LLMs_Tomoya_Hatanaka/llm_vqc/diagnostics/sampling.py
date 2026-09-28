"""Parameter sampling, the fixed diagnostic-input policy, and statevector
generation shared by all three diagnostics.

State generation goes through Phase 1's compilers only — it never
re-implements gate application — so diagnostics measure exactly the same
circuit semantics that training does. A single `statevector(ir, inputs,
weights)` produces the PennyLane statevector; `statevector_qiskit(...)`
produces the independent Qiskit statevector for the cross-backend
validation tests.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml

from llm_vqc.ir.compiler_pennylane import apply_program_ops
from llm_vqc.ir.compiler_qiskit import to_qiskit_bound
from llm_vqc.ir.expand import CircuitProgram, build_program
from llm_vqc.ir.schema import CircuitIR


def sample_weight_vectors(
    program: CircuitProgram, n: int, low: float, high: float, seed: int
) -> np.ndarray:
    """`n` weight vectors, each of length `program.num_parameters`, i.i.d.
    uniform on `[low, high)`. Shape `(n, num_parameters)`.

    For a circuit with zero trainable parameters this returns shape
    `(n, 0)` — every sample is the empty vector, so the circuit produces
    the same state every time (correct: a parameterless circuit is a
    single fixed state).
    """
    rng = np.random.default_rng(seed)
    return rng.uniform(low, high, size=(n, program.num_parameters))


def diagnostic_input(program: CircuitProgram, seed: int) -> np.ndarray:
    """The FIXED data input held constant across all parameter samples.

    - Angle encoding: zeros. With zero input, angle-encoding rotation
      gates become the identity, so the measured circuit is exactly the
      pure variational ansatz `U_variational(theta)|0...0>` — Sim et al.'s
      own setup. (This is why the idle/parameterless circuit correctly
      scores worst on expressibility: it produces the single state
      |0...0> for every weight sample.)
    - Amplitude encoding: a fixed, seeded, L2-normalized real vector.
      Amplitude-encoding inputs are amplitudes, not angles, so "zeros" is
      not meaningful (the zero vector cannot be normalized); a fixed
      seeded unit vector is used instead and held constant across all
      samples. Documented as an asymmetry in `DECISIONS.md`.
    """
    if program.encoding_type == "angle":
        return np.zeros(program.num_inputs, dtype=np.float64)
    # amplitude
    rng = np.random.default_rng(seed)
    vec = rng.uniform(-1.0, 1.0, size=program.num_inputs)
    norm = np.linalg.norm(vec)
    if norm == 0.0:  # astronomically unlikely; guard anyway
        vec = np.ones(program.num_inputs, dtype=np.float64)
        norm = np.linalg.norm(vec)
    return vec / norm


def _state_qnode(ir: CircuitIR, program: CircuitProgram) -> qml.QNode:
    dev = qml.device("default.qubit", wires=ir.n_qubits)

    def state_circuit(inputs, weights):
        # apply_program_ops queues ONLY gates (no observable operators),
        # so the prepared state is not corrupted by queued measurements.
        apply_program_ops(program, inputs, weights)
        return qml.state()

    return qml.QNode(state_circuit, dev, diff_method=None)


def statevector(
    ir: CircuitIR,
    inputs: np.ndarray,
    weights: np.ndarray,
    program: CircuitProgram | None = None,
) -> np.ndarray:
    """The PennyLane statevector of `ir` at the given inputs and weights."""
    program = program if program is not None else build_program(ir)
    qnode = _state_qnode(ir, program)
    return np.asarray(qnode(np.asarray(inputs), np.asarray(weights)), dtype=np.complex128)


def statevector_qiskit(
    ir: CircuitIR,
    inputs: np.ndarray,
    weights: np.ndarray,
    program: CircuitProgram | None = None,
) -> np.ndarray:
    """The Qiskit statevector of `ir` — the independent cross-backend reference.

    Qiskit and PennyLane use opposite qubit-ordering conventions for the
    statevector index; `to_qiskit_bound` already builds the circuit
    through the shared wire layer, and here we reverse the qubit order so
    the returned amplitude vector is indexed identically to PennyLane's
    (little-endian vs big-endian reconciliation), making the two directly
    comparable up to global phase.
    """
    from qiskit.quantum_info import Statevector

    program = program if program is not None else build_program(ir)
    circuit = to_qiskit_bound(ir, inputs=list(inputs), weights=list(weights), program=program)
    # Reverse qubit order so index ordering matches PennyLane's convention.
    sv = Statevector(circuit.reverse_bits())
    return np.asarray(sv.data, dtype=np.complex128)


def fidelity(state_a: np.ndarray, state_b: np.ndarray) -> float:
    """|<a|b>|^2 — global-phase invariant by construction."""
    overlap = np.vdot(state_a, state_b)
    return float(np.abs(overlap) ** 2)
