"""IR -> Qiskit compiler and measurement-observable exporter.

Walks the same `CircuitProgram` (`llm_vqc.ir.expand.build_program`) that
the PennyLane compiler walks, so the two backends cannot silently diverge
on gate order or parameter assignment.

Stages are kept explicitly separate, per the Phase 1 design brief:

- `to_qiskit_symbolic`: state-prep circuit with `qiskit.circuit.Parameter`
  placeholders for weights (and, for angle encoding, for inputs too) —
  no numeric values bound. Suitable for inspection, export, and
  transpilation experiments that don't need concrete values yet.
- `to_qiskit_bound`: the same circuit with concrete, validated (correct
  length, all-finite) numeric values substituted in. This is
  "parameter binding" as a distinct stage from compilation.
- `qiskit_observables`: the measurement spec as a list of
  `SparsePauliOp`, independent of the state-prep circuit. Deliberately
  separate because Qiskit has no single object that bundles "circuit +
  what to measure on it" the way a PennyLane QNode does; keeping this
  split also keeps evaluation-metric concerns (which belong to a future
  Estimator-based evaluation harness, not here) out of the compiler.

Transpilation and hardware mapping are out of Phase 1 scope: these
functions return backend-native circuit objects only.
"""

from __future__ import annotations

from collections.abc import Sequence

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp

from llm_vqc.ir.compiler_pennylane import CompilerError
from llm_vqc.ir.expand import (
    CircuitProgram,
    EntangleInstruction,
    RotationInstruction,
    build_program,
)
from llm_vqc.ir.schema import CircuitIR

_ROTATION_METHODS = {"RX": "rx", "RY": "ry", "RZ": "rz"}
_NONPARAM_ROTATION_METHODS = {"H": "h"}
_ENTANGLE_METHODS = {"CNOT": "cx", "CZ": "cz"}
_ENTANGLE_PARAM_METHODS = {"CRZ": "crz"}


def _apply_body(circuit: QuantumCircuit, program: CircuitProgram, weight_params: Sequence) -> None:
    for instr in program.body:
        if isinstance(instr, RotationInstruction):
            if instr.param_index is not None:
                getattr(circuit, _ROTATION_METHODS[instr.gate])(
                    weight_params[instr.param_index], instr.wire
                )
            else:
                getattr(circuit, _NONPARAM_ROTATION_METHODS[instr.gate])(instr.wire)
        elif isinstance(instr, EntangleInstruction):
            if instr.param_index is not None:
                getattr(circuit, _ENTANGLE_PARAM_METHODS[instr.gate])(
                    weight_params[instr.param_index], instr.control, instr.target
                )
            else:
                getattr(circuit, _ENTANGLE_METHODS[instr.gate])(instr.control, instr.target)
        else:  # pragma: no cover - exhaustive over BodyInstruction
            raise CompilerError(f"unknown instruction: {instr!r}")


def to_qiskit_symbolic(ir: CircuitIR, program: CircuitProgram | None = None) -> QuantumCircuit:
    """State-prep circuit with symbolic Parameter placeholders (no bound values).

    Amplitude encoding has no symbolic representation here (see module
    docstring) — this raises CompilerError for `encoding.type ==
    "amplitude"`; use `to_qiskit_bound` with concrete input values instead.
    """
    program = program if program is not None else build_program(ir)
    if program.encoding_type == "amplitude":
        raise CompilerError(
            "to_qiskit_symbolic does not support amplitude encoding "
            "(no symbolic Qiskit state-preparation representation); "
            "use to_qiskit_bound with concrete input values instead."
        )

    circuit = QuantumCircuit(program.n_qubits)
    input_params = [Parameter(f"x_{i}") for i in range(program.num_inputs)]
    for instr in program.encoding_instructions:
        getattr(circuit, _ROTATION_METHODS[instr.gate])(
            input_params[instr.input_index], instr.wire
        )

    weight_params = [Parameter(f"theta_{i}") for i in range(program.num_parameters)]
    _apply_body(circuit, program, weight_params)
    return circuit


def to_qiskit_bound(
    ir: CircuitIR,
    inputs: Sequence[float],
    weights: Sequence[float],
    program: CircuitProgram | None = None,
) -> QuantumCircuit:
    """State-prep circuit with concrete, validated numeric values bound.

    Validates (raising CompilerError with a clear message on failure):
    - `len(weights) == program.num_parameters`
    - `len(inputs) == program.num_inputs` (for angle encoding: one value
      per encoding wire; for amplitude encoding: 2**len(amplitude_wires))
    - every value in `inputs` and `weights` is finite (no NaN/Inf)
    """
    program = program if program is not None else build_program(ir)

    if len(weights) != program.num_parameters:
        raise CompilerError(
            f"expected {program.num_parameters} weights, got {len(weights)}"
        )
    if len(inputs) != program.num_inputs:
        raise CompilerError(f"expected {program.num_inputs} inputs, got {len(inputs)}")
    for i, v in enumerate(weights):
        if not _is_finite(v):
            raise CompilerError(f"weights[{i}] is not finite: {v!r}")
    for i, v in enumerate(inputs):
        if not _is_finite(v):
            raise CompilerError(f"inputs[{i}] is not finite: {v!r}")

    circuit = QuantumCircuit(program.n_qubits)

    if program.encoding_type == "angle":
        for instr in program.encoding_instructions:
            getattr(circuit, _ROTATION_METHODS[instr.gate])(
                float(inputs[instr.input_index]), instr.wire
            )
    elif program.encoding_type == "amplitude":
        circuit.prepare_state(list(inputs), list(program.amplitude_wires), normalize=True)
    else:  # pragma: no cover - schema already restricts this
        raise CompilerError(f"unknown encoding type: {program.encoding_type!r}")

    _apply_body(circuit, program, [float(w) for w in weights])
    return circuit


def _is_finite(x: float) -> bool:
    try:
        fx = float(x)
    except (TypeError, ValueError):
        return False
    return fx == fx and fx not in (float("inf"), float("-inf"))  # NaN != NaN


_PAULI_LETTERS = {"X": "X", "Y": "Y", "Z": "Z"}


def qiskit_observables(
    ir: CircuitIR, program: CircuitProgram | None = None
) -> list[SparsePauliOp]:
    """The measurement spec as one single-qubit SparsePauliOp per measured wire.

    Each returned operator acts on the full `n_qubits`-qubit register with
    identity everywhere except the measured wire, so it can be used
    directly with a Qiskit Estimator against the state-prep circuits
    above without further padding.
    """
    program = program if program is not None else build_program(ir)
    observables = []
    for m in program.measurements:
        label = ["I"] * program.n_qubits
        # Qiskit Pauli labels are big-endian: index 0 is the *rightmost* char.
        label[program.n_qubits - 1 - m.wire] = _PAULI_LETTERS[m.observable]
        observables.append(SparsePauliOp("".join(label)))
    return observables
