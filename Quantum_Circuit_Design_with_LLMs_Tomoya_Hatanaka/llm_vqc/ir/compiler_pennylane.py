"""IR -> PennyLane compiler.

Walks the same `CircuitProgram` (`llm_vqc.ir.expand.build_program`) that
the Qiskit compiler walks, so the two backends cannot silently diverge on
gate order or parameter assignment.

Signature convention: the compiled circuit function takes `(inputs,
weights)`, matching the `circuit(inputs, weights)` PennyLane QNode
convention used throughout Knipfer et al. 2026 (the mentors' own prior
work) and its tool docstrings — chosen deliberately for continuity with
that reference, not invented fresh.

This module only builds circuits; it does not train them, compute losses,
or touch task-specific data (that is explicitly evaluation-harness/Phase 2
territory — see llm_vqc/ir/README.md "What Phase 1 does not do").
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pennylane as qml

from llm_vqc.ir.expand import (
    CircuitProgram,
    EntangleInstruction,
    RotationInstruction,
    build_program,
)
from llm_vqc.ir.schema import CircuitIR

_ROTATION_GATES: dict[str, Callable[..., Any]] = {"RX": qml.RX, "RY": qml.RY, "RZ": qml.RZ}
_NONPARAM_ROTATION_GATES: dict[str, Callable[..., Any]] = {"H": qml.Hadamard}
_ENTANGLE_GATES: dict[str, Callable[..., Any]] = {"CNOT": qml.CNOT, "CZ": qml.CZ}
_ENTANGLE_PARAM_GATES: dict[str, Callable[..., Any]] = {"CRZ": qml.CRZ}
_OBSERVABLES: dict[str, Callable[..., Any]] = {"X": qml.PauliX, "Y": qml.PauliY, "Z": qml.PauliZ}


class CompilerError(Exception):
    """Raised when a CircuitProgram cannot be compiled or bound."""


def apply_program_ops(
    program: CircuitProgram, inputs: Sequence[float], weights: Sequence[float]
) -> None:
    """Queue only the circuit's *gates* (encoding + body) into the active
    PennyLane tape — no measurement, no observable.

    Use this when you want the prepared state or a custom measurement,
    rather than the IR's own measurement spec. It is critical that this
    does NOT create bare observable operators: instantiating e.g.
    ``qml.PauliZ(0)`` inside a QNode's quantum function queues it as a
    *gate*, which would silently corrupt the prepared state. `to_qnode`
    and `build_circuit_fn` layer the measurement on top of this.
    """
    _apply_program(program, inputs, weights)


def _apply_program(
    program: CircuitProgram, inputs: Sequence[float], weights: Sequence[float]
) -> None:
    if isinstance(inputs, list):
        # Plain lists don't support `[..., i]` ellipsis indexing (needed
        # below for batch-safe feature selection); arrays and tensors do.
        # A bare list of floats/ints is never part of a torch autograd
        # graph, so converting it to a numpy array here is lossless.
        inputs = np.asarray(inputs)

    if program.encoding_type == "angle":
        for instr in program.encoding_instructions:
            # `...` (not a bare index) so this is correct whether `inputs`
            # is a single sample (1D) or a batch (2D, batch dim first) —
            # see DECISIONS.md Phase 2 "Changes to the Phase 1 IR" for why
            # this matters: bare `inputs[i]` silently sliced the *batch*
            # axis instead of the *feature* axis for batched torch input,
            # producing wrong (not erroring) results.
            _ROTATION_GATES[instr.gate](inputs[..., instr.input_index], wires=instr.wire)
    elif program.encoding_type == "amplitude":
        qml.AmplitudeEmbedding(
            inputs, wires=list(program.amplitude_wires), normalize=True, pad_with=0.0
        )
    else:  # pragma: no cover - schema already restricts this
        raise CompilerError(f"unknown encoding type: {program.encoding_type!r}")

    for instr in program.body:
        if isinstance(instr, RotationInstruction):
            if instr.param_index is not None:
                _ROTATION_GATES[instr.gate](weights[instr.param_index], wires=instr.wire)
            else:
                _NONPARAM_ROTATION_GATES[instr.gate](wires=instr.wire)
        elif isinstance(instr, EntangleInstruction):
            if instr.param_index is not None:
                _ENTANGLE_PARAM_GATES[instr.gate](
                    weights[instr.param_index], wires=[instr.control, instr.target]
                )
            else:
                _ENTANGLE_GATES[instr.gate](wires=[instr.control, instr.target])
        else:  # pragma: no cover - exhaustive over BodyInstruction
            raise CompilerError(f"unknown instruction: {instr!r}")


def build_circuit_fn(ir: CircuitIR, program: CircuitProgram | None = None) -> Callable[..., list]:
    """Return a plain `circuit(inputs, weights)` function (no @qml.qnode).

    Undecorated so callers can wrap it with their own device, interface,
    or diff_method (e.g. a future evaluation harness choosing torch vs
    autograd). Use `to_qnode` below for a ready-to-use default.
    """
    program = program if program is not None else build_program(ir)

    def circuit(inputs: Sequence[float], weights: Sequence[float]) -> list:
        _apply_program(program, inputs, weights)
        return [_OBSERVABLES[m.observable](m.wire) for m in program.measurements]

    return circuit


def to_qnode(
    ir: CircuitIR,
    device: qml.devices.Device | None = None,
    diff_method: str = "best",
) -> qml.QNode:
    """Compile ir into a ready-to-call PennyLane QNode: `qnode(inputs, weights)`.

    Returns expectation values (one per measured wire), matching the
    Knipfer et al. tool convention `[qml.expval(...), ...]`.
    """
    program = build_program(ir)
    circuit_fn = build_circuit_fn(ir, program)

    def qnode_circuit(inputs: Sequence[float], weights: Sequence[float]) -> list:
        return [qml.expval(obs) for obs in circuit_fn(inputs, weights)]

    if device is None:
        device = qml.device("default.qubit", wires=ir.n_qubits)
    return qml.QNode(qnode_circuit, device, diff_method=diff_method)
