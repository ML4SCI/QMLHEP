"""Deterministic linearization of a valid CircuitIR into a flat instruction program.

This is the single source of truth for: repeat-block flattening, entangle
pattern -> (control, target) pair resolution, and canonical parameter-slot
ordering. Both backend compilers (`compiler_pennylane.py`,
`compiler_qiskit.py`) walk the same `CircuitProgram` produced here, so they
cannot silently diverge on gate order or parameter assignment — this is
what makes "the same IR input produces semantically equivalent compiled
circuits across backends" true by construction rather than by convention.

This module assumes its input is already a valid, pydantic-parsed
`CircuitIR` (post `llm_vqc.ir.validators.validate_proposal`). It does not
re-validate; callers are expected to validate first.
"""

from __future__ import annotations

from dataclasses import dataclass

from llm_vqc.ir.schema import (
    NONPARAMETERIZED_ROTATION_GATES,
    PARAMETERIZED_ENTANGLE_GATES,
    PARAMETERIZED_ROTATION_GATES,
    CircuitIR,
    EncodingSpec,
    EntangleLayer,
    Layer,
    RotationLayer,
)
from llm_vqc.ir.wires import resolve_wires


@dataclass(frozen=True)
class EncodingInstruction:
    gate: str
    wire: int
    input_index: int


@dataclass(frozen=True)
class RotationInstruction:
    gate: str
    wire: int
    param_index: int | None


@dataclass(frozen=True)
class EntangleInstruction:
    gate: str
    control: int
    target: int
    param_index: int | None


@dataclass(frozen=True)
class MeasurementInstruction:
    observable: str
    wire: int


BodyInstruction = RotationInstruction | EntangleInstruction


@dataclass(frozen=True)
class CircuitProgram:
    """Fully linearized, backend-agnostic circuit program."""

    n_qubits: int
    encoding_type: str
    encoding_instructions: tuple[EncodingInstruction, ...]
    amplitude_wires: tuple[int, ...] | None
    body: tuple[BodyInstruction, ...]
    measurements: tuple[MeasurementInstruction, ...]
    num_parameters: int
    num_inputs: int


def expand_layers(layers: list[Layer]) -> list[RotationLayer | EntangleLayer]:
    """Flatten RepeatBlocks into repeated copies of their body, in order."""
    expanded: list[RotationLayer | EntangleLayer] = []
    for layer in layers:
        if layer.type == "repeat":
            for _ in range(layer.times):
                expanded.extend(layer.body)
        else:
            expanded.append(layer)
    return expanded


def entangle_pairs_for_layer(layer: EntangleLayer, n_qubits: int) -> list[tuple[int, int]]:
    """Resolve an EntangleLayer's pattern into an ordered (control, target) list.

    Definitions (deterministic, order matters for non-commuting gates):
    - "none": no pairs.
    - "pairs": the proposer-supplied list, verbatim.
    - "ring": consecutive pairs over the resolved wire list, including the
      wraparound edge (last, first) — n_wires edges for n_wires wires.
    - "line": consecutive pairs without wraparound — n_wires - 1 edges.
    - "star": (center, w) for every resolved wire w != center.
    - "all_to_all": every distinct unordered pair (i, j), i < j in resolved
      wire *list position* order, emitted as (wires[i], wires[j]).
    """
    if layer.pattern == "none":
        return []
    if layer.pattern == "pairs":
        return [(c, t) for c, t in (layer.pairs or [])]

    wires = resolve_wires(layer.wires, n_qubits)
    if layer.pattern == "ring":
        return [(wires[i], wires[(i + 1) % len(wires)]) for i in range(len(wires))]
    if layer.pattern == "line":
        return [(wires[i], wires[i + 1]) for i in range(len(wires) - 1)]
    if layer.pattern == "star":
        return [(layer.center, w) for w in wires if w != layer.center]
    if layer.pattern == "all_to_all":
        return [
            (wires[i], wires[j]) for i in range(len(wires)) for j in range(i + 1, len(wires))
        ]
    raise ValueError(f"unknown entangle pattern: {layer.pattern!r}")  # pragma: no cover


def _encoding_wires(encoding: EncodingSpec, n_qubits: int) -> list[int]:
    return resolve_wires(encoding.wires, n_qubits)


def build_program(ir: CircuitIR) -> CircuitProgram:
    """Linearize a valid CircuitIR into a flat, deterministic CircuitProgram."""
    encoding_instructions: list[EncodingInstruction] = []
    amplitude_wires: tuple[int, ...] | None = None
    num_inputs = 0

    if ir.encoding.type == "angle":
        wires = _encoding_wires(ir.encoding, ir.n_qubits)
        num_inputs = len(wires)
        repeats = ir.encoding.reupload + 1
        for _ in range(repeats):
            for input_index, wire in enumerate(wires):
                encoding_instructions.append(
                    EncodingInstruction(gate=ir.encoding.gate, wire=wire, input_index=input_index)
                )
    elif ir.encoding.type == "amplitude":
        wires = _encoding_wires(ir.encoding, ir.n_qubits)
        amplitude_wires = tuple(wires)
        num_inputs = 2 ** len(wires)
    else:  # pragma: no cover - schema already restricts this
        raise ValueError(f"unknown encoding type: {ir.encoding.type!r}")

    body: list[BodyInstruction] = []
    param_index = 0
    for layer in expand_layers(ir.layers):
        if layer.type == "rot":
            wires = resolve_wires(layer.wires, ir.n_qubits)
            for gate in layer.gates:
                is_parameterized = gate in PARAMETERIZED_ROTATION_GATES
                for wire in wires:
                    idx = None
                    if is_parameterized:
                        idx = param_index
                        param_index += 1
                    elif gate not in NONPARAMETERIZED_ROTATION_GATES:  # pragma: no cover
                        raise ValueError(f"unknown rotation gate: {gate!r}")
                    body.append(RotationInstruction(gate=gate, wire=wire, param_index=idx))
        elif layer.type == "entangle":
            is_parameterized = layer.gate in PARAMETERIZED_ENTANGLE_GATES
            for control, target in entangle_pairs_for_layer(layer, ir.n_qubits):
                idx = None
                if is_parameterized:
                    idx = param_index
                    param_index += 1
                body.append(
                    EntangleInstruction(
                        gate=layer.gate, control=control, target=target, param_index=idx
                    )
                )
        else:  # pragma: no cover - schema already restricts this
            raise ValueError(f"unknown layer type: {layer.type!r}")

    measurement_wires = resolve_wires(ir.measurements.wires, ir.n_qubits)
    measurements = tuple(
        MeasurementInstruction(observable=ir.measurements.observable, wire=w)
        for w in measurement_wires
    )

    return CircuitProgram(
        n_qubits=ir.n_qubits,
        encoding_type=ir.encoding.type,
        encoding_instructions=tuple(encoding_instructions),
        amplitude_wires=amplitude_wires,
        body=tuple(body),
        measurements=measurements,
        num_parameters=param_index,
        num_inputs=num_inputs,
    )


def count_parameters(ir: CircuitIR) -> int:
    """Number of trainable parameters a valid CircuitIR requires to bind."""
    return build_program(ir).num_parameters


def count_gate_applications(ir: CircuitIR) -> int:
    """Total gate applications (encoding + body), used for size-limit checks."""
    program = build_program(ir)
    return len(program.encoding_instructions) + len(program.body)
