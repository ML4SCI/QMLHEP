"""Typed circuit intermediate representation (IR) schema.

This module defines *structural* typing only: field names, literal enums of
allowed values, and basic per-field numeric bounds that pydantic can check
context-free. Cross-field and index-bound semantic rules (e.g. "a `star`
pattern requires `center`", "wire indices must be < n_qubits") are
deliberately NOT implemented here — they live in `llm_vqc.ir.validators`,
which collects *all* violations in one pass rather than failing on the
first one, which is what a search algorithm or LLM agent needs from
feedback (see that module's docstring).

Design constants and gate sets are documented in `llm_vqc/ir/README.md`.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

# --- Constants -------------------------------------------------------------

MIN_QUBITS = 2
MAX_QUBITS = 10

# Parameterized single-qubit rotation gates (consume one flat-vector
# parameter per application) plus one fixed non-parameterized gate (H),
# needed to faithfully re-encode published reference circuits that use a
# Hadamard superposition-initialization layer (see llm_vqc/ir/README.md,
# "Grammar extensions beyond the master plan's illustrative example").
ROTATION_GATE_NAMES = ("RX", "RY", "RZ", "H")
PARAMETERIZED_ROTATION_GATES = frozenset({"RX", "RY", "RZ"})
NONPARAMETERIZED_ROTATION_GATES = frozenset({"H"})

ENTANGLE_GATE_NAMES = ("CNOT", "CZ", "CRZ")
PARAMETERIZED_ENTANGLE_GATES = frozenset({"CRZ"})
NONPARAMETERIZED_ENTANGLE_GATES = frozenset({"CNOT", "CZ"})

ENTANGLE_PATTERNS = ("ring", "line", "star", "all_to_all", "pairs", "none")

ENCODING_TYPES = ("angle", "amplitude")
ENCODING_GATE_NAMES = ("RX", "RY", "RZ")

OBSERVABLES = ("X", "Y", "Z")

RotationGateName = Literal["RX", "RY", "RZ", "H"]
EntangleGateName = Literal["CNOT", "CZ", "CRZ"]
EntanglePattern = Literal["ring", "line", "star", "all_to_all", "pairs", "none"]
EncodingType = Literal["angle", "amplitude"]
EncodingGateName = Literal["RX", "RY", "RZ"]
Observable = Literal["X", "Y", "Z"]

# A wire group is either the literal string "all" (meaning every wire
# 0..n_qubits-1) or an explicit, proposer-specified list of qubit indices.
# This is the Phase 1 interpretation of the master plan's "arbitrary gate
# subsets and wire groups" requirement — see README for the rationale
# (an earlier illustrative example in the master plan uses a bare string
# "data" as a wire-group name without ever defining a named-group
# mini-language; explicit index lists are strictly more expressive and
# unambiguous, and cover that example by substitution).
WireSpec = Literal["all"] | list[int]


class EncodingSpec(BaseModel):
    """How classical input data is embedded into the circuit."""

    model_config = ConfigDict(extra="forbid")

    type: EncodingType
    gate: EncodingGateName | None = None
    wires: WireSpec = "all"
    reupload: int = Field(default=0, ge=0)


class RotationLayer(BaseModel):
    """Apply a fixed sequence of rotation gates to each wire in `wires`."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["rot"] = "rot"
    gates: list[RotationGateName] = Field(min_length=1)
    wires: WireSpec = "all"


class EntangleLayer(BaseModel):
    """Apply a two-qubit entangling gate according to a structural pattern."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["entangle"] = "entangle"
    pattern: EntanglePattern
    gate: EntangleGateName
    wires: WireSpec = "all"
    center: int | None = None
    pairs: list[tuple[int, int]] | None = None


NonRepeatLayer = Annotated[
    RotationLayer | EntangleLayer, Field(discriminator="type")
]


class RepeatBlock(BaseModel):
    """Repeat a fixed sub-sequence of (non-repeat) layers `times` times.

    Nesting a RepeatBlock inside another RepeatBlock's body is intentionally
    not supported in Phase 1 (kept out of scope to bound complexity; see
    README "Known limitations"). Use a flat `times` count instead.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["repeat"] = "repeat"
    times: int = Field(ge=1)
    body: list[NonRepeatLayer] = Field(min_length=1)


Layer = Annotated[
    RotationLayer | EntangleLayer | RepeatBlock, Field(discriminator="type")
]


class MeasurementSpec(BaseModel):
    """Measure a single Pauli observable on each wire in `wires`."""

    model_config = ConfigDict(extra="forbid")

    observable: Observable = "Z"
    wires: WireSpec = "all"


class CircuitIR(BaseModel):
    """A complete, backend-agnostic variational circuit proposal."""

    model_config = ConfigDict(extra="forbid")

    n_qubits: int = Field(ge=MIN_QUBITS, le=MAX_QUBITS)
    encoding: EncodingSpec
    layers: list[Layer] = Field(default_factory=list)
    measurements: MeasurementSpec
    metadata: dict[str, str] = Field(default_factory=dict)
