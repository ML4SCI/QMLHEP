"""Capacity-controlled search space (experiment: capacity-controlled T1 v1).

**Why this exists.** In the legacy unrestricted pilot the classical embedding
width is `raw_feature_dim -> circuit.num_inputs`, and `num_inputs` is the number
of angle-encoding wires or `2**n_wires` for amplitude encoding. So circuits with
different encodings/qubit counts had wildly different *total trainable parameter
counts* (audited range 24-22,609, a 942x swing). "Which search arm won" was
therefore confounded by model capacity, not isolated to VQC architecture.

This module defines a search space in which **every** candidate has *identical*
capacity, so that only the internal VQC architecture varies:

- fixed 4 qubits;
- fixed angle encoding, gate RY, on all 4 wires  -> embed is always Linear(21->4);
- fixed Z measurement on all 4 wires             -> head is always Linear(4->1);
- **exactly** `QUANTUM_PARAM_COUNT` trainable quantum angles;
- fixed structural limits (depth / gates / two-qubit gates / blocks).

Only these vary: parameterized gate *types* (RX/RY/RZ vs CRZ), fixed gate types
(H/CNOT/CZ), gate ordering, entanglement connectivity, and layer organization.

**Representation.** A candidate is a `ControlledGenome`: an ordered list of block
tokens, of which *exactly* `N_PARAM_BLOCKS` are parameterized blocks (each
contributing `PARAM_BLOCK_SIZE` angles) and the rest are zero-parameter free
blocks. Building the `CircuitIR` from a genome, and every genome operator
(random draw, greedy neighbor, mutation, crossover), preserves the exact
parameter count **by construction** — so no operator needs rejection sampling to
hit the count. `validate_controlled` is an independent second gate that the
shared harness runs on every proposal, so any bug still surfaces as a
transparently-recorded INVALID rather than a silent repair.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from pydantic import BaseModel

from llm_vqc.ir.expand import build_program
from llm_vqc.ir.metrics import circuit_cost_summary
from llm_vqc.ir.schema import (
    CircuitIR,
    EncodingSpec,
    EntangleLayer,
    Layer,
    MeasurementSpec,
    RotationLayer,
)
from llm_vqc.ir.validators import ValidationIssue

# --- Fixed controlled configuration ----------------------------------------

SPACE_NAME = "T1_capacity_controlled_v1"
N_QUBITS = 4
ENCODING_GATE = "RY"
MEASUREMENT_OBSERVABLE = "Z"
RAW_FEATURE_DIM = 21  # T1

#: Exactly this many trainable quantum angles in every candidate. Chosen (see
#: SPACE_SPEC.md) as the smallest multiple of PARAM_BLOCK_SIZE that yields
#: N_PARAM_BLOCKS>=3 -> lets a candidate mix single-qubit rotations AND multiple
#: CRZ entangling layers, i.e. real entanglement variation among *parameterized*
#: gates, while staying computationally light.
QUANTUM_PARAM_COUNT = 12
#: A parameterized block spans all 4 wires: a single-qubit rotation layer (one of
#: RX/RY/RZ on every wire = 4 angles) or a CRZ ring (4 edges = 4 angles).
PARAM_BLOCK_SIZE = N_QUBITS  # = 4
N_PARAM_BLOCKS = QUANTUM_PARAM_COUNT // PARAM_BLOCK_SIZE  # = 3

MAX_FREE_BLOCKS = 3
MAX_BLOCKS = N_PARAM_BLOCKS + MAX_FREE_BLOCKS  # = 6

# Structural limits (identical for every arm). Set safely above what the
# generator can produce (verified in tests/experiment logs) so the generator is
# always valid, while still rejecting pathological hand-built circuits.
MAX_DEPTH = 30
MAX_TOTAL_GATES = 32
MAX_TWO_QUBIT_GATES = 24

# Derived, fixed classical capacity (asserted in tests):
#   embed Linear(21->4) = 21*4 + 4 = 88 ; head Linear(4->1) = 4 + 1 = 5.
EMBED_PARAMS = RAW_FEATURE_DIM * N_QUBITS + N_QUBITS   # 88
HEAD_PARAMS = N_QUBITS * 1 + 1                          # 5
TOTAL_TRAINABLE_PARAMS = EMBED_PARAMS + QUANTUM_PARAM_COUNT + HEAD_PARAMS  # 105

# --- Block token vocabulary -------------------------------------------------

#: Parameterized block kinds (each = PARAM_BLOCK_SIZE angles).
PARAM_BLOCK_KINDS = ("RX", "RY", "RZ", "CRZ")
#: Free (zero-parameter) block templates: (kind, gate, pattern, center).
FREE_BLOCK_TEMPLATES = (
    ("entangle", "CNOT", "ring", None),
    ("entangle", "CNOT", "line", None),
    ("entangle", "CNOT", "star", 0),
    ("entangle", "CZ", "ring", None),
    ("entangle", "CZ", "line", None),
    ("entangle", "CZ", "star", 0),
    ("hadamard", "H", None, None),
)


class ParamBlock(BaseModel):
    role: str = "param"
    kind: str  # one of PARAM_BLOCK_KINDS


class FreeBlock(BaseModel):
    role: str = "free"
    template_index: int  # index into FREE_BLOCK_TEMPLATES


Block = ParamBlock | FreeBlock


@dataclass
class ControlledGenome:
    """An ordered list of blocks with exactly N_PARAM_BLOCKS parameterized blocks."""

    blocks: list[Block] = field(default_factory=list)

    def n_param_blocks(self) -> int:
        return sum(1 for b in self.blocks if b.role == "param")

    def n_free_blocks(self) -> int:
        return sum(1 for b in self.blocks if b.role == "free")

    def to_dict(self) -> dict:
        return {"blocks": [b.model_dump() for b in self.blocks]}

    @classmethod
    def from_dict(cls, d: dict) -> ControlledGenome:
        blocks: list[Block] = []
        for b in d["blocks"]:
            blocks.append(ParamBlock(**b) if b.get("role") == "param" else FreeBlock(**b))
        return cls(blocks=blocks)


# --- Genome -> CircuitIR -----------------------------------------------------

def _param_block_to_layer(kind: str) -> Layer:
    if kind == "CRZ":
        return EntangleLayer(pattern="ring", gate="CRZ", wires="all")
    return RotationLayer(gates=[kind], wires="all")


def _free_block_to_layer(template_index: int) -> Layer:
    kind, gate, pattern, center = FREE_BLOCK_TEMPLATES[template_index]
    if kind == "hadamard":
        return RotationLayer(gates=["H"], wires="all")
    if pattern == "star":
        return EntangleLayer(pattern="star", gate=gate, wires="all", center=center)
    return EntangleLayer(pattern=pattern, gate=gate, wires="all")


def genome_to_ir(genome: ControlledGenome) -> CircuitIR:
    layers: list[Layer] = []
    for b in genome.blocks:
        if b.role == "param":
            layers.append(_param_block_to_layer(b.kind))
        else:
            layers.append(_free_block_to_layer(b.template_index))
    return CircuitIR(
        n_qubits=N_QUBITS,
        encoding=EncodingSpec(type="angle", gate=ENCODING_GATE, wires="all"),
        layers=layers,
        measurements=MeasurementSpec(observable=MEASUREMENT_OBSERVABLE, wires="all"),
    )


# --- The controlled invariant validator -------------------------------------

def validate_controlled(ir: CircuitIR) -> list[ValidationIssue]:
    """Return every controlled-space invariant this IR violates (empty = valid).

    Never repairs. The shared harness records a non-empty result as a
    transparently-logged INVALID proposal, exactly like any other invalid IR.
    """
    issues: list[ValidationIssue] = []

    def bad(code: str, msg: str, path: str = "") -> None:
        issues.append(ValidationIssue(code=f"controlled.{code}", message=msg, path=path))

    if ir.n_qubits != N_QUBITS:
        bad("n_qubits", f"must be exactly {N_QUBITS} qubits, got {ir.n_qubits}", "n_qubits")

    enc = ir.encoding
    if enc.type != "angle":
        bad("encoding_type", f"encoding must be 'angle', got {enc.type!r} "
            "(amplitude encoding is forbidden in the controlled space)", "encoding.type")
    if enc.gate != ENCODING_GATE:
        bad("encoding_gate", f"encoding gate must be {ENCODING_GATE!r}, got {enc.gate!r}",
            "encoding.gate")
    if enc.wires != "all":
        bad("encoding_wires", f"encoding wires must be 'all', got {enc.wires!r}", "encoding.wires")
    if enc.reupload != 0:
        bad("encoding_reupload", f"encoding reupload must be 0, got {enc.reupload}",
            "encoding.reupload")

    m = ir.measurements
    if m.observable != MEASUREMENT_OBSERVABLE:
        bad("measurement_obs", f"measurement observable must be {MEASUREMENT_OBSERVABLE!r}, "
            f"got {m.observable!r}", "measurements.observable")
    if m.wires != "all":
        bad("measurement_wires", f"measurement wires must be 'all', got {m.wires!r}",
            "measurements.wires")

    # No repeat blocks in the controlled space (keeps structure/counting flat).
    for i, layer in enumerate(ir.layers):
        if getattr(layer, "type", None) == "repeat":
            bad("no_repeat", "repeat blocks are not allowed in the controlled space",
                f"layers[{i}]")
    if len(ir.layers) > MAX_BLOCKS:
        bad("too_many_blocks", f"{len(ir.layers)} blocks exceeds MAX_BLOCKS={MAX_BLOCKS}", "layers")

    # Exact quantum parameter count and structural limits require expansion;
    # only attempt when the IR is structurally sane so far (n_qubits ok).
    if ir.n_qubits == N_QUBITS:
        try:
            program = build_program(ir)
            cost = circuit_cost_summary(ir, program)
        except Exception as exc:  # noqa: BLE001 - surface as an invalid, not a crash
            bad("uncompilable", f"could not expand/compile: {exc}")
            return issues
        if program.num_parameters != QUANTUM_PARAM_COUNT:
            bad("param_count", f"must have exactly {QUANTUM_PARAM_COUNT} trainable quantum "
                f"params, got {program.num_parameters}")
        if program.num_inputs != N_QUBITS:
            bad("encoding_width", f"encoding input width must be {N_QUBITS} (fixes embed "
                f"Linear(21->{N_QUBITS})), got {program.num_inputs}")
        if len(program.measurements) != N_QUBITS:
            bad("measurement_width", f"must measure {N_QUBITS} wires (fixes head "
                f"Linear({N_QUBITS}->1)), got {len(program.measurements)}")
        if cost.depth > MAX_DEPTH:
            bad("depth", f"depth {cost.depth} exceeds MAX_DEPTH={MAX_DEPTH}")
        if cost.gate_count > MAX_TOTAL_GATES:
            bad("gate_count", f"gate_count {cost.gate_count} exceeds "
                f"MAX_TOTAL_GATES={MAX_TOTAL_GATES}")
        if cost.two_qubit_gate_count > MAX_TWO_QUBIT_GATES:
            bad("two_qubit", f"two_qubit_gate_count {cost.two_qubit_gate_count} exceeds "
                f"MAX_TWO_QUBIT_GATES={MAX_TWO_QUBIT_GATES}")

    return issues


def is_controlled_valid(ir: CircuitIR) -> bool:
    return not validate_controlled(ir)


# --- Genome operators (validity + exact param count by construction) ---------

def random_genome(rng: np.random.Generator) -> ControlledGenome:
    param_blocks = [ParamBlock(kind=str(rng.choice(PARAM_BLOCK_KINDS)))
                    for _ in range(N_PARAM_BLOCKS)]
    n_free = int(rng.integers(0, MAX_FREE_BLOCKS + 1))
    free_blocks = [FreeBlock(template_index=int(rng.integers(0, len(FREE_BLOCK_TEMPLATES))))
                   for _ in range(n_free)]
    blocks: list[Block] = param_blocks + free_blocks
    rng.shuffle(blocks)
    return ControlledGenome(blocks=blocks)


def deterministic_seed_genome() -> ControlledGenome:
    """The fixed valid circuit every controlled greedy run grows from:
    three RY rotation layers (12 angles), no free blocks."""
    return ControlledGenome(blocks=[ParamBlock(kind="RY") for _ in range(N_PARAM_BLOCKS)])


def _param_positions(genome: ControlledGenome) -> list[int]:
    return [i for i, b in enumerate(genome.blocks) if b.role == "param"]


def _free_positions(genome: ControlledGenome) -> list[int]:
    return [i for i, b in enumerate(genome.blocks) if b.role == "free"]


def neighbors(genome: ControlledGenome, rng: np.random.Generator) -> list[ControlledGenome]:
    """All single-edit neighbors that preserve the controlled invariants.

    Edits: retype a parameterized block; retype a free block; add a free block
    (if under the cap); remove a free block; swap two adjacent blocks. Every
    result has exactly N_PARAM_BLOCKS parameterized blocks (12 angles).
    """
    out: list[ControlledGenome] = []

    # retype parameterized blocks
    for pos in _param_positions(genome):
        for kind in PARAM_BLOCK_KINDS:
            if kind != genome.blocks[pos].kind:
                blocks = [b.model_copy() for b in genome.blocks]
                blocks[pos] = ParamBlock(kind=kind)
                out.append(ControlledGenome(blocks=blocks))

    # retype free blocks
    for pos in _free_positions(genome):
        for ti in range(len(FREE_BLOCK_TEMPLATES)):
            if ti != genome.blocks[pos].template_index:
                blocks = [b.model_copy() for b in genome.blocks]
                blocks[pos] = FreeBlock(template_index=ti)
                out.append(ControlledGenome(blocks=blocks))

    # add a free block (any position, any template) if under cap
    if genome.n_free_blocks() < MAX_FREE_BLOCKS:
        for insert_at in range(len(genome.blocks) + 1):
            for ti in range(len(FREE_BLOCK_TEMPLATES)):
                blocks = [b.model_copy() for b in genome.blocks]
                blocks.insert(insert_at, FreeBlock(template_index=ti))
                out.append(ControlledGenome(blocks=blocks))

    # remove a free block
    for pos in _free_positions(genome):
        blocks = [b.model_copy() for i, b in enumerate(genome.blocks) if i != pos]
        out.append(ControlledGenome(blocks=blocks))

    # swap adjacent blocks
    for i in range(len(genome.blocks) - 1):
        blocks = [b.model_copy() for b in genome.blocks]
        blocks[i], blocks[i + 1] = blocks[i + 1], blocks[i]
        out.append(ControlledGenome(blocks=blocks))

    return out


def sample_neighbor(genome: ControlledGenome, rng: np.random.Generator) -> ControlledGenome:
    candidates = neighbors(genome, rng)
    if not candidates:
        return genome
    return candidates[int(rng.integers(0, len(candidates)))]


def mutate(genome: ControlledGenome, rng: np.random.Generator, mutation_rate: float) -> ControlledGenome:
    """Apply zero or one invariant-preserving edit with probability mutation_rate."""
    if rng.random() > mutation_rate:
        return genome
    return sample_neighbor(genome, rng)


def crossover(a: ControlledGenome, b: ControlledGenome, rng: np.random.Generator) -> ControlledGenome:
    """Recombine two genomes, then REPAIR to exactly N_PARAM_BLOCKS param blocks.

    One-point crossover on the block lists can yield the wrong number of
    parameterized blocks; the repair restores exactly N_PARAM_BLOCKS by taking
    the child's parameterized blocks in order and padding from parent A's if
    short / truncating if long, and clamping free blocks to the cap. The repair
    only ever changes *counts*, never introduces an out-of-space block type, so
    the scientific search space is unchanged.
    """
    cut_a = int(rng.integers(0, len(a.blocks) + 1))
    cut_b = int(rng.integers(0, len(b.blocks) + 1))
    child = [x.model_copy() for x in a.blocks[:cut_a]] + [y.model_copy() for y in b.blocks[cut_b:]]

    params = [x for x in child if x.role == "param"]
    frees = [x for x in child if x.role == "free"]

    # restore exactly N_PARAM_BLOCKS parameterized blocks
    if len(params) > N_PARAM_BLOCKS:
        params = params[:N_PARAM_BLOCKS]
    while len(params) < N_PARAM_BLOCKS:
        donor = [x for x in a.blocks if x.role == "param"]
        params.append(donor[len(params) % len(donor)].model_copy())
    # clamp free blocks to cap
    frees = frees[:MAX_FREE_BLOCKS]

    blocks: list[Block] = params + frees
    rng.shuffle(blocks)
    return ControlledGenome(blocks=blocks)
