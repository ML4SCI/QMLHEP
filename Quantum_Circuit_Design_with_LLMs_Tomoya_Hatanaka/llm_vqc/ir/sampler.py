"""Random IR sampler.

This is not merely a testing convenience: per LLM-VQC_MASTER_PLAN.md
Section 9 (Phase 1 deliverables), this sampler *is* the `random` search
arm's proposal generator. Every future search arm — informed or not —
draws circuits from the same IR grammar (Section 5.2), and this is the
uninformed baseline's draw function.

Guarantee: `sample_random_ir` always returns a valid `CircuitIR` (never a
raw dict that might fail validation). It builds a candidate, validates it
through the same `validate_proposal` every other proposal source must go
through, and resamples on the rare case a combination is invalid (e.g. an
`all_to_all` pattern degenerately drawn over a single wire) rather than
special-casing every combinatorial edge case by hand — the acceptance
criterion ("sampler produces 100% valid IRs") is enforced this way by
construction, and is also checked empirically in
`tests/test_ir_sampler.py` over many draws.
"""

from __future__ import annotations

import numpy as np

from llm_vqc.ir.schema import (
    ENCODING_GATE_NAMES,
    ENTANGLE_GATE_NAMES,
    ENTANGLE_PATTERNS,
    OBSERVABLES,
    ROTATION_GATE_NAMES,
    CircuitIR,
    EncodingSpec,
    EntangleLayer,
    Layer,
    MeasurementSpec,
    RepeatBlock,
    RotationLayer,
)
from llm_vqc.ir.schema import MAX_QUBITS as _MAX_QUBITS
from llm_vqc.ir.schema import MIN_QUBITS as _MIN_QUBITS
from llm_vqc.ir.validators import validate_proposal

MAX_RESAMPLE_ATTEMPTS = 50

_MAX_TOP_LEVEL_LAYERS = 5
_MAX_GATES_PER_ROTATION_LAYER = 3
_MAX_REPEAT_TIMES = 2
_MAX_REPEAT_BODY_LEN = 2
_REPEAT_LAYER_PROBABILITY = 0.15
_AMPLITUDE_ENCODING_PROBABILITY = 0.2


class SamplerError(Exception):
    """Raised if a valid IR cannot be produced within the resample budget."""


def _sample_wire_subset(rng: np.random.Generator, n_qubits: int, min_size: int = 1) -> list[int]:
    size = int(rng.integers(min_size, n_qubits + 1))
    return sorted(rng.choice(n_qubits, size=size, replace=False).tolist())


def _sample_wires_or_all(
    rng: np.random.Generator, n_qubits: int, min_size: int = 1
) -> list[int] | str:
    if rng.random() < 0.5:
        return "all"
    return _sample_wire_subset(rng, n_qubits, min_size=min_size)


def _sample_encoding(rng: np.random.Generator, n_qubits: int) -> EncodingSpec:
    if rng.random() < _AMPLITUDE_ENCODING_PROBABILITY:
        wires = _sample_wires_or_all(rng, n_qubits, min_size=1)
        return EncodingSpec(type="amplitude", wires=wires, reupload=0)
    gate = str(rng.choice(ENCODING_GATE_NAMES))
    wires = _sample_wires_or_all(rng, n_qubits, min_size=1)
    reupload = int(rng.integers(0, 2))
    return EncodingSpec(type="angle", gate=gate, wires=wires, reupload=reupload)


def _sample_rotation_layer(rng: np.random.Generator, n_qubits: int) -> RotationLayer:
    n_gates = int(rng.integers(1, _MAX_GATES_PER_ROTATION_LAYER + 1))
    gates = rng.choice(ROTATION_GATE_NAMES, size=n_gates, replace=True).tolist()
    wires = _sample_wires_or_all(rng, n_qubits, min_size=1)
    return RotationLayer(gates=gates, wires=wires)


def _sample_entangle_layer(rng: np.random.Generator, n_qubits: int) -> EntangleLayer:
    pattern = str(rng.choice(ENTANGLE_PATTERNS))
    gate = str(rng.choice(ENTANGLE_GATE_NAMES))

    if pattern == "star":
        wires = _sample_wire_subset(rng, n_qubits, min_size=2)
        center = int(rng.choice(wires))
        return EntangleLayer(pattern=pattern, gate=gate, wires=wires, center=center)
    if pattern == "pairs":
        n_pairs = int(rng.integers(1, 4))
        pairs = []
        for _ in range(n_pairs):
            control, target = rng.choice(n_qubits, size=2, replace=False)
            pairs.append((int(control), int(target)))
        return EntangleLayer(pattern=pattern, gate=gate, pairs=pairs)
    if pattern == "none":
        return EntangleLayer(pattern=pattern, gate=gate)
    # ring, line, all_to_all
    wires = _sample_wire_subset(rng, n_qubits, min_size=2)
    return EntangleLayer(pattern=pattern, gate=gate, wires=wires)


def _sample_layer(rng: np.random.Generator, n_qubits: int) -> Layer:
    if rng.random() < _REPEAT_LAYER_PROBABILITY:
        body_len = int(rng.integers(1, _MAX_REPEAT_BODY_LEN + 1))
        body = [_sample_non_repeat_layer(rng, n_qubits) for _ in range(body_len)]
        times = int(rng.integers(1, _MAX_REPEAT_TIMES + 1))
        return RepeatBlock(times=times, body=body)
    return _sample_non_repeat_layer(rng, n_qubits)


def _sample_non_repeat_layer(rng: np.random.Generator, n_qubits: int):
    if rng.random() < 0.5:
        return _sample_rotation_layer(rng, n_qubits)
    return _sample_entangle_layer(rng, n_qubits)


def _sample_measurement(rng: np.random.Generator, n_qubits: int) -> MeasurementSpec:
    observable = str(rng.choice(OBSERVABLES))
    wires = _sample_wires_or_all(rng, n_qubits, min_size=1)
    return MeasurementSpec(observable=observable, wires=wires)


def _sample_candidate(rng: np.random.Generator) -> CircuitIR:
    n_qubits = int(rng.integers(_MIN_QUBITS, _MAX_QUBITS + 1))
    encoding = _sample_encoding(rng, n_qubits)
    n_layers = int(rng.integers(0, _MAX_TOP_LEVEL_LAYERS + 1))
    layers = [_sample_layer(rng, n_qubits) for _ in range(n_layers)]
    measurements = _sample_measurement(rng, n_qubits)
    return CircuitIR(
        n_qubits=n_qubits, encoding=encoding, layers=layers, measurements=measurements
    )


def sample_single_layer(rng: np.random.Generator, n_qubits: int) -> RotationLayer | EntangleLayer:
    """Sample one non-repeat layer (rotation or entangle), from the exact
    same distribution `sample_random_ir` draws individual layers from.

    Exposed publicly (unlike the other `_sample_*` helpers) because the
    `greedy` and `evolutionary` search arms (Phase 4) need to draw a
    single-layer mutation/extension from the identical grammar the
    `random` arm uses -- per master plan Section 6.4, every arm must draw
    from the same IR grammar, so arm-specific layer generation would be an
    unmatchable, unfair action space.
    """
    return _sample_non_repeat_layer(rng, n_qubits)


def sample_random_ir(rng: np.random.Generator) -> CircuitIR:
    """Draw one valid, uniformly-structured random CircuitIR.

    `rng` must be an explicit `numpy.random.Generator` (e.g.
    `numpy.random.default_rng(seed)`) — no hidden global random state, so
    draws are fully reproducible given a seed, per the master plan's
    reproducibility requirements (Section 5.5 / 11).
    """
    for _ in range(MAX_RESAMPLE_ATTEMPTS):
        candidate = _sample_candidate(rng)
        result = validate_proposal(candidate)
        if result.valid:
            return result.ir
    raise SamplerError(
        f"failed to sample a valid CircuitIR within {MAX_RESAMPLE_ATTEMPTS} attempts"
    )
