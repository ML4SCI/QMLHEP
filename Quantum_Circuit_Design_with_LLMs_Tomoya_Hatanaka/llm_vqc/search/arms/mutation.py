"""Hand-coded mutation and crossover operators for the `evolutionary` arm
(master plan Section 6.2: "add/remove/perturb layer, change pattern,
change measurement, grow/shrink qubits").

Every operator is defensive: it may produce an invalid `CircuitIR` (e.g.
shrinking qubits below an existing layer's wire indices), in which case
`mutate`/`crossover` fall back to returning the *original* (already-valid)
input unchanged rather than propagating an invalid candidate or raising.
This never hides a validation failure from the shared runner -- it only
means the mutation had no effect that round; the runner's own
`evaluate_candidate` call independently validates whatever the arm
proposes regardless.
"""

from __future__ import annotations

import numpy as np

from llm_vqc.ir.sampler import sample_single_layer
from llm_vqc.ir.schema import MAX_QUBITS, MIN_QUBITS, OBSERVABLES, CircuitIR, MeasurementSpec
from llm_vqc.ir.validators import validate_proposal

MUTATION_OPERATORS = (
    "add_layer",
    "remove_layer",
    "perturb_layer",
    "change_measurement",
    "grow_qubits",
    "shrink_qubits",
)


def _add_layer(ir: CircuitIR, rng: np.random.Generator) -> CircuitIR:
    new_layer = sample_single_layer(rng, ir.n_qubits)
    return ir.model_copy(update={"layers": [*ir.layers, new_layer]})


def _remove_layer(ir: CircuitIR, rng: np.random.Generator) -> CircuitIR:
    if not ir.layers:
        return ir
    index = int(rng.integers(0, len(ir.layers)))
    new_layers = list(ir.layers)
    del new_layers[index]
    return ir.model_copy(update={"layers": new_layers})


def _perturb_layer(ir: CircuitIR, rng: np.random.Generator) -> CircuitIR:
    if not ir.layers:
        return ir
    index = int(rng.integers(0, len(ir.layers)))
    if ir.layers[index].type == "repeat":
        return ir  # repeat-block internals are not perturbed (Phase 1 scope limit)
    new_layers = list(ir.layers)
    new_layers[index] = sample_single_layer(rng, ir.n_qubits)
    return ir.model_copy(update={"layers": new_layers})


def _change_measurement(ir: CircuitIR, rng: np.random.Generator) -> CircuitIR:
    observable = str(rng.choice(OBSERVABLES))
    return ir.model_copy(
        update={"measurements": MeasurementSpec(observable=observable, wires=ir.measurements.wires)}
    )


def _grow_qubits(ir: CircuitIR, rng: np.random.Generator) -> CircuitIR:
    if ir.n_qubits >= MAX_QUBITS:
        return ir
    return ir.model_copy(update={"n_qubits": ir.n_qubits + 1})


def _shrink_qubits(ir: CircuitIR, rng: np.random.Generator) -> CircuitIR:
    if ir.n_qubits <= MIN_QUBITS:
        return ir
    return ir.model_copy(update={"n_qubits": ir.n_qubits - 1})


_OPERATORS = {
    "add_layer": _add_layer,
    "remove_layer": _remove_layer,
    "perturb_layer": _perturb_layer,
    "change_measurement": _change_measurement,
    "grow_qubits": _grow_qubits,
    "shrink_qubits": _shrink_qubits,
}


def mutate(ir: CircuitIR, rng: np.random.Generator, mutation_rate: float) -> CircuitIR:
    """Apply zero or one mutation operator, with probability `mutation_rate`."""
    if rng.random() > mutation_rate:
        return ir
    operator_name = str(rng.choice(MUTATION_OPERATORS))
    candidate = _OPERATORS[operator_name](ir, rng)
    result = validate_proposal(candidate)
    return result.ir if result.valid else ir


def crossover(parent_a: CircuitIR, parent_b: CircuitIR, rng: np.random.Generator) -> CircuitIR:
    """Single-point crossover over the two parents' layer lists.

    The child keeps `parent_a`'s `n_qubits`/`encoding`/`measurements`
    (crossing those over independently would multiply edge cases for
    limited benefit at this scope); only the layer sequence is recombined.
    """
    cut = int(rng.integers(0, len(parent_a.layers) + 1))
    child_layers = [*parent_a.layers[:cut], *parent_b.layers[cut:]]
    candidate = parent_a.model_copy(update={"layers": child_layers})
    result = validate_proposal(candidate)
    return result.ir if result.valid else parent_a
