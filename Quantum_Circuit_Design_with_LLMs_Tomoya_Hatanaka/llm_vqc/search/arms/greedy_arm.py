"""The `greedy` search arm (master plan Section 6.2): "layer-wise growth:
start minimal, add the best of k sampled single-layer extensions each
round."

**Reproducible start:** every run begins from the same fixed minimal
circuit (`MIN_QUBITS` qubits, angle-RY encoding, no layers, Z measurement
on all wires) -- a constant, not a draw, so "start minimal" is
deterministic even before any RNG is consulted.

**Explicit neighborhood:** each round samples `k` single-layer extensions
of the current incumbent circuit, using `llm_vqc.ir.sampler.
sample_single_layer` -- the identical layer grammar every other arm draws
from (Section 6.4 parity).

**Every neighbor counted against budget:** each of the `k` neighbors is
one proposal through the shared runner/harness, so it consumes budget
exactly like any other arm's proposal (or is free if INVALID, same rule
for everyone).

**Deterministic tie-breaking:** the winner of a round is chosen with
`is_strictly_better`, which never lets a tie replace the first-seen
incumbent.

**Stops at budget exhaustion, not free exhaustive evaluation:** a round
in progress when the run's budget runs out simply never completes; the
shared runner's loop (not this arm) is what stops proposing, so no
"finish the current round for free" special case exists.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field

from llm_vqc.evaluation.seeds import derive_child_seed
from llm_vqc.ir.sampler import sample_single_layer
from llm_vqc.ir.schema import MIN_QUBITS, CircuitIR, EncodingSpec, Layer, MeasurementSpec
from llm_vqc.ir.validators import validate_proposal
from llm_vqc.search.arm import SearchArm
from llm_vqc.search.comparison import is_strictly_better
from llm_vqc.search.feedback import SearchFeedback

_NEIGHBOR_RESAMPLE_ATTEMPTS = 20


def minimal_ir(n_qubits: int = MIN_QUBITS) -> CircuitIR:
    """The fixed, deterministic starting circuit every `greedy` run grows from."""
    return CircuitIR(
        n_qubits=n_qubits,
        encoding=EncodingSpec(type="angle", gate="RY", wires="all"),
        layers=[],
        measurements=MeasurementSpec(observable="Z", wires="all"),
    )


def _sample_neighbor(current_ir: CircuitIR, rng: np.random.Generator) -> CircuitIR:
    """One single-layer extension of `current_ir`. Falls back to
    `current_ir` unchanged (itself always valid) if `_NEIGHBOR_RESAMPLE_ATTEMPTS`
    sampled extensions all fail validation -- vanishingly rare, but this
    guarantees `propose()` always returns something well-formed rather
    than raising out of a search arm's control flow."""
    for _ in range(_NEIGHBOR_RESAMPLE_ATTEMPTS):
        new_layer: Layer = sample_single_layer(rng, current_ir.n_qubits)
        candidate = current_ir.model_copy(update={"layers": [*current_ir.layers, new_layer]})
        result = validate_proposal(candidate)
        if result.valid:
            return result.ir
    return current_ir


class _RoundObservation(BaseModel):
    ir: CircuitIR
    val_metric: float


class GreedyArmState(BaseModel):
    seed: int
    n_proposed: int = 0
    current_ir: CircuitIR = Field(default_factory=minimal_ir)
    round_index: int = 0
    round_observations: list[_RoundObservation] = []
    best_hash: str | None = None
    best_metric: float | None = None


class GreedyArm(SearchArm[GreedyArmState]):
    name = "greedy"

    def __init__(self, lower_is_better: bool, k: int = 4, start_n_qubits: int = MIN_QUBITS) -> None:
        self.lower_is_better = lower_is_better
        self.k = k
        self.start_n_qubits = start_n_qubits

    def initialize(self, seed: int) -> GreedyArmState:
        return GreedyArmState(seed=seed, current_ir=minimal_ir(self.start_n_qubits))

    def propose(self, state: GreedyArmState) -> dict:
        rng = np.random.default_rng(
            derive_child_seed(
                state.seed, "greedy_propose", str(state.round_index), str(state.n_proposed)
            )
        )
        return _sample_neighbor(state.current_ir, rng).model_dump()

    def update_state(
        self, state: GreedyArmState, proposal: dict, feedback: SearchFeedback
    ) -> GreedyArmState:
        best_hash, best_metric = state.best_hash, state.best_metric
        if feedback.val_metric_value is not None and is_strictly_better(
            feedback.val_metric_value, best_metric, self.lower_is_better
        ):
            best_hash, best_metric = feedback.structural_hash, feedback.val_metric_value

        round_observations = list(state.round_observations)
        if feedback.val_metric_value is not None:
            round_observations.append(
                _RoundObservation(
                    ir=CircuitIR.model_validate(proposal),
                    val_metric=feedback.val_metric_value,
                )
            )

        current_ir = state.current_ir
        round_index = state.round_index
        if len(round_observations) >= self.k:
            best_obs = round_observations[0]
            for obs in round_observations[1:]:
                if is_strictly_better(obs.val_metric, best_obs.val_metric, self.lower_is_better):
                    best_obs = obs
            current_ir = best_obs.ir
            round_observations = []
            round_index += 1

        return state.model_copy(
            update={
                "n_proposed": state.n_proposed + 1,
                "current_ir": current_ir,
                "round_index": round_index,
                "round_observations": round_observations,
                "best_hash": best_hash,
                "best_metric": best_metric,
            }
        )

    def select_final(self, state: GreedyArmState) -> str | None:
        return state.best_hash

    def deserialize_state(self, raw_json: str) -> GreedyArmState:
        return GreedyArmState.model_validate_json(raw_json)
