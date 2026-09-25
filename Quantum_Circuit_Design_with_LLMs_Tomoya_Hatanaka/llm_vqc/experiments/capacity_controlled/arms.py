"""Controlled-space search arms: Random, Greedy, Evolutionary.

All three propose circuits from the *same* controlled grammar
(`llm_vqc.experiments.capacity_controlled.space`), so every candidate has
identical qubits/encoding/measurement/classical-capacity and *exactly*
`QUANTUM_PARAM_COUNT` quantum angles. Only the internal VQC architecture varies.

Each arm is a drop-in `SearchArm`: it proposes an IR-shaped dict and the shared
`SearchRunner`/harness validates it (including the controlled `extra_validator`),
trains it under the shared protocol + explicit init policy, and hands back
validation-only feedback. Arms operate on `ControlledGenome`s whose every
operator preserves the invariants *by construction*, so the arm's own proposals
are valid without free rejection sampling; the harness's controlled validator is
an independent second gate that records any escape as a transparent INVALID.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel

from llm_vqc.evaluation.seeds import derive_child_seed
from llm_vqc.experiments.capacity_controlled import space as S
from llm_vqc.search.arm import SearchArm
from llm_vqc.search.comparison import is_strictly_better
from llm_vqc.search.feedback import SearchFeedback


def _genome_dict_to_ir_dict(genome: S.ControlledGenome) -> dict:
    return S.genome_to_ir(genome).model_dump()


# --- Controlled Random -------------------------------------------------------

class ControlledRandomState(BaseModel):
    seed: int
    n_proposed: int = 0
    best_hash: str | None = None
    best_metric: float | None = None
    n_generation_rejections: int = 0  # kept 0: generator is valid by construction


class ControlledRandomArm(SearchArm[ControlledRandomState]):
    name = "controlled_random"

    def __init__(self, lower_is_better: bool) -> None:
        self.lower_is_better = lower_is_better

    def initialize(self, seed: int) -> ControlledRandomState:
        return ControlledRandomState(seed=seed)

    def propose(self, state: ControlledRandomState) -> dict:
        rng = np.random.default_rng(
            derive_child_seed(state.seed, "controlled_random", str(state.n_proposed))
        )
        return _genome_dict_to_ir_dict(S.random_genome(rng))

    def update_state(self, state, proposal, feedback: SearchFeedback):
        best_hash, best_metric = state.best_hash, state.best_metric
        if feedback.val_metric_value is not None and is_strictly_better(
            feedback.val_metric_value, best_metric, self.lower_is_better
        ):
            best_hash, best_metric = feedback.structural_hash, feedback.val_metric_value
        return state.model_copy(update={
            "n_proposed": state.n_proposed + 1, "best_hash": best_hash, "best_metric": best_metric,
        })

    def select_final(self, state) -> str | None:
        return state.best_hash

    def deserialize_state(self, raw_json: str) -> ControlledRandomState:
        return ControlledRandomState.model_validate_json(raw_json)


# --- Controlled Greedy -------------------------------------------------------

class _GreedyObs(BaseModel):
    genome: dict
    val_metric: float


class ControlledGreedyState(BaseModel):
    seed: int
    n_proposed: int = 0
    current_genome: dict
    round_index: int = 0
    round_observations: list[_GreedyObs] = []
    best_hash: str | None = None
    best_metric: float | None = None


class ControlledGreedyArm(SearchArm[ControlledGreedyState]):
    name = "controlled_greedy"

    def __init__(self, lower_is_better: bool, k: int = 5) -> None:
        self.lower_is_better = lower_is_better
        self.k = k

    def initialize(self, seed: int) -> ControlledGreedyState:
        return ControlledGreedyState(seed=seed, current_genome=S.deterministic_seed_genome().to_dict())

    def _propose_genome(self, state: ControlledGreedyState) -> S.ControlledGenome:
        rng = np.random.default_rng(
            derive_child_seed(state.seed, "controlled_greedy", str(state.round_index),
                              str(state.n_proposed))
        )
        current = S.ControlledGenome.from_dict(state.current_genome)
        return S.sample_neighbor(current, rng)

    def propose(self, state: ControlledGreedyState) -> dict:
        return _genome_dict_to_ir_dict(self._propose_genome(state))

    def update_state(self, state, proposal, feedback: SearchFeedback):
        best_hash, best_metric = state.best_hash, state.best_metric
        if feedback.val_metric_value is not None and is_strictly_better(
            feedback.val_metric_value, best_metric, self.lower_is_better
        ):
            best_hash, best_metric = feedback.structural_hash, feedback.val_metric_value

        proposed = self._propose_genome(state)  # deterministic re-derivation
        observations = list(state.round_observations)
        if feedback.val_metric_value is not None:
            observations.append(_GreedyObs(genome=proposed.to_dict(),
                                           val_metric=feedback.val_metric_value))

        current_genome = state.current_genome
        round_index = state.round_index
        if len(observations) >= self.k:
            best_obs = observations[0]
            for obs in observations[1:]:
                if is_strictly_better(obs.val_metric, best_obs.val_metric, self.lower_is_better):
                    best_obs = obs
            current_genome = best_obs.genome
            observations = []
            round_index += 1

        return state.model_copy(update={
            "n_proposed": state.n_proposed + 1, "current_genome": current_genome,
            "round_index": round_index, "round_observations": observations,
            "best_hash": best_hash, "best_metric": best_metric,
        })

    def select_final(self, state) -> str | None:
        return state.best_hash

    def deserialize_state(self, raw_json: str) -> ControlledGreedyState:
        return ControlledGreedyState.model_validate_json(raw_json)


# --- Controlled Evolutionary -------------------------------------------------

class ControlledEvoConfig(BaseModel):
    mu: int = 5
    lambda_: int = 5
    crossover_rate: float = 0.5
    mutation_rate: float = 0.5
    tournament_size: int = 2


class _EvoIndividual(BaseModel):
    genome: dict
    val_metric: float | None = None


class ControlledEvoState(BaseModel):
    seed: int
    n_proposed: int = 0
    generation: int = 0
    population: list[_EvoIndividual] = []
    pending: list[dict] = []
    evaluated: list[_EvoIndividual] = []
    best_hash: str | None = None
    best_metric: float | None = None


def _rank_key(ind: _EvoIndividual, lower_is_better: bool) -> float:
    if ind.val_metric is None:
        return float("inf")
    return ind.val_metric if lower_is_better else -ind.val_metric


def _tournament(pool, rng, size, lower_is_better) -> _EvoIndividual:
    size = min(size, len(pool))
    idx = rng.choice(len(pool), size=size, replace=False)
    contenders = [pool[i] for i in idx]
    best = contenders[0]
    for c in contenders[1:]:
        if c.val_metric is not None and (
            best.val_metric is None
            or is_strictly_better(c.val_metric, best.val_metric, lower_is_better)
        ):
            best = c
    return best


class ControlledEvolutionaryArm(SearchArm[ControlledEvoState]):
    name = "controlled_evolutionary"

    def __init__(self, lower_is_better: bool, config: ControlledEvoConfig | None = None) -> None:
        self.lower_is_better = lower_is_better
        self.config = config or ControlledEvoConfig()

    def initialize(self, seed: int) -> ControlledEvoState:
        rng = np.random.default_rng(derive_child_seed(seed, "controlled_evo_init"))
        pending = [S.random_genome(rng).to_dict() for _ in range(self.config.mu)]
        return ControlledEvoState(seed=seed, pending=pending)

    def propose(self, state: ControlledEvoState) -> dict:
        index = len(state.evaluated)
        genome = S.ControlledGenome.from_dict(state.pending[index])
        return _genome_dict_to_ir_dict(genome)

    def update_state(self, state, proposal, feedback: SearchFeedback):
        best_hash, best_metric = state.best_hash, state.best_metric
        if feedback.val_metric_value is not None and is_strictly_better(
            feedback.val_metric_value, best_metric, self.lower_is_better
        ):
            best_hash, best_metric = feedback.structural_hash, feedback.val_metric_value

        index = len(state.evaluated)
        evaluated = [*state.evaluated,
                     _EvoIndividual(genome=state.pending[index], val_metric=feedback.val_metric_value)]

        population = state.population
        pending = state.pending
        generation = state.generation

        if len(evaluated) >= len(state.pending):
            combined = [*population, *evaluated]
            new_population = sorted(combined, key=lambda i: _rank_key(i, self.lower_is_better))[
                : self.config.mu
            ]
            rng = np.random.default_rng(derive_child_seed(state.seed, "controlled_evo_gen",
                                                          str(generation + 1)))
            offspring: list[dict] = []
            for _ in range(self.config.lambda_):
                if len(new_population) >= 2 and rng.random() < self.config.crossover_rate:
                    pa = _tournament(new_population, rng, self.config.tournament_size, self.lower_is_better)
                    pb = _tournament(new_population, rng, self.config.tournament_size, self.lower_is_better)
                    child = S.crossover(S.ControlledGenome.from_dict(pa.genome),
                                        S.ControlledGenome.from_dict(pb.genome), rng)
                else:
                    parent = _tournament(new_population, rng, self.config.tournament_size, self.lower_is_better)
                    child = S.ControlledGenome.from_dict(parent.genome)
                child = S.mutate(child, rng, self.config.mutation_rate)
                offspring.append(child.to_dict())
            population = new_population
            pending = offspring
            evaluated = []
            generation += 1

        return state.model_copy(update={
            "n_proposed": state.n_proposed + 1, "generation": generation,
            "population": population, "pending": pending, "evaluated": evaluated,
            "best_hash": best_hash, "best_metric": best_metric,
        })

    def select_final(self, state) -> str | None:
        return state.best_hash

    def deserialize_state(self, raw_json: str) -> ControlledEvoState:
        return ControlledEvoState.model_validate_json(raw_json)
