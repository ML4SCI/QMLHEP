"""The `evolutionary` search arm (master plan Section 6.2): "(mu+lambda)
evolution over IR with hand-coded mutation/crossover operators."

**(mu+lambda) selection is elitism.** Each generation's next parent
population is the top `mu` individuals from the *combined* pool of the
current parents plus the `lambda` newly evaluated offspring -- a parent
can only be displaced by something at least as good, which is exactly
what "elitism" means in an evolution strategy. No separate elitism
parameter is needed on top of this.

**Deterministic under a fixed seed:** every generation's tournament
selection, crossover, and mutation draws from one RNG stream derived as
`derive_child_seed(seed, "evo_gen", generation)` -- a pure function of
the run's search seed and the generation index, so replaying the same
state sequence always reproduces the same offspring.

**Exact checkpoint/resume:** `propose()` is a pure function of state (it
indexes into `state.pending` by `len(state.evaluated)`, never mutating
anything itself); all state transitions happen in `update_state()`, so
serializing `EvolutionaryArmState` after every proposal (done by
`SearchRunner`) captures everything needed to resume mid-generation.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel

from llm_vqc.evaluation.seeds import derive_child_seed
from llm_vqc.ir.sampler import sample_random_ir
from llm_vqc.ir.schema import CircuitIR
from llm_vqc.search.arm import SearchArm
from llm_vqc.search.arms.mutation import crossover, mutate
from llm_vqc.search.comparison import is_strictly_better
from llm_vqc.search.feedback import SearchFeedback


class EvolutionaryArmConfig(BaseModel):
    mu: int = 4
    lambda_: int = 4
    crossover_rate: float = 0.5
    mutation_rate: float = 0.5
    tournament_size: int = 2


class _Individual(BaseModel):
    ir: CircuitIR
    val_metric: float | None = None


class EvolutionaryArmState(BaseModel):
    seed: int
    n_proposed: int = 0
    generation: int = 0
    population: list[_Individual] = []  # evaluated parents; empty until generation 0 completes
    pending: list[CircuitIR] = []  # this generation's individuals, in propose order
    evaluated: list[_Individual] = []  # this generation's individuals evaluated so far
    best_hash: str | None = None
    best_metric: float | None = None


def _rank_key(individual: _Individual, lower_is_better: bool) -> float:
    if individual.val_metric is None:
        return float("inf")
    return individual.val_metric if lower_is_better else -individual.val_metric


def _tournament_select(
    pool: list[_Individual], rng: np.random.Generator, size: int, lower_is_better: bool
) -> _Individual:
    size = min(size, len(pool))
    indices = rng.choice(len(pool), size=size, replace=False)
    contenders = [pool[i] for i in indices]
    best = contenders[0]
    for contender in contenders[1:]:
        if contender.val_metric is not None and (
            best.val_metric is None
            or is_strictly_better(contender.val_metric, best.val_metric, lower_is_better)
        ):
            best = contender
    return best


class EvolutionaryArm(SearchArm[EvolutionaryArmState]):
    name = "evolutionary"

    def __init__(self, lower_is_better: bool, config: EvolutionaryArmConfig | None = None) -> None:
        self.lower_is_better = lower_is_better
        self.config = config or EvolutionaryArmConfig()

    def initialize(self, seed: int) -> EvolutionaryArmState:
        rng = np.random.default_rng(derive_child_seed(seed, "evo_init"))
        initial_population = [sample_random_ir(rng) for _ in range(self.config.mu)]
        return EvolutionaryArmState(seed=seed, pending=initial_population)

    def propose(self, state: EvolutionaryArmState) -> dict:
        index = len(state.evaluated)
        return state.pending[index].model_dump()

    def update_state(
        self, state: EvolutionaryArmState, proposal: dict, feedback: SearchFeedback
    ) -> EvolutionaryArmState:
        best_hash, best_metric = state.best_hash, state.best_metric
        if feedback.val_metric_value is not None and is_strictly_better(
            feedback.val_metric_value, best_metric, self.lower_is_better
        ):
            best_hash, best_metric = feedback.structural_hash, feedback.val_metric_value

        evaluated = [
            *state.evaluated,
            _Individual(
                ir=CircuitIR.model_validate(proposal),
                val_metric=feedback.val_metric_value,
            ),
        ]

        population = state.population
        pending = state.pending
        generation = state.generation

        if len(evaluated) >= len(state.pending):
            combined = [*population, *evaluated]
            combined_sorted = sorted(
                combined, key=lambda ind: _rank_key(ind, self.lower_is_better)
            )
            new_population = combined_sorted[: self.config.mu]

            rng = np.random.default_rng(
                derive_child_seed(state.seed, "evo_gen", str(generation + 1))
            )
            offspring: list[CircuitIR] = []
            for _ in range(self.config.lambda_):
                if len(new_population) >= 2 and rng.random() < self.config.crossover_rate:
                    parent_a = _tournament_select(
                        new_population, rng, self.config.tournament_size, self.lower_is_better
                    )
                    parent_b = _tournament_select(
                        new_population, rng, self.config.tournament_size, self.lower_is_better
                    )
                    child = crossover(parent_a.ir, parent_b.ir, rng)
                else:
                    parent = _tournament_select(
                        new_population, rng, self.config.tournament_size, self.lower_is_better
                    )
                    child = parent.ir
                child = mutate(child, rng, self.config.mutation_rate)
                offspring.append(child)

            population = new_population
            pending = offspring
            evaluated = []
            generation += 1

        return state.model_copy(
            update={
                "n_proposed": state.n_proposed + 1,
                "generation": generation,
                "population": population,
                "pending": pending,
                "evaluated": evaluated,
                "best_hash": best_hash,
                "best_metric": best_metric,
            }
        )

    def select_final(self, state: EvolutionaryArmState) -> str | None:
        return state.best_hash

    def deserialize_state(self, raw_json: str) -> EvolutionaryArmState:
        return EvolutionaryArmState.model_validate_json(raw_json)
