"""The `random` search arm (master plan Section 6.2 and Phase 1's
`sample_random_ir` -- "this *is* the `random` arm's generator").

Uninformed baseline: every proposal is an independent uniform draw from
the IR grammar. Tracks best-so-far by validation metric for
`select_final`, but nothing about the draw itself depends on feedback --
that is what makes it the uninformed control every other arm is compared
against.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel

from llm_vqc.evaluation.seeds import derive_child_seed
from llm_vqc.ir.sampler import sample_random_ir
from llm_vqc.search.arm import SearchArm
from llm_vqc.search.comparison import is_strictly_better
from llm_vqc.search.feedback import SearchFeedback


class RandomArmState(BaseModel):
    seed: int
    n_proposed: int = 0
    best_hash: str | None = None
    best_metric: float | None = None


class RandomArm(SearchArm[RandomArmState]):
    name = "random"

    def __init__(self, lower_is_better: bool) -> None:
        self.lower_is_better = lower_is_better

    def initialize(self, seed: int) -> RandomArmState:
        return RandomArmState(seed=seed)

    def propose(self, state: RandomArmState) -> dict:
        rng = np.random.default_rng(
            derive_child_seed(state.seed, "random_propose", str(state.n_proposed))
        )
        return sample_random_ir(rng).model_dump()

    def update_state(
        self, state: RandomArmState, proposal: dict, feedback: SearchFeedback
    ) -> RandomArmState:
        best_hash, best_metric = state.best_hash, state.best_metric
        if feedback.val_metric_value is not None and is_strictly_better(
            feedback.val_metric_value, best_metric, self.lower_is_better
        ):
            best_hash, best_metric = feedback.structural_hash, feedback.val_metric_value
        return state.model_copy(
            update={
                "n_proposed": state.n_proposed + 1,
                "best_hash": best_hash,
                "best_metric": best_metric,
            }
        )

    def select_final(self, state: RandomArmState) -> str | None:
        return state.best_hash

    def deserialize_state(self, raw_json: str) -> RandomArmState:
        return RandomArmState.model_validate_json(raw_json)
