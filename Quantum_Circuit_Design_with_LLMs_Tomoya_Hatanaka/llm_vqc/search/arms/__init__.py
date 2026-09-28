"""Concrete search-arm implementations (master plan Section 6.2,
Phase 4 non-LLM arms): `random`, `greedy`, `evolutionary`.

Every arm here is a `llm_vqc.search.arm.SearchArm` subclass, driven only
by `llm_vqc.search.runner.SearchRunner` -- see that module for the
fairness guarantees (identical validator, evaluator, budget accounting,
duplicate policy across all arms).
"""

from __future__ import annotations

from llm_vqc.search.arms.evolutionary_arm import (
    EvolutionaryArm,
    EvolutionaryArmConfig,
    EvolutionaryArmState,
)
from llm_vqc.search.arms.greedy_arm import GreedyArm, GreedyArmState, minimal_ir
from llm_vqc.search.arms.llm_evo_arm import LLMEvoArm, LLMEvoState
from llm_vqc.search.arms.llm_iter_arm import LLMIterArm, LLMIterState
from llm_vqc.search.arms.random_arm import RandomArm, RandomArmState

__all__ = [
    "EvolutionaryArm",
    "EvolutionaryArmConfig",
    "EvolutionaryArmState",
    "GreedyArm",
    "GreedyArmState",
    "LLMEvoArm",
    "LLMEvoState",
    "LLMIterArm",
    "LLMIterState",
    "RandomArm",
    "RandomArmState",
    "minimal_ir",
]
