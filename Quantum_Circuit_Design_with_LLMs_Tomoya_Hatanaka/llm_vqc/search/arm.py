"""`SearchArm`: the one typed interface every search method (random,
evolutionary, greedy, and the LLM arms) implements, so that `SearchRunner`
(`llm_vqc.search.runner`) can drive all of them through an identical loop.

An arm never talks to the evaluator, the result store, or the IR validator
directly -- it only ever produces a raw proposal (a dict, in the same
shape any other proposal source uses) and consumes the `SearchFeedback`
the runner hands back. This is what makes "no arm may have a private,
unmatchable action space" (master plan Section 5.2) structurally true
rather than a convention: an arm that wanted to skip validation or read
test data would have no object in its method signatures through which to
reach either one.

Arm state is always a pydantic `BaseModel` subclass, so `serialize_state`
is one generic implementation shared by every arm; only deserialization
needs the concrete state type, hence the one abstract `deserialize_state`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from pydantic import BaseModel

from llm_vqc.search.feedback import SearchFeedback

StateT = TypeVar("StateT", bound=BaseModel)


class SearchArm(ABC, Generic[StateT]):
    """Base class for every search method the master plan calls an "arm"."""

    #: Stable machine-readable identifier (e.g. "random", "evolutionary",
    #: "greedy", "llm_iter", "llm_evo") -- used as part of `run_id`
    #: composition and in every results table.
    name: str

    @abstractmethod
    def initialize(self, seed: int) -> StateT:
        """Build the arm's fresh initial state for one (task, seed) run.

        `seed` is the arm's own search-RNG seed -- distinct from the task's
        data-split seed and from any candidate's training seed (three
        independent streams per master plan Section 11).
        """

    @abstractmethod
    def propose(self, state: StateT) -> dict:
        """Produce one raw circuit proposal (an IR-shaped dict).

        The returned dict is not assumed valid -- the runner always routes
        it through `llm_vqc.ir.validators.validate_proposal` via
        `llm_vqc.evaluation.harness.evaluate_candidate`, exactly as it
        would any other arm's proposal.
        """

    @abstractmethod
    def update_state(self, state: StateT, proposal: dict, feedback: SearchFeedback) -> StateT:
        """Incorporate one proposal's outcome and return the new state.

        `feedback` is the only channel back from the evaluator; it never
        carries a test metric (see `SearchFeedback`'s docstring).
        """

    @abstractmethod
    def select_final(self, state: StateT) -> str | None:
        """Return the `structural_hash` of the arm's chosen best candidate,
        or `None` if no candidate was ever successfully evaluated.

        Called only once, after the run's budget is exhausted -- an arm
        must not use this to keep searching, and the runner never calls it
        mid-run.
        """

    @abstractmethod
    def deserialize_state(self, raw_json: str) -> StateT:
        """Reconstruct this arm's state type from a prior `serialize_state` call."""

    def serialize_state(self, state: StateT) -> str:
        return state.model_dump_json()
