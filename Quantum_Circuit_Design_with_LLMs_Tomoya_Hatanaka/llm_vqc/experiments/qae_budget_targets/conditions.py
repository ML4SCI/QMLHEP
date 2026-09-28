"""The budget-target conditions and their declared anchors.

These live in their own registry on purpose. `qae_robustness.CONDITIONS`
is the frozen one-factor-at-a-time matrix of the previous study and is left
exactly as it was; adding a second-anchor condition to it would silently
change what `tests/test_qae_robustness_manifest.py` asserts.
"""
from __future__ import annotations

from dataclasses import dataclass

from llm_vqc.experiments.qae_robustness.conditions import (
    CONDITIONS_BY_KEY,
    REFERENCE,
    Condition,
)

# Frozen by the protocol. `B` must be even so that n_warm = B/2 keeps the
# pre-registered 1:1 exploration/refinement split.
ADMISSIBLE_BUDGETS = (4, 6, 8, 10, 16)

METHODS = ("Random", "Greedy", "LLM-Open", "LLM-Closed")


class InadmissibleBudgetError(ValueError):
    """Raised for a budget outside the pre-registered admissible grid."""


@dataclass(frozen=True)
class TargetCell:
    """One budget-target experiment: a condition plus its declared anchor.

    `methods` is explicit because a boundary probe may legitimately run a
    single method; recording it stops a later reader from manufacturing a
    same-budget cross-method comparison that was never executed.
    """

    condition: Condition
    anchor: Condition
    methods: tuple[str, ...]
    rationale: str

    @property
    def key(self) -> str:
        return self.condition.key

    @property
    def budget(self) -> int:
        return self.condition.budget

    @property
    def runs_open(self) -> bool:
        return "LLM-Open" in self.methods

    @property
    def runs_closed(self) -> bool:
        return "LLM-Closed" in self.methods

    def describe(self) -> dict:
        return {
            "key": self.key,
            "label": self.condition.label,
            "anchor_key": self.anchor.key,
            "factors": self.condition.factors,
            "n_warm": self.condition.n_warm,
            "n_refine": self.condition.budget - self.condition.n_warm,
            "methods": list(self.methods),
            "rationale": self.rationale,
        }


def make_condition(
    key: str, *, budget: int, anchor: Condition, label: str = ""
) -> Condition:
    """A condition that differs from `anchor` in the budget alone."""
    if budget not in ADMISSIBLE_BUDGETS:
        raise InadmissibleBudgetError(
            f"budget {budget} is not on the pre-registered grid "
            f"{ADMISSIBLE_BUDGETS}; the grid may not be extended after "
            "seeing a result"
        )
    return Condition(
        key=key,
        factor="budget",
        n_qubits=anchor.n_qubits,
        family=anchor.family,
        budget=budget,
        model=anchor.model,
        label=label or f"B = {budget}",
    )


XXZ_ANCHOR = CONDITIONS_BY_KEY["hamiltonian_xxz"]

TARGET_CELLS: tuple[TargetCell, ...] = (
    TargetCell(
        condition=make_condition(
            "target_tfim_b6", budget=6, anchor=REFERENCE,
            label="4-qubit TFIM, reference model, B = 6",
        ),
        anchor=REFERENCE,
        methods=METHODS,
        rationale=(
            "TFIM reference anchor: B=4 fails (LLM-Open 0/12, LLM-Closed 8/12) "
            "and B=8 passes (10/12, 12/12) at F_val >= 0.95, so B=6 is the "
            "first-priority unverified even budget between them. All four "
            "methods run at the same B so the controls are budget-matched."
        ),
    ),
    TargetCell(
        condition=make_condition(
            "target_xxz_b10", budget=10, anchor=XXZ_ANCHOR,
            label="4-qubit XXZ, reference model, B = 10 (LLM-Closed only)",
        ),
        anchor=XXZ_ANCHOR,
        methods=("LLM-Closed",),
        rationale=(
            "XXZ anchor boundary probe: XXZ B=8 LLM-Closed is 9/12 at "
            "F_val >= 0.95, one seed short of the rule. Declared as its own "
            "XXZ-anchored protocol, not a one-factor change from the TFIM "
            "reference. Only LLM-Closed is executed, so no same-budget "
            "cross-method comparison exists at this cell."
        ),
    ),
)

TARGET_CELLS_BY_KEY = {cell.key: cell for cell in TARGET_CELLS}
