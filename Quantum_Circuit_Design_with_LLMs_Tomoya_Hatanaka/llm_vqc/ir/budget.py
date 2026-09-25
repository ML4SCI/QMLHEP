"""Budget-accounting interfaces and data structures.

`BudgetLedger` establishes the vocabulary and counters a search-arm runner
(Phase 4+) builds on: "how many circuits were proposed, how many were
valid, how many were unique (by structural hash), how many were
duplicates, how many failed compilation/training" — the exact bookkeeping
LLM-VQC_MASTER_PLAN.md Section 6.3 needs to make Knipfer et al.'s
"exploration collapse" measurable rather than anecdotal, and to make
budget-matched comparison across search arms valid.

Per Section 6.3: invalid IR proposals do not consume simulation budget
(they are counted separately as a secondary metric); valid, duplicate,
and failed-evaluation proposals all consume one budget unit each, because
each of them caused (or attempted) one candidate evaluation.

`BudgetLedger.from_store()` reconstructs a ledger identical to one that
never crashed, by replaying every persisted `ProposalRecord` for a given
`run_id` from a durable store in the exact order they were recorded. This
is a hard prerequisite (governance decision G2-C) for any budget-matched
search comparison: without it, a restarted search could silently receive
extra budget, or lose track of which structural hashes it had already
seen (breaking duplicate detection across a restart).
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import Enum
from typing import Protocol

from pydantic import BaseModel

from llm_vqc.ir.metrics import CircuitCostSummary
from llm_vqc.ir.validators import ValidationIssue


class ProposalOutcome(str, Enum):
    VALID = "valid"
    INVALID = "invalid"
    DUPLICATE = "duplicate"
    # Valid, non-duplicate IR whose compilation or training failed. This
    # still consumes budget (a real evaluation attempt was made), unlike
    # INVALID (rejected before any simulation was attempted).
    FAILED = "failed"


#: Outcomes that consume one unit of the candidate-evaluation budget.
#: INVALID is the only outcome that is free, per master plan Section 6.3.
_BUDGET_CONSUMING_OUTCOMES = frozenset(
    {ProposalOutcome.VALID, ProposalOutcome.DUPLICATE, ProposalOutcome.FAILED}
)


class ProposalRecord(BaseModel):
    """One proposal event, as it would be logged to a future run's ledger."""

    outcome: ProposalOutcome
    structural_hash: str | None = None
    issues: list[ValidationIssue] = []
    cost: CircuitCostSummary | None = None

    @property
    def consumes_budget(self) -> bool:
        return self.outcome in _BUDGET_CONSUMING_OUTCOMES


class ProposalEventStore(Protocol):
    """Minimal durable-storage contract `BudgetLedger` depends on.

    Defined as a Protocol (rather than importing a concrete store class)
    so that `llm_vqc.ir` never depends on `llm_vqc.evaluation` — the
    concrete implementation (`llm_vqc.evaluation.store.ResultStore`)
    depends on `ir`, not the other way around.
    """

    def append_proposal_event(
        self, run_id: str, proposal_index: int, record: ProposalRecord
    ) -> None: ...

    def iter_proposal_events(self, run_id: str) -> Iterable[ProposalRecord]: ...

    def count_proposal_events(self, run_id: str) -> int: ...


class BudgetLedger:
    """In-memory counters for proposal bookkeeping, durably reconstructible.

    A structural hash is counted as a duplicate the second and later time
    it is recorded with outcome=VALID; the caller decides the outcome
    (VALID vs DUPLICATE) at the call site — this class only counts, it
    does not do hash lookups or decide duplicate status itself, keeping it
    a pure recorder rather than an implicit cache (that belongs in the
    Phase 2 evaluation harness's candidate cache).

    Two ways to record a proposal:

    - `record_proposal()` — in-memory only (Phase 1 behavior; still used
      directly by tests and by any caller that manages its own
      persistence).
    - `record_and_persist(store, run_id, record)` — durably appends the
      event to `store` *before* updating in-memory state, then updates
      in-memory state. If the process crashes between the two steps, the
      event is already durable and `from_store()` will reconstruct it on
      the next run; if it crashes before the store write, the event never
      happened and nothing is double-counted. Either way a crash can
      never reset or inflate the budget.
    """

    def __init__(self) -> None:
        self._records: list[ProposalRecord] = []
        self._seen_hashes: set[str] = set()

    def record_proposal(self, record: ProposalRecord) -> None:
        self._records.append(record)
        if record.structural_hash is not None:
            self._seen_hashes.add(record.structural_hash)

    def record_and_persist(
        self, store: ProposalEventStore, run_id: str, record: ProposalRecord
    ) -> None:
        """Append `record` durably to `store`, then update in-memory state.

        The persisted index is this ledger's current `num_proposed`
        (i.e. exactly where the durable log left off when this ledger was
        constructed, whether fresh or via `from_store`), so resuming a
        crashed run and continuing to call this method reproduces a
        gapless, correctly-ordered event log.
        """
        store.append_proposal_event(run_id, self.num_proposed, record)
        self.record_proposal(record)

    @classmethod
    def from_store(cls, store: ProposalEventStore, run_id: str) -> BudgetLedger:
        """Reconstruct a ledger by replaying every persisted event for `run_id`.

        Events are replayed in the exact order they were originally
        recorded (`iter_proposal_events` is required to yield them ordered
        by `proposal_index`), so the resulting ledger is indistinguishable
        from one that had been running in a single unbroken process —
        including which structural hashes have been seen (duplicate
        detection continues correctly) and the exact consumed-budget
        count (a resumed search cannot receive extra budget).
        """
        ledger = cls()
        for record in store.iter_proposal_events(run_id):
            ledger.record_proposal(record)
        return ledger

    @property
    def num_proposed(self) -> int:
        return len(self._records)

    @property
    def num_valid(self) -> int:
        return sum(1 for r in self._records if r.outcome == ProposalOutcome.VALID)

    @property
    def num_invalid(self) -> int:
        return sum(1 for r in self._records if r.outcome == ProposalOutcome.INVALID)

    @property
    def num_duplicate(self) -> int:
        return sum(1 for r in self._records if r.outcome == ProposalOutcome.DUPLICATE)

    @property
    def num_failed(self) -> int:
        return sum(1 for r in self._records if r.outcome == ProposalOutcome.FAILED)

    @property
    def num_unique(self) -> int:
        """Distinct structural hashes among all recorded proposals."""
        return len(self._seen_hashes)

    @property
    def consumed_budget(self) -> int:
        """Total candidate-evaluation budget consumed so far.

        Equals `num_valid + num_duplicate + num_failed` — every outcome
        except INVALID, per master plan Section 6.3.
        """
        return sum(1 for r in self._records if r.consumes_budget)

    def remaining_budget(self, budget_limit: int) -> int:
        """Budget left before a search run at `budget_limit` must terminate."""
        return max(0, budget_limit - self.consumed_budget)

    def is_exhausted(self, budget_limit: int) -> bool:
        return self.consumed_budget >= budget_limit

    def summary(self) -> dict[str, int]:
        return {
            "num_proposed": self.num_proposed,
            "num_valid": self.num_valid,
            "num_invalid": self.num_invalid,
            "num_duplicate": self.num_duplicate,
            "num_failed": self.num_failed,
            "num_unique": self.num_unique,
            "consumed_budget": self.consumed_budget,
        }
