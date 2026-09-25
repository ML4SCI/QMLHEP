"""SQLite-backed result store with resume.

Deliberately plain `sqlite3` (standard library, no new dependency) rather
than a client-server database — the master plan's own reproducibility
requirements (Section 11) name SQLite explicitly as the recommended
choice, and nothing in Phase 2's access pattern (one writer, occasional
reads, a few thousand rows per run at most) needs anything heavier.

Guarantees this module provides:

- **Atomic writes:** every `put_cached` is one SQLite transaction
  (`INSERT OR REPLACE` + commit); a process killed mid-write leaves the
  previous row (if any) or no row, never a half-written one.
- **Resume without repeating work:** `get_cached` is exactly the
  `CandidateCache.get` a fresh `evaluate_candidate` call needs to skip
  re-training a `(task, structural_hash, train_seed)` triple that was
  already evaluated in a prior process.
- **Incompatible-resume detection:** opening an existing store computes a
  hash of the reproducibility-critical parts of the new config and
  compares it to what is stored; a mismatch raises
  `IncompatibleResumeError` rather than silently mixing results from two
  different experiment conditions in one run directory.
- **Failures are rows, not silence:** a failed training run is recorded
  with its `EvaluationResult.training_outcome == "failed"` like any other
  result — `put_cached` does not distinguish success from failure, it
  persists whatever `EvaluationResult` it is given.
- **Durable budget ledger:** `append_proposal_event` persists one search
  proposal event (append-only, one SQLite transaction) so that
  `BudgetLedger.from_store()` (`llm_vqc.ir.budget`) can reconstruct exact
  consumed-budget accounting after a crash or restart — see that module's
  docstring for the crash-safety argument.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llm_vqc.evaluation.results import EvaluationResult
from llm_vqc.ir.budget import ProposalRecord
from llm_vqc.llm.records import LLMCallRecord

_SCHEMA = """
CREATE TABLE IF NOT EXISTS run_meta (
    run_id TEXT PRIMARY KEY,
    config_json TEXT NOT NULL,
    config_compat_hash TEXT NOT NULL,
    git_sha TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS evaluations (
    task_name TEXT NOT NULL,
    structural_hash TEXT NOT NULL,
    train_seed INTEGER NOT NULL,
    result_json TEXT NOT NULL,
    trained_weights_json TEXT,
    created_at TEXT NOT NULL,
    PRIMARY KEY (task_name, structural_hash, train_seed)
);

CREATE TABLE IF NOT EXISTS proposal_events (
    run_id TEXT NOT NULL,
    proposal_index INTEGER NOT NULL,
    outcome TEXT NOT NULL,
    structural_hash TEXT,
    record_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (run_id, proposal_index)
);

CREATE TABLE IF NOT EXISTS run_state (
    run_id TEXT PRIMARY KEY,
    state_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS llm_calls (
    run_id TEXT NOT NULL,
    proposal_id TEXT NOT NULL,
    call_index INTEGER NOT NULL,
    record_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (run_id, proposal_id, call_index)
);
"""


class IncompatibleResumeError(Exception):
    """Raised when resuming into a store whose config differs in a
    reproducibility-critical way from the config being used now."""


class ResultStoreError(Exception):
    """Raised on store-level errors not covered by a more specific exception."""


def compute_config_compat_hash(config_reproducibility_fields: dict[str, Any]) -> str:
    """Hash of the parts of a config that must match to safely resume.

    Callers pass only the *reproducibility-critical* subset (task name,
    training hyperparameters, seed) — cosmetic fields (e.g. a run
    description) should not be included, or every trivial edit would
    force a fresh store.
    """
    canonical = json.dumps(config_reproducibility_fields, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ResultStore:
    """One SQLite file per run directory."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    @classmethod
    def open_or_create(
        cls,
        db_path: str | Path,
        run_id: str,
        config_json: str,
        config_reproducibility_fields: dict[str, Any],
        git_sha: str | None,
        created_at: str,
        allow_incompatible: bool = False,
    ) -> ResultStore:
        """Open an existing store (validating compatibility) or create a new one.

        Raises IncompatibleResumeError if a run_meta row for `run_id`
        already exists with a different `config_compat_hash`, unless
        `allow_incompatible=True` is passed explicitly (an intentional,
        non-silent override — the caller had to opt in).
        """
        store = cls(db_path)
        new_hash = compute_config_compat_hash(config_reproducibility_fields)

        existing = store._conn.execute(
            "SELECT config_compat_hash FROM run_meta WHERE run_id = ?", (run_id,)
        ).fetchone()

        if existing is None:
            store._conn.execute(
                "INSERT INTO run_meta "
                "(run_id, config_json, config_compat_hash, git_sha, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (run_id, config_json, new_hash, git_sha, created_at),
            )
            store._conn.commit()
        elif existing[0] != new_hash and not allow_incompatible:
            store.close()
            raise IncompatibleResumeError(
                f"run_id {run_id!r} already exists in {db_path} with a different "
                "reproducibility-critical configuration (task, training hyperparameters, "
                "or seed changed). Resuming would silently mix results from incompatible "
                "conditions. Use a new run_id, or pass allow_incompatible=True if this is "
                "an intentional override."
            )
        return store

    def put_cached(
        self,
        task_name: str,
        structural_hash: str,
        train_seed: int,
        result: EvaluationResult,
        trained_weights: dict | None,
    ) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO evaluations "
            "(task_name, structural_hash, train_seed, result_json, "
            "trained_weights_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                task_name,
                structural_hash,
                train_seed,
                result.model_dump_json(),
                json.dumps(trained_weights) if trained_weights is not None else None,
                result.created_at,
            ),
        )
        self._conn.commit()

    def get_cached(
        self, task_name: str, structural_hash: str, train_seed: int
    ) -> EvaluationResult | None:
        row = self._conn.execute(
            "SELECT result_json FROM evaluations "
            "WHERE task_name = ? AND structural_hash = ? AND train_seed = ?",
            (task_name, structural_hash, train_seed),
        ).fetchone()
        if row is None:
            return None
        return EvaluationResult.model_validate_json(row[0])

    def get_trained_weights(
        self, task_name: str, structural_hash: str, train_seed: int
    ) -> dict | None:
        row = self._conn.execute(
            "SELECT trained_weights_json FROM evaluations "
            "WHERE task_name = ? AND structural_hash = ? AND train_seed = ?",
            (task_name, structural_hash, train_seed),
        ).fetchone()
        if row is None or row[0] is None:
            return None
        return json.loads(row[0])

    def all_evaluations(self, task_name: str | None = None) -> list[EvaluationResult]:
        if task_name is None:
            rows = self._conn.execute("SELECT result_json FROM evaluations").fetchall()
        else:
            rows = self._conn.execute(
                "SELECT result_json FROM evaluations WHERE task_name = ?", (task_name,)
            ).fetchall()
        return [EvaluationResult.model_validate_json(row[0]) for row in rows]

    def count_evaluations(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM evaluations").fetchone()
        return int(row[0])

    def append_proposal_event(
        self, run_id: str, proposal_index: int, record: ProposalRecord
    ) -> None:
        """Durably append one search-proposal event.

        Plain `INSERT` (not `INSERT OR REPLACE`): `(run_id, proposal_index)`
        is an append-only log's primary key, so accidentally reusing an
        index raises `sqlite3.IntegrityError` instead of silently
        overwriting a prior event — a resumed run that mis-derives its
        next index fails loudly rather than corrupting history.
        """
        self._conn.execute(
            "INSERT INTO proposal_events "
            "(run_id, proposal_index, outcome, structural_hash, record_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                run_id,
                proposal_index,
                record.outcome.value,
                record.structural_hash,
                record.model_dump_json(),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self._conn.commit()

    def iter_proposal_events(self, run_id: str) -> Iterator[ProposalRecord]:
        """Yield every persisted proposal event for `run_id`, ordered by
        `proposal_index` ascending — the exact order required for
        `BudgetLedger.from_store()` to reconstruct duplicate detection and
        consumed-budget counts identically to the original run."""
        rows = self._conn.execute(
            "SELECT record_json FROM proposal_events "
            "WHERE run_id = ? ORDER BY proposal_index ASC",
            (run_id,),
        ).fetchall()
        for row in rows:
            yield ProposalRecord.model_validate_json(row[0])

    def count_proposal_events(self, run_id: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM proposal_events WHERE run_id = ?", (run_id,)
        ).fetchone()
        return int(row[0])

    def save_run_state(self, run_id: str, state_json: str) -> None:
        """Overwrite the latest checkpoint of a search arm's state.

        Unlike `proposal_events`, this is `INSERT OR REPLACE`: it is the
        *current* checkpoint, not an append-only log, so each call is
        meant to supersede the previous one.
        """
        self._conn.execute(
            "INSERT OR REPLACE INTO run_state (run_id, state_json, updated_at) VALUES (?, ?, ?)",
            (run_id, state_json, datetime.now(timezone.utc).isoformat()),
        )
        self._conn.commit()

    def load_run_state(self, run_id: str) -> str | None:
        row = self._conn.execute(
            "SELECT state_json FROM run_state WHERE run_id = ?", (run_id,)
        ).fetchone()
        return row[0] if row is not None else None

    def append_llm_call(self, run_id: str, record: LLMCallRecord) -> None:
        """Durably persist one LLM call's full provenance (prompts, raw
        response, parsed proposal, token usage, cost, latency).

        Unlike `append_proposal_event`, this is `INSERT OR REPLACE`, not a
        strict append-only log: a proposal's ledger entry is only written
        after its *entire* driver retry chain completes, so a crash
        mid-chain means the next resume re-runs that chain from attempt 0
        and re-derives the same `(run_id, proposal_id, call_index)` keys.
        With `MockLLMProvider` this is exactly idempotent (the response is
        a pure function of the prompt text, which resume reproduces
        exactly). **Known limitation for a real provider:** a crash
        mid-repair-chain would re-issue any already-paid-for call on
        resume -- costing slightly more than the theoretical minimum,
        never bypassing the per-call `LLMApiBudget.check_can_afford` cap.
        Not exercised in this work cycle (no real provider is used).
        """
        self._conn.execute(
            "INSERT OR REPLACE INTO llm_calls "
            "(run_id, proposal_id, call_index, record_json, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                run_id,
                record.proposal_id,
                record.call_index,
                record.model_dump_json(),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self._conn.commit()

    def iter_llm_calls(self, run_id: str) -> Iterator[LLMCallRecord]:
        rows = self._conn.execute(
            "SELECT record_json FROM llm_calls WHERE run_id = ? "
            "ORDER BY proposal_id ASC, call_index ASC",
            (run_id,),
        ).fetchall()
        for row in rows:
            yield LLMCallRecord.model_validate_json(row[0])

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> ResultStore:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
