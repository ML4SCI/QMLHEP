"""SQLite-backed diagnostic result store with resume.

Mirrors `llm_vqc.evaluation.store.ResultStore`'s design (plain sqlite3,
atomic commit-per-write, incompatible-resume detection) but with a
DISTINCT cache identity appropriate to diagnostics.

**Cache identity** for a diagnostic result is the tuple

    (structural_hash, diagnostic_name, diagnostic_version, config_hash, run_seed)

— NOT the metric name alone. Two diagnostics that share a metric name but
differ in version or config (e.g. a different bin count, a different
sample count, a different estimator version) are DIFFERENT measurements
and get different rows; serving one where the other was requested would
silently mix protocols. `config_hash` is a hash of the fully-resolved
diagnostic config, and `diagnostic_version` is bumped whenever the meaning
of the output changes.

**Syntactic vs physical identity:** `structural_hash` is the Phase 1
syntactic circuit hash. As documented there, two physically-equivalent but
syntactically-different circuits hash differently and are cached
separately — Phase 3 does not define any physical-equivalence analysis, so
this is correct (a physically equivalent circuit is a legitimately
separate cache entry).

**Failures are rows:** a failed diagnostic (e.g. all-gradients-non-finite)
is stored like any successful one, so a resumed run does not blindly retry
a computation that deterministically fails.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

from llm_vqc.diagnostics.results import DiagnosticResult

_SCHEMA = """
CREATE TABLE IF NOT EXISTS diagnostic_run_meta (
    run_id TEXT PRIMARY KEY,
    config_json TEXT NOT NULL,
    config_compat_hash TEXT NOT NULL,
    git_sha TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS diagnostics (
    structural_hash TEXT NOT NULL,
    diagnostic_name TEXT NOT NULL,
    diagnostic_version TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    run_seed INTEGER NOT NULL,
    result_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (structural_hash, diagnostic_name, diagnostic_version, config_hash, run_seed)
);
"""


class IncompatibleDiagnosticResumeError(Exception):
    """Raised when resuming into a store whose run config differs in a
    reproducibility-critical way from the config being used now."""


def config_hash(config_dict: dict[str, Any]) -> str:
    """Deterministic hash of a fully-resolved diagnostic config dict."""
    canonical = json.dumps(config_dict, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class DiagnosticStore:
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
    ) -> DiagnosticStore:
        store = cls(db_path)
        new_hash = config_hash(config_reproducibility_fields)
        existing = store._conn.execute(
            "SELECT config_compat_hash FROM diagnostic_run_meta WHERE run_id = ?", (run_id,)
        ).fetchone()

        if existing is None:
            store._conn.execute(
                "INSERT INTO diagnostic_run_meta "
                "(run_id, config_json, config_compat_hash, git_sha, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (run_id, config_json, new_hash, git_sha, created_at),
            )
            store._conn.commit()
        elif existing[0] != new_hash and not allow_incompatible:
            store.close()
            raise IncompatibleDiagnosticResumeError(
                f"run_id {run_id!r} already exists in {db_path} with a different "
                "reproducibility-critical diagnostic configuration. Resuming would silently "
                "mix results from incompatible protocols. Use a new run_id, or pass "
                "allow_incompatible=True to override intentionally."
            )
        return store

    def get_cached(
        self,
        structural_hash: str,
        diagnostic_name: str,
        diagnostic_version: str,
        cfg_hash: str,
        run_seed: int,
    ) -> DiagnosticResult | None:
        row = self._conn.execute(
            "SELECT result_json FROM diagnostics WHERE structural_hash = ? AND diagnostic_name = ? "
            "AND diagnostic_version = ? AND config_hash = ? AND run_seed = ?",
            (structural_hash, diagnostic_name, diagnostic_version, cfg_hash, run_seed),
        ).fetchone()
        if row is None:
            return None
        return DiagnosticResult.model_validate_json(row[0])

    def put_cached(self, result: DiagnosticResult, run_seed: int) -> None:
        if result.structural_hash is None:
            # Invalid-circuit results have no structural hash and are not
            # cacheable by circuit identity; they are returned to the caller
            # but not persisted as a reusable measurement.
            return
        self._conn.execute(
            "INSERT OR REPLACE INTO diagnostics "
            "(structural_hash, diagnostic_name, diagnostic_version, config_hash, run_seed, "
            "result_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                result.structural_hash,
                result.diagnostic_name,
                result.diagnostic_version,
                result.config_hash,
                run_seed,
                result.model_dump_json(),
                result.created_at,
            ),
        )
        self._conn.commit()

    def all_results(self) -> list[DiagnosticResult]:
        rows = self._conn.execute("SELECT result_json FROM diagnostics").fetchall()
        return [DiagnosticResult.model_validate_json(row[0]) for row in rows]

    def count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM diagnostics").fetchone()[0])

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> DiagnosticStore:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
