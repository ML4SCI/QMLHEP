"""Diagnostic store tests: cache identity completeness, resume,
incompatible-resume rejection, failure persistence, crash recovery."""

from __future__ import annotations

import pytest

from llm_vqc.diagnostics.config import EXPRESSIBILITY_VERSION
from llm_vqc.diagnostics.results import (
    DiagnosticFailureCategory,
    DiagnosticResult,
)
from llm_vqc.diagnostics.store import (
    DiagnosticStore,
    IncompatibleDiagnosticResumeError,
    config_hash,
)

REPRO = {"config": {"n_pairs": 500, "n_bins": 75}}


def _result(
    structural_hash="h1",
    name="expressibility",
    version=EXPRESSIBILITY_VERSION,
    cfg_hash="cfg1",
    failure=DiagnosticFailureCategory.NONE,
    kl=0.05,
):
    return DiagnosticResult(
        structural_hash=structural_hash,
        circuit_canonical_json="{}",
        diagnostic_name=name,
        diagnostic_version=version,
        config_hash=cfg_hash,
        config={"n_bins": 75},
        metrics={"expressibility_kl": kl},
        failure_category=failure,
    )


def test_config_hash_is_deterministic_and_order_independent():
    a = config_hash({"a": 1, "b": 2})
    b = config_hash({"b": 2, "a": 1})
    assert a == b
    assert a != config_hash({"a": 1, "b": 3})


def test_put_then_get_round_trips(tmp_path):
    store = DiagnosticStore.open_or_create(
        tmp_path / "d.sqlite", run_id="r1", config_json="{}",
        config_reproducibility_fields=REPRO, git_sha=None, created_at="t0",
    )
    r = _result()
    store.put_cached(r, run_seed=0)
    got = store.get_cached("h1", "expressibility", EXPRESSIBILITY_VERSION, "cfg1", 0)
    assert got is not None
    assert got.metrics["expressibility_kl"] == 0.05
    store.close()


@pytest.mark.parametrize(
    "lookup",
    [
        ("h1", "expressibility", EXPRESSIBILITY_VERSION, "cfg1", 999),  # wrong seed
        ("h1", "expressibility", EXPRESSIBILITY_VERSION, "OTHER", 0),  # wrong config hash
        ("h1", "expressibility", "9.9.9", "cfg1", 0),  # wrong version
        ("h1", "entanglement", EXPRESSIBILITY_VERSION, "cfg1", 0),  # wrong diagnostic name
        ("OTHER", "expressibility", EXPRESSIBILITY_VERSION, "cfg1", 0),  # wrong circuit
    ],
)
def test_cache_identity_includes_every_component(tmp_path, lookup):
    """Changing ANY component of the identity tuple must miss the cache —
    proves cache-key completeness (a metric name alone is not identity)."""
    store = DiagnosticStore.open_or_create(
        tmp_path / "d.sqlite", run_id="r1", config_json="{}",
        config_reproducibility_fields=REPRO, git_sha=None, created_at="t0",
    )
    store.put_cached(_result(), run_seed=0)
    assert store.get_cached(*lookup) is None
    # sanity: the exact key hits
    assert store.get_cached("h1", "expressibility", EXPRESSIBILITY_VERSION, "cfg1", 0) is not None
    store.close()


def test_failed_diagnostic_is_persisted(tmp_path):
    store = DiagnosticStore.open_or_create(
        tmp_path / "d.sqlite", run_id="r1", config_json="{}",
        config_reproducibility_fields=REPRO, git_sha=None, created_at="t0",
    )
    failed = _result(
        name="gradient_variance", failure=DiagnosticFailureCategory.ALL_GRADIENTS_NON_FINITE
    )
    store.put_cached(failed, run_seed=0)
    got = store.get_cached("h1", "gradient_variance", EXPRESSIBILITY_VERSION, "cfg1", 0)
    assert got is not None
    assert got.failure_category == DiagnosticFailureCategory.ALL_GRADIENTS_NON_FINITE
    store.close()


def test_invalid_circuit_result_is_not_persisted(tmp_path):
    """A result with no structural hash (invalid circuit) is returned to
    the caller but not stored as a reusable measurement."""
    store = DiagnosticStore.open_or_create(
        tmp_path / "d.sqlite", run_id="r1", config_json="{}",
        config_reproducibility_fields=REPRO, git_sha=None, created_at="t0",
    )
    store.put_cached(_result(structural_hash=None), run_seed=0)
    assert store.count() == 0
    store.close()


def test_resume_identical_config_preserves_results(tmp_path):
    db = tmp_path / "d.sqlite"
    s1 = DiagnosticStore.open_or_create(
        db, run_id="r1", config_json="{}", config_reproducibility_fields=REPRO,
        git_sha=None, created_at="t0",
    )
    s1.put_cached(_result(), run_seed=0)
    s1.close()
    s2 = DiagnosticStore.open_or_create(
        db, run_id="r1", config_json="{}", config_reproducibility_fields=REPRO,
        git_sha=None, created_at="t1",
    )
    assert s2.count() == 1
    s2.close()


def test_resume_incompatible_config_is_rejected(tmp_path):
    db = tmp_path / "d.sqlite"
    DiagnosticStore.open_or_create(
        db, run_id="r1", config_json="{}", config_reproducibility_fields=REPRO,
        git_sha=None, created_at="t0",
    ).close()
    with pytest.raises(IncompatibleDiagnosticResumeError):
        DiagnosticStore.open_or_create(
            db, run_id="r1", config_json="{}",
            config_reproducibility_fields={"config": {"n_pairs": 999, "n_bins": 75}},
            git_sha=None, created_at="t1",
        )


def test_partial_results_survive_abrupt_close_and_reopen(tmp_path):
    db = tmp_path / "d.sqlite"
    s1 = DiagnosticStore.open_or_create(
        db, run_id="r1", config_json="{}", config_reproducibility_fields=REPRO,
        git_sha=None, created_at="t0",
    )
    s1.put_cached(_result(), run_seed=0)
    s1.close()
    s2 = DiagnosticStore(db)  # raw reopen, simulating crash recovery
    assert s2.count() == 1
    assert s2.get_cached("h1", "expressibility", EXPRESSIBILITY_VERSION, "cfg1", 0) is not None
    s2.close()
