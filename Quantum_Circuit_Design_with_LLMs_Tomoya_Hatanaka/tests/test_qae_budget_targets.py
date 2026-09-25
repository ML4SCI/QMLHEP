"""Tests for the minimum-budget-to-target study.

Covers what the protocol promises: threshold equality, the 10-of-12 rule,
non-attainment, fallback accounting, missing seeds, resume, cost-cap
enforcement, test-set protection, and backward compatibility of the
arbitrary-budget path with the frozen B = 4 / 8 / 16 behaviour.
"""
from __future__ import annotations

import importlib.util
import json
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from llm_vqc.experiments.qae_budget_targets import conditions as BT
from llm_vqc.experiments.qae_budget_targets import runner as R
from llm_vqc.experiments.qae_robustness import arms
from llm_vqc.experiments.qae_robustness import conditions as C
from llm_vqc.experiments.qae_robustness import manifest as M
from llm_vqc.experiments.qae_robustness import prompts as P
from llm_vqc.llm.budget import LLMApiBudget, LLMBudgetExceededError

_SPEC = importlib.util.spec_from_file_location(
    "analyze_budget_targets", Path("scripts/qae/analyze_budget_targets.py")
)
audit = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(audit)


# ------------------------------------------------------ decision rule ------

def _trace_rows(values_by_seed: dict[int, list[float]], method="LLM-Closed",
                fallback_seeds=()) -> list[dict]:
    rows = []
    for seed, values in values_by_seed.items():
        for order, value in enumerate(values, 1):
            row = {column: "" for column in audit.SAFE_COLUMNS}
            row.update(seed=str(seed), method=method, order=str(order),
                       val_fid=str(value),
                       fallback="True" if seed in fallback_seeds else "False")
            rows.append(row)
    return rows


def _cell(values_by_seed, budget, method="LLM-Closed", fallback_seeds=()):
    candidates = _trace_rows(values_by_seed, method, fallback_seeds)
    selected = []
    for seed, values in values_by_seed.items():
        row = {column: "" for column in audit.SAFE_COLUMNS}
        row.update(seed=str(seed), method=method, order=str(budget),
                   val_fid=str(max(values)), fallback="False")
        selected.append(row)
    return audit.audit_condition("cell", budget, candidates, selected, (method,))


def test_threshold_equality_counts_as_a_hit():
    """F exactly equal to the target passes; one ulp below does not."""
    assert audit.first_hit([0.95], 0.95) == 1
    assert audit.first_hit([np.nextafter(0.95, 0.0)], 0.95) is None
    exactly = {s: [0.95] for s in range(12)}
    endpoints = _cell(exactly, 1)[1]
    assert [e["successes"] for e in endpoints if e["target"] == 0.95] == [12]


def test_ten_of_twelve_is_the_boundary():
    for hits, expected in ((9, False), (10, True), (12, True)):
        values = {s: [0.96 if s < hits else 0.80] for s in range(12)}
        endpoints = _cell(values, 1)[1]
        row = next(e for e in endpoints if e["target"] == 0.95)
        assert row["successes"] == hits
        assert row["passes"] is expected
        assert row["required"] == 10 and row["n_seeds"] == 12


def test_mean_above_target_does_not_pass():
    """Nine excellent seeds and three poor ones: mean > 0.95, rule says fail."""
    values = {s: [1.0 if s < 9 else 0.90] for s in range(12)}
    row = next(e for e in _cell(values, 1)[1] if e["target"] == 0.95)
    assert row["mean_validation"] > 0.95
    assert row["successes"] == 9 and row["passes"] is False


def test_non_attainment_is_right_censored_not_imputed():
    values = {s: [0.90, 0.94] for s in range(12)}
    curves, endpoints, hits, prefixes, _ = _cell(values, 2)
    assert all(h["right_censored"] and h["first_hit_in_this_trace"] is None
               for h in hits)
    assert all(h["censor_at"] == 2 for h in hits)
    assert all(p["first_k_with_10_of_12_in_this_trace"] is None for p in prefixes)
    assert all(e["successes"] == 0 for e in endpoints)


def test_best_so_far_is_monotone_and_never_looks_ahead():
    assert audit.running_best([0.8, 0.7, 0.96, 0.9]) == [0.8, 0.8, 0.96, 0.96]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.01, 1.01])
def test_non_finite_or_out_of_range_validation_is_rejected(bad):
    with pytest.raises(ValueError):
        audit.running_best([bad])


def test_fallback_candidates_stay_in_the_primary_analysis():
    """A random fallback consumes budget and counts; the exclusion is only
    reported as a separate diagnostic column."""
    values = {s: [0.96] for s in range(12)}
    endpoints = _cell(values, 1, fallback_seeds=tuple(range(4)))[1]
    row = next(e for e in endpoints if e["target"] == 0.95)
    assert row["successes"] == 12
    assert row["successes_without_fallback_candidates_diagnostic_only"] == 8


def test_a_missing_seed_is_an_error_not_a_failed_seed():
    values = {s: [0.96] for s in range(11)}  # seed 11 absent
    with pytest.raises(ValueError, match="Incomplete or unexpected"):
        _cell(values, 1)


def test_a_missing_candidate_within_a_seed_is_an_error():
    values = {s: [0.9, 0.96] for s in range(12)}
    candidates = _trace_rows(values)
    selected = [r for r in candidates if r["order"] == "2"]
    with pytest.raises(ValueError, match="Missing, duplicate, or noncontiguous"):
        audit.audit_condition("cell", 2, candidates[:-1], selected, ("LLM-Closed",))


def test_single_method_cell_is_audited_without_inventing_the_others():
    """The XXZ probe runs LLM-Closed alone; that must not be reported as
    incomplete, and no row may appear for a method that never ran."""
    endpoints = _cell({s: [0.96] for s in range(12)}, 1)[1]
    assert {e["method"] for e in endpoints} == {"LLM-Closed"}


def test_historical_default_condition_set_is_unchanged():
    assert audit.CONDITIONS == {
        "budget_b4": 4, "reference": 8, "budget_b16": 16, "qubits_n6": 8,
        "qubits_n8": 8, "hamiltonian_xxz": 8, "model_alt": 8}
    assert audit.TARGETS == (0.95, 0.99) and audit.REQUIRED == 10


# ------------------------------------------------------- admissibility -----

def test_admissible_grid_and_fifty_fifty_split():
    for budget in BT.ADMISSIBLE_BUDGETS:
        condition = BT.make_condition("x", budget=budget, anchor=C.REFERENCE)
        assert condition.n_warm == budget // 2
        assert condition.budget - condition.n_warm == budget // 2


@pytest.mark.parametrize("budget", [3, 5, 7, 12, 20])
def test_budgets_off_the_frozen_grid_are_refused(budget):
    with pytest.raises(BT.InadmissibleBudgetError):
        BT.make_condition("x", budget=budget, anchor=C.REFERENCE)


def test_odd_budget_is_refused_by_the_condition_itself():
    with pytest.raises(ValueError, match="even"):
        C.Condition(key="odd", factor="budget", budget=7)


def test_declared_cells_match_the_protocol():
    b6 = BT.TARGET_CELLS_BY_KEY["target_tfim_b6"]
    assert (b6.budget, b6.condition.family, b6.condition.n_qubits) == (6, "TFIM", 4)
    assert b6.methods == BT.METHODS and b6.anchor.key == "reference"
    assert b6.condition.model == C.REFERENCE_MODEL

    b10 = BT.TARGET_CELLS_BY_KEY["target_xxz_b10"]
    assert (b10.budget, b10.condition.family) == (10, "XXZ")
    assert b10.methods == ("LLM-Closed",)
    assert b10.anchor.key == "hamiltonian_xxz"
    assert b10.condition.model == C.REFERENCE_MODEL


# ---------------------------------------------------------- anchoring ------

def test_the_original_one_factor_verifier_still_holds_for_every_old_condition():
    assert M.all_manifests()["violations"] == {c.key: [] for c in C.CONDITIONS}


def test_each_target_cell_is_one_factor_from_its_declared_anchor():
    for cell in BT.TARGET_CELLS:
        assert M.verify(cell.condition, cell.anchor) == []
        assert C.changed_factors(cell.condition, cell.anchor) == ["budget"]


def test_the_xxz_probe_is_not_silently_admitted_against_the_tfim_reference():
    cell = BT.TARGET_CELLS_BY_KEY["target_xxz_b10"]
    problems = M.verify(cell.condition, C.REFERENCE)
    assert any("varies 2 factors" in p for p in problems)


def test_an_anchor_must_itself_be_a_clean_condition():
    dirty = C.Condition(key="dirty", factor="qubits", n_qubits=6, budget=16)
    probe = C.Condition(key="probe", factor="budget", n_qubits=6, budget=10)
    assert any("is not itself a clean condition" in p for p in M.verify(probe, dirty))


def test_target_cells_inherit_the_frozen_block_byte_for_byte():
    reference = M.condition_manifest(C.REFERENCE)["frozen"]
    for cell in BT.TARGET_CELLS:
        manifest = M.condition_manifest(cell.condition, cell.anchor)
        assert C.canonical_json(manifest["frozen"]) == C.canonical_json(reference)
        assert manifest["derived"] == M.derive(manifest["factors"])


# ------------------------------------------------------ test protection ----

def test_no_allowlisted_column_mentions_the_test_set():
    assert not any("test" in column for column in R.SAFE_COLUMNS)


def test_the_runner_hands_the_trainer_an_empty_test_array():
    train, val, test = R.validation_only_state_sets(0, "TFIM", 4)
    assert test.shape == (0, 16)
    assert train.shape[0] == 32 and val.shape[0] == 12


def test_evaluating_with_the_empty_test_array_produces_no_test_number():
    condition = BT.make_condition("b4", budget=4, anchor=C.REFERENCE)
    train, val, test = R.validation_only_state_sets(0, "TFIM", 4)
    space = C.space_for(condition)
    architecture = C.sample_neutral_arch(np.random.default_rng(0), space)
    result = C.evaluate_architecture(architecture, train, val, test, seed=1, n=4)
    assert np.isfinite(result.val_fid) and 0 <= result.val_fid <= 1
    assert not np.isfinite(result.test_fid)


def test_a_finite_test_value_trips_the_guard():
    with pytest.raises(AssertionError, match="never be evaluated"):
        R.assert_no_test_values([{"seed": 0, "test_fid": 0.9}])
    R.assert_no_test_values([{"seed": 0, "test_fid": float("nan")}])


def test_projection_drops_every_non_allowlisted_column(tmp_path):
    rows = [{"seed": 0, "method": "Random", "candidate": "R1", "order": 1,
             "phase": "explore", "fallback": False, "invalid_errors": "",
             "edit_distance": float("nan"), "strategy": "",
             "val_loss": 0.1, "val_fid": 0.9, "train_loss": 0.1,
             "train_fid": 0.9, "test_loss": float("nan"),
             "test_fid": float("nan")}]
    projected = R.project(rows)
    assert set(projected[0]) == set(R.SAFE_COLUMNS)
    R.write_csv(tmp_path / "c.csv", projected)
    header = (tmp_path / "c.csv").read_text().splitlines()[0]
    assert "test" not in header


def test_selection_uses_validation_loss_only():
    rows = [
        {"seed": 0, "method": "Random", "order": 1, "val_loss": 0.30},
        {"seed": 0, "method": "Random", "order": 2, "val_loss": 0.10},
        {"seed": 0, "method": "Random", "order": 3, "val_loss": 0.20},
    ]
    assert R.select(rows)[0]["order"] == 2


# ------------------------------------------------ legacy regression --------

@pytest.mark.parametrize("key,budget", [("budget_b4", 4), ("budget_b16", 16)])
def test_arbitrary_budget_path_reproduces_the_frozen_legacy_behaviour(key, budget):
    """The new arbitrary-budget condition must be the SAME object as the old
    one at a legacy budget, and must reproduce the committed non-API rows."""
    condition = BT.make_condition(key, budget=budget, anchor=C.REFERENCE)
    legacy = C.CONDITIONS_BY_KEY[key]
    assert condition.factors == legacy.factors
    assert condition.n_warm == legacy.n_warm
    assert (M.condition_manifest(condition)["derived"]
            == M.condition_manifest(legacy)["derived"])


def test_legacy_b4_non_api_arms_are_reproduced_bit_for_bit():
    """One seed of the committed B = 4 run, recomputed through the new
    validation-only path."""
    condition = BT.make_condition("budget_b4", budget=4, anchor=C.REFERENCE)
    train, val, test = R.validation_only_state_sets(0, "TFIM", 4)
    rows = (arms.run_random_seed(condition, 0, train, val, test)
            + arms.run_greedy_seed(condition, 0, train, val, test))
    frame = pd.DataFrame(rows).sort_values(["method", "order"])
    committed = pd.read_csv("outputs/qae_robustness/budget_b4/candidate_results.csv")
    committed = committed[(committed.seed == 0)
                          & committed.method.isin(["Random", "Greedy"])]
    committed = committed.sort_values(["method", "order"])
    assert list(frame.candidate) == list(committed.candidate)
    assert np.allclose(frame.val_fid.values, committed.val_fid.values, atol=1e-12)
    assert np.allclose(frame.val_loss.values, committed.val_loss.values, atol=1e-12)


# ------------------------------------------------------------ cost cap -----

def test_no_cap_configured_means_no_paid_call(monkeypatch, tmp_path):
    monkeypatch.delenv("LLM_API_BUDGET_USD", raising=False)
    assert LLMApiBudget.from_env() is None
    condition = BT.make_condition("b6", budget=6, anchor=C.REFERENCE)
    with pytest.raises(RuntimeError, match="LLM_API_BUDGET_USD"):
        arms.generate_open_pool(condition, tmp_path)


@pytest.mark.parametrize("raw", ["", "0", "-1", "abc"])
def test_an_absent_zero_or_unparseable_cap_is_refused(raw):
    assert LLMApiBudget.from_env({"LLM_API_BUDGET_USD": raw}) is None


def test_the_cap_stops_the_run_before_it_is_exceeded():
    budget = LLMApiBudget(cap_usd=0.02)
    budget.check_can_afford(0.01)
    budget.record_spend(0.01)
    budget.check_can_afford(0.01)
    budget.record_spend(0.01)
    with pytest.raises(LLMBudgetExceededError):
        budget.check_can_afford(0.005)
    assert budget.spent_usd == pytest.approx(0.02)


# ------------------------------------------------------------- resume ------

class _CountingProvider:
    """Returns valid, distinct architectures and counts the calls made."""

    def __init__(self, condition):
        self.condition = condition
        self.calls = 0
        self._rng = np.random.default_rng(7)

    def complete(self, system, prompt, temperature):
        self.calls += 1
        space = C.space_for(self.condition)
        wanted = prompt.count("Output 1 candidate") or 1
        n = 1 if "exactly 1 candidate" in prompt else self.condition.n_warm
        candidates = [
            {"name": f"c{self.calls}_{i}", "strategy": "global_redesign",
             "gates": P._arch_to_gate_list(C.sample_neutral_arch(self._rng, space))}
            for i in range(max(n, wanted))
        ]
        return types.SimpleNamespace(
            raw_text=json.dumps({"candidates": candidates}),
            model=self.condition.model, input_tokens=800, output_tokens=350,
            estimated_cost_usd=None, latency_seconds=0.1,
        )


def test_a_resumed_seed_costs_nothing_and_returns_the_stored_rows(
        monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_API_BUDGET_USD", "1.0")
    condition = BT.make_condition("b6", budget=6, anchor=C.REFERENCE)
    provider = _CountingProvider(condition)
    monkeypatch.setattr(arms, "build_provider", lambda c, n: provider)
    train, val, test = R.validation_only_state_sets(0, "TFIM", 4)

    first = arms.run_closed_seed(condition, 0, tmp_path, train, val, test)
    calls_after_first = provider.calls
    assert calls_after_first > 0
    assert len(first) == condition.budget
    assert (tmp_path / "llm_closed_seed0.json").exists()

    second = arms.run_closed_seed(condition, 0, tmp_path, train, val, test)
    assert provider.calls == calls_after_first, "resume must not re-call the API"
    assert [r["candidate"] for r in second] == [r["candidate"] for r in first]
    assert [r["val_fid"] for r in second] == [r["val_fid"] for r in first]


def test_resume_status_reports_only_seeds_actually_on_disk(tmp_path):
    cell = BT.TARGET_CELLS_BY_KEY["target_tfim_b6"]
    root = tmp_path / "root"
    (root / cell.key).mkdir(parents=True)
    assert R.completed_seeds(cell, root) == []
    (root / cell.key / "llm_closed_seed3.json").write_text("{}")
    assert R.completed_seeds(cell, root) == [3]


# -------------------------------------------------------------- costs ------

def test_cost_reconstruction_uses_official_list_prices(tmp_path):
    condition = BT.make_condition("b6", budget=6, anchor=C.REFERENCE)
    calls = tmp_path / "llm_calls"
    calls.mkdir()
    for i in range(3):
        (calls / f"closed_s0_call{i}.json").write_text(json.dumps(
            {"input_tokens": 1000, "output_tokens": 500, "model": condition.model,
             "call_index": i}))
    record = R.cost_record(tmp_path, condition)
    assert record["n_calls"] == 3
    assert record["n_repair_or_retry_calls"] == 2  # call_index > 0
    assert record["input_tokens"] == 3000 and record["output_tokens"] == 1500
    expected = round(3000 / 1e6 * 0.75 + 1500 / 1e6 * 4.50, 6)
    assert record["estimated_cost_usd_at_list_price"] == expected
    assert record["declared_model"] == C.REFERENCE_MODEL


def test_the_model_snapshot_is_never_silently_substituted():
    for cell in BT.TARGET_CELLS:
        assert cell.condition.model == cell.anchor.model == C.REFERENCE_MODEL
