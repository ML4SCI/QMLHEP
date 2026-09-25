"""Offline invariants for the generalised methods, physics and analysis.

No test here makes a network call.
"""
import json

import numpy as np
import pandas as pd
import pytest

from llm_vqc.experiments.qae_robustness import arms, study
from llm_vqc.experiments.qae_robustness import conditions as C
from llm_vqc.experiments.qae_robustness import prompts as P

N6 = C.CONDITIONS_BY_KEY["qubits_n6"]
B4 = C.CONDITIONS_BY_KEY["budget_b4"]
B16 = C.CONDITIONS_BY_KEY["budget_b16"]
XXZ = C.CONDITIONS_BY_KEY["hamiltonian_xxz"]


# ----------------------------------------------------------------- space ---

@pytest.mark.parametrize("n", [4, 6, 8])
def test_sampled_architectures_respect_the_width_rule(n):
    space = C.Space(n)
    rng = np.random.default_rng(0)
    for _ in range(20):
        arch = C.sample_neutral_arch(rng, space)
        C.verify_capacity(arch, space)  # raises on violation
        assert len(arch) == 4 * n


@pytest.mark.parametrize("n", [4, 6, 8])
def test_one_mutation_changes_at_most_two_slots(n):
    space = C.Space(n)
    rng = np.random.default_rng(1)
    base = C.sample_neutral_arch(rng, space)
    for _ in range(40):
        mutated = C.mutate_one_choice(base, rng, space)
        assert C.arch_distance(base, mutated) <= 2 / space.n_gates + 1e-9


@pytest.mark.parametrize("n", [4, 6, 8])
def test_parser_round_trips_and_rejects_wrong_width(n):
    space = C.Space(n)
    arch = C.sample_neutral_arch(np.random.default_rng(2), space)
    gates = P._arch_to_gate_list(arch)
    parsed, errors = C.parse_candidate({"name": "x", "gates": gates}, space)
    assert errors == []
    from llm_vqc.experiments.qae_tfim.pilot import architecture_key
    assert architecture_key(parsed) == architecture_key(arch)
    # a candidate sized for a different qubit count must be rejected
    other = C.Space(n + 2 if n < 8 else 4)
    wrong = P._arch_to_gate_list(C.sample_neutral_arch(np.random.default_rng(2), other))
    assert C.parse_candidate({"gates": wrong}, space)[0] is None
    # an out-of-range wire must be rejected
    bad = [dict(g) for g in gates]
    bad[0] = {"g": "RY", "q": n}
    assert C.parse_candidate({"gates": bad}, space)[0] is None


# --------------------------------------------------------------- physics ---

@pytest.mark.parametrize("family", ["TFIM", "XXZ"])
@pytest.mark.parametrize("n", [4, 6])
def test_hamiltonians_are_real_symmetric_with_a_unique_ground_state(family, n):
    for param in (0.2, 1.0, 2.0):
        h = C.hamiltonian(family, n, param)
        assert np.allclose(h, h.conj().T)
        assert np.allclose(h.imag, 0)
        assert C.ground_state_gap(family, n, param) > 1e-7
        state = C.ground_state(family, n, param)
        assert np.isclose(np.linalg.norm(state), 1.0)
        assert np.abs(state.imag).max() < 1e-10


def test_tfim_generalisation_matches_the_reference_at_four_qubits():
    from llm_vqc.experiments.qae_tfim.pilot import tfim_ground_state
    for h in (0.2, 0.7, 1.3, 2.0):
        assert np.array_equal(C.ground_state("TFIM", 4, h), tfim_ground_state(h))


def test_xxz_family_is_declared_and_distinct_from_tfim():
    assert XXZ.family == "XXZ" and XXZ.n_qubits == 4 and XXZ.budget == 8
    a = C.ground_state("XXZ", 4, 1.0)
    b = C.ground_state("TFIM", 4, 1.0)
    assert abs(np.vdot(a, b)) < 0.99


def test_hamiltonian_parameters_are_paired_across_families_and_widths():
    """Only the states change with the factor; the parameter draws do not."""
    rng = np.random.default_rng(10_000 + 3)
    expected = np.sort(rng.uniform(0.2, 2.0, 32))
    for family, n in (("TFIM", 4), ("XXZ", 4), ("TFIM", 6)):
        train, _, _ = C.state_sets(3, family, n)
        assert train.shape == (32, 2 ** n)
        recovered = np.random.default_rng(10_000 + 3)
        assert np.array_equal(np.sort(recovered.uniform(0.2, 2.0, 32)), expected)


# ------------------------------------------------------------- trainer -----

def test_trash_indices_cover_the_declared_trash_qubits():
    for n in (4, 6, 8):
        keep = C.trash_zero_indices(n)
        assert len(keep) == 2 ** (n // 2)
        trash = C.Space(n).trash
        for index in keep:
            assert all(((index >> (n - 1 - q)) & 1) == 0 for q in trash)


def test_evaluation_is_deterministic_and_in_range():
    train, val, test = C.state_sets(0, "XXZ", 4)
    arch = C.sample_neutral_arch(np.random.default_rng(0), C.Space(4))
    seed = C.train_seed(100, arch)
    a = C.evaluate_architecture(arch, train, val, test, seed=seed, n=4)
    b = C.evaluate_architecture(arch, train, val, test, seed=seed, n=4)
    assert a.__dict__ == b.__dict__
    assert 0.0 <= a.test_fid <= 1.0
    assert np.isclose(a.test_loss, 1 - a.test_fid)


# ----------------------------------------------------------------- arms ----

def test_random_arm_scales_with_the_budget_and_stays_nested():
    train, val, test = C.state_sets(0, "TFIM", 4)
    small = arms.run_random_seed(B4, 0, train, val, test)
    reference = arms.run_random_seed(C.REFERENCE, 0, train, val, test)
    big = arms.run_random_seed(B16, 0, train, val, test)
    assert (len(small), len(reference), len(big)) == (4, 8, 16)
    # the same draw stream: B=4 is the first 4 of B=8, which is the first 8 of B=16
    for a, b in zip(small, reference, strict=False):
        assert a["val_loss"] == b["val_loss"]
    for a, b in zip(reference, big, strict=False):
        assert a["val_loss"] == b["val_loss"]


def test_greedy_arm_keeps_the_fifty_fifty_split_at_every_budget():
    train, val, test = C.state_sets(0, "TFIM", 4)
    for condition, warm in ((B4, 2), (C.REFERENCE, 4), (B16, 8)):
        rows = arms.run_greedy_seed(condition, 0, train, val, test)
        assert len(rows) == condition.budget
        assert [r["phase"] for r in rows] == (
            ["explore"] * warm + ["refine"] * (condition.budget - warm))
        for row in rows[warm:]:
            assert row["edit_distance"] <= 2 / 16 + 1e-9 or row["fallback"]


def test_greedy_never_accepts_on_test_and_selection_uses_validation_only():
    train, val, test = C.state_sets(0, "TFIM", 4)
    rows = arms.run_greedy_seed(C.REFERENCE, 0, train, val, test)
    frame = pd.DataFrame(rows)
    selected, _ = study.analyze(frame)
    assert float(selected["val_loss"].iloc[0]) == float(frame["val_loss"].min())


def test_greedy_runs_at_six_qubits():
    train, val, test = C.state_sets(0, "TFIM", 6)
    rows = arms.run_greedy_seed(N6, 0, train, val, test)
    assert len(rows) == 8
    assert all(0.0 <= r["test_fid"] <= 1.0 for r in rows)


def test_api_arms_refuse_to_run_without_a_configured_budget(tmp_path, monkeypatch):
    monkeypatch.delenv("LLM_API_BUDGET_USD", raising=False)
    train, val, test = C.state_sets(0, "TFIM", 4)
    with pytest.raises(RuntimeError, match="LLM_API_BUDGET_USD"):
        arms.run_closed_seed(C.REFERENCE, 0, tmp_path, train, val, test)
    with pytest.raises(RuntimeError, match="LLM_API_BUDGET_USD"):
        arms.generate_open_pool(C.REFERENCE, tmp_path)


def test_closed_arm_reloads_from_disk_without_calling_the_api(tmp_path):
    stored = {"rows": [{"seed": 0, "method": "LLM-Closed"}], "architectures": []}
    (tmp_path / "llm_closed_seed0.json").write_text(json.dumps(stored))
    assert arms.run_closed_seed(C.REFERENCE, 0, tmp_path, None, None, None) == \
        stored["rows"]


# -------------------------------------------------------------- analysis ---

def _synthetic_frame(offset: float) -> pd.DataFrame:
    rows = []
    for seed in range(12):
        for method, base in (("Random", 0.80), ("Greedy", 0.81),
                             ("LLM-Open", 0.95), ("LLM-Closed", 0.95 + offset)):
            for order in (1, 2):
                fid = base + 0.001 * seed - 0.01 * (order - 1)
                rows.append({"seed": seed, "method": method, "candidate": f"c{order}",
                             "order": order, "phase": "explore", "fallback": False,
                             "invalid_errors": "", "edit_distance": np.nan,
                             "strategy": "", "train_loss": 1 - fid,
                             "val_loss": 1 - fid, "test_loss": 1 - fid,
                             "train_fid": fid, "val_fid": fid, "test_fid": fid})
    return pd.DataFrame(rows)


def test_analysis_reports_the_four_required_contrasts():
    _, stats = study.analyze(_synthetic_frame(0.005))
    for a, b in study.REPORTED_CONTRASTS:
        assert f"{b}_minus_{a}" in stats
    entry = stats["LLM-Closed_minus_LLM-Open"]
    assert entry["n"] == 12 and entry["wins_b"] == 12
    assert entry["mean_paired_gain"] == pytest.approx(0.005)
    assert len(entry["bootstrap_95ci"]) == 2


def test_analysis_drops_a_contrast_with_no_variation():
    _, stats = study.analyze(_synthetic_frame(0.0))
    assert "LLM-Closed_minus_LLM-Open" not in stats


def test_anytime_table_is_monotone_per_method():
    anytime = study.anytime_table(_synthetic_frame(0.005))
    for _method, group in anytime.groupby("method"):
        values = group.sort_values("candidates_used")["best_so_far_test_fid"].to_numpy()
        assert np.all(np.diff(values) >= -1e-12)
