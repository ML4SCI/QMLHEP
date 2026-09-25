"""The robustness code must reduce EXACTLY to the previous QAE study at the
reference condition (4 qubits, TFIM, B=8).

If any of these fail, the reference cell may no longer be reused as the
baseline column of the robustness matrix.
"""
import numpy as np
import pytest

from llm_vqc.experiments.qae_robustness import arms
from llm_vqc.experiments.qae_robustness import conditions as C
from llm_vqc.experiments.qae_robustness import prompts as P
from llm_vqc.experiments.qae_tfim import neutral_v3 as v3
from llm_vqc.experiments.qae_tfim import neutral_v4 as v4
from llm_vqc.experiments.qae_tfim import neutral_v5 as v5
from llm_vqc.experiments.qae_tfim import pilot

REF = C.REFERENCE
SPACE = C.space_for(REF)


def test_reference_width_matches_the_previous_contract():
    assert (SPACE.n_rotations, SPACE.n_cnots, SPACE.n_gates) == (12, 4, 16)
    assert SPACE.latent == (0, 1)
    assert SPACE.trash == (2, 3)
    assert REF.n_warm == 4


def test_width_rule_is_three_rotations_and_one_cnot_per_qubit():
    for n in (4, 6, 8):
        space = C.Space(n)
        assert space.n_rotations == 3 * n
        assert space.n_cnots == n
        assert space.latent == tuple(range(n // 2))
        assert space.trash == tuple(range(n // 2, n))


@pytest.mark.parametrize("seed", [0, 5, 11])
def test_state_sets_are_bit_identical_to_the_previous_study(seed):
    new = C.state_sets(seed, "TFIM", 4)
    old = pilot.state_sets(seed)
    for a, b in zip(new, old, strict=True):
        assert np.array_equal(a, b)


def test_trash_indices_and_train_seed_rule_are_unchanged():
    assert np.array_equal(C.trash_zero_indices(4), pilot.TRASH00_INDICES)
    arch = C.sample_neutral_arch(np.random.default_rng(3), SPACE)
    assert C.train_seed(100, arch) == pilot._train_seed(100, arch)


def test_architecture_and_mutation_streams_are_unchanged():
    a, b = np.random.default_rng(62_000), np.random.default_rng(62_000)
    assert all(C.sample_neutral_arch(a, SPACE) == v3.sample_neutral_arch(b)
               for _ in range(25))
    base = C.sample_neutral_arch(np.random.default_rng(1), SPACE)
    a, b = np.random.default_rng(7), np.random.default_rng(7)
    assert all(C.mutate_one_choice(base, a, SPACE) == v3.mutate_one_choice(base, b)
               for _ in range(25))


def test_trainer_is_bit_identical_to_the_previous_study():
    train, val, test = pilot.state_sets(0)
    for k in range(3):
        arch = C.sample_neutral_arch(np.random.default_rng(200 + k), SPACE)
        seed = pilot._train_seed(100, arch)
        assert (C.evaluate_architecture(arch, train, val, test, seed=seed, n=4).__dict__
                == pilot.evaluate_architecture(arch, train, val, test, seed=seed).__dict__)


def test_every_prompt_is_byte_identical_to_the_previous_study():
    arch = C.sample_neutral_arch(np.random.default_rng(0), SPACE)
    assert P.SYSTEM_PROMPT == v3.SYSTEM_PROMPT
    assert P.task_card(REF) == v3.TASK_CARD
    assert P.json_schema_card(REF) == v3.JSON_SCHEMA_CARD
    assert P.open_loop_prompt(REF) == v3.OPEN_LOOP_PROMPT
    assert P.warmstart_prompt(REF) == v4.WARMSTART_PROMPT
    assert P.redesign_prompt(REF, arch, 0.9123) == v5.redesign_prompt(arch, 0.9123)


def test_sampling_controls_and_retry_policy_are_unchanged():
    assert P.TEMPERATURE == v3.TEMPERATURE == 0.7
    assert P.MAX_REPAIR_ATTEMPTS == v3.MAX_REPAIR_ATTEMPTS == 2
    assert P.max_output_tokens(REF, REF.budget) == 6000
    assert P.max_output_tokens(REF, REF.n_warm) == 6000
    assert P.max_output_tokens(REF, 1) == 6000
    assert arms.POOL_CALL_COST_ESTIMATE_USD == v3.POOL_CALL_COST_ESTIMATE_USD
    assert arms.WARM_CALL_COST_ESTIMATE_USD == v4.WARM_CALL_COST_ESTIMATE_USD
    assert arms.REFINE_CALL_COST_ESTIMATE_USD == v5.REFINE_CALL_COST_ESTIMATE_USD


def test_rng_stream_offsets_match_the_previous_study():
    assert arms.RANDOM_STREAM == 62_000       # neutral_v5.run_random_seed
    assert arms.GREEDY_STREAM == 93_000       # neutral_v5.run_greedy_seed
    assert arms.CLOSED_FALLBACK_STREAM == 86_000
    assert arms.WARM_DEFICIT_STREAM == 95_000
    assert arms.OPEN_DEFICIT_STREAM == 999_999


@pytest.mark.parametrize("seed", [0, 6])
def test_random_and_greedy_arms_reproduce_the_previous_study_rows(seed):
    train, val, test = C.state_sets(seed, "TFIM", 4)
    for new_rows, old_rows in (
        (arms.run_random_seed(REF, seed, train, val, test),
         v5.run_random_seed(seed, train, val, test)),
        (arms.run_greedy_seed(REF, seed, train, val, test),
         v5.run_greedy_seed(seed, train, val, test)),
    ):
        assert len(new_rows) == 8
        for new, old in zip(new_rows, old_rows, strict=True):
            for key in ("candidate", "order", "phase", "fallback",
                        "train_loss", "val_loss", "test_loss",
                        "train_fid", "val_fid", "test_fid"):
                assert new[key] == old[key], key
