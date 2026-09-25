import numpy as np

from llm_vqc.experiments.qae_tfim.pilot import (
    SEMANTIC_CANDIDATES,
    sample_random_arch,
    tfim_ground_state,
    verify_capacity,
)


def test_semantic_candidates_have_identical_capacity():
    assert len(SEMANTIC_CANDIDATES) == 8
    for architecture in SEMANTIC_CANDIDATES.values():
        verify_capacity(architecture)


def test_random_sampler_has_identical_capacity():
    rng = np.random.default_rng(123)
    for _ in range(20):
        verify_capacity(sample_random_arch(rng))
        verify_capacity(sample_random_arch(rng, ry_only=True))


def test_tfim_ground_state_is_normalized_and_effectively_real():
    state = tfim_ground_state(1.0)
    assert np.isclose(np.linalg.norm(state), 1.0)
    assert np.max(np.abs(state.imag)) < 1e-10
