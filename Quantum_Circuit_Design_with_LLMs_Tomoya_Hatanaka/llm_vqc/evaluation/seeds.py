"""Deterministic seed derivation.

No single ambiguous global seed anywhere: every distinct source of
randomness in the evaluation pipeline (data splitting, preprocessing,
minibatch ordering, circuit parameter initialization, training, synthetic
test fixtures, backend sampling) gets its own reproducible sub-seed,
derived from a base seed via `numpy.random.SeedSequence` — the
NumPy-recommended mechanism for splitting one seed into many independent,
statistically-uncorrelated child streams (as opposed to ad hoc string
hashing, which does not carry the same independence guarantee).

The one seed relationship the master plan makes load-bearing (Section
6.4): "per-candidate training seed drawn deterministically from run seed
+ structural hash, so the same circuit gets the same training result in
every arm." `train_seed_for_circuit` implements exactly that — it is a
pure function of `(run_seed, structural_hash)` alone, nothing else, which
is also what makes it usable as a cache key component (`llm_vqc.
evaluation.store`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _stable_int_from_string(text: str) -> int:
    """Stable string -> int, independent of PYTHONHASHSEED (unlike hash())."""
    import hashlib

    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def derive_child_seed(parent_seed: int, *labels: str) -> int:
    """Deterministically derive one child seed from a parent seed + labels."""
    entropy = [int(parent_seed) & 0xFFFFFFFF] + [_stable_int_from_string(label) for label in labels]
    seq = np.random.SeedSequence(entropy)
    return int(seq.generate_state(1, dtype=np.uint32)[0])


def train_seed_for_circuit(run_seed: int, structural_hash: str) -> int:
    """The single per-candidate training seed (master plan Section 6.4).

    Deterministic given (run_seed, structural_hash) alone — no seed index,
    no arm identity, no wall-clock component — so the same circuit
    trained under the same run_seed always gets the same training seed
    regardless of which search arm proposed it.
    """
    return derive_child_seed(run_seed, "train_seed", structural_hash)


def data_split_seed(run_seed: int, task_name: str) -> int:
    return derive_child_seed(run_seed, "data_split", task_name)


def preprocessing_seed(run_seed: int, task_name: str) -> int:
    return derive_child_seed(run_seed, "preprocessing", task_name)


def synthetic_fixture_seed(run_seed: int, fixture_name: str) -> int:
    return derive_child_seed(run_seed, "synthetic_fixture", fixture_name)


@dataclass(frozen=True)
class TrainingSeeds:
    """Independent sub-seeds for one training run, all derived from one
    `train_seed` via `SeedSequence.spawn()` (guaranteed independent child
    streams, not re-derived by ad hoc hashing of the same seed four
    times)."""

    param_init: int
    minibatch: int
    training: int
    backend_sampling: int

    @classmethod
    def from_train_seed(cls, train_seed: int) -> TrainingSeeds:
        parent = np.random.SeedSequence(int(train_seed) & 0xFFFFFFFF)
        child_param_init, child_minibatch, child_training, child_backend = parent.spawn(4)
        return cls(
            param_init=int(child_param_init.generate_state(1, dtype=np.uint32)[0]),
            minibatch=int(child_minibatch.generate_state(1, dtype=np.uint32)[0]),
            training=int(child_training.generate_state(1, dtype=np.uint32)[0]),
            backend_sampling=int(child_backend.generate_state(1, dtype=np.uint32)[0]),
        )
