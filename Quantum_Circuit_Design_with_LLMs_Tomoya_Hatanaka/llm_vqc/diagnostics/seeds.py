"""Deterministic seed derivation for diagnostics.

Same philosophy as `llm_vqc.evaluation.seeds` (no single ambiguous global
seed; independent child streams via `numpy.random.SeedSequence`), but
scoped to the distinct sources of randomness a diagnostic uses:

- **parameter_sampling** — the random weight vectors.
- **diagnostic_input** — the fixed diagnostic data input (drawn once, then
  held constant across all parameter samples).
- **backend_sampling** — finite-shot backend randomness, if ever enabled
  (statevector simulation is exact and uses none; reserved for forward
  compatibility so the seed contract does not change later).
- **bootstrap** — resampling for uncertainty estimation.
- **replicate** — the base for the R independent replicate seed-sets used
  for repeated-seed dispersion (each replicate spawns its own full set of
  the above streams).

`diagnostic_base_seed(run_seed, structural_hash, diagnostic_name)` mirrors
the Phase 2 `train_seed_for_circuit` contract: a diagnostic's randomness
is a pure function of `(run_seed, structural_hash, diagnostic_name)`, so
the same circuit measured under the same run seed gives the same
diagnostic result regardless of which search arm proposed it — and this
same triple is part of the store's cache identity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from llm_vqc.evaluation.seeds import derive_child_seed


def diagnostic_base_seed(run_seed: int, structural_hash: str, diagnostic_name: str) -> int:
    """Base seed for one (circuit, diagnostic) measurement.

    Pure function of (run_seed, structural_hash, diagnostic_name) — no
    arm identity, no wall-clock — so it is reproducible and usable as part
    of a cache key.
    """
    return derive_child_seed(run_seed, "diagnostic", diagnostic_name, structural_hash)


@dataclass(frozen=True)
class DiagnosticSeeds:
    """One replicate's independent sub-seeds, all spawned from a base seed."""

    parameter_sampling: int
    diagnostic_input: int
    backend_sampling: int
    bootstrap: int

    @classmethod
    def from_base_seed(cls, base_seed: int) -> DiagnosticSeeds:
        parent = np.random.SeedSequence(int(base_seed) & 0xFFFFFFFF)
        params, inp, backend, boot = parent.spawn(4)
        return cls(
            parameter_sampling=int(params.generate_state(1, dtype=np.uint32)[0]),
            diagnostic_input=int(inp.generate_state(1, dtype=np.uint32)[0]),
            backend_sampling=int(backend.generate_state(1, dtype=np.uint32)[0]),
            bootstrap=int(boot.generate_state(1, dtype=np.uint32)[0]),
        )


def replicate_seeds(base_seed: int, n_replicates: int) -> list[DiagnosticSeeds]:
    """R independent replicate seed-sets for repeated-seed dispersion.

    Each replicate gets a fully independent `DiagnosticSeeds` (its own
    parameter/input/backend/bootstrap streams), spawned from an
    independent child of `base_seed` — so replicates are statistically
    independent, not merely offset by a constant.
    """
    parent = np.random.SeedSequence(int(base_seed) & 0xFFFFFFFF)
    children = parent.spawn(n_replicates)
    return [
        DiagnosticSeeds.from_base_seed(int(child.generate_state(1, dtype=np.uint32)[0]))
        for child in children
    ]
