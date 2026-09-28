"""Explicit, versioned parameter-initialization policy for the controlled experiment.

The legacy runs used PyTorch/PennyLane *dependency-default* initialization under a
seeded RNG (documented as such, never changed). This experiment introduces an
**explicit, versioned** policy so the initialization is a declared scientific
choice rather than an implicit dependency default — and so it is identical for
every arm (never tuned per arm, never chosen after seeing results).

Policy `explicit_kaiming_uniform_qc_v1`:

- classical embed / head Linear weights and biases: uniform on
  `[-1/sqrt(fan_in), +1/sqrt(fan_in)]` with `fan_in = in_features` — exactly the
  distribution of PyTorch's default `nn.Linear` reset (Kaiming-uniform weight
  with `a=sqrt(5)` reduces to this bound; bias uses the same bound), made
  explicit and version-robust rather than relying on the framework default;
- quantum gate angles: `U(-pi, +pi)` — the full rotation-angle range, symmetric
  about 0, declared before any results were seen.

All draws come from one `torch.Generator` seeded by the run's `param_init`
sub-seed, in a fixed order (embed.weight, embed.bias, q_layer.weights,
head.weight, head.bias), so initialization is fully deterministic and
reproducible.
"""

from __future__ import annotations

import math

import torch

INIT_POLICY_NAME = "explicit_kaiming_uniform_qc_v1"
INIT_POLICY_VERSION = "v1"
QUANTUM_INIT_LOW = -math.pi
QUANTUM_INIT_HIGH = math.pi
DIFF_METHOD = "best"  # to_qnode default; resolves to backprop on default.qubit


def _init_linear(layer: torch.nn.Linear, gen: torch.Generator) -> None:
    fan_in = layer.in_features
    bound = 1.0 / math.sqrt(fan_in)
    with torch.no_grad():
        layer.weight.uniform_(-bound, bound, generator=gen)
        if layer.bias is not None:
            layer.bias.uniform_(-bound, bound, generator=gen)


def apply_explicit_init(model: torch.nn.Module, seed: int) -> None:
    """Deterministically (re)initialize a HybridQNNModel under this policy."""
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    # order matters for reproducibility: embed -> quantum -> head
    _init_linear(model.embed, gen)
    with torch.no_grad():
        model.q_layer.weights.uniform_(QUANTUM_INIT_LOW, QUANTUM_INIT_HIGH, generator=gen)
    _init_linear(model.head, gen)


def init_policy_metadata() -> dict:
    return {
        "name": INIT_POLICY_NAME,
        "version": INIT_POLICY_VERSION,
        "classical_weight": "uniform[-1/sqrt(fan_in), +1/sqrt(fan_in)] (PyTorch Linear semantics, explicit)",
        "classical_bias": "uniform[-1/sqrt(fan_in), +1/sqrt(fan_in)]",
        "quantum_angles": f"uniform[{QUANTUM_INIT_LOW}, {QUANTUM_INIT_HIGH}]",
        "seed_source": "TrainingSeeds.param_init (SeedSequence-derived)",
        "draw_order": ["embed.weight", "embed.bias", "q_layer.weights", "head.weight", "head.bias"],
        "diff_method": DIFF_METHOD,
    }
