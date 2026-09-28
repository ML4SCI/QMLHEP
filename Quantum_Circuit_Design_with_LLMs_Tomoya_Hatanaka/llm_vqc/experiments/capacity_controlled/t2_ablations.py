"""Paired quantum-ablation builders for T2-v1.

- **A. frozen vs trainable**: 20 predeclared valid controlled architectures; each
  trained with quantum angles trainable vs frozen (identical init via the same
  train seed; only `requires_grad` differs).
- **B. product vs entangled**: 20 predeclared *entangled* architectures (>=1
  two-qubit gate), each deterministically mapped to a **parameter-matched product
  counterpart** — every CRZ param block -> RY rotation block (4 params each, count
  preserved) and every free entangling block -> H block (0 params) — so the pair
  shares encoding/measurement/classical-capacity/param-count and differs only in
  entanglement. Any gate-count mismatch is reported.
"""

from __future__ import annotations

import numpy as np

from llm_vqc.experiments.capacity_controlled import space as S
from llm_vqc.ir.expand import build_program
from llm_vqc.ir.metrics import circuit_cost_summary

# index of the Hadamard free-block template in space.FREE_BLOCK_TEMPLATES
_H_TEMPLATE = next(i for i, t in enumerate(S.FREE_BLOCK_TEMPLATES) if t[0] == "hadamard")
_ENTANGLE_TEMPLATES = {i for i, t in enumerate(S.FREE_BLOCK_TEMPLATES) if t[0] == "entangle"}

FREEZE_ARCH_SEEDS = tuple(range(1000, 1020))   # 20 architectures for ablation A


def predeclared_architectures(seeds=FREEZE_ARCH_SEEDS) -> list[S.ControlledGenome]:
    """20 structurally-distinct valid controlled architectures (fixed seeds)."""
    return [S.random_genome(np.random.default_rng(s)) for s in seeds]


def two_qubit_count(genome: S.ControlledGenome) -> int:
    return circuit_cost_summary(S.genome_to_ir(genome)).two_qubit_gate_count


def product_counterpart(genome: S.ControlledGenome) -> S.ControlledGenome:
    """Parameter-matched product (no-entanglement) counterpart of `genome`.

    CRZ param block -> RY rotation block; free entangling block -> H block; all
    other blocks unchanged. Preserves exactly 3 param blocks (12 quantum params)
    and the block count; removes every two-qubit gate.
    """
    blocks: list[S.Block] = []
    for b in genome.blocks:
        if b.role == "param":
            blocks.append(S.ParamBlock(kind="RY" if b.kind == "CRZ" else b.kind))
        else:
            ti = _H_TEMPLATE if b.template_index in _ENTANGLE_TEMPLATES else b.template_index
            blocks.append(S.FreeBlock(template_index=ti))
    return S.ControlledGenome(blocks=blocks)


def entangled_pairs(n: int = 20, start_seed: int = 2000) -> list[tuple[S.ControlledGenome, S.ControlledGenome]]:
    """`n` predeclared (entangled, product) pairs. Entangled members are the first
    `n` seeded random genomes that contain >=1 two-qubit gate; each is mapped to
    its product counterpart."""
    pairs = []
    s = start_seed
    while len(pairs) < n:
        g = S.random_genome(np.random.default_rng(s))
        s += 1
        if two_qubit_count(g) >= 1:
            p = product_counterpart(g)
            # only accept if the product truly has zero two-qubit gates (it always will)
            if two_qubit_count(p) == 0:
                pairs.append((g, p))
    return pairs


def pair_gate_mismatch(entangled: S.ControlledGenome, product: S.ControlledGenome) -> dict:
    ce = circuit_cost_summary(S.genome_to_ir(entangled))
    cp = circuit_cost_summary(S.genome_to_ir(product))
    pe = build_program(S.genome_to_ir(entangled)).num_parameters
    pp = build_program(S.genome_to_ir(product)).num_parameters
    return {"entangled_depth": ce.depth, "product_depth": cp.depth,
            "entangled_gates": ce.gate_count, "product_gates": cp.gate_count,
            "entangled_two_qubit": ce.two_qubit_gate_count, "product_two_qubit": cp.two_qubit_gate_count,
            "gate_count_mismatch": ce.gate_count - cp.gate_count,
            "param_count_preserved": pe == pp}
