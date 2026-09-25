"""Quantum ablation conditions for v2 (capacity-controlled).

Separates *where* any VQC benefit comes from:
- **Q0 no-quantum**: the controlled classical shell with the VQC replaced by an
  identity — is the quantum layer useful at all?
- **QF frozen random quantum features** (built via `train_model(..., freeze_quantum=True)`
  on the fixed QR architecture) — is *training* the quantum angles useful?
- **QP product-state** (this module) — is *entanglement* useful?
- **QE fixed entangled** (v1 Q1 HEA) / **QR fixed random** (v1 Q2) — is
  architecture *search* useful beyond a fixed valid VQC?
- **QS searched** — the primary Random/Evolutionary/Greedy condition.

QE/QR reuse `baselines.fixed_hea_genome` / `baselines.fixed_random_genome`.
Q0 trains through `baselines.train_classical_baseline` (it is a torch Module with
`apply_init`). QF trains through `train_model(..., freeze_quantum=True)`.
"""

from __future__ import annotations

import math

import torch

from llm_vqc.experiments.capacity_controlled import space as S
from llm_vqc.experiments.capacity_controlled.init_policy import _init_linear


def product_state_genome() -> S.ControlledGenome:
    """QP: three single-qubit rotation layers (RX, RY, RZ on all wires) — 12
    trainable quantum angles, **no two-qubit gates** (product state)."""
    return S.ControlledGenome(blocks=[
        S.ParamBlock(kind="RX"), S.ParamBlock(kind="RY"), S.ParamBlock(kind="RZ"),
    ])


class NoQuantumModel(torch.nn.Module):
    """Q0: the controlled shell with the VQC replaced by identity.

    `embed(21->4) -> sigmoid*pi -> [identity] -> head(4->1) -> sigmoid`.
    Keeps the controlled encoding activation exactly (unlike C1's plain sigmoid
    bottleneck). 93 trainable params, 0 quantum params.
    """

    def __init__(self) -> None:
        super().__init__()
        self.embed = torch.nn.Linear(S.RAW_FEATURE_DIM, S.N_QUBITS, dtype=torch.float64)
        self.head = torch.nn.Linear(S.N_QUBITS, 1, dtype=torch.float64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = torch.sigmoid(self.embed(x)) * math.pi
        return torch.sigmoid(self.head(encoded))

    def apply_init(self, seed: int) -> None:
        gen = torch.Generator()
        gen.manual_seed(int(seed))
        _init_linear(self.embed, gen)
        _init_linear(self.head, gen)
