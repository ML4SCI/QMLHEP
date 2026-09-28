"""Hybrid classical-quantum torch model: linear embed -> VQC -> linear + sigmoid.

Matches Knipfer et al. 2026's "Simple QNN" wrapper convention exactly
(their `TrainCustomSimpleQNNTool` docstring, quoted in
`LLM-VQC_MASTER_PLAN.md` Section 2.1): "a linear layer that maps the
input to `q_enc_size` and scales it to `[0, pi]` ... a VQC ... a linear
layer that maps `q_out_size` to the final output size (1) ... a sigmoid
activation at the end."

**Generalization beyond T1:** the master plan's T1 description explicitly
calls for this wrapper; T2's description ("PCA -> 8-16 features, angle
encoding") does not explicitly mention a linear embed. This module
applies the same wrapper to both tasks, for a structural reason rather
than a scientific one: IR circuits proposed by any future search arm can
have a different qubit count and a different number of encoding wires
from each other, so a task's fixed-dimensionality feature vector cannot
be wired directly into an arbitrary circuit's `inputs` — a small
trainable adapter (the embed layer) is the necessary glue. Its output
width is derived per-circuit from `expand.build_program(ir).num_inputs`,
never hard-coded.

This module builds the model only. Training (optimizer, loop, epochs) is
`llm_vqc.evaluation.training`; nothing here touches a training loop, a
task's raw data loading, or test data.
"""

from __future__ import annotations

import math

import pennylane as qml
import torch

from llm_vqc.ir.compiler_pennylane import to_qnode
from llm_vqc.ir.expand import CircuitProgram, build_program
from llm_vqc.ir.schema import CircuitIR


class HybridQNNModel(torch.nn.Module):
    """`raw_features -> embed -> [scale] -> VQC -> head -> sigmoid`."""

    def __init__(self, ir: CircuitIR, raw_feature_dim: int, head_out_dim: int) -> None:
        super().__init__()
        program: CircuitProgram = build_program(ir)

        self.encoding_type = program.encoding_type
        self.embed = torch.nn.Linear(raw_feature_dim, program.num_inputs, dtype=torch.float64)

        qnode = to_qnode(ir)
        weight_shapes = {"weights": (program.num_parameters,)}
        self.q_layer = qml.qnn.TorchLayer(qnode, weight_shapes)

        n_measurements = len(program.measurements)
        self.head = torch.nn.Linear(n_measurements, head_out_dim, dtype=torch.float64)

    def _encode(self, raw_features: torch.Tensor) -> torch.Tensor:
        embedded = self.embed(raw_features)
        if self.encoding_type == "angle":
            # Bound the otherwise-unbounded linear output into [0, pi] via
            # sigmoid before it becomes a rotation angle — Knipfer et al.'s
            # tool convention assumes inputs are "already scaled to
            # [0, pi]"; sigmoid is the standard, differentiable way to get
            # an unconstrained linear layer's output into a bounded range.
            return torch.sigmoid(embedded) * math.pi
        # Amplitude encoding: PennyLane's AmplitudeEmbedding (see
        # compiler_pennylane.py) normalizes internally, so the embed's raw
        # output is used as-is — no [0, pi] bound applies.
        return embedded

    def forward(self, raw_features: torch.Tensor) -> torch.Tensor:
        encoded = self._encode(raw_features)
        q_out = self.q_layer(encoded)
        prediction = torch.sigmoid(self.head(q_out))
        return prediction
