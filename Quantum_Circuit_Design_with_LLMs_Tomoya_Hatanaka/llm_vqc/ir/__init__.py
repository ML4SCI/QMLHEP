"""Shared typed circuit intermediate representation (IR) for LAQS-Bench.

See llm_vqc/ir/README.md for the full design write-up. This package is the
common proposal format every future search arm (random, evolutionary,
greedy, LLM-guided) must emit through, so that comparisons across arms are
fair and not attributable to unequal circuit expressivity or backend
privileges (LLM-VQC_MASTER_PLAN.md Section 5.2).
"""

from llm_vqc.ir.schema import (
    CircuitIR,
    EncodingSpec,
    EntangleLayer,
    Layer,
    MeasurementSpec,
    RepeatBlock,
    RotationLayer,
)
from llm_vqc.ir.validators import ValidationIssue, ValidationResult, validate_proposal

__all__ = [
    "CircuitIR",
    "EncodingSpec",
    "RotationLayer",
    "EntangleLayer",
    "RepeatBlock",
    "Layer",
    "MeasurementSpec",
    "ValidationIssue",
    "ValidationResult",
    "validate_proposal",
]
