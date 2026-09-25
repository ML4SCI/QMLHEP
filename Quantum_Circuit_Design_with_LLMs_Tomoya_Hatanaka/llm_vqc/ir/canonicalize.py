"""Canonical serialization and structural identity hashing.

**Scope note (read before use):** the canonical form and structural hash
defined here represent **syntactic identity**, not physical equivalence.
Two `CircuitIR` values that describe physically identical circuits via
different structural encodings — e.g. an `entangle` layer with
`pattern="star", center=0` versus an explicit `pattern="pairs"` layer
listing the same edges — are **syntactically different** and will produce
**different** canonical JSON and **different** structural hashes.

Detecting *physical* equivalence of parameterized circuits is a
substantially harder problem than the discrete Clifford-state equivalence
solved by `llm_vqc.circuit_explorer.state_key` (Phase 0): a parameterized
circuit's physical behavior depends on the parameter values it will
eventually be bound to, so "are these two circuits physically the same"
is really "are they the same for every possible parameter assignment",
which has no general closed-form test. This is explicitly out of scope
for Phase 1.

What syntactic-identity hashing *is* useful for (and the reason it exists,
per LLM-VQC_MASTER_PLAN.md Section 5.2 point 3): detecting when a search
method — especially an LLM agent — resubmits the exact same architecture
it already proposed. That is an exploration-collapse signal Knipfer et al.
observed qualitatively; structural hashing makes it a measurable duplicate
rate. It is not a mechanism for merging circuits that merely *behave* the
same.
"""

from __future__ import annotations

import hashlib
import json

from llm_vqc.ir.schema import CircuitIR


def canonical_json(ir: CircuitIR) -> str:
    """Deterministic, sorted-key JSON serialization of a valid CircuitIR.

    Dict keys are sorted for stability; list order (layers, gates, wires,
    pairs) is preserved exactly as given, since it is semantically
    significant (gate application order) and must not be silently
    reordered. The IR schema has no float fields (parameter values are
    bound externally, not embedded in the IR), so there is no
    floating-point formatting-stability concern here.
    """
    data = ir.model_dump(mode="json")
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def structural_hash(ir: CircuitIR) -> str:
    """SHA-256 hex digest of `canonical_json(ir)`. See module docstring."""
    canonical = canonical_json(ir)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
