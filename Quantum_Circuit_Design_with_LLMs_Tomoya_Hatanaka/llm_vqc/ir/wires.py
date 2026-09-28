"""Wire-group resolution shared by validators, expansion, and compilers."""

from __future__ import annotations

from llm_vqc.ir.schema import WireSpec


def resolve_wires(spec: WireSpec, n_qubits: int) -> list[int]:
    """Resolve a WireSpec into an explicit, ordered list of qubit indices.

    "all" expands to [0, 1, ..., n_qubits - 1]. An explicit list is returned
    exactly as given (not sorted, not deduplicated) — order and duplicates
    are the caller's business to validate; this function only resolves the
    "all" shorthand.
    """
    if spec == "all":
        return list(range(n_qubits))
    return list(spec)
