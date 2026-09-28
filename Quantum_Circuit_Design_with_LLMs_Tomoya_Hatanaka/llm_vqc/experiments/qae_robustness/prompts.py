"""The frozen QAE prompt family, parameterised ONLY by the allowed factors.

Every string below is the reference (previous-study) prompt with the
qubit count, gate counts, requested-candidate count and Hamiltonian facts
substituted. No hint, heuristic, instruction, schema field or ordering is
added, removed or reworded. `tests/test_qae_robustness_frozen.py` asserts
byte equality with the committed reference prompts at n=4, TFIM, B=8.
"""
from __future__ import annotations

import json

from llm_vqc.experiments.qae_robustness.conditions import Condition, Space, space_for

TEMPERATURE = 0.7
MAX_REPAIR_ATTEMPTS = 2
REFERENCE_MAX_OUTPUT_TOKENS = 6000
REFERENCE_GATES = 16

NUMBER_WORDS = {2: "two", 4: "four", 6: "six", 8: "eight", 10: "ten", 12: "twelve"}

# Only the Hamiltonian FACTS differ between families: the name/formula and
# the parameter symbol/range. Both families are real in the computational
# basis and both are nearest-neighbour on the open chain, so the two
# physical-structure sentences are literally unchanged.
FAMILY_DESCRIPTOR = {
    "TFIM": ("open-chain transverse-field Ising Hamiltonian "
             "H = -sum_i Z_i Z_{i+1} - h sum_i X_i, h in [0.2, 2.0]"),
    "XXZ": ("open-chain XXZ Heisenberg Hamiltonian "
            "H = sum_i (X_i X_{i+1} + Y_i Y_{i+1} + D Z_i Z_{i+1}), D in [0.2, 2.0]"),
}

# The one other place the prompt names the interaction type (redesign call).
FAMILY_INTERACTION = {
    "TFIM": "nearest-neighbor Ising interactions",
    "XXZ": "nearest-neighbor XXZ interactions",
}

SYSTEM_PROMPT = (
    "You are a quantum-autoencoder circuit architect. Your job is architecture "
    "proposal, not numerical optimization. Respect the exact resource budget. "
    "Use the physical structure in the task description to propose circuit "
    "hypotheses. Return circuit structure only."
)


def max_output_tokens(condition: Condition, requested_candidates: int) -> int:
    """Deterministic, pre-registered token-limit scaling.

    The reference limit (6000) is kept whenever the requested JSON is no
    larger than the reference request; larger requests (more candidates or
    more gates per candidate) scale linearly so that a mechanical response
    truncation cannot masquerade as a model or budget effect. At the
    reference condition, and for EVERY call of the model-factor condition,
    this returns exactly the reference 6000.
    """
    space = space_for(condition)
    scaled = 500 * requested_candidates * space.n_gates // REFERENCE_GATES
    return max(REFERENCE_MAX_OUTPUT_TOKENS, scaled)


def _qubit_list(wires: tuple[int, ...]) -> str:
    return ",".join(f"q{q}" for q in wires)


def task_card(condition: Condition) -> str:
    space = space_for(condition)
    return (
        f"Compress ground states of the {space.n}-qubit "
        f"{FAMILY_DESCRIPTOR[condition.family]}, into "
        f"latent qubits {_qubit_list(space.latent)} while trash qubits "
        f"{_qubit_list(space.trash)} should end in |{'0' * len(space.trash)}>. "
        "The Hamiltonian is real in the computational basis, so its ground-state "
        "amplitudes can be chosen real. Interactions are nearest-neighbor on the "
        f"open chain {'-'.join(f'q{q}' for q in space.wires)}, so nearest-neighbor "
        "correlations are important. "
        "Resource contract (exact, non-negotiable): the encoder is an ordered "
        f"sequence of exactly {space.n_gates} gates - exactly {space.n_rotations} "
        "single-qubit rotations, each "
        "RX, RY, or RZ on any qubit (no per-axis quota), and exactly "
        f"{space.n_cnots} CNOTs, "
        "each with any control and any different target (the same pair may repeat "
        "at different positions). You choose the gate ORDER freely. You do not "
        f"choose rotation angles: a shared numerical optimizer trains all "
        f"{space.n_rotations} "
        "angles identically for every method."
    )


def json_schema_card(condition: Condition) -> str:
    space = space_for(condition)
    top = space.n - 1
    return (
        "Return ONLY a single JSON object, no prose and no markdown fences, "
        "with this schema:\n"
        '{"candidates": [{"name": "<short-id>", "gates": ['
        '{"g": "RY", "q": 0}, {"g": "CNOT", "c": 3, "t": 2}, ...]}]}\n'
        f'Each gates list holds exactly {space.n_gates} entries in execution order: '
        'a rotation is '
        f'{{"g": "RX"|"RY"|"RZ", "q": 0-{top}}} and a CNOT is '
        f'{{"g": "CNOT", "c": 0-{top}, "t": 0-{top}, c != t}}. '
        f"Exactly {space.n_rotations} rotations and exactly {space.n_cnots} "
        "CNOTs per candidate."
    )


def open_loop_prompt(condition: Condition) -> str:
    """LLM-Open: one batch of B candidates, generated before any score."""
    return (
        task_card(condition)
        + f"\nOutput {condition.budget} distinct candidates. Prefer hypotheses "
        "that (1) preserve a "
        "real-valued representation when useful and (2) route correlations/information "
        "from the trash qubits toward the latent qubits.\n"
        + json_schema_card(condition)
    )


def warmstart_prompt(condition: Condition) -> str:
    """LLM-Closed phase 1: B/2 diverse semantic warm starts, no feedback."""
    return (
        task_card(condition)
        + f"\nOutput {condition.n_warm} distinct candidates. Aim for structural "
        "DIVERSITY "
        "across the candidates (different orderings, wirings, and axis patterns), "
        "while preferring hypotheses that (1) preserve a real-valued representation "
        "when useful and (2) route correlations/information from the trash qubits "
        "toward the latent qubits.\n"
        + json_schema_card(condition)
    )


def repair_prompt(condition: Condition, complaint: str, deficit: int) -> str:
    """Bounded capacity repair (frozen policy)."""
    space = space_for(condition)
    return (
        task_card(condition)
        + "\nYour earlier candidates below were REJECTED for violating the exact "
        f"resource contract (exactly {space.n_rotations} rotations and exactly "
        f"{space.n_cnots} CNOTs, {space.n_gates} gates):\n"
        + complaint
        + f"\nOutput exactly {deficit} NEW distinct candidates that satisfy the "
        "contract exactly. Count the gates before answering.\n"
        + json_schema_card(condition)
    )


def _arch_to_gate_list(architecture: list[dict]) -> list[dict]:
    out = []
    for op in architecture:
        if op["type"] == "R":
            out.append({"g": "R" + op["axis"], "q": op["q"]})
        else:
            out.append({"g": "CNOT", "c": op["c"], "t": op["t"]})
    return out


def redesign_prompt(
    condition: Condition, incumbent: list[dict], val_fid: float
) -> str:
    """LLM-Closed phase 2: free-form redesign of the current best, conditioned
    on the incumbent architecture and its validation trash fidelity."""
    space: Space = space_for(condition)
    gate_list = json.dumps(_arch_to_gate_list(incumbent), separators=(",", ":"))
    return (
        "You are improving the currently best quantum-autoencoder architecture.\n"
        + task_card(condition)
        + f"\n\nCurrent best architecture (execution order): {gate_list}\n"
        f"Current validation trash fidelity: {val_fid:.4f}\n\n"
        "You may redesign the architecture freely. Your next proposal may modify "
        "one gate choice, several coordinated choices, or the entire architecture, "
        "provided the exact resource budget is preserved: exactly "
        f"{space.n_rotations} single-qubit "
        f"rotations (each RX, RY, or RZ), exactly {space.n_cnots} CNOTs, exactly "
        f"{space.n_gates} gates total, "
        f"{NUMBER_WORDS[space.n]} qubits, arbitrary valid ordering, arbitrary "
        "valid CNOT control/target. "
        "You may retain the incumbent conceptually, modify it locally, or redesign "
        "it substantially. Do not propose numerical rotation angles - a shared "
        f"optimizer trains all {space.n_rotations} angles. Do not repeat the "
        "incumbent or any "
        "previously evaluated architecture exactly.\n"
        "Use the physical structure of the task - the real-valued ground states, "
        f"{FAMILY_INTERACTION[condition.family]}, and latent/trash roles - to propose "
        "an architecture you expect to outperform the incumbent.\n"
        "Return one complete new architecture.\n"
        + json_schema_card(condition)
        + "\nThe candidates array must contain exactly 1 candidate. In addition to "
        '"name" and "gates", include: "strategy" (one of "local_adjustment", '
        '"topology_redesign", "rotation_redesign", "global_redesign"), '
        '"rationale" (one short sentence), "preserved" (short phrase: what you '
        'kept from the incumbent), "changed" (short phrase: what you changed). '
        "This metadata is recorded for analysis only."
    )
