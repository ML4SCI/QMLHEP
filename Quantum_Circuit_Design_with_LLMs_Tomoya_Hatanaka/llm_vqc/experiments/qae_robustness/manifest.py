"""Per-condition manifests: machine-checkable evidence that each condition
differs from the previous-study reference in EXACTLY ONE allowed factor.

A manifest has three parts.

`factors`  the four things a condition is allowed to vary
           (n_qubits, family, budget, model).
`frozen`   fingerprints of everything that must be byte- or value-identical
           in EVERY condition: the LLM workflow, prompt skeleton, output
           schema, validation/retry behaviour, trainer, optimizer, splits,
           selection rule, gate set, method logic (hashed source), RNG
           stream offsets, and the analysis conventions.
`derived`  quantities that are a deterministic FUNCTION of `factors`
           (rendered prompt hashes, circuit width, latent/trash split,
           token limit). These may differ between conditions, but only in
           the way the factor forces; `derive()` recomputes them from
           `factors` alone, so a manifest that carries a hand-edited
           derived value fails verification.

`tests/test_qae_robustness_manifest.py` asserts, for every condition:
  frozen == reference.frozen   and   |changed factors| == 1
  derived == derive(factors)
"""
from __future__ import annotations

import inspect

import numpy as np

from llm_vqc.experiments.qae_robustness import arms
from llm_vqc.experiments.qae_robustness import conditions as C
from llm_vqc.experiments.qae_robustness import prompts as P

ANALYSIS_CONVENTIONS = {
    "primary_metric": "held_out_test_trash_fidelity",
    "selection": "lowest validation loss among the B evaluated candidates",
    "feedback": "validation only",
    "test_reads": "once, after selection",
    "paired_seeds": list(C.VERIFY_SEEDS),
    "bootstrap_resamples": 10_000,
    "bootstrap_seed": 0,
    "wilcoxon": "exact, two-sided",
    "sign_test": "binomtest, two-sided",
    "effect_size": "cohens_dz",
    "reported_contrasts": ["LLM-Closed_minus_LLM-Open", "LLM-Open_minus_Random",
                           "LLM-Closed_minus_Random", "Greedy_minus_Random"],
}

def _CANONICAL_INCUMBENT(condition: C.Condition) -> list[dict]:
    """A fixed incumbent used only to render the redesign prompt for hashing."""
    return C.sample_neutral_arch(np.random.default_rng(0), C.space_for(condition))


def _source_hash(*objects) -> str:
    return C.sha("\n".join(inspect.getsource(o) for o in objects))


def prompt_template_sha() -> str:
    """Hash of the prompt TEMPLATE itself (source of every prompt builder plus
    the two Hamiltonian fact tables and the number-word table).

    This is condition-independent by construction: it changes if and only if
    the wording, instruction set, schema description or substitution rules
    are edited. Per-condition RENDERED prompt hashes live in `derived`.
    """
    return _source_hash(
        P.task_card, P.json_schema_card, P.open_loop_prompt, P.warmstart_prompt,
        P.repair_prompt, P.redesign_prompt, P._arch_to_gate_list,
        P.max_output_tokens, P._qubit_list,
    ) + ":" + C.sha(C.canonical_json({
        "system": P.SYSTEM_PROMPT,
        "family_descriptor": P.FAMILY_DESCRIPTOR,
        "family_interaction": P.FAMILY_INTERACTION,
        "number_words": {str(k): v for k, v in P.NUMBER_WORDS.items()},
    }))


def hamiltonian_substitution_diff(other: C.Condition) -> dict:
    """Prove that switching the Hamiltonian family substitutes FACTS only.

    Render every prompt for `other` and for a copy of `other` at the
    reference family, undo the two fact substitutions, and report whether
    anything else moved. A non-empty `residual_difference` means a hint (or
    any other wording change) rode along with the family switch.
    """
    reference_family = C.Condition(
        key=other.key + "_ctrl", factor=other.factor, n_qubits=other.n_qubits,
        family=C.REFERENCE_FAMILY, budget=other.budget, model=other.model,
    )
    space = C.space_for(other)
    incumbent = C.sample_neutral_arch(np.random.default_rng(0), space)

    def render(condition: C.Condition) -> str:
        return "\n@@\n".join([
            P.task_card(condition),
            P.json_schema_card(condition),
            P.open_loop_prompt(condition),
            P.warmstart_prompt(condition),
            P.repair_prompt(condition, "<COMPLAINT>", 1),
            P.redesign_prompt(condition, incumbent, 0.5),
        ])

    raw_other, raw_control = render(other), render(reference_family)

    def undo(text: str) -> str:
        for family in (other.family, C.REFERENCE_FAMILY):
            text = text.replace(P.FAMILY_DESCRIPTOR[family], "{HAMILTONIAN}")
            text = text.replace(P.FAMILY_INTERACTION[family], "{INTERACTION}")
        return text

    normalised, control = undo(raw_other), undo(raw_control)
    return {
        "identical_after_undoing_fact_substitution": normalised == control,
        "residual_difference": (
            "" if normalised == control
            else "\n".join(
                line for line in __import__("difflib").unified_diff(
                    control.splitlines(), normalised.splitlines(), lineterm="", n=0
                )
            )[:2000]
        ),
        "length_delta_bytes": len(raw_other) - len(raw_control),
    }


def frozen_fingerprint() -> dict:
    """Everything that must be identical in every condition."""
    return {
        "llm_workflow": {
            "system_prompt_sha": C.sha(P.SYSTEM_PROMPT),
            "temperature": P.TEMPERATURE,
            "max_repair_attempts": P.MAX_REPAIR_ATTEMPTS,
            "reference_max_output_tokens": P.REFERENCE_MAX_OUTPUT_TOKENS,
            "call_and_retry_logic_sha": _source_hash(arms.complete_json),
            "prompt_template_sha": prompt_template_sha(),
            "candidate_absorption_sha": _source_hash(arms._absorb, arms._generate_batch),
            "deficit_policy": "bounded capacity repair, then flagged random draws",
        },
        "output_schema": {
            "validator_sha": _source_hash(C.parse_candidate, C.verify_capacity),
            "gate_set": ["RX", "RY", "RZ", "CNOT"],
            "angles_chosen_by": "shared numerical optimizer (never the LLM)",
        },
        "trainer": {
            "source_sha": _source_hash(C.evaluate_architecture),
            "optimizer": "Adam",
            "learning_rate": 0.05,
            "epochs": 60,
            "init": "uniform(-0.05, 0.05)",
            "dtype": "complex128/float64",
            "checkpoint": "best validation loss",
            "loss": "1 - mean trash fidelity",
            "train_seed_rule_sha": _source_hash(C.train_seed),
        },
        "splits": {
            "source_sha": _source_hash(C.state_sets),
            "n_train": 32, "n_val": 12, "n_test": 64,
            "train_val_draw": "uniform(0.2, 2.0), sorted",
            "test_grid": "linspace(0.215, 1.985, 64)",
            "parameter_stream": "default_rng(10_000 + seed)",
        },
        "method_logic": {
            "random_sha": _source_hash(arms.run_random_seed),
            "greedy_sha": _source_hash(arms.run_greedy_seed),
            "open_sha": _source_hash(arms.run_open_seed),
            "closed_sha": _source_hash(arms.run_closed_seed),
            "space_sha": _source_hash(C.sample_neutral_arch, C.mutate_one_choice),
            "exploration_refinement_split": "50/50",
        },
        "rng_streams": {
            "random": arms.RANDOM_STREAM,
            "greedy": arms.GREEDY_STREAM,
            "closed_fallback": arms.CLOSED_FALLBACK_STREAM,
            "warm_deficit": arms.WARM_DEFICIT_STREAM,
            "open_deficit": arms.OPEN_DEFICIT_STREAM,
        },
        "analysis": ANALYSIS_CONVENTIONS,
        "width_rule": "3 trainable rotations + 1 CNOT per qubit, for every method",
        "latent_trash_rule": "first n/2 qubits latent, last n/2 trash",
    }


def derive(factors: dict) -> dict:
    """Recompute every factor-determined quantity from the factors alone."""
    condition = C.Condition(key="_derive", factor="_derive", **factors)
    space = C.space_for(condition)
    return {
        "n_rotations": space.n_rotations,
        "n_cnots": space.n_cnots,
        "n_gates": space.n_gates,
        "latent_qubits": list(space.latent),
        "trash_qubits": list(space.trash),
        "n_warm": condition.n_warm,
        "n_refine": condition.budget - condition.n_warm,
        "task_card_sha": C.sha(P.task_card(condition)),
        "schema_card_sha": C.sha(P.json_schema_card(condition)),
        "open_prompt_sha": C.sha(P.open_loop_prompt(condition)),
        "warm_prompt_sha": C.sha(P.warmstart_prompt(condition)),
        "redesign_prompt_template_sha": C.sha(
            P.redesign_prompt(condition, _CANONICAL_INCUMBENT(condition), 0.5)
        ),
        "max_output_tokens_open": P.max_output_tokens(condition, condition.budget),
        "max_output_tokens_warm": P.max_output_tokens(condition, condition.n_warm),
        "max_output_tokens_refine": P.max_output_tokens(condition, 1),
    }


def condition_manifest(
    condition: C.Condition, anchor: C.Condition | None = None
) -> dict:
    """Manifest of `condition` relative to `anchor` (default: the reference).

    `anchor_key` and `changed_factors` are the only anchor-dependent
    entries; `frozen` and `derived` are anchor-independent by construction,
    so a manifest written against a second anchor is still directly
    comparable with every reference-anchored manifest.
    """
    anchor = anchor or C.REFERENCE
    return {
        "key": condition.key,
        "factor": condition.factor,
        "label": condition.label,
        "anchor_key": anchor.key,
        "anchor_factors": anchor.factors,
        "anchor_changed_factors_vs_study_reference": C.changed_factors(anchor),
        "changed_factors": C.changed_factors(condition, anchor),
        "factors": condition.factors,
        "frozen": frozen_fingerprint(),
        "derived": derive(condition.factors),
        "hamiltonian_substitution_check": hamiltonian_substitution_diff(condition),
    }


def verify(condition: C.Condition, anchor: C.Condition | None = None) -> list[str]:
    """Return the list of violations (empty means the condition is clean).

    The rule is unchanged and is NOT relaxed by the `anchor` argument: a
    condition must still differ from its declared anchor in exactly one
    declared factor, and its frozen block must still be byte-identical to
    the study reference's. Passing a non-default anchor declares a second
    protocol (e.g. a budget probe anchored at XXZ); it does not permit two
    factors to move at once.
    """
    problems: list[str] = []
    anchor = anchor or C.REFERENCE
    manifest = condition_manifest(condition, anchor)
    reference = condition_manifest(C.REFERENCE)
    if anchor.key != C.REFERENCE.key:
        # A second anchor is only admissible if it is ITSELF a clean
        # one-factor condition of the reference study. That keeps the chain
        # back to the reference explicit and auditable instead of letting an
        # arbitrary two-factor cell be declared as its own baseline.
        anchor_problems = verify(anchor)
        if anchor_problems:
            problems.extend(
                f"anchor '{anchor.key}' is not itself a clean condition: {p}"
                for p in anchor_problems
            )

    if C.canonical_json(manifest["frozen"]) != C.canonical_json(reference["frozen"]):
        for key in reference["frozen"]:
            if C.canonical_json(manifest["frozen"][key]) != C.canonical_json(
                reference["frozen"][key]
            ):
                problems.append(f"frozen component '{key}' differs from the reference")

    changed = manifest["changed_factors"]
    if condition.factor == "reference":
        if changed:
            problems.append(f"reference condition varies {changed}")
    elif len(changed) != 1:
        problems.append(
            f"{condition.key} varies {len(changed)} factors: {changed} "
            f"(anchor '{anchor.key}')"
        )
    elif not changed[0].startswith(_FACTOR_TO_FIELD[condition.factor]):
        problems.append(
            f"{condition.key} declares factor '{condition.factor}' but varies "
            f"'{changed[0]}'"
        )

    if C.canonical_json(manifest["derived"]) != C.canonical_json(
        derive(manifest["factors"])
    ):
        problems.append(f"{condition.key} derived block is not a function of its factors")

    check = manifest["hamiltonian_substitution_check"]
    if not check["identical_after_undoing_fact_substitution"]:
        problems.append(
            f"{condition.key} prompts differ from the reference family beyond the "
            "Hamiltonian fact substitution"
        )
    return problems


_FACTOR_TO_FIELD = {
    "budget": "budget",
    "qubits": "n_qubits",
    "hamiltonian": "family",
    "model": "model",
    "reference": "",
}


def all_manifests() -> dict:
    return {
        "reference_key": C.REFERENCE.key,
        "conditions": {c.key: condition_manifest(c) for c in C.CONDITIONS},
        "violations": {c.key: verify(c) for c in C.CONDITIONS},
    }
