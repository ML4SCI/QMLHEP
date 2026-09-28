"""Fixed, versioned prompt templates for the LLM search arms.

Per master plan Section 6.3 ("Prompt templates versioned in-repo") and
the work brief's "Prompt/feedback policy fixed across seeds unless prompt
variation is itself a declared ablation": every prompt in this module is
a plain string template, not something generated differently per seed.
`PROMPT_VERSION` is recorded in every `LLMCallRecord`'s `system_prompt`
verbatim (the whole prompt IS the provenance) so a future prompt-wording
change is auditable as a diff to this file, not a silent behavior change.

Only master-plan-approved feedback fields appear in any prompt built
here: validation performance, training outcome, circuit cost (depth/gate
count/param count), duplicate/validation-failure flags. Never a test
metric or label (impossible anyway -- `SearchFeedback` has no such
field), never another arm's results, never a privileged hand-authored
circuit.
"""

from __future__ import annotations

PROMPT_VERSION = "v1"

_SCHEMA_DESCRIPTION = """\
Propose ONE variational quantum circuit as a single JSON object with this shape:
{
  "n_qubits": <int, 2-10>,
  "encoding": {"type": "angle"|"amplitude", "gate": "RX"|"RY"|"RZ" (angle only), \
"wires": "all" | [int, ...], "reupload": <int, angle only>},
  "layers": [
    {"type": "rot", "gates": ["RX"|"RY"|"RZ"|"H", ...], "wires": "all" | [int, ...]},
    {"type": "entangle", "pattern": "ring"|"line"|"star"|"all_to_all"|"pairs"|"none", \
"gate": "CNOT"|"CZ"|"CRZ", "wires": "all" | [int, ...], "center": <int, star only>, \
"pairs": [[int, int], ...] (pairs only)}
  ],
  "measurements": {"observable": "X"|"Y"|"Z", "wires": "all" | [int, ...]}
}
Respond with ONLY that JSON object -- no prose, no markdown fences, no explanation.
"""


def build_system_prompt(task_description: str) -> str:
    return (
        f"You are designing variational quantum circuits for the task: {task_description}.\n"
        f"{_SCHEMA_DESCRIPTION}"
    )


def build_iter_user_prompt(history_lines: list[str]) -> str:
    """Closed-loop (`llm_iter`) user turn: full conversation history of
    approved-fields-only feedback, then a request for the next proposal."""
    if not history_lines:
        return "This is your first proposal. Propose a circuit."
    history = "\n".join(history_lines)
    return f"Feedback so far:\n{history}\n\nPropose your next circuit, using this feedback."


def build_open_loop_user_prompt(proposal_index: int) -> str:
    """Open-loop (ablation A5): no feedback from earlier proposals at all,
    not even a count -- each call is prompted identically."""
    return "Propose a circuit."


def build_evo_user_prompt(archive_lines: list[str]) -> str:
    """`llm_evo` user turn: top-k archive (IR summaries + scores +
    a diversity note), then a request for one offspring proposal."""
    if not archive_lines:
        return "The archive is empty. Propose an initial circuit."
    archive = "\n".join(archive_lines)
    return (
        f"Current top-performing archive:\n{archive}\n\n"
        "Propose ONE new circuit that could improve on or diversify this archive."
    )
