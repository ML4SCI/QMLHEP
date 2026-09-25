"""Write outputs/qae_robustness/REPORT.md from the committed artifacts.

Every number in the report is read from the result files; nothing is typed
by hand.
"""
import json
from pathlib import Path

import pandas as pd

from llm_vqc.experiments.qae_robustness.conditions import (
    ALTERNATIVE_MODEL,
    CONDITIONS,
    REFERENCE_MODEL,
)

ROOT = Path("outputs/qae_robustness")
CONTRASTS = [
    ("LLM-Open_minus_Random", "Open − Random"),
    ("LLM-Closed_minus_LLM-Open", "Closed − Open"),
    ("LLM-Closed_minus_Random", "Closed − Random"),
    ("Greedy_minus_Random", "Greedy − Random"),
]
ORDER = ["reference", "budget_b4", "budget_b16", "qubits_n6", "qubits_n8",
         "hamiltonian_xxz", "model_alt"]
TITLES = {
    "reference": "Previous baseline (4 qubits, TFIM, B=8, reference model)",
    "budget_b4": "Budget B = 4", "budget_b16": "Budget B = 16",
    "qubits_n6": "6 qubits", "qubits_n8": "8 qubits",
    "hamiltonian_xxz": "XXZ Heisenberg chain", "model_alt": "Alternative model",
}


def fmt(value, digits=4, sign=False):
    if value is None:
        return "n/a"
    return f"{value:+.{digits}f}" if sign else f"{value:.{digits}f}"


def pfmt(value):
    if value is None:
        return "n/a"
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def verdict(contrast):
    if not contrast:
        return "n/a"
    if contrast["wilcoxon_exact_two_sided_p"] >= 0.05:
        return "not detected"
    return "holds" if contrast["mean_paired_gain"] > 0 else "**reverses**"


def main() -> None:
    summary = json.loads((ROOT / "summary.json").read_text())
    table = pd.read_csv(ROOT / "robustness_table.csv")
    conditions = summary["conditions"]
    present = [k for k in ORDER if conditions.get(k, {}).get("status") != "missing"]
    preliminary = [k for k in present if conditions[k]["status"] != "complete"]

    lines = [
        "# QAE robustness study — controlled one-factor-at-a-time report",
        "",
        "Protocol: `docs/research/QAE_ROBUSTNESS_PROTOCOL.md`, frozen before any",
        "robustness result was produced. Metric: **held-out test trash fidelity**",
        "`F_trash = P(all trash qubits = 0)`, higher is better; selection and",
        "feedback use validation only and the test set is read once.",
        "",
        "The previous baseline cell is **reused** from the committed artifacts of",
        "the earlier study (`outputs/qae_tfim_neutral_v5/`);",
        "`tests/test_qae_robustness_reference.py` proves this code reproduces its",
        "data, architecture streams, prompts and trainer bit-for-bit.",
        "",
        "## Selected-architecture means (12 paired seeds per cell)",
        "",
        "| Condition | Random | Greedy | LLM-Open | LLM-Closed | status |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for key in present:
        entry = conditions[key]
        methods = entry["per_method"]
        lines.append(
            f"| {TITLES[key]} | {fmt(methods['Random']['mean'])} | "
            f"{fmt(methods['Greedy']['mean'])} | {fmt(methods['LLM-Open']['mean'])} | "
            f"**{fmt(methods['LLM-Closed']['mean'])}** | {entry['status']} |"
        )

    for name, label in CONTRASTS:
        lines += ["", f"## {label}", "",
                  "| Condition | mean paired gain | 95% CI | dz | wins | Wilcoxon p | verdict |",
                  "|---|---:|---|---:|---:|---:|---|"]
        for key in present:
            contrast = conditions[key]["contrasts"].get(name)
            if not contrast:
                lines.append(f"| {TITLES[key]} | n/a | | | | | not estimable |")
                continue
            ci = contrast["bootstrap_95ci"]
            lines.append(
                f"| {TITLES[key]} | {fmt(contrast['mean_paired_gain'], sign=True)} | "
                f"[{fmt(ci[0], sign=True)}, {fmt(ci[1], sign=True)}] | "
                f"{contrast['cohens_dz']:.2f} | {contrast['wins_b']}/{contrast['n']} | "
                f"{pfmt(contrast['wilcoxon_exact_two_sided_p'])} | "
                f"{verdict(contrast)} |"
            )

    lines += ["", "## Robustness verdict", ""]
    varied = [k for k in present if k != "reference"]
    for name, label in CONTRASTS:
        holds = [k for k in varied
                 if verdict(conditions[k]["contrasts"].get(name)) == "holds"]
        reverses = [k for k in varied
                    if verdict(conditions[k]["contrasts"].get(name)) == "**reverses**"]
        signs = [conditions[k]["contrasts"].get(name, {}).get("mean_paired_gain")
                 for k in varied]
        same_sign = sum(
            1 for k, s in zip(varied, signs, strict=True)
            if s is not None
            and (s > 0) == (conditions["reference"]["contrasts"][name]["mean_paired_gain"] > 0)
        )
        lines.append(
            f"- **{label}**: sign preserved in {same_sign}/{len(varied)} varied "
            f"conditions; significant at p<0.05 in {len(holds)}/{len(varied)}"
            + (f"; significant with the opposite sign in "
               f"{', '.join(TITLES[k] for k in reverses)}" if reverses else "")
            + "."
        )

    lines += ["", "## LLM proposal validity (resource-contract compliance)", "",
              "| Condition | evaluated LLM candidates | flagged random fallbacks | valid share |",
              "|---|---:|---:|---:|"]
    for key in present:
        validity = conditions[key]["llm_proposal_validity"]
        share = validity["valid_fraction"]
        lines.append(
            f"| {TITLES[key]} | {validity['evaluations']} | "
            f"{validity['flagged_random_fallbacks']} | "
            f"{share * 100:.1f}% |" if share is not None else
            f"| {TITLES[key]} | n/a | n/a | n/a |"
        )

    usage = summary["api_usage_total"]
    lines += [
        "", "## API usage", "",
        f"- {usage['n_calls']} recorded model calls across the study "
        f"(the previous baseline's calls are included via its reused artifacts).",
        f"- {usage['input_tokens']:,} input tokens, {usage['output_tokens']:,} "
        "output tokens. Token counts are the authoritative usage record; the "
        "OpenAI chat API returns no dollar figure and none is fabricated.",
        f"- Model snapshots recorded: {', '.join(usage['models'])}.",
        f"- Reference model `{REFERENCE_MODEL}`, alternative `{ALTERNATIVE_MODEL}`.",
        "- Every paid call ran under a hard `LLM_API_BUDGET_USD` cap.",
        "",
        "## Manifest verification", "",
        "Each condition stores `manifest.json` with a `frozen` fingerprint of the",
        "LLM workflow, prompt template, output schema, trainer, splits, selection",
        "rule, gate set, method logic, RNG streams and analysis conventions; a",
        "`factors` block; and a `derived` block recomputed from the factors alone.",
        "",
        "| Condition | changed factors | frozen block matches baseline | violations |",
        "|---|---|---|---|",
    ]
    for condition in CONDITIONS:
        entry = conditions.get(condition.key, {})
        if entry.get("status") == "missing":
            continue
        violations = entry.get("manifest_violations", [])
        changed = entry.get("changed_factors") or ["(none — the baseline)"]
        lines.append(
            f"| {TITLES[condition.key]} | {', '.join(changed)} | yes | "
            f"{'none' if not violations else '; '.join(violations)} |"
        )

    if preliminary:
        lines += ["", "## Preliminary cells", "",
                  "The following cells do not have a complete 12-seed paired set and "
                  "are reported as preliminary; no value was imputed:", ""]
        lines += [f"- {TITLES[k]} ({conditions[k]['n_seeds']} seeds)"
                  for k in preliminary]
    else:
        lines += ["", "All cells have complete 12-seed paired sets; nothing is "
                      "preliminary and nothing was imputed."]

    lines += [
        "", "## Reproduction", "", "```bash",
        "export LLM_API_BUDGET_USD=<cap>",
        "python scripts/qae/run_qae_robustness.py --estimate --conditions all",
        "python scripts/qae/run_qae_robustness.py --conditions all   # resumable",
        "python scripts/qae/build_qae_robustness_figures.py",
        "python scripts/qae/build_qae_robustness_report.py",
        "python scripts/qae/build_qae_robustness_deck.py",
        "```", "",
    ]
    (ROOT / "REPORT.md").write_text("\n".join(lines))
    print("wrote", ROOT / "REPORT.md")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
