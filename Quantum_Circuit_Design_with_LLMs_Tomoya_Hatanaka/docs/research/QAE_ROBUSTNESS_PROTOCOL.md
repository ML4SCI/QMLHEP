# QAE robustness study — controlled one-factor-at-a-time stress test (pre-registered)

Frozen **before** any robustness result was produced or inspected.
Date: 2026-09-04.

## Question

The previous QAE study found, at 4 qubits / TFIM / B = 8 / a single LLM
snapshot, that semantic LLM proposals beat non-semantic search by a large
margin, and that closed-loop free-form redesign beat the open-loop batch
by a small but consistent margin. **Does any of that survive when we
change one thing at a time?**

Four factors, each varied alone:

1. candidate-evaluation budget `B`,
2. qubit count `n`,
3. Hamiltonian family,
4. underlying LLM model.

## Reference condition (read from committed artifacts)

`outputs/qae_tfim_neutral_v5/` — 4 qubits, open-chain TFIM, `B = 8`,
12 paired seeds, `gpt-5.4-mini-2026-03-17`, held-out test trash fidelity
`F_trash = <00|rho_trash|00> = P(q2 q3 = 00)`, higher is better.

The reference cell is **reused, not re-run**. `tests/test_qae_robustness_reference.py`
proves that the code in `llm_vqc/experiments/qae_robustness/` reproduces
the previous study's data, architecture streams, prompts and trainer
bit-for-bit at the reference settings.

## What is frozen (identical in every condition)

LLM workflow; system prompt; prompt template, wording and instruction set;
output schema; validation/retry behaviour (2 JSON repair attempts, bounded
capacity repair, then flagged random fill); temperature 0.7; trainer
(Adam, lr 0.05, 60 epochs, `uniform(-0.05, 0.05)` init, complex128,
best-validation checkpoint, loss `1 - mean trash fidelity`); the
train/val/test split sizes and Hamiltonian-parameter streams; the
selection rule (lowest validation loss); the gate set
{RX, RY, RZ, CNOT}; every method's logic; the RNG stream offsets; and the
analysis conventions. `llm_vqc/experiments/qae_robustness/manifest.py`
fingerprints all of it and `tests/test_qae_robustness_manifest.py` asserts
that each condition's frozen block is byte-identical to the reference's
and that **exactly one** factor differs.

## Methods (unchanged)

- **Random** — `B` independent uniform candidates.
- **Greedy** — `B/2` random warm starts + `B/2` one-local-mutation
  refinements of the best warm start; accept only on validation
  improvement.
- **LLM-Open** — `B` semantic proposals generated in one batch *before*
  any score is seen.
- **LLM-Closed** — `B/2` semantic warm starts + `B/2` full free-form
  redesigns of the current best, conditioned on the incumbent
  architecture and its **validation** trash fidelity.

Selection and feedback use validation only. Held-out test is read once,
after selection.

## Matrix

| Factor | Levels | Non-target settings |
|---|---|---|
| Budget | `B = 4, 8*, 16` | n = 4, TFIM, reference model |
| Qubits | `n = 4*, 6, 8` | B = 8, TFIM, reference model |
| Hamiltonian | TFIM*, XXZ | n = 4, B = 8, reference model |
| Model | `gpt-5.4-mini-2026-03-17`*, `gpt-4.1-mini-2025-04-14` | n = 4, TFIM, B = 8 |

`*` = the reference level (the previous study's cell, reused).
Adaptive methods always keep the 50/50 exploration/refinement split, so
`B = 4` is 2 + 2, `B = 8` is 4 + 4, `B = 16` is 8 + 8.

### Width rule (deterministic, all methods)

**3 trainable single-qubit rotations + 1 CNOT per qubit**, i.e. `3n`
rotations and `n` CNOTs in a `4n`-gate ordered sequence. At `n = 4` this
is exactly the previous study's 12R + 4CX contract, so the reference cell
is a special case of the rule rather than a separate design. This is a
declared width **scaling**, not a gate-count study: circuit width is a
deterministic function of `n` and is identical across all four methods
within a condition.

### Latent/trash split

The first `n/2` qubits are latent, the last `n/2` are trash; the metric
is `P(all trash qubits = 0)`. At `n = 4` this is the reference latent
q0,q1 / trash q2,q3.

### Alternative Hamiltonian (declared before the run)

Open-chain XXZ Heisenberg chain
`H = sum_i (X_i X_{i+1} + Y_i Y_{i+1} + D Z_i Z_{i+1})`, anisotropy
`D in [0.2, 2.0]`, with the reference's split sizes and parameter
streams (train 32 / val 12 uniform draws, test 64-point grid). The
Hamiltonian parameter draws are the reference streams, so the physical
parameters are paired across conditions.

Only **Hamiltonian facts** are substituted into the frozen prompt: the
name/formula/parameter range, and the phrase naming the interaction type.
No hint, heuristic or extra instruction is added.
`manifest.hamiltonian_substitution_diff` renders every prompt for both
families, undoes those two substitutions and asserts the results are
byte-identical; the check is stored in every condition manifest.

Pre-run data-validity checks (properties of the data, not of any method):
the XXZ ground state must be non-degenerate and the family smooth over
the declared range. Measured before the run: minimum gap 1.51 at `n = 4`
(0.86 at `n = 8`), maximum imaginary amplitude `7.4e-17`, minimum
consecutive-state overlap 0.99997.

### Alternative model (declared before the run)

`gpt-4.1-mini-2025-04-14` — a different generation and family from the
reference snapshot, deliberately chosen as the harder generality test.
Prompt bytes, temperature, retry policy, schema and token limit are
identical to the reference for every call of this condition; only the
model identifier in the provider wiring changes.

### Token-limit scaling (declared before the run)

`max_output_tokens = max(6000, 500 * requested_candidates * n_gates / 16)`.
This returns exactly the reference 6000 at the reference condition and for
**every** call of the model condition, and only grows where a larger JSON
response is physically required (`B = 16` open batch, `n = 8` open batch),
so a mechanical response truncation cannot masquerade as a factor effect.
Truncations and schema failures are logged and reported either way.

## Pre-declared analysis

Identical to the previous study.

- Primary: held-out test `F_trash` of the selected candidate per method
  per seed; all 12 seed points shown, never only summaries.
- Paired per-seed differences with bootstrap 95% CIs (10,000 resamples,
  seed 0), exact two-sided Wilcoxon, two-sided sign test, Cohen's dz,
  win counts.
- Reported contrasts in every condition: **Closed − Open**,
  **Open − Random**, **Closed − Random**, **Greedy − Random**.
- Anytime best-so-far curves over evaluations `1..B`.
- Refinement gains for Greedy and LLM-Closed.
- LLM-Closed redesign diagnostics: edit distance from the incumbent,
  local vs global acceptance, duplicate/invalid rates, strategy metadata.
- Cross-condition summary: the sign, size and significance of each
  contrast against the reference value.

## Honesty commitments

Outcomes are not assumed. Any of the following will be reported as found:
the semantic advantage shrinking or vanishing at larger `n`; the
closed-loop advantage failing to replicate under any factor change (it
was a single small effect and is the most fragile claim in the previous
study); the alternative model failing the schema; or the alternative
Hamiltonian being trivially easy or impossible. Cells that cannot be
completed are marked **preliminary** and never imputed. All figures and
statistics are produced from committed artifacts by committed scripts.

Out of scope for this study, deliberately: any prompt-semantic ablation.

## Reproduction

```bash
export LLM_API_BUDGET_USD=<cap>
python scripts/qae/run_qae_robustness.py --conditions all
python scripts/qae/build_qae_robustness_figures.py
python scripts/qae/build_qae_robustness_deck.py
```
