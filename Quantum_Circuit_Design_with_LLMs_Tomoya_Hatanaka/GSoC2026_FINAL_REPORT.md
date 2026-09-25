# Google Summer of Code 2026 — Final Work Product

**When does language-model reasoning actually help variational quantum circuit architecture search?**

---

## Project information

| | |
|---|---|
| **Project title** | Quantum Circuit Design with LLMs (ML4SCI project slot **QMLHEP16**) |
| **Contributor** | Tomoya Hatanaka — GitHub [@dorakingx](https://github.com/dorakingx) |
| **Organization** | [ML4SCI](https://ml4sci.org/) — Machine Learning for Science |
| **Mentors** | Marco Knipfer, Jogi Suda Neto, Konstantin Matchev, Katia Matcheva |
| **Program** | Google Summer of Code 2026 |
| **Repository** | https://github.com/dorakingx/llm-vqc |
| **Canonical branch** | `main` — the single development branch |
| **Final snapshot** | tag [`gsoc-2026-final`](https://github.com/dorakingx/llm-vqc/tree/gsoc-2026-final) |

---

## 1. Project goal

A **variational quantum circuit** (VQC) is a parameterised quantum circuit whose
angles are trained by a classical optimizer. Its *architecture* — which gates,
on which qubits, in which order — is not differentiable and must be **searched**.
This is quantum architecture search (QAS), and it is hard for three reasons:

1. the space is combinatorial and discrete;
2. every candidate must be **trained** before it can be scored, so evaluations
   are expensive and budgets are small;
3. good architectures depend on the *physics* of the target problem in ways that
   a black-box optimizer has no access to.

Point 3 is the opening for a large language model: an LLM has read the physics
literature, so it may carry a useful **semantic prior** over circuit structure.
The existing literature — including the mentoring group's own paper, *AI Agents
for Variational Quantum Circuit Design*
([arXiv:2602.19387](https://arxiv.org/abs/2602.19387), analysed in
[`LLM-VQC_MASTER_PLAN.md`](./LLM-VQC_MASTER_PLAN.md)) — demonstrates
that an LLM *can* design VQCs, but does so qualitatively: single runs, no random
or evolutionary baselines, no seed replication, no statistical tests, and in at
least one case feedback computed on the test set.

**The project therefore did not set out to build another agent.** The research
question it converged on is evaluative:

> Under a **budget-matched, capacity-controlled, protocol-frozen** comparison
> against non-semantic search, does an LLM's semantic prior actually buy better
> circuit architectures — and does iterative feedback add anything on top of it?

This framing produces a publishable answer whichever way the result falls, and it
fills the gap ("controlled evaluation") that the existing system-demonstration
papers leave open.

---

## 2. What I implemented

All of the following is present in this repository.

**Search and evaluation framework**

- A circuit **intermediate representation** with an exact, machine-checked
  capacity contract, so every candidate from every method has identical
  circuit width, gate set and trainable-parameter count — [`llm_vqc/ir/`](./llm_vqc/ir)
- A **deterministic shared evaluator/trainer**: Adam (lr 0.05, 60 epochs,
  `U(−0.05, 0.05)` init, complex128, best-validation checkpoint), with the
  training seed derived from `(seed, architecture-hash)` so every method's
  candidates are trained identically — [`llm_vqc/evaluation/`](./llm_vqc/evaluation)
- Task packages and Hamiltonian families (TFIM, XXZ) — [`llm_vqc/tasks/`](./llm_vqc/tasks)
- Search drivers and run orchestration — [`llm_vqc/search/`](./llm_vqc/search), [`llm_vqc/runner.py`](./llm_vqc/runner.py)

**Search methods (four arms, all budget-matched)**

- **Random** — `B` independent uniform candidates
- **Greedy** — `B/2` random warm starts + `B/2` one-local-mutation refinements, accepted only on validation improvement
- **Evolutionary** — population baseline used in the v2 API verification
- **LLM open-loop** — `B` semantic proposals generated in **one batch before any score is seen**
- **LLM closed-loop** — `B/2` semantic warm starts + `B/2` free-form redesigns of the incumbent, conditioned on the incumbent architecture and its **validation** score
- Hand-designed reference ansätze as a further control
- Arms and prompts: [`llm_vqc/experiments/qae_robustness/arms.py`](./llm_vqc/experiments/qae_robustness/arms.py), [`prompts.py`](./llm_vqc/experiments/qae_robustness/prompts.py)

**LLM API layer**

- OpenAI provider with **strict structured outputs**, bounded JSON repair
  (≤2 retries), bounded capacity repair, then a **flagged random fallback**
  that still consumes budget and stays in the primary analysis — [`llm_vqc/llm/`](./llm_vqc/llm)
- **Version-pinned model snapshots** (e.g. `gpt-5.4-mini-2026-03-17`), recorded per call
- **Hard cost control**: paid calls are refused unless `LLM_API_BUDGET_USD` is
  set to an explicit nonzero cap, and the remaining cap is checked *before*
  every request; a cumulative ledger survives interruption and resume
- Rate-limit-aware backoff and client-side pacing

**Scientific controls**

- **Capacity control** — deterministic width rule (3 rotations + 1 CNOT per
  qubit) verified per candidate; a method cannot win by proposing a bigger circuit
- **Protected-test quarantine** — selection and feedback read **validation only**;
  the test split is read once, after selection. The budget-target study goes
  further and hands the trainer an **empty test array**, with a guard that fails
  the run if any recorded test quantity is finite
- **Protocol freezing in Git** — every study's protocol is written, committed and
  frozen *before* any result is produced; amendments are recorded before
  inspection. This is a Git-verifiable ordering claim, **not** a registration
  with an external preregistration service (see §4)
- **Manifest fingerprinting** — [`llm_vqc/experiments/qae_robustness/manifest.py`](./llm_vqc/experiments/qae_robustness/manifest.py)
  hashes the LLM workflow, prompt template, output schema, trainer, splits,
  selection rule, gate set, method logic, RNG streams and analysis conventions;
  a test asserts each condition's frozen block is **byte-identical** to the
  reference's and that **exactly one** factor differs
- **Provenance** — per-call records, file hashes, run timelines, and a
  bit-for-bit reproduction test proving the robustness code reproduces the
  earlier study's data, prompts and trainer exactly
  ([`tests/test_qae_robustness_reference.py`](./tests/test_qae_robustness_reference.py))

**Analysis, figures and reporting**

- Paired per-seed statistics: bootstrap 95% CIs (10,000 resamples), exact
  two-sided Wilcoxon, sign test, Cohen's `dz`, win counts — all 12 seed points
  always shown, never only summaries
- Figure, report and deck builders — [`scripts/qae/`](./scripts/qae)
- **53 test modules** under [`tests/`](./tests)
- A LaTeX manuscript — [`paper/main.tex`](./paper/main.tex)

**Origin tooling (retained).** The project began with an LLM agent that explores
**Clifford-circuit equivalence classes** by BFS with phase-normalised statevector
hashing — [`llm_vqc/agent.py`](./llm_vqc/agent.py),
[`llm_vqc/circuit_explorer.py`](./llm_vqc/circuit_explorer.py),
[`llm_vqc/visualization.py`](./llm_vqc/visualization.py). It is documented in the
README and still works, but it is **not** the final scientific contribution.

---

## 3. Research evolution

The project pivoted several times. Each pivot was forced by a gate the previous
stage failed — not by preference.

```
Clifford equivalence-class exploration (origin tooling)
   │  the LLM had no measurable role; counts are known in closed form
   ▼
LAQS-Bench: LLM vs classical VQC architecture search
   │  early "LLM wins" was traced to the LLM proposing LARGER circuits
   ▼
Capacity-controlled benchmarks (T1 / T2)
   │  with size, training and budget matched, Random ≈ Greedy ≈ Evolutionary
   │  → so the search problem itself must be non-degenerate to be informative
   ▼
Task qualification (HIGGS v1 → HIGGS data-scale)
   │  HIGGS at n=500 with PCA(8) was degenerate: frozen/product circuits
   │  beat trainable/entangled ones and the VQC sat at baseline level
   │  → design rule: qualify the task BEFORE comparing search strategies
   ▼
Rigorous QAS benchmark v2  ──►  BLOCKED (API account had no quota)
   │  protocol, parity proof and spend ledger survive; LLM cells never ran
   ▼
Semantic QAE benchmark (quantum autoencoder)
   │  a task with a DIRECT quantum objective (state-compression fidelity)
   │  and an exactly controlled circuit budget — affordable and non-degenerate
   ▼
v2 API verification → v3 neutral space → v4 multi-start → v5 free-form redesign
   │  v5: first detected closed-loop advantage (+0.0066, p = 0.012)
   ▼
Controlled one-factor robustness study    ◄── PRIMARY RESULT
   │  does any of it survive changing one thing at a time?
   ▼
Minimum-budget-to-target study            ◄── most recent confirmatory work
```

Two pivots deserve emphasis because they are the project's main scientific
lessons:

- **Capacity confounding.** The first apparent LLM advantage disappeared once
  circuit size was matched. Everything afterwards is capacity-controlled by
  construction and machine-verified per candidate.
- **Task qualification.** A search comparison on a task where the model cannot
  learn measures nothing. The QAE task was chosen precisely because its objective
  is a direct quantum quantity with an exactly controllable circuit budget.

Full narrative: [`docs/research/RESEARCH_ROADMAP_QAE.md`](./docs/research/RESEARCH_ROADMAP_QAE.md) ·
[`docs/research-log.md`](./docs/research-log.md) ·
[`docs/research/BRANCH_STATUS.md`](./docs/research/BRANCH_STATUS.md)

---

## 4. Final experimental setup

The **primary experiment** is the controlled one-factor-at-a-time robustness
study, whose protocol was **frozen and committed before execution** at
[`docs/research/QAE_ROBUSTNESS_PROTOCOL.md`](./docs/research/QAE_ROBUSTNESS_PROTOCOL.md)
— commit [`6933d3f`](https://github.com/dorakingx/llm-vqc/commit/6933d3f), which
lands before any robustness result existed.

> **What "protocol-frozen" means here, precisely.** Throughout this report it
> means: the protocol was pre-specified, committed to Git, and frozen *before*
> the corresponding experiment was run, with any amendment recorded before the
> results were inspected. The commit order is independently checkable by anyone
> with a clone. It does **not** mean the study was registered with an external
> preregistration registry such as OSF or AsPredicted — **no external
> preregistration service was used at any point in this project.**

| Element | Value |
|---|---|
| **Task** | Quantum autoencoder — compress the ground states of a spin chain into a latent register |
| **Qubits** | `n = 4` (reference); `n ∈ {4, 6, 8}` across conditions |
| **State family** | Open-chain transverse-field Ising model (TFIM); pre-declared alternative: open-chain XXZ Heisenberg chain, anisotropy `Δ ∈ [0.2, 2.0]` |
| **Data** | 32 train / 12 validation parameter draws + a 64-point test grid, per seed; identical parameter streams across conditions (paired) |
| **Gate grammar** | `{RX, RY, RZ, CNOT}`, free ordering, free target/control choice |
| **Circuit capacity** | Deterministic width rule: **3 trainable rotations + 1 CNOT per qubit** (`3n` rotations, `n` CNOTs, `4n` ordered slots). At `n = 4` this is exactly 12 rotations + 4 CNOTs |
| **Trainable parameters** | `3n` rotation angles (12 at `n = 4`) — identical for every method; verified per candidate |
| **Search budget** | `B = 8` candidate evaluations per method per seed (reference); `B ∈ {4, 8, 16}` across conditions. Adaptive methods always split 50/50 exploration/refinement |
| **Seeds** | 12 paired seeds (0–11), identical state streams across methods and conditions |
| **Training** | Adam, lr 0.05, 60 epochs, init `U(−0.05, 0.05)`, complex128, best-validation checkpoint, loss `1 − mean trash fidelity`; training seed from `(seed, architecture-hash)` |
| **Model** | `gpt-5.4-mini-2026-03-17` (reference, pinned snapshot); pre-declared alternative `gpt-4.1-mini-2025-04-14` |
| **LLM protocol** | Strict structured outputs, temperature 0.7, ≤2 JSON-repair retries, bounded capacity repair, then flagged random fallback (which **still consumes budget** and stays in the primary analysis). Token limit scales only where a larger JSON response is physically required |
| **Selection metric** | Lowest **validation** loss among the method's `B` candidates |
| **Protected-test metric** | Held-out test trash fidelity `F_trash = P(all trash qubits = 0)`, read **once**, after selection |
| **Baselines** | Random, Greedy, LLM-Open, LLM-Closed — all budget-matched and capacity-matched |
| **Statistics** | Paired per-seed differences; bootstrap 95% CI (10,000 resamples, seed 0); exact two-sided Wilcoxon; two-sided sign test; Cohen's `dz`; win counts. Contrasts pre-declared: Closed−Open, Open−Random, Closed−Random, Greedy−Random |

Simulation is **noiseless state-vector**; no hardware noise model is used.

Implementation: [`llm_vqc/experiments/qae_robustness/`](./llm_vqc/experiments/qae_robustness) ·
runner [`scripts/qae/run_qae_robustness.py`](./scripts/qae/run_qae_robustness.py)

---

## 5. Main results

Full report with all four contrast tables, per-condition validity rates and the
manifest verification: **[`outputs/qae_robustness/REPORT.md`](./outputs/qae_robustness/REPORT.md)**
· data [`robustness_table.csv`](./outputs/qae_robustness/robustness_table.csv),
[`contrast_table.csv`](./outputs/qae_robustness/contrast_table.csv)
· figures [`outputs/qae_robustness/figures/`](./outputs/qae_robustness/figures)

### 5.1 Selected-architecture means (held-out test `F_trash`, 12 paired seeds per cell)

| Condition | Random | Greedy | LLM-Open | LLM-Closed |
|---|---:|---:|---:|---:|
| Reference (4 qubits, TFIM, B=8, reference model) | 0.8520 | 0.8696 | 0.9605 | **0.9671** |
| Budget B = 4 | 0.8001 | 0.7520 | 0.8682 | **0.9494** |
| Budget B = 16 | 0.8893 | 0.9069 | **0.9681** | 0.9659 |
| 6 qubits | 0.7254 | 0.6936 | **0.9222** | 0.8728 |
| 8 qubits | 0.5924 | 0.5245 | **0.8765** | 0.8013 |
| XXZ Heisenberg chain | 0.8094 | 0.7338 | 0.9215 | **0.9563** |
| Alternative model | 0.8520 | 0.8696 | **0.9503** | 0.9166 |

### 5.2 PRIMARY RESULT — the semantic advantage is robust

**LLM-Open − Random keeps its sign in 6/6 varied conditions, and is significant
at p < 0.05 in 4/6.** It is largest where the search problem is hardest:

| Condition | mean paired gain | 95% CI | dz | wins | Wilcoxon p | verdict |
|---|---:|---|---:|---:|---:|---|
| Reference | +0.1084 | [+0.0726, +0.1471] | 1.58 | 11/12 | <0.001 | holds |
| Budget B = 4 | +0.0681 | [+0.0096, +0.1408] | 0.56 | 8/12 | 0.064 | not detected |
| Budget B = 16 | +0.0789 | [+0.0558, +0.1013] | 1.87 | 11/12 | <0.001 | holds |
| **6 qubits** | **+0.1968** | [+0.1798, +0.2135] | 6.26 | 12/12 | <0.001 | holds |
| **8 qubits** | **+0.2841** | [+0.2572, +0.3125] | 5.55 | 12/12 | <0.001 | holds |
| XXZ chain | +0.1121 | [+0.0152, +0.2208] | 0.59 | 5/12 | 0.791 | not detected |
| Alternative model | +0.0982 | [+0.0623, +0.1366] | 1.44 | 11/12 | <0.001 | holds |

**LLM-Closed − Random holds in 6/6 and is significant in 6/6.** So *some* LLM
arm beats non-semantic search in every condition tested.

By contrast **Greedy − Random keeps its sign in only 2/6**, is significant in
**0/6**, and significantly **reverses** at `B = 4`. The gain is therefore
attributable to the semantic prior, not merely to "refining a good candidate".

### 5.3 NEGATIVE RESULT — iterative feedback does *not* generalise

This is the finding that overturns the previous headline and is reported as
prominently as the positive one.

| Condition | Closed − Open | 95% CI | dz | wins | p | verdict |
|---|---:|---|---:|---:|---:|---|
| Reference | +0.0066 | [+0.0023, +0.0112] | 0.81 | 11/12 | 0.012 | holds |
| Budget B = 4 | +0.0812 | [+0.0602, +0.0987] | 2.25 | 11/12 | <0.001 | holds |
| Budget B = 16 | −0.0023 | [−0.0078, +0.0017] | −0.25 | 9/12 | 0.677 | not detected |
| **6 qubits** | **−0.0494** | [−0.0783, −0.0239] | −0.97 | 0/12 | <0.001 | **reverses** |
| **8 qubits** | **−0.0752** | [−0.1320, −0.0271] | −0.78 | 1/12 | 0.007 | **reverses** |
| XXZ chain | +0.0348 | [+0.0297, +0.0406] | 3.45 | 12/12 | <0.001 | holds |
| Alternative model | −0.0336 | [−0.0551, −0.0114] | −0.83 | 4/12 | 0.052 | not detected |

**Closed − Open keeps its sign in only 2/6 varied conditions and significantly
reverses at 6 and 8 qubits (0/12 and 1/12 seeds).** The closed-loop advantage
first detected in v5 is a property of that specific operating point — small
circuits, mid-range budget — not a general property of feedback. It appears to
help when the budget is *tight* (`B = 4`: +0.081) and to hurt when the space is
*large* (6–8 qubits), which is consistent with closed-loop search sacrificing
exploration breadth for refinement depth.

### 5.4 NEGATIVE RESULT — resource-contract compliance is a real failure mode

Newly quantified: the share of LLM proposals that satisfy the exact gate contract
without falling back to a flagged random draw.

| Condition | evaluated LLM candidates | flagged random fallbacks | valid share |
|---|---:|---:|---:|
| Reference | 192 | 8 | 95.8% |
| 6 qubits | 192 | 29 | 84.9% |
| 8 qubits | 192 | 22 | 88.5% |
| **Alternative model** | 192 | **67** | **65.1%** |

Compliance degrades with circuit size and, most sharply, with the weaker model.
Because fallbacks stay inside the primary analysis, every LLM number reported
here is an **operational** score that already pays for the model's own
specification failures. Architecture quality and generation validity are
therefore entangled and are not separated by this study.

### 5.5 SUPPORTING RESULT — minimum budget to a target fidelity

Most recent study. Protocol frozen and committed before execution at
[`docs/research/QAE_BUDGET_TARGET_PROTOCOL.md`](./docs/research/QAE_BUDGET_TARGET_PROTOCOL.md)
(commit [`f9a8d81`](https://github.com/dorakingx/llm-vqc/commit/f9a8d81));
full report **[`outputs/qae_budget_targets_v2_20260908/REPORT.md`](./outputs/qae_budget_targets_v2_20260908/REPORT.md)**.
Endpoint is **validation** trash fidelity; a cell passes iff ≥10 of the same 12
paired seeds reach the target. **No candidate was evaluated on the test set
anywhere in this study.**

Attainment at target 0.95, 4-qubit TFIM (seeds of 12 reaching the target;
**P** = passes the 10-of-12 rule):

| Configured B | Source | Random | Greedy | LLM-Open | LLM-Closed |
|---:|---|---:|---:|---:|---:|
| 4 | re-analysed | 0/12 | 0/12 | 0/12 | 8/12 |
| **6** | **measured now** | 0/12 | 2/12 | **12/12 P** | 8/12 |
| 8 | re-analysed | 1/12 | 1/12 | 10/12 **P** | 12/12 **P** |
| 16 | re-analysed | 2/12 | 3/12 | 12/12 **P** | 11/12 **P** |

Smallest **verified** passing budget: **`B = 6` for LLM-Open, `B = 8` for
LLM-Closed**; `B = 4` is a verified failure for both. Random and Greedy pass at
no budget tested. This independently reproduces §5.3's direction: at a tight
budget the open loop is *ahead*, because the closed loop spends half of six
evaluations on three warm starts and half on three redesigns.

Two cells were newly executed against the live API (**USD 0.2902** total,
138 calls); seven earlier conditions were re-analysed from committed logs
with **zero** model calls, reproducing the prior audit **byte-for-byte**.

### 5.6 Limitations

- **12 paired seeds**, noiseless state-vector simulation, one circuit-capacity
  rule, one task family pair (TFIM / XXZ), two model snapshots.
- **Attainment counts are not monotone in budget** — each budget is an
  independently executed policy, so no minimum-budget *interval* is claimed, and
  none is reported.
- **No method reaches target 0.99 anywhere**, at any budget or condition. Whether
  search, training length or circuit capacity is the binding constraint is
  **undetermined**.
- The **XXZ** cell is a knife-edge: the closed loop went 9/12 at `B = 8` to 5/12
  at `B = 10` while the mean final validation fidelity moved only −0.0076. Pass
  counts near a threshold are dominated by run-to-run variation.
- The **open-loop pool is shared across seeds** within a condition, so 12
  successes mean 12 seeds evaluated on one pool, not 12 independent generations.
- **Random fallbacks are inside the score** (§5.4), so architecture quality and
  generation validity are not separable here.
- **No claim of quantum advantage** is made anywhere. The comparison is between
  *search strategies* on a fixed quantum task, not between quantum and classical
  computing.
- **No claim of general LLM superiority** is made. The evidence supports a
  semantic prior being useful for *this* QAS problem under *these* controls.

---

## 6. Open-loop vs closed-loop: what the evidence does and does not support

This distinction is the centre of the final result, so it is spelled out exactly.

**What LLM-Open receives.** One batch call per seed, issued **before any
candidate has been trained or scored**. The prompt contains: the physics/task
card (Hamiltonian family, formula, parameter range, qubit count, latent/trash
split), the exact capacity contract, the gate set, and an instruction not to
repeat itself. It returns `B` distinct architectures in one structured response.
It **never sees a score**. Any advantage it shows is therefore attributable to
the model's **prior knowledge of physics and circuit structure alone**.

**What LLM-Closed receives.** The first `B/2` calls are semantic warm starts —
identical in kind to LLM-Open's batch, and *also* score-free. Only then does the
loop begin. Each of the remaining `B/2` calls receives: the same physics/task
card, the same capacity contract, the **current incumbent architecture** `C*`,
its **validation** trash fidelity `F_val(C*)`, and a statement that it may
redesign freely within capacity. It returns one full free-form redesign, which
is trained by the same shared optimizer and becomes the new incumbent if it
improves validation.

**What "feedback" actually means here.** A single scalar — the incumbent's
validation fidelity — plus the incumbent's structure. It is **not** a gradient,
not a training curve, not a history of all prior candidates, and **never a test
score**. The test split is read once, after selection is complete.

**How candidates are selected.** Within a method and seed, the selected
architecture is the one with the **lowest validation loss** among that method's
`B` candidates. Identical rule for all four methods.

**What the final evidence supports.**

1. The **semantic prior is real and robust**. LLM-Open, which never sees a
   score, beats Random in 6/6 varied conditions, by up to +0.284 (`dz = 5.55`)
   at 8 qubits. This cannot be an artifact of feedback, because there is none.
2. **Some LLM arm beats non-semantic search in every condition tested**
   (Closed − Random: 6/6, significant 6/6).
3. Feedback helps **specifically when the budget is tight** (`B = 4`: +0.081,
   p < 0.001) — the one condition where the open loop's single batch is too small.

**What the final evidence does *not* support.**

1. **That closed-loop feedback is generally better than open-loop.** It is not:
   Closed − Open holds in 2/6 and *significantly reverses* at 6 and 8 qubits.
   The v5 headline ("first detected closed-loop advantage") is true **only at
   its own operating point** and is explicitly superseded as a general claim.
2. **That the closed loop's mechanism is understood.** The reading that splitting
   a small budget into exploration + refinement costs more than feedback gains is
   an *interpretation* of the observed split, labelled `MODEL_INFERENCE` in
   [`docs/research-log.md`](./docs/research-log.md) and flagged as requiring a
   dedicated experiment. It is not an isolated cause.
3. **That LLMs are good circuit designers in general.** Compliance fell to 65.1%
   with a weaker model (§5.4), and the absolute ceiling (target 0.99) was reached
   by no method anywhere.

---

## 7. Reproducibility

### 7.1 Setup from a fresh clone (no secrets required)

```bash
git clone https://github.com/dorakingx/llm-vqc.git
cd llm-vqc

python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

./scripts/check.sh
```

`./scripts/check.sh` runs `ruff check .` then the full pytest suite. **It needs
no API key and makes no network calls.** See §8 for the verified result.

### 7.2 Deterministic reproduction — no API, no cost

Every table, statistic and figure in §5 is regenerated from the **committed**
artifacts, with zero model calls:

```bash
# Primary study — figures, report, slice figures, deck
python scripts/qae/build_qae_robustness_figures.py
python scripts/qae/build_qae_robustness_report.py
python scripts/qae/build_qae_robustness_slice_figures.py

# Supporting study — validation-only audit over committed logs
python scripts/qae/analyze_budget_targets.py \
  --root outputs/qae_robustness \
  --out outputs/qae_budget_targets_v2_20260908/audit \
  --extra-condition target_tfim_b6:6:all:outputs/qae_budget_targets_v2_20260908 \
  --extra-condition target_xxz_b10:10:LLM-Closed:outputs/qae_budget_targets_v2_20260908

python scripts/qae/build_budget_target_figures.py \
  --audit outputs/qae_budget_targets_v2_20260908/audit
```

The audit step reproduces the previously published analysis **byte-for-byte**
(verified by `diff -r` of the whole output tree), and drops test columns at
ingestion via an explicit allowlist.

### 7.3 Paid reproduction — ⚠ consumes API credits

> **Warning.** The commands below call the OpenAI API and **will spend money**.
> Everything in §5 is already reproducible without them (§7.2). Run these only
> to re-generate the raw candidate pools from scratch.

```bash
cp .env.example .env         # then put your own key in .env — never commit it
export LLM_API_BUDGET_USD=2.00   # REQUIRED: paid calls are refused without an explicit nonzero cap

python scripts/qae/run_qae_robustness.py --estimate --conditions all   # dry-run cost bound, no calls
python scripts/qae/run_qae_robustness.py --conditions all              # resumable

python scripts/qae/run_budget_target.py --status     # which seeds are already on disk
python scripts/qae/run_budget_target.py --estimate   # upper bound on calls/tokens/cost
LLM_API_BUDGET_USD=2.00 python scripts/qae/run_budget_target.py --cells all
```

Safeguards that are always active: paid calls are **refused** unless
`LLM_API_BUDGET_USD` is an explicit nonzero value; the remaining cap is checked
**before** each request; stored artifacts are **reused, not re-paid**, when the
budget, model snapshot, prompt, seed, initialisation and data all match exactly.
Recorded cost of the last paid study: **USD 0.2902** over 138 calls.

### 7.4 Recorded resource usage

- Robustness study: **511** model calls, 428,651 input / 225,577 output tokens.
  Token counts are the authoritative record — the chat API returns no dollar
  figure, and **none is fabricated**.
- Budget-target study: **138** calls (17 repair/retry), 109,516 input / 46,246
  output tokens, **USD 0.2902** at the provider's published list price read on
  2026-09-08.

---

## 8. Tests and validation

Verified on the final commit, in a fresh virtual environment created by the
commands in §7.1:

```bash
./scripts/check.sh          # ruff check . && pytest tests/ -q
python -m pytest tests/ -q  # full suite on its own
```

<!-- TEST_RESULTS_START -->
**Result: `603 passed` — full suite green, lint clean.**

| Step | Command | Result |
|---|---|---|
| Lint | `ruff check .` | **All checks passed** |
| Tests | `pytest tests/ -q` | **603 passed** in 168 s |
| Combined | `./scripts/check.sh` | **all checks passed** (603 passed in 163 s) |

Environment: Python 3.14.7, macOS (arm64), fresh `.venv` created by the §7.1
commands. The count is **603 passed**, measured on the final commit — earlier
figures quoted in this project's history (445, 557, …) are superseded and are
not reused here.

**Independently reproduced in CI.** The same `./scripts/check.sh` runs on every
push to `main` and every pull request via
[`.github/workflows/ci.yml`](./.github/workflows/ci.yml), on **Ubuntu with
Python 3.12** — a different OS and Python version from the local run — installing
the project from scratch exactly as §7.1 describes. It reports the identical
**`All checks passed!` / `603 passed`**, so the fresh-clone setup path is verified
on two independent platforms and is not an artifact of one machine.

Two genuine regressions were found and fixed while verifying this, rather than
worked around:

1. **`python-pptx` was imported by three deck builders and two test modules but
   declared in no dependency list.** A fresh clone following §7.1 could not
   collect the test suite at all. It is now a declared dependency in
   [`pyproject.toml`](./pyproject.toml).
2. **38 lint errors** had entered with the most recent study's scripts. The
   genuine ones were fixed. `scripts/qae/analyze_budget_targets.py` was instead
   given a documented, style-only exemption: it copies **itself** into the audit
   directory as a provenance record whose SHA-256 is pinned in
   `outputs/qae_budget_targets_v2_20260908/file_hashes.json`, so reformatting it
   would break that hash and invalidate a published reproducibility claim.
   `archive/` and `outputs/` are excluded from lint for the same reason — they
   are historical records, not source.

### Reproduction verified

The deterministic, no-API path in §7.2 was executed on the final commit with no
API key present:

- The validation-only audit **reproduces every attainment count exactly**
  (`target_tfim_b6` LLM-Open 12/12, LLM-Closed 8/12, Greedy 2/12, Random 0/12;
  `target_xxz_b10` LLM-Closed 5/12; all 0/12 at target 0.99) and every CSV
  byte-for-byte.
- All **31 figures regenerate byte-identically** to the committed ones.
- The only differences in the whole output tree are two `run_record.json`
  **copies** inside the audit snapshot, which differ solely in wall-clock
  metadata (`wall_clock_seconds` vs the later
  `wall_clock_seconds_this_invocation` + note added when the cells were
  regenerated in reuse-only mode). No scientific quantity differs. The committed
  snapshot is deliberately left as-is, because it records what actually ran.
<!-- TEST_RESULTS_END -->

Archived code under [`archive/`](./archive) is intentionally **excluded** from
both lint and test collection (`testpaths = ["tests"]` in
[`pyproject.toml`](./pyproject.toml)); it is a read-only historical record, not
part of the build.

---

## 9. Major code contributions

| Contribution | PR / commit | Description | Status |
|---|---|---|---|
| Publish archived output graphs | [#1](https://github.com/dorakingx/llm-vqc/pull/1) | First sanitized publication of exploration artifacts and plots | Merged |
| Unify research line + semantic QAE benchmark | [#4](https://github.com/dorakingx/llm-vqc/pull/4) | Consolidated capacity-controlled VQC work with the new QAE benchmark; established `main` as the single line | Merged |
| QAE v2 — version-pinned API verification | [#5](https://github.com/dorakingx/llm-vqc/pull/5) | Real-API benchmark with evolutionary and hand-designed baselines; pinned model snapshots | Merged |
| Record final deliverable locations | [#6](https://github.com/dorakingx/llm-vqc/pull/6) | Deck, Overleaf and reference-manager provenance | Merged |
| QAE v3 — neutral-space four-method benchmark | [#8](https://github.com/dorakingx/llm-vqc/pull/8) | Random / Greedy / LLM-Open / LLM-Closed in a neutral capacity-controlled space | Merged |
| QAE v4 — pre-registered multi-start (4+4) | [#9](https://github.com/dorakingx/llm-vqc/pull/9) | Multi-start protocol; rescued Greedy to parity; v3 preserved as diagnostic | Merged |
| QAE v5 — incumbent-based free-form redesign | [#10](https://github.com/dorakingx/llm-vqc/pull/10) | Free-form closed-loop redesign; first detected closed-loop advantage (later shown not to generalise, §5.3) | Merged |
| **QAE robustness study (PRIMARY)** | [#11](https://github.com/dorakingx/llm-vqc/pull/11) | One-factor-at-a-time stress test across budget, qubits, Hamiltonian and model; manifest fingerprinting; bit-for-bit reference reproduction | Merged |
| **Minimum budget to target fidelity** | [#12](https://github.com/dorakingx/llm-vqc/pull/12) | Protocol-frozen `B = 6` and XXZ `B = 10` boundary cells (both **executed**); validation-only, test-quarantined audit | Merged |
| Final GSoC consolidation | [`main`](https://github.com/dorakingx/llm-vqc/commits/main) | Archived the disconnected research lines onto `main`, this report, README and branch-status rewrite, `gsoc-2026-final` tag | Merged |

Closed without merging (superseded, kept for the record):
[#2](https://github.com/dorakingx/llm-vqc/pull/2),
[#3](https://github.com/dorakingx/llm-vqc/pull/3),
[#7](https://github.com/dorakingx/llm-vqc/pull/7).

Key protocol-freezing commits — each lands in Git **before** its corresponding run:

| Protocol | Frozen at | In `git log main`? |
|---|---|---|
| QAE robustness (**primary**) | [`6933d3f`](https://github.com/dorakingx/llm-vqc/commit/6933d3f) | **Yes** (fast-forwarded) |
| Budget-target | [`f9a8d81`](https://github.com/dorakingx/llm-vqc/commit/f9a8d81) | **Yes** (fast-forwarded) |
| QAE v5 | `33740dc` | No — squash-merged via PR [#10](https://github.com/dorakingx/llm-vqc/pull/10) |
| QAE v3 | `6620cd4` | No — squash-merged via PR [#8](https://github.com/dorakingx/llm-vqc/pull/8) |

The primary study's protocol-freezing commit is a **first-class commit on `main`**, so a
reviewer can verify from a plain clone that the protocol predates the results.
The two older squash-merged SHAs are not commit objects on `main`; their protocol
*files* are (`docs/research/QAE_PROTOCOL_V3.md`, `QAE_PROTOCOL_V5.md`), and the
original commits remain viewable through their pull requests.

---

## 10. Repository map

```
GSoC2026_FINAL_REPORT.md      ← you are here (the submitted work product)
README.md                     project overview, architecture, setup
LLM-VQC_MASTER_PLAN.md        approved research design, literature analysis
DECISIONS.md                  dated decision log (incl. dataset licensing diligence)

llm_vqc/
  ir/                         circuit IR + machine-checked capacity contract
  evaluation/                 deterministic shared trainer / evaluator
  tasks/                      TFIM, XXZ and classification task packages
  search/                     search drivers
  llm/                        OpenAI provider, structured outputs, budget ledger
  experiments/
    qae_robustness/           ★ PRIMARY experiment (arms, prompts, manifest, study)
    qae_budget_targets/       supporting minimum-budget study
    qae_tfim/                 QAE v2 → v5 lineage
    capacity_controlled/      T1 / T2 capacity-controlled benchmarks
  agent.py, circuit_explorer.py, visualization.py   origin Clifford tooling

scripts/
  check.sh                    ← authoritative lint + test entry point
  qae/                        runners, analysis, figure and deck builders

docs/research/
  QAE_ROBUSTNESS_PROTOCOL.md      ★ primary protocol (frozen before execution)
  QAE_BUDGET_TARGET_PROTOCOL.md   supporting protocol (frozen before execution)
  QAE_PROTOCOL{,_V3,_V4,_V5}.md   QAE lineage protocols (each frozen first)
  RESEARCH_ROADMAP_QAE.md         the scientific story
  HIGGS_ARCHIVE_SYNTHESIS.md      what the archived HIGGS studies contributed
  BRANCH_STATUS.md                post-consolidation branch/provenance record
docs/research-log.md              phase status, gates, evidence states

outputs/
  qae_robustness/                 ★ PRIMARY results, figures, tables, deck
  qae_budget_targets_v2_20260908/ supporting results, audit, deck
  qae_tfim_neutral_v5/            v5 (the primary study's reference cell)
  qae_tfim_neutral_v3, _v4/       diagnostics
  qae_tfim_api_v2/                earlier rigid-layout API verification
  capacity_controlled_t1_v1, _t1_v2, _t2_v1/   capacity-controlled benchmarks

paper/                        LaTeX manuscript + bibliography
tests/                        53 test modules (the only pytest root)
archive/                      read-only copies of superseded, disconnected lines
                              (see archive/README.md for provenance SHAs)
```

---

## 11. Current state

**The technical work product is complete.** Verified on `main`:

- ✅ Capacity-controlled, budget-matched, protocol-frozen QAS evaluation framework
- ✅ Four search arms + hand-designed reference, all capacity- and budget-matched
- ✅ Real-API execution with pinned model snapshots, hard spend caps and per-call provenance
- ✅ Protected-test quarantine, escalating to an empty-test-array guard in the latest study
- ✅ **Primary result** (§5.2/5.3): semantic advantage robust 6/6; closed-loop advantage does **not** generalise
- ✅ **Supporting result** (§5.5): smallest verified passing budget — `B = 6` open-loop, `B = 8` closed-loop
- ✅ Negative results reported as prominently as positive ones
- ✅ Full test suite and lint green (§8)
- ✅ Deterministic, no-API reproduction path for every published number
- ✅ Superseded research lines archived onto `main`; `main` is the single branch
- ✅ Secret scan clean across working tree and full Git history (§13)

The one open administrative item is the **reuse licence**, which awaits ML4SCI /
mentor confirmation (§13). It does not affect the completeness or validity of
the work product above.

---

## 12. Remaining work

Genuinely unfinished. **None of the following was executed; none is reported as
a result anywhere in this repository.**

1. **Repeat the two boundary cells with fresh generation seeds.** Only one run
   of each exists. The `B = 6` open-loop pass (12/12) and the XXZ `B = 10` drop
   (9/12 → 5/12) are each a single measurement. *(Unexecuted — decision D1 in
   [`docs/research-log.md`](./docs/research-log.md).)*
2. **Diagnose the 0.99 ceiling.** No method reaches target 0.99 at any budget or
   condition tested. Whether search, training length or circuit capacity binds is
   **undetermined**, and nothing in this repository identifies it.
   *(Unexecuted — decision D2.)*
3. **Resolve the deliberately unresolved boundaries.** The 6-qubit, 8-qubit and
   alternative-model budget boundaries were left unmeasured on purpose; on the
   4-qubit TFIM ladder, budgets 2, 10, 12, 14 and everything above 16 remain
   **unverified**. *(Unexecuted.)*
4. **Separate generation validity from architecture quality.** Random fallbacks
   currently sit inside the operational score (§5.4). Disentangling them needs a
   new design, not a filter over existing logs. *(Unexecuted.)*
5. **Test a mechanism for the closed-loop reversal.** The exploration/refinement
   split explanation is an interpretation, not a measured cause. *(Unexecuted.)*
6. **Unblock or retire the `bench_v2` line.** Its LLM cells never ran
   (account-level `insufficient_quota`); recorded spend is $0.000000. Protocol
   and machinery are archived at
   [`archive/pre-consolidation/docs/research/BENCHMARK_V2_PROTOCOL.md`](./archive/pre-consolidation/docs/research/BENCHMARK_V2_PROTOCOL.md).
   *(Blocked, never executed.)*
7. **Noise and hardware.** Everything here is noiseless state-vector simulation.
   *(Out of scope for this project; unexecuted.)*
8. **Confirm the reuse licence with ML4SCI / the mentors and add a `LICENSE` file**
   (§13). This is an administrative follow-up, not unfinished research, and does
   not affect the completeness of the work product.

---

## 13. Security, privacy and licensing

**Secret scan — clean.** Performed on the final commit across both the working
tree and the **complete Git history** (161 commits, ~54 MB of blobs):

- `gitleaks git --log-opts="--all"` → 7 findings, **all false positives**: each is
  a 64-character SHA-256 file-integrity hash in
  `outputs/qae_budget_targets_v2_20260908/file_hashes.json`, flagged only for
  high entropy.
- Targeted pattern sweep over every text blob in history for OpenAI (`sk-`),
  Groq (`gsk_`), GitHub (`ghp_`, `github_pat_`), Google (`AIza`), Slack (`xox*`),
  AWS (`AKIA`) keys and PEM private keys → **no real credential**. The only
  matches were self-evident placeholders (`sk-FAKE…`, `sk-not-a-real-…`) in test
  fixtures on branches that no longer exist.
- No `.env`, service-account JSON, `client_secret`, or credential file was ever
  committed on any branch.
- No credentials embedded in URLs; no `Authorization`/`Bearer` headers, org IDs
  or project IDs in any tracked artifact.
- Stored API provenance contains only call counts, token counts and model
  snapshot identifiers — no prompts-with-secrets, no keys, no response headers.

**No history rewrite was performed**, because none was warranted. `.env` is
gitignored and [`.env.example`](./.env.example) contains placeholders only.

**Privacy.** Every commit on the final `main` line is authored as
`Doraking <120563040+dorakingx@users.noreply.github.com>`. Personal-email commits
existed only on the disconnected pre-rewrite branches, which are now deleted;
their content is preserved under [`archive/`](./archive) as part of the current
history. The contributor's name appears in the repository only as **deliberate
authorship attribution** on GSoC deliverables.

**Data provenance.** All datasets used in the final results are **generated
locally** (TFIM and XXZ ground states by exact diagonalisation) — no external
dataset is redistributed. [`DECISIONS.md`](./DECISIONS.md) records the diligence
on the ML4SCI electron-photon ECAL dataset, including the decision **not** to use
an unofficial Kaggle re-upload of unclear licence, and the permissively licensed
scikit-learn fallback actually used.

**Reuse licence: awaiting organizational confirmation.**

Two separate things, deliberately not conflated:

| | Status |
|---|---|
| **Technical work product** | **Complete.** The code, experiments, results, protocols, documentation and reproduction paths described in this report are finished, verified and submitted. |
| **Repository reuse licence** | **Not yet confirmed.** Awaiting ML4SCI / mentor confirmation of the licence the organization requires. |

The licence is an **administrative follow-up, not a defect in the work and not a
blocker for this submission.** The work product stands on its own; only the terms
under which third parties may *reuse* the code remain to be set.

Concretely: this repository has no `LICENSE` file, and GitHub detects none. No
licence is specified in `LLM-VQC_MASTER_PLAN.md`, `DECISIONS.md`, or any other
project record, so under default copyright no reuse rights are granted yet — which
is very likely not the long-term intent.

A licence has deliberately **not** been invented here, because ML4SCI or the
mentors may already require a specific one, and picking one unilaterally could
conflict with that. **Next step for the contributor and mentors:** confirm the
organization's required licence and add the corresponding `LICENSE` file. This can
be done at any time, before or after submission, without changing any of the work
described above.

Third-party dependencies (Qiskit, PennyLane, PyTorch, scikit-learn, pandas,
SciPy, matplotlib, openai, pydantic) are used **unmodified** via their public
package distributions under their own permissive licences; none is vendored into
this repository.

---

## 14. Challenges and lessons learned

**1. Capacity confounding destroys naive comparisons.** The project's first
apparent "LLM beats random search" result evaporated once circuit size was
matched — the LLM had simply been proposing *larger* circuits. Every subsequent
experiment enforces an exact capacity contract, verified per candidate, and the
final study makes circuit width a deterministic function of qubit count so no
method can win on size. *This single control changed the project's conclusion.*

**2. Qualify the task before you compare search strategies.** HIGGS at n=500
with PCA(8) was degenerate: frozen parameters beat trainable ones and product
circuits beat entangled ones, meaning the "search" was measuring noise. Scaling
the data flipped those signs. The design rule that came out of it — *do not
evaluate a search strategy until the underlying learning task is demonstrably
non-degenerate* — is why the final work uses a quantum autoencoder, whose
objective is a direct quantum quantity.

**3. Fair baselines are harder to build than the method under test.** Greedy had
to be rescued twice (v3 → v4 multi-start) before it was a credible opponent, and
the eventual finding that Greedy − Random is significant in **0/6** conditions is
only meaningful *because* Greedy was given every structural advantage the LLM had:
same budget, same capacity, same warm-start count, same trainer, same selection
rule.

**4. The interesting quantity was the semantic prior, not the agent loop.**
LLM-Open never sees a score, yet it is the arm whose advantage survives
everywhere. The elaborate feedback machinery — the part that *looks* like an
agent — turned out to be the fragile component. Building the open-loop control
properly was worth more than any amount of agent engineering.

**5. Search breadth beats refinement depth as the space grows.** Closed-loop
feedback wins at `B = 4` (+0.081) and reverses at 6–8 qubits (0/12 and 1/12
seeds). Spending half a small budget on refinement is a real cost, and it grows
with the size of the space being given up.

**6. Reproducibility must survive a dead credential.** The `bench_v2` line was
fully built, protocol-frozen, parity-proved and budget-ledgered — and then died on
an account-level `insufficient_quota` with $0.000000 spent. That loss forced the
discipline that the final work depends on: commit raw candidate logs, make every
published number regenerable from them with **zero** API calls, and treat paid
execution as a separate, explicitly-capped step. Every table in §5 survives the
API disappearing tomorrow.

**7. Protected-test isolation has to be enforced mechanically.** "Don't look at
the test set" is not a policy, it is a code path. The final study hands the
trainer an **empty test array**, guards that any recorded test quantity is
non-finite, and applies a validation-only column allowlist at ingestion — after
an earlier audit found historical `anytime_mean.csv` files that aggregated *test*
fidelity and could have contaminated a budget decision.

**8. Negative results are the deliverable.** The most valuable output of this
project is not "LLMs help" — it is the precise statement of *which part* helps
(the score-free semantic prior, robustly) and *which part does not* (iterative
feedback, which reverses as the space grows), plus a newly quantified failure
mode nobody was measuring (resource-contract compliance falling to 65%). Writing
up the v5 headline as superseded, rather than quietly keeping the better-sounding
claim, is the outcome the whole control apparatus was built to make possible.

---

## 15. GSoC final snapshot

**Final GSoC tag: [`gsoc-2026-final`](https://github.com/dorakingx/llm-vqc/tree/gsoc-2026-final)**

Annotated tag on the exact commit whose tests and checks are reported in §8.
Browse the repository at that immutable snapshot:
https://github.com/dorakingx/llm-vqc/tree/gsoc-2026-final

This report at the fixed snapshot:
https://github.com/dorakingx/llm-vqc/blob/gsoc-2026-final/GSoC2026_FINAL_REPORT.md

The canonical submission URL remains the `main` copy, which stays current:
**https://github.com/dorakingx/llm-vqc/blob/main/GSoC2026_FINAL_REPORT.md**

---

## 16. Additional material

- **Manuscript** — [`paper/main.tex`](./paper/main.tex) with
  [`paper/references.bib`](./paper/references.bib), in this repository and
  therefore public and stable. An Overleaf mirror exists but is **not** a
  public link and is not the submission target.
- **Presentation decks** — committed in-repo (no external permissions needed):
  [`outputs/qae_robustness/deck/`](./outputs/qae_robustness/deck) (primary study,
  2026-09-04, with earlier drafts under `deck/archive/`) and
  [`outputs/qae_budget_targets_v2_20260908/deck/`](./outputs/qae_budget_targets_v2_20260908/deck)
  (supporting study, 2026-09-08). Google Drive/Slides copies exist but are
  permission-gated; the committed PPTX/PDF files are authoritative.
- **Japanese summary** of the supporting study —
  [`outputs/qae_budget_targets_v2_20260908/SUMMARY_JA.md`](./outputs/qae_budget_targets_v2_20260908/SUMMARY_JA.md)
- **Archived research lines** — [`archive/README.md`](./archive/README.md)

**This Markdown file is the canonical GSoC 2026 work-product submission page.**
