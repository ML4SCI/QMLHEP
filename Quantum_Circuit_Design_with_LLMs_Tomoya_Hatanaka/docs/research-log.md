# Research log — llm-vqc

Maintained by the research-workflow skill and the researcher. Append-only:
newest entry last. English.

## Project
- **Goal**: Determine whether and when physics-informed language-model
  proposals are a useful circuit-architecture search strategy for a quantum
  autoencoder, under budget-matched comparison against non-semantic search.
- **Current research question**: What is the smallest candidate-evaluation
  budget that reaches a fixed target validation trash fidelity in at least
  10 of 12 paired seeds?
- **Hypothesis and competitors**: H1 a budget below 8 suffices on the Ising
  chain; H_c1 attainment counts near a threshold are dominated by run-to-run
  variation, so budget changes move the count without moving the fidelity
  distribution. Diagnostic prediction: under H1 the count rises monotonically
  with budget; under H_c1 it does not.
- **Novelty status**: recombined (closest prior work: this project's own
  one-factor robustness study of 2026-09-04).
- **Design lock**: locked 2026-09-08
  (`docs/research/QAE_BUDGET_TARGET_PROTOCOL.md`; no amendments).
- **Started**: 2026-06 (this log opened 2026-09-08)
- **Resources**: Paperpile folder `llm-vqc` · GitHub `dorakingx/llm-vqc`
  (private) · Overleaf `llm-vqc` · Drive `Progress/llm-vqc`

## Phase status
| Phase | Status | Gate date | Deck |
|---|---|---|---|
| 1 Literature landscape & research gap | approved | 2026-06-23 | 20260623_llm-vqc |
| 2 Experiment design & baseline | approved | 2026-07-31 | 20260731_llm-vqc |
| 3 Initial results | approved | 2026-08-29 | 20260829_llm-vqc |
| 4 Main results | in progress | 2026-09-08 | 20260908_llm-vqc |
| 5 Paper-ready analysis | pending | | |

## Entries

### 2026-09-08 — Main results — Minimum budget to a target validation fidelity
- **Done**: Pre-registered and committed the budget-target protocol before any
  new candidate was generated. Reproduced the prior validation-only audit of
  seven historical conditions byte-for-byte with no model calls. Implemented an
  arbitrary-even-budget, validation-only runner with resume, an anchor-aware
  manifest verifier that does not weaken the original one-factor rule, and a
  reuse-only mode that regenerates tables from paid artefacts with no API key
  and no cost cap configured. Executed the two pre-registered boundary cells.
- **Results** (measured): Ising chain, budget 6, target 0.95 — LLM-Open 12/12
  (passes), LLM-Closed 8/12, Greedy 2/12, Random 0/12, all on the same 12
  paired seeds. XXZ chain, budget 10, closed loop only — 5/12, against 9/12 at
  budget 8. Smallest verified passing budget on the Ising ladder: 6 for the
  open loop, 8 for the closed loop; budget 4 is a verified failure for both.
  XXZ: no verified passing budget among those run.
- **Negative or null results**: The closed loop fails the rule at budget 6,
  where the open loop passes. The XXZ boundary probe did not rescue the cell;
  the count fell while the mean final fidelity moved only -0.0076. Target 0.99
  is reached by 0 of 12 seeds in both new cells, as in every earlier condition.
  Attainment counts are not monotone in budget, so no interval for a minimum
  is reported.
- **Commits**: on branch `claude/llm-vqc-min-budget-search-eb95ad`, opened from
  `analysis/qae-budget-target-20260908`.
- **Manuscript**: not updated this run; the manuscript covers the earlier
  robustness study and this checkpoint's decisions are still open.
- **Deck**: `My Drive/Progress/llm-vqc/20260908_llm-vqc` (https://docs.google.com/presentation/d/1vAIwMlvsQusalNrKdYEGFGt8ts4Id2osqez_n0maFuc/edit). Rebuilt 2026-09-08 after review feedback that the type was too small, the whitespace unbalanced and the figures unreadable; the house template was changed rather than this project's deck alone. The first native file was superseded and trashed, so its URL no longer resolves.
- **Open questions**: whether the budget-6 pass survives a repeat with fresh
  generation seeds; whether the 0.99 ceiling is search, training length or
  circuit capacity; the 6-qubit, 8-qubit and alternative-model boundaries,
  left unresolved on purpose.
- **Evidence states**: CONFIRMED — the attainment counts above, the
  byte-for-byte reproduction of the earlier audit, and that no candidate was
  evaluated on the test set. MODEL_INFERENCE — that the budget-6 closed-loop
  failure is caused by splitting a small budget into 3 warm starts plus 3
  redesigns, and that the XXZ count change is threshold sensitivity rather
  than a fidelity change. EXPERIMENT_CHECK_REQUIRED — both of those readings.
  UNDETERMINED — what binds at target 0.99.
- **Self-review**: SELF_PASS. Skeptic: the result could exist under a false H1
  only if counts were monotone by construction; they are not, since each budget
  is an independent run, so the test was genuine — and H_c1 is in fact
  supported. Could not check: whether a repeat of either cell lands the same
  way; only one run per cell exists.
- **Decision requested**: (D1) repeat both boundary cells with fresh generation
  seeds, or accept budget-6 open-loop as the working answer; (D2) diagnose
  capacity and training length for the 0.99 target, or record it out of scope;
  (D3) whether any confirmatory read of held-out data is wanted. What would
  change the answer to D1: a repeat moving either count by more than two seeds.
- **Recommendation** (moderate confidence): D1 repeat the Ising cell only, as
  it carries the reported pass; D2 diagnose rather than sweep, since a budget
  sweep cannot separate the candidate causes; D3 (high confidence) do not read
  held-out data yet.
- **Human decision** (quote, date): pending.
- **Next**: awaiting the decisions above; no further paid run is authorised.
