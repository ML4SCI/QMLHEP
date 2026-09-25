# QAE-TFIM v4 — multi-start four-method protocol (pre-registered)

Status: **frozen before any v4 protected-test inspection.** This file is
committed before the v4 matrix runs; any later change would be visible as a
git diff. The v3 protocol (`QAE_PROTOCOL_V3.md`) is NOT edited retroactively;
v3 remains a valid, archived, reproducible experiment.

## Why v4 exists (motivation, stated honestly)

This design change is made AFTER the v3 results are known. v3 found:

- Greedy (1 random start + 7 local mutations) reached mean held-out test
  trash fidelity 0.7667 — significantly below Random's 0.8721;
- LLM-Closed (fully sequential proposals) showed no detectable advantage
  over LLM-Open (paired difference −0.0100, p ≈ 0.68) and produced 26/96
  duplicate/invalid proposals.

Diagnosis: at B=8 in the large neutral space, **single-start adaptive search
sacrificed exploration breadth** — Greedy committed to one (usually
mediocre) random start, and LLM-Closed appeared to focus quickly around
previously successful structures (premature exploitation / loss of proposal
diversity). v3 does not isolate "feedback" causally, because the adaptive
arms also differ from the non-adaptive arms in allocation policy.

**v4 hypothesis (pre-declared):** giving the adaptive methods a multi-start
warm-up (4 diverse candidates) before refinement (4 adaptive candidates)
removes the single-start handicap. Whether feedback then adds value over the
open-loop baselines is an open question; the outcome is not assumed.

## What is unchanged from v3

- Task: 4-qubit open-chain TFIM ground-state compression,
  `H(h) = -Σ Z_i Z_{i+1} - h Σ X_i`, `h ∈ [0.2, 2.0]`; latent q0,q1;
  trash q2,q3.
- Metric: **held-out test trash fidelity**
  `F_trash = ⟨00| ρ_trash |00⟩ = P(q2q3 = 00)` where `ρ_trash` is the
  reduced density matrix of q2,q3 after the encoder; higher is better.
  "Held-out/protected test" refers only to the DATA SPLIT (the 64 unseen
  h values); the fidelity formula is identical on train/validation/test.
  F_trash is a compression-side trash/reference fidelity, not automatically
  identical to end-to-end reconstruction fidelity.
- Neutral circuit space: ordered sequence of exactly 16 gates — exactly 12
  single-qubit rotations (axis ∈ {RX,RY,RZ} free, qubit free) + exactly 4
  CNOTs (control ≠ target, repeats allowed), ordering completely free.
  Rationale: 12 rotations = 3 local rotations per qubit, 4 CNOTs ≈ 1
  entangler per qubit — a benchmark capacity choice, NOT a claim that this
  ratio is physically optimal for QAE.
- Trainer: Adam lr 0.05, 60 epochs, init U(−0.05, 0.05), best epoch by
  validation loss; training seed derived from (seed, architecture-hash) as
  in v1/v2/v3. Identical for every candidate of every method.
- 12 paired verification seeds (0..11); identical state streams as v3.
- Budget: exactly **B = 8 candidate evaluations** per method per seed.
- Selection: lowest validation loss among the method's 8 evaluated
  candidates; held-out test read once, after selection; test values never
  appear in prompts or search decisions.
- LLM: model from `OPENAI_MODEL` with returned snapshot id recorded per
  call, temperature 0.7, ≤2 JSON-repair retries per call, bounded capacity
  repair for batch generation (as amended in v3), hard cost cap via
  `LLM_API_BUDGET_USD`, full per-call provenance. The LLM proposes
  architecture only, never angles.

## v4 method definitions

**Random (unchanged).** 8 independent uniform draws from the neutral
space. No semantics, no feedback. Breadth-only baseline.

**Greedy (NEW 4+4).**
- Phase 1 (exploration): draw and evaluate 4 independent Random
  architectures; the best by validation loss becomes the incumbent.
- Phase 2 (refinement): 4 further evaluations; each candidate applies
  exactly ONE structural change to the incumbent (one rotation axis, one
  rotation qubit, one CNOT control, one CNOT target, or one position swap),
  chosen uniformly; accepted only if validation improves, else the
  incumbent stays. Every evaluated mutation consumes budget. A mutation
  duplicating an already-evaluated architecture is resampled without cost
  (bounded attempts, then a flagged fresh random draw).

**LLM-Open (unchanged design).** One batch call requesting 8 distinct
candidates from the physics/task card; no validation feedback; pool frozen
and evaluated on every seed. Bounded capacity repair as in v3.

**LLM-Closed (NEW 4+4).**
- Phase 1 (semantic exploration): one batch call requesting 4 distinct
  candidates with NO validation feedback (bounded capacity repair allowed;
  deficits after repair filled by flagged random draws). All 4 trained and
  evaluated. This warm-start batch is generated once per seed.
- Phase 2 (semantic refinement): 4 sequential calls; each prompt contains
  the validation trash fidelities and duplicate flags of ALL candidates
  evaluated so far in this seed, and requests ONE new candidate. Invalid or
  duplicate proposals consume budget and are replaced by flagged random
  draws.

## Pre-declared analysis

- Primary: held-out test trash fidelity of the validation-selected
  candidate, all four methods; paired per-seed differences; exact two-sided
  Wilcoxon; bootstrap 95% CIs of mean paired gains; Cohen's dz; win counts.
- Anytime best-so-far curves (by validation; report the selected
  candidate's test fidelity) vs evaluations 1..8.
- Duplicate/invalid/fallback accounting per method.
- **Greedy refinement gain**: selected-candidate performance (validation
  and test) minus the best of Greedy's first 4 warm-start candidates
  (validation-selected within the first 4, its test value read only at
  analysis time from stored per-candidate results — no extra training).
- **LLM-Closed refinement gain**: same definition over its first 4
  warm-start candidates.
- Proposal diversity of LLM-Open vs LLM-Closed: number of unique
  architectures and mean pairwise structural distance (fraction of
  differing gate slots after canonical serialization) among first-4 and
  last-4 proposals.
- Presentation language: the four methods are a CONCEPTUAL organization
  (semantics × search policy), not a strict causal factorial; Random vs
  Greedy differ in allocation policy as well as feedback, and LLM-Open vs
  LLM-Closed compare open-loop batch search vs adaptive closed-loop search.

## Honesty commitments

- No condition is tuned after v4 protected-test inspection; any mechanical
  API/schema repair will be documented and committed before restarting and
  before inspecting outcomes.
- If Random or Greedy wins, or if LLM-Closed again fails to beat LLM-Open,
  that is the reported result.
- All figures/statistics generated from committed machine-readable
  artifacts by committed scripts.
