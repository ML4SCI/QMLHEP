# QAE-TFIM v3 — neutral-space four-method protocol (pre-registered)

Status: **frozen before any protected-test inspection** of v3 results.
This file is committed before the experiment matrix runs; any later change
would be visible as a git diff.

## Research question

When does task-aware language-model reasoning help QAE/VQC architecture
search under exact capacity control?

## Design: a 2×2 over semantics × feedback

|              | no validation feedback | validation feedback |
|--------------|------------------------|---------------------|
| no semantics | **Random**             | **Greedy**          |
| semantics    | **LLM-Open**           | **LLM-Closed**      |

These four are the ONLY primary methods. RY-only random, Evolutionary, and
the hand-designed reference from v2 are historical context, not part of the
v3 primary comparison.

## Task (unchanged from v1/v2)

Compress the ground-state family of the 4-qubit open-chain transverse-field
Ising model `H(h) = -Σ_{i=0}^{2} Z_i Z_{i+1} - h Σ_{i=0}^{3} X_i`,
`h ∈ [0.2, 2.0]`, into latent qubits q0,q1; trash qubits q2,q3 must approach
|00⟩. Score: trash fidelity `F_trash = P(q2q3 = 00)`; higher is better.

## Neutral circuit space (NEW in v3)

Every candidate is an **ordered sequence of exactly 16 gates**:

- exactly **12 single-qubit rotations**, each with axis ∈ {RX, RY, RZ}
  (free — no per-axis quota) and target qubit ∈ {0,1,2,3} (free);
- exactly **4 CNOTs**, each with control ≠ target ∈ {0,1,2,3}
  (repeats of the same directed pair at different positions are allowed);
- **gate ordering is completely free** — the rigid
  4-rot/2-CNOT/4-rot/2-CNOT/4-rot layering of v1/v2 is dropped, since it
  was a convenience of the original pilot, not a technical necessity.

Rationale to record: 12 rotations = 3 trainable local rotations per qubit
and 4 CNOTs ≈ 1 two-qubit entangler per qubit. This is a benchmark
**capacity choice**, not a claim that this ratio is physically optimal for
QAE.

Capacity is verified programmatically for every candidate of every method
(16 ops, 12 R, 4 CX, wire bounds, control ≠ target).

## Shared protocol (identical for all four methods)

- 12 verification seeds (0..11), **paired across methods** (same
  train/validation/test states per seed; same rng-stream construction as
  v1/v2: `default_rng(10_000+seed)`, 32 train h, 12 validation h, fixed
  64-point protected-test grid in [0.215, 1.985]).
- Candidate budget **B = 8 evaluations** per method per seed (matches
  v1/v2; a pre-run audit found no technical reason to change it).
- Identical trainer for every candidate: Adam, lr 0.05, 60 epochs,
  parameters init U(-0.05, 0.05), best epoch by validation loss;
  training seed derived from (seed, architecture-hash) exactly as v1/v2.
- Selection: the candidate with the lowest validation loss among the
  method's 8 evaluated candidates. Protected test is read once, after
  selection. Protected-test values never appear in any prompt or any
  search decision.
- Analysis: paired per-seed differences, exact two-sided Wilcoxon
  signed-rank tests, bootstrap 95% CIs of mean paired gains, win counts,
  anytime best-so-far curves (by validation, reporting the selected
  candidate's test fidelity), duplicate/invalid-proposal rates.

## Method definitions

**Random.** 8 architectures sampled independently and uniformly from the
neutral space: 12 rotations with uniform axis and uniform qubit, 4 CNOTs
with uniform ordered (control, target) pair drawn with replacement, then a
uniform random interleaving of the 16 gates. Structural duplicates within
one seed are resampled (never observed in practice at this space size).
No task knowledge, no feedback.

**Greedy.** Candidate 1 is one Random draw. Each subsequent candidate
(up to 8 total evaluations) is the incumbent with exactly ONE structural
change, chosen uniformly among applicable moves:
1. change one rotation's axis (to a different axis),
2. change one rotation's target qubit (to a different qubit),
3. change one CNOT's control (to a different valid wire),
4. change one CNOT's target (to a different valid wire),
5. swap the positions of two gates in the sequence.
The mutated candidate is trained and its validation loss compared to the
incumbent's; the mutation is **accepted only if validation improves**,
else the incumbent stays. Every evaluated mutation consumes budget whether
accepted or not. A mutation that reproduces an already-evaluated
architecture is resampled without cost (bounded attempts, then a fresh
random draw flagged as fallback). No task knowledge; DOES use validation
feedback.

**LLM-Open.** One chat-completion call to the version-pinned model
requesting 8 distinct candidates in a structured JSON schema over the
neutral space. The prompt contains the physics/task card (Hamiltonian,
real-ground-state property, nearest-neighbour interaction structure,
latent/trash roles) and the exact resource contract. No validation result
is ever shown. Invalid or duplicate entries are logged; if fewer than 8
valid unique candidates result after the bounded repair policy, the
deficit is filled with flagged Random draws (charged to the LLM arm).

**LLM-Closed.** Same pinned model, same task card and contract. Per seed,
8 sequential calls; after each candidate is trained, the next prompt
appends that seed's validation trash fidelities and duplicate flags —
never protected-test values. Invalid/duplicate proposals consume budget
and are replaced by flagged Random draws.

**LLM specifics (both arms).** Model from `OPENAI_MODEL` with the exact
returned snapshot id recorded per call; temperature 0.7; max 2 JSON-repair
retries per call, every attempt stored as an `LLMCallRecord` (system
prompt, user prompt, raw response, tokens, latency, snapshot). Hard
cumulative cost cap via `LLM_API_BUDGET_USD` checked before every call.
The LLM proposes architecture only — never continuous angles.

## Pre-inspection amendment 1 (2026-08-28, before any protected-test inspection)

The first LLM-Open pool call returned 8 candidates of which 4 were rejected
for a mechanical miscount (13 rotations + 3 CNOTs). The original harness
only repaired JSON-parse failures, so the pool would have been half random
fallbacks — measuring schema arithmetic, not architecture reasoning.
**Amendment:** the bounded repair policy for LLM-Open is extended to
capacity-invalid/duplicate entries: up to 2 additional calls may request
replacement candidates, quoting the rejected entries and the exact
capacity errors. Only after that does the flagged-random fallback fill any
remaining deficit. Call count stays bounded (≤3 pool calls) and every call
is stored. No protected-test value of any v3 candidate had been computed
or inspected when this amendment was committed; the stale pool and the
aborted partial run were deleted and the matrix restarted from scratch.

## Honesty commitments

- No condition is tuned after protected-test inspection.
- If Random or Greedy wins, that is the reported result.
- All figures and statistics are generated from committed machine-readable
  artifacts by committed scripts.
