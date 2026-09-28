# QAE-TFIM v5 — incumbent-based free-form LLM redesign (pre-registered)

Status: **frozen before any v5 protected-test inspection.** Committed before
the v5 matrix runs; any later change would be visible as a git diff.
v3 (`QAE_PROTOCOL_V3.md`) and v4 (`QAE_PROTOCOL_V4.md`) are completed,
inspected experiments; neither is edited retroactively.

## Versioning audit (recorded)

Before this file was written we audited whether an incumbent-based protocol
already existed: it had only been discussed, never committed or run
(case A). v4 results HAVE been inspected, so the design below is a new
version rather than an amendment of v4.

## Research question

**Does semantic LLM-guided adaptive architecture proposal outperform simple
local Greedy refinement under equal candidate-evaluation and
circuit-capacity budgets?**

The v4 LLM-Closed refinement was conditioned on the full evaluation history
but, like Greedy, was implicitly compared as if it were a mutation policy.
v5 removes an artificial restriction: the LLM refinement step receives the
CURRENT BEST architecture and its validation score and may **redesign the
architecture freely** within the fixed capacity — one small change, several
coordinated changes, or a complete redesign. Greedy remains deliberately
restricted to one structural change per step. The comparison is therefore
NOT "random mutation vs LLM choosing the same mutation" but **restricted
local search vs semantic global redesign conditioned on the current best
solution** — this asymmetry in search expressivity is intentional and will
be stated in the paper and slides.

## Unchanged shared elements

- Task, metric, and terminology as v4: 4-qubit open-chain TFIM ground-state
  compression; latent q0,q1; trash q2,q3; **held-out test trash fidelity**
  `F_trash = ⟨00|ρ_trash|00⟩ = P(q2q3=00)`, higher is better; "held-out /
  protected" refers only to the data split.
- Neutral capacity: exactly 12 rotations (RX/RY/RZ free, target free) +
  exactly 4 CNOTs (control ≠ target) = 16 gates, free ordering; verified
  programmatically for every candidate. No method proposes angles: every
  architecture goes through the identical shared Adam trainer (lr 0.05,
  60 epochs, init U(−0.05,0.05), best epoch by validation loss, training
  seed derived from (seed, architecture-hash)).
- 12 paired seeds (0..11), identical state streams; B = 8 evaluations per
  method per seed; selection = lowest validation loss among the method's 8;
  held-out test read once after selection; test values never influence
  generation, training, feedback, or selection.
- LLM: model from `OPENAI_MODEL`, snapshot recorded per call, temperature
  0.7, ≤2 JSON-repair retries per call, bounded capacity repair for batch
  generation, hard `LLM_API_BUDGET_USD` cap, full per-call provenance.

## v5 method definitions

**Random.** 8 independent uniform draws (fresh v5 streams — independent
replication, not a replay).

**Greedy (unchanged design, fresh streams).** Phase 1: 4 independent
Random draws; best by validation becomes the incumbent. Phase 2: 4
evaluations; each proposal changes exactly ONE structural property of the
current incumbent (one rotation axis, one rotation qubit, one CNOT
control, one CNOT target, or one position swap); accepted only if
validation improves. Deliberately local one-step hill climbing.

**LLM-Open (unchanged design, fresh pool).** One batch call for 8 diverse
candidates from the physics/task card; no feedback; bounded capacity
repair; deficits (if any) filled by flagged random draws.

**LLM-Closed (NEW: incumbent-based free-form redesign).**
- Phase 1 (semantic exploration): one batch call per seed for 4 distinct
  no-feedback candidates (diversity-encouraging wording, as v4); all four
  trained/evaluated; the best by validation becomes the incumbent C*.
- Phase 2 (free-form semantic redesign), 4 evaluations: each call gives
  the LLM (1) the full physics/task card, (2) the exact capacity contract,
  (3) the CURRENT BEST architecture C* as an explicit gate list, (4) its
  validation trash fidelity F_val(C*), (5) a statement that the incumbent
  may be retained conceptually, modified locally, or redesigned
  substantially, and (6) an instruction to propose ONE new complete
  architecture expected to improve validation. **No restriction to one
  change; no instruction to stay close to the incumbent.** The proposal is
  trained by the shared optimizer; if its validation loss beats the
  incumbent's, it becomes the new incumbent; otherwise the incumbent is
  kept. The next call sees ONLY the resulting current best (not the full
  history).
- Optional reasoning metadata requested alongside the architecture
  (strategy ∈ {local_adjustment, topology_redesign, rotation_redesign,
  global_redesign}; short rationale; preserved aspects; changed aspects).
  Metadata is logged for analysis only and never alters scoring.

## Duplicates / invalid proposals (frozen policy)

Per refinement call: ≤2 JSON-repair retries (parse failures only). A
proposal that is capacity-invalid after retries, an exact duplicate of the
incumbent, or a duplicate of any previously evaluated candidate in that
seed consumes the evaluation and is replaced by a flagged random draw
(fresh v5 fallback stream). Each failure type is logged separately
(invalid gate count / invalid CNOT / schema failure / duplicate-of-
incumbent / duplicate-of-other). Warm-batch deficits use the bounded
capacity-repair-then-flagged-random policy of v4.

## Pre-declared analysis

- Primary: held-out test F_trash of the selected candidate, four methods;
  paired per-seed differences; exact two-sided Wilcoxon; bootstrap 95% CIs;
  Cohen's dz; win counts.
- Anytime best-so-far curves (evaluations 1..8).
- Refinement gains for Greedy and LLM-Closed:
  `Delta_refine = F_final − F_best-warm-start` (validation and test),
  fraction of seeds improved, accepted refinements, per-step improvements.
- LLM-Closed redesign diagnostics: per-proposal structural edit distance
  from the incumbent (fraction of differing gate slots, position-wise on
  the 16-slot serialization), number of changed slots, acceptance vs edit
  distance (do successful proposals tend to be local or global?), proposal
  diversity, duplicate/invalid rates, and the strategy metadata
  distribution (with acceptance rate per strategy).
- Fairness framing (pre-declared): equal circuit capacity, gate set,
  trainable-parameter count, angle optimizer, data, B=8, warm-start count
  4, refinement count 4, and accept/reject policy; NOT equal search
  expressivity (restricted one-change neighbourhood vs free redesign) —
  deliberate, and reported as such.

## Honesty commitments

Outcome not assumed. If Closed > Open: adaptive semantic redesign adds
value. If Closed ≈ Open: the semantic batch remains sufficient at B=8. If
Closed < Open: adaptive redesign does not compensate for reduced
exploration breadth at this budget. Whichever occurs is reported. All
figures/statistics from committed artifacts by committed scripts.
