# Research roadmap: from controlled VQC search to semantic QAE search

## Central research question

**When does language-model reasoning actually help variational quantum-circuit architecture search?**

The project originally asked whether an LLM can propose better VQC gate structures than random or evolutionary search. The controlled experiments showed that this question is too broad: when the search task contains little semantic structure, different search strategies can be practically equivalent.

The new roadmap makes the role of semantics explicit.

## Step 1 — Establish a fair search benchmark (completed)

Capacity-controlled T1/T2 fixed the main confounders:
- identical parameter budgets across candidate circuits,
- identical training and validation pipelines,
- protected test data not exposed during search,
- equal candidate-evaluation budgets,
- explicit classical and analytical baselines.

On T1-v2, Random / Evolutionary / Greedy were practically equivalent within the frozen margin, even though selecting the best of a budget of candidates helped. This says that *architecture selection can matter while the particular search algorithm does not*.

## Step 2 — Diagnose task degeneracy before comparing searchers (completed)

The first HIGGS study showed that a difficult-looking real dataset can still be a bad architecture-search benchmark if the learning pipeline is under-qualified. The follow-up data-scale study traced the near-baseline behavior to:
1. too few training examples,
2. an inherited learning rate that was too large,
3. information loss from PCA(8).

After increasing the data scale, using raw features, and auditing the optimizer, trainable quantum angles and entanglement became mildly useful again. The lesson is methodological: **qualify the task before judging the search strategy**.

The HIGGS branches are retained as archives because their Git ancestry is disconnected from `main`. Their scientific conclusions are summarized here instead of force-merging their history.

## Step 3 — Give the LLM a decision that can use meaning (new QAE study)

A Quantum Autoencoder (QAE) provides an architecture-search task with explicit semantic structure:
- which qubits are latent and which are trash,
- which correlations should be routed toward the latent subsystem,
- which gate axes match the state family,
- which entanglement topology matches the Hamiltonian interaction graph.

The first QAE task compresses 4-qubit transverse-field Ising ground states to 2 latent qubits. All methods receive exactly 12 trainable rotations and 4 CNOTs. The LLM is given the Hamiltonian, the latent/trash assignment, and the fact that the ground states can be chosen real.

This isolates a new hypothesis:

> **Semantic-prior hypothesis.** An LLM gains sample efficiency when the prompt contains task information that can be converted into useful circuit-architecture priors.

## Step 4 — Controls that distinguish LLM reasoning from a trivial heuristic

The QAE pilot includes:
- **Random**: axes and CNOT topology both random.
- **RY-only random**: all rotations fixed to RY, but CNOT topology remains random.
- **LLM semantic pool**: uses the same exact capacity but may use the Hamiltonian and latent/trash roles.

RY-only random is important. If the LLM only beats ordinary random, the result could be explained by one simple heuristic: "the target states are real, so use RY." Beating the RY-only control tests whether topology reasoning adds value beyond that heuristic.

## Step 5 — Confirm before generalizing

The current QAE result is an exploratory/frozen verification pilot, not a final publication claim. Before claiming general LLM superiority:
1. ~~replay the exact prompt through version-pinned APIs and save raw provenance~~ — **done** (v2: `gpt-5.4-mini-2026-03-17`, open- and closed-loop, full call provenance in `outputs/qae_tfim_api_v2/`),
2. ~~add Evolutionary and a hand-designed QAE baseline~~ — **done** (v2; both underperform even unconstrained Random at B=8),
3. repeat on 6–8 qubits and other Hamiltonians,
4. add noise and hardware connectivity,
5. freeze a confirmatory multi-model protocol (several pinned models, committed to Git in advance) before inspecting protected test results.

The v2 verification confirmed the pilot: the API open-loop pool (mean test trash
fidelity 0.9700) beats Random (+0.0518, 11/12 seeds), the RY-only control
(+0.0276, 12/12), Evolutionary (+0.0695, 11/12), and the hand-designed
reference (+0.0811, 12/12). Closed-loop feedback added no measurable value over
open-loop priors at this budget (23/96 proposals were duplicates).

## Step 6 — Neutral-space 2×2 (v3, completed; now a DIAGNOSTIC)

v3 (`docs/research/QAE_PROTOCOL_V3.md`, frozen before the run) removed the rigid
layer layout (free ordering of 12 rotations + 4 CNOTs) and reduced the
comparison to a clean 2×2 over semantics × validation feedback:
Random / Greedy / LLM-Open / LLM-Closed, B=8, paired seeds, pinned
`gpt-5.4-mini-2026-03-17`. Result (`outputs/qae_tfim_neutral_v3/REPORT.md`):
**semantics is the decisive axis** — LLM-Open 0.9653 vs Random 0.8721
(+0.0932, 12/12 seeds, p=0.00049, dz=1.73); LLM-Closed 0.9553 beats both
non-semantic arms 12/12; Greedy 0.7667 lands *significantly below* Random —
single-start local refinement sacrificed exploration breadth at B=8 — and
no additional closed-loop benefit over open-loop was detected (p=0.68,
26/96 duplicate/invalid proposals). v3 is preserved unchanged as the
single-start diagnostic.

## Step 7 — Multi-start 4+4 follow-up (v4, completed; now a DIAGNOSTIC)

Motivated by the v3 diagnosis, with its protocol frozen and committed before running
(`docs/research/QAE_PROTOCOL_V4.md`, commit 69f8335): Greedy and
LLM-Closed get 4 diverse warm starts before 4 refinement evaluations;
Random and LLM-Open unchanged in design (fresh draws/pool). Result
(`outputs/qae_tfim_neutral_v4/REPORT.md`): the diagnosis holds —
**Greedy recovers to Random-parity** (0.8793 vs 0.8677, p=0.42) with a
genuine refinement gain (+0.0635, 10/12 seeds, p=0.009); the **semantic
advantage replicates with an independent pool** (LLM-Open 0.9643, 12/12
vs Random, dz=3.5); LLM-Closed's duplicates drop 26/96 → 8/96 and its
refinement gain is small but positive (+0.0185), yet **no detectable
closed-loop benefit over the open-loop batch** (−0.0085, p=0.13).

## Step 8 — Incumbent-based free-form redesign (v5, completed; now the REFERENCE CELL of Step 9)

Protocol frozen and committed before running (`docs/research/QAE_PROTOCOL_V5.md`, commit
33740dc): the LLM-Closed refinement step now receives ONLY the current
best architecture and its validation trash fidelity, and may redesign
freely within the 12R+4CX capacity (no one-change restriction, no
stay-close instruction); Greedy stays one-change local search. Result
(`outputs/qae_tfim_neutral_v5/REPORT.md`): **first detected closed-loop
advantage** — LLM-Closed 0.9671 > LLM-Open 0.9605 (+0.0066 [0.0023,
0.0112], 11/12 seeds, p=0.012, dz=0.81). Redesigns are global (median
11/16 slots changed; 6/7 acceptances were global); zero duplicate
proposals (8 capacity-invalid → flagged fallbacks); Greedy again at
Random-parity (p=0.52). Framing: Greedy vs LLM-Closed differ in search
expressivity as well as semantics — deliberate, and reported as such.

## Step 9 — Controlled one-factor robustness study (2026-09-04) — ★ PRIMARY EXPERIMENT

Protocol frozen and committed at `docs/research/QAE_ROBUSTNESS_PROTOCOL.md`
(commit `6933d3f`) before any robustness result existed. The v5 cell of Step 8 is **reused,
not re-run**, and `tests/test_qae_robustness_reference.py` proves the new code
reproduces it bit-for-bit. Four factors varied one at a time: budget
`B ∈ {4,8,16}`, qubits `n ∈ {4,6,8}`, Hamiltonian (TFIM vs XXZ), and the LLM
snapshot. Everything else is frozen and machine-verified per condition by a
manifest fingerprint. Result: `outputs/qae_robustness/REPORT.md`.

**This step changed the headline, deliberately.**

- **Open − Random keeps its sign in 6/6 varied conditions** (significant in 4/6),
  and is *largest where search is hardest*: +0.197 at 6 qubits, **+0.284 at 8
  qubits** (dz = 5.55). Closed − Random holds 6/6, significant 6/6.
- **Closed − Open holds in only 2/6 and significantly REVERSES at 6 and 8
  qubits** (0/12 and 1/12 seeds). Step 8's "first closed-loop advantage" is real
  *at its own operating point* and does **not** generalise. It is therefore no
  longer the headline — keeping it would have been an overclaim.
- Greedy − Random is significant in **0/6**, so the gain is attributable to the
  semantic prior, not merely to refining a good candidate.
- New failure mode quantified: LLM resource-contract compliance falls to
  **65.1%** with the weaker model, and fallbacks stay inside the score.

## Step 10 — Minimum budget to a target fidelity (2026-09-08, supporting)

Protocol frozen and committed at `docs/research/QAE_BUDGET_TARGET_PROTOCOL.md`
(commit `f9a8d81`) before either boundary cell was run. Stops ranking methods at one budget and instead asks for the smallest
budget reaching a target *validation* fidelity in ≥10 of 12 paired seeds. Seven
earlier conditions re-analysed with **zero** model calls (reproducing the prior
audit byte-for-byte); two boundary cells newly executed (USD 0.2902).
Result: `outputs/qae_budget_targets_v2_20260908/REPORT.md`.

Smallest **verified** passing budget at target 0.95 on the 4-qubit Ising ladder:
**B = 6 for LLM-Open** (12/12) and **B = 8 for LLM-Closed**; Random and Greedy
pass at no budget tested. This independently reproduces Step 9's direction. No
method reaches 0.99 anywhere, and attainment counts are **not monotone** in
budget, so no minimum-budget interval is claimed.

## Paper story

The paper is no longer "LLM beats random at VQC design."

The stronger story is:

1. Controlled benchmarks reveal that search strategies are interchangeable when the task exposes little semantic information.
2. Task qualification explains why misleading negative or positive results appear.
3. A QAE benchmark exposes meaningful physical structure to the proposer.
4. Under exact capacity control, a task-aware language-model proposal pool becomes more sample-efficient than random controls.
5. The boundary of the claim is explicit: the benefit is **conditional on useful semantic priors**.
6. **The useful component is the score-free semantic prior, not the agent loop.** The
   open-loop pool — which never sees a score — is what survives every varied
   condition. Iterative feedback helps only when the budget is tight and
   *reverses* as the space grows. This is the project's central, and partly
   negative, finding.
