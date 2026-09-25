# LLM-VQC Research Master Plan

**Prepared:** 2026-07-12 · **Role:** Principal research architect review · **Intended executor:** Claude Sonnet 5
**Project:** GSoC 2026, ML4SCI QMLHEP16 — "Quantum Circuit Design with LLMs" (Tomoya Hatanaka)

Throughout this document, statements are tagged:
**[FACT]** established by direct inspection of the repository or documents; **[RESULT]** conclusion supported by checks run during this review; **[ASSUMPTION]** believed but not verified; **[PROPOSAL]** recommended work; **[SPECULATIVE]** optional research opportunity.

---

## 1. Current-State Audit

### 1.1 What exists

**[FACT]** The repository is small (~1,000 LOC of Python, 11 commits, June–July 2026) and contains four modules:

- `llm_vqc/circuit_explorer.py` — BFS over circuits built from `{H, X, Y, Z, CX, CY, CZ}` up to depth ≤ 12, with equivalence-class pruning via a phase-normalized statevector hash (`state_key`), Clifford-composition simulation, self-inverse-gate pruning, and summary statistics (per-depth discovery counts, gate distribution, "best circuit", "most complex state" by measurement entropy).
- `llm_vqc/agent.py` — a minimal OpenAI function-calling loop exposing exactly one tool (`explore_circuit_space(num_qubits, max_depth)`). The LLM's entire role is to parse a natural-language question into two integers and then narrate the returned JSON.
- `llm_vqc/visualization.py` — matplotlib dashboards (trajectory, gate distribution, pruning efficiency, state probabilities, circuit diagram).
- `llm_vqc/main.py` — CLI entry point running a single hard-coded prompt (N=3, G=5).

**[FACT]** 12 unit tests exist and all pass. The repo is private, has no open issues, no CI, no experiment configuration system, no seed management, no result store, and no datasets.

### 1.2 Defects found during this review

**[RESULT] The equivalence hashing has a correctness bug that invalidates all previously reported counts.** `state_key` rounds real/imaginary parts and hashes raw bytes; NumPy's `-0.0` and `+0.0` have different byte representations, so physically identical states receive different keys. Verified consequences:

- N=2, G=8: engine reports **82** "unique states"; only **60** two-qubit stabilizer states exist. With a corrected key (adding `0.0` to kill negative zeros) the count is exactly 60.
- N=3, G=5: engine reports **893** states with per-depth counts {1, 6, 21, 77, 240, 548}; the correct values are {1, 6, 21, 74, 191, 373} (666 total ≤ depth 5). The numbers in the June 22 slide deck and in `outputs/` are therefore quantitatively wrong.

**[RESULT] The scientific question the engine answers has a known closed-form answer.** The number of n-qubit stabilizer states is 2ⁿ·∏ₖ₌₁ⁿ(2ᵏ+1) = 6, 60, 1080, 36720 for n = 1–4. With the corrected hash, the BFS reaches **all 1080** stabilizer states at N=3 (saturating at depth 8). "How many unique states are reachable" is therefore a rediscovery exercise, not a discovery.

**[RESULT] The gate set is not Clifford-complete on one qubit.** Lacking S, the N=1 reachable set is only 4 of 6 stabilizer states (|±i⟩ unreachable). At N≥2, CY supplies the missing phases and full coverage is restored. This asymmetry is undocumented.

### 1.3 What claims are currently supported

- **Supported:** the engineering claims — BFS with dedup-pruning works, Clifford composition is fast, minimum-depth retention is guaranteed by BFS, the agent loop reliably invokes the tool.
- **Not supported:** any claim about counts of equivalence classes (bug), any claim of "mapping Hilbert-space boundaries" as a contribution (known combinatorics), and any claim that the system is a *closed-loop design agent* — there is no feedback loop; the LLM makes one tool call and summarizes. The proposal's core promise (iterative generate → evaluate → refine of VQCs) is **not yet implemented at all**.

### 1.4 Verdict

**[RESULT]** The current codebase is a competent warm-up exercise, not a research vehicle. Continuing to polish discrete Clifford enumeration cannot produce a publishable result: the answers are known analytically, the LLM contributes nothing measurable, and the headline numbers produced so far are wrong. A pivot to the direction described in Section 3 is required. Parts of the engine remain useful in a supporting role (Section 5.4).

---

## 2. Literature and Novelty Assessment

### 2.1 Closest prior work

1. **Knipfer, Roman, Matchev, Matcheva, Gleyzer (Feb 2026), "AI Agents for Variational Quantum Circuit Design" (arXiv:2602.19387).** **[FACT — read in full]** This is the ML4SCI mentors' own paper and the direct predecessor of this project. Single LLM agent (Claude 3.7 Sonnet / Llama 3.3 70B) in the Orchestral AI framework writes free-form PennyLane VQC code for three QNN wrappers (Simple QNN, QuanvNN, Full-Quantum QNN) on a **synthetic 1D Gaussian-peak regression task** (21 points, predict peak position), plus a Lie-EQGNN jet-tagging benchmark where the agent designs only the internal VQC ansatz. Feedback = test RMSE + gate/parameter counts. Findings are **qualitative**: single runs, mid-run human steering, no random-search or evolutionary baselines, no seed replication, no statistical tests, no comparison against standard ansätze, and feedback computed on the **test set** (a leakage the paper does not address). The paper itself lists as limitations: exploration collapse, context-window forgetting, no principled exploration/exploitation mechanism; and as future work: benchmarking, more sophisticated agent architectures.

2. **arXiv:2606.13380 (June 2026), "An LLM System for Autonomous VQC Design."** **[FACT — abstract read]** A seven-component agentic pipeline (exploration → generation → … → review) for quantum feature maps and VQE ansätze, claiming wins over representative feature maps and competitive VQE accuracy. **[ASSUMPTION]** Based on the abstract, it is a system-demonstration paper; it does not appear to perform budget-matched comparisons against non-LLM search baselines.

3. **Roman et al. (Mar 2026), "Pareto Frontiers of Magic and Entanglement: Two Qubits" (arXiv:2603.24902).** **[FACT — read in part]** Analytic frontiers of stabilizer Rényi-2 magic vs concurrence for 2 qubits, by the same group. Relevant as a possible discrete-search extension (Section 13.4), not as competing work.

4. **Classical QAS literature.** **[ASSUMPTION — from training knowledge, spot-checked by search]** Quantum architecture search via RL (Ostaszewski et al.; Kuo et al.), evolutionary methods, differentiable QAS, MCTS, greedy operator growth (ADAPT-style), predictor-based QAS; GQE/GPT-QE (Nakaji et al. 2024) trains a transformer as a circuit generator; FunSearch/AlphaEvolve demonstrate LLM-as-mutation-operator inside evolutionary loops in classical domains. Recent related arXiv entries: [QASER](https://arxiv.org/pdf/2511.16272), [topology-driven QAS](https://arxiv.org/pdf/2502.14265), [RL for molecular PECs](https://arxiv.org/pdf/2511.16559).

5. **Supporting methodology papers in the mentors' folder:** Sim et al. 2019 (expressibility / entangling capability metrics), Nielsen et al. 2025 (the Conductor / Orchestral AI framework), QiboAgent 2026, HEPTAPOD / ASTER / Agentic Diagrammatica (the group's broader agentic-science program).

### 2.2 The gap

**[RESULT]** Across all of the above, **no work answers the quantitative question: does the LLM actually help, relative to what, at what cost?** Specifically missing everywhere:

- **Budget-matched baselines.** No paper compares LLM-guided search against random search, evolutionary search, or greedy growth *given the same number of candidate-circuit trainings*. Without this, "the agent improved over iterations" is uninterpretable — random search also improves over iterations.
- **Statistical treatment.** No seed replication, no variance reporting, no significance tests. Knipfer et al.'s headline observations (e.g., Llama beating Claude) rest on single runs of a noisy pipeline.
- **Feedback-channel ablation.** Nobody has measured *which information* in the closed loop matters (final loss only? training curves? gate counts? quantum diagnostics like gradient variance or expressibility?).
- **Causal motif analysis.** Knipfer et al. observe recurring motifs (star-topology entanglement, data/compute qubit separation) but never test whether the motifs cause the performance.
- **Clean evaluation hygiene.** Feedback on validation data with test data quarantined for final reporting.

### 2.3 The defensible contribution

**[PROPOSAL]** A rigorous, budget-matched, statistically controlled evaluation framework and study of LLM-guided VQC architecture search, on HEP-relevant QML tasks, with feedback-channel and search-scaffold ablations, and causal tests of discovered motifs. This is:

- **Novel** — the evaluation layer does not exist in the literature (Section 2.2);
- **Aligned** — it is literally the mentors' own stated future work, executed on their task suite plus ML4SCI datasets;
- **Robust to outcomes** — a well-powered negative result ("LLM priors do not beat evolutionary search at equal budget") is a publishable and useful finding; a positive result is stronger;
- **Feasible** — small circuits (≤10 qubits), simulator-only, ~7 remaining GSoC weeks.

**Duplication risk:** **[ASSUMPTION]** moderate but manageable. arXiv:2606.13380 and the mentors' paper occupy the "system demonstration" slot; the "controlled evaluation / benchmark" slot is open. The framing must be explicitly evaluative ("when and why do LLM priors help QAS?") rather than "we built an agent," which would collide with both.

---

## 3. Recommended Research Direction

**[PROPOSAL] Primary direction — "Budget-Matched Evaluation of LLM-Guided Quantum Architecture Search" (working name: LAQS-Bench).**

Build one evaluation harness in which *every* search method — LLM-iterative agent, LLM-as-mutation-operator evolutionary search, random search, classical evolutionary search, greedy layer growth, and fixed reference ansätze — proposes circuits in the **same typed circuit representation**, is charged against the **same evaluation budget** (number of candidate trainings), is trained by the **same deterministic pipeline**, and is scored by the **same validation-then-test protocol** with seed replication and significance testing.

**Why this beats the alternatives:**

- *Stay with discrete Clifford exploration:* rejected as the primary direction. The counts are known in closed form, the LLM has no measurable role, and the mentors have already moved to parameterized circuits. (Retained in a supporting role: Section 5.4.)
- *Build a bigger autonomous agent (multi-agent, literature-RAG, à la arXiv:2606.13380):* rejected. It competes head-on with a 63-page paper and the mentors' own system, and adds engineering surface without adding evidence. Feature count is not a contribution.
- *Pivot to magic/entanglement Pareto discovery with Clifford+T:* scientifically attractive but risky as the primary GSoC deliverable (depends on whether search finds anything analytics hasn't; the group's analytic machinery is strong). Kept as a **[SPECULATIVE]** secondary track (Section 13.4).
- *The chosen direction* converts the project's existing weakness (no baselines anywhere in this literature) into its central contribution, produces a paper regardless of which method wins, reuses the mentors' task for direct comparability, and adds ML4SCI's own HEP datasets for relevance to QMLHEP.

**Relationship to the original proposal:** the proposal promised a closed-loop LLM system that generates, evaluates, and refines VQCs with benchmarking against heuristic ansätze (Phase 3, Week 11). This plan implements exactly that promise, but elevates the benchmarking from an afterthought to the scientific core.

---

## 4. Research Questions and Hypotheses

**Main question (RQ1).** At a fixed budget of candidate-circuit evaluations, does LLM-guided architecture search find better-generalizing VQCs than uninformed (random) and classical informed (evolutionary, greedy) search on QML tasks?

**Secondary questions.**
- **RQ2 (scaffold):** Does embedding the LLM in an archive/population scaffold (LLM-as-mutation-operator over a top-k archive) outperform the standard full-conversation iterative agent, and does it mitigate the exploration collapse reported by Knipfer et al.?
- **RQ3 (feedback):** Which feedback channels change outcomes — validation loss only, + circuit cost statistics, + quantum diagnostics (gradient variance at init, expressibility, entanglement)?
- **RQ4 (motifs):** Are the motifs that agents converge on (star entanglement, data/compute separation, ring entanglement) causally beneficial, tested by transplanting motifs into fixed ansätze?
- **RQ5 (calibration):** On a discrete task with a *certified optimum* (from the corrected BFS engine), what is the measured regret of each method at each budget?

**Testable hypotheses.**
- **H1:** At small budgets (≤ 25 evaluations) LLM-guided methods beat random search (the prior helps early). *Direction: expected true.*
- **H2:** At larger budgets (≥ 50–100 evaluations) the advantage over evolutionary search shrinks or vanishes. *Direction: genuinely uncertain — this is the interesting one.*
- **H3:** Archive-scaffolded LLM search beats conversation-history LLM search on best-found performance and on explored-architecture diversity.
- **H4:** Adding quantum diagnostics to feedback primarily prevents proposals of untrainable circuits (fewer low-gradient-variance candidates) rather than raising the ceiling.
- **H5:** Motif transplants explain only a minority of the performance gap between agent-found and reference circuits (i.e., motifs are correlated, weakly causal).

Each hypothesis is falsifiable by the experiments in Sections 6–7; none presumes the LLM wins.

---

## 5. Required System Design

### 5.1 Architecture overview

```
configs/*.yaml ──► ExperimentRunner ──► RunDir (results.sqlite/parquet + artifacts + manifest)
                        │
        ┌───────────────┼────────────────────┐
        ▼               ▼                    ▼
  SearchDriver     EvaluationHarness     Diagnostics
  (proposes IR)    (trains candidate)    (expr., ent., grad-var)
        │               │
        ▼               ▼
   Circuit IR ──► Compiler (IR → PennyLane QNode / Qiskit)
        │
   Validators + Canonicalizer + StructuralHash (dedup, cache keys)
```

### 5.2 The circuit IR (the single most important design decision)

**[PROPOSAL]** All searchers emit candidates in a **typed JSON intermediate representation**, not free-form Python:

```json
{"n_qubits": 6,
 "encoding": {"type": "angle", "gate": "RY", "wires": "data", "reupload": 0},
 "layers": [
   {"type": "rot", "gates": ["RY","RZ"], "wires": "all"},
   {"type": "entangle", "pattern": "star", "center": 0, "gate": "CNOT"},
   {"type": "rot", "gates": ["RX"], "wires": "all"}
 ],
 "measurements": {"observable": "Z", "wires": [0,1,2,3,4]}}
```

Rationale (scientifically important, not a style choice):
1. **Fair comparison.** Random and evolutionary baselines can only be budget-matched to the LLM if all methods draw from the *same* search space. Free-form code gives the LLM a private, unmatchable action space.
2. **Determinism and validation.** IR → circuit compilation is deterministic; schema validation replaces "the agent fixes its own syntax errors over 1–3 iterations" (Knipfer), which contaminates budget accounting.
3. **Dedup and caching.** Canonicalization + structural hash detects resubmitted architectures (Knipfer's exploration collapse becomes *measurable*).
4. **Comparability ablation.** A free-form-code mode is kept as a P1 ablation arm so the study can also quantify what the IR constraint costs the LLM (and stay comparable to Knipfer et al.).

The IR grammar must cover: qubit count (2–10); encodings (angle RX/RY/RZ, re-uploading count, optional amplitude encoding); rotation layers with arbitrary gate subsets and wire groups; entanglement layers with patterns {ring, line, star(center), all-to-all, pairs(list), none} and gates {CNOT, CZ, CRZ}; layer repetition blocks; measurement of arbitrary wire subsets with Pauli observables. **[ASSUMPTION]** This grammar covers ≥95% of what Claude/Llama produced in Knipfer et al. (their best circuits — star topology, data/compute split, ring entanglement — are all expressible; verify during Phase 1 by re-encoding their published best circuits).

### 5.3 Components

| Component | Responsibility | Key requirements |
|---|---|---|
| `ir/` | schema (pydantic), validators, canonicalizer, structural hash, IR→PennyLane compiler, IR→Qiskit exporter | round-trip tests; compile determinism |
| `tasks/` | task registry: dataset loading, splits, metric definitions | fixed train/val/test splits stored on disk with hashes |
| `evaluation/` | one fixed training pipeline (torch + PennyLane `default.qubit`, AdamW, fixed schedule); returns val metric history, test metric (computed but quarantined), cost stats | per-candidate seeding; result cache keyed by (task, structural hash, train seed) |
| `diagnostics/` | expressibility (Sim et al. KL vs Haar), entangling capability (Meyer–Wallach), gradient variance at k random inits, parameter/gate/depth/CNOT counts | pure functions of IR + task dims; unit-tested against known ansätze |
| `search/` | drivers: `random`, `evolutionary`, `greedy_growth`, `llm_iterative`, `llm_evolutionary`; all implement `propose(history) -> IR` against a common budget ledger | identical mutation operator *definitions* shared between `evolutionary` and `llm_evolutionary` prompts |
| `llm/` | provider abstraction (OpenAI + Anthropic), full transcript logging, token/cost accounting, response caching, retry policy | every call persisted; temperature and model pinned in config |
| `runner/` | config → run; resumability (budget ledger checkpoint after every evaluation); parallel dispatch of independent runs | a killed run resumes without re-spending budget |
| `analysis/` | best-so-far curves, stats tests, tables, figures — all generated from the result store by script | no hand-made figures |

### 5.4 Fate of existing components

| Component | Decision |
|---|---|
| `circuit_explorer.py` BFS | **Retain, fix, repurpose.** Fix the −0.0 hash bug (and add regression tests asserting 60/1080 saturation). Repurpose as (a) the *certified-optimum discrete task* generator for RQ5 and (b) a canonicalization utility. Stop presenting state counting as a contribution. |
| `agent.py` | **Replace.** Generalize into the `llm/` provider layer + `llm_iterative` driver. The single-tool loop is a special case of the new design. |
| `visualization.py` | **Replace** with the `analysis/` module (paper-grade, script-generated from the result store). Keep only the Qiskit circuit renderer. |
| `main.py` | **Replace** with `runner/` CLI (`python -m llm_vqc.run experiment=...`). |
| Tests | Keep and extend; the 12 existing tests still pass and remain valid. |

### 5.5 Reproducibility requirements (binding)

Every run directory must contain: resolved config (YAML), git SHA + dirty flag, `pip freeze`, all RNG seeds (search seed, per-candidate train seeds, data-split seed), the complete LLM transcript (prompts, responses, token counts, cost), the budget ledger, and the result table. Any figure in the paper must be regenerable by one command from stored results. LLM non-determinism is handled by treating each LLM run as one seed-level replicate and reporting distributions, never single runs.

---

## 6. Experimental Design

### 6.1 Tasks

| ID | Task | Why | Size |
|---|---|---|---|
| **T1** | Gaussian-peak 1D regression (replicate Knipfer et al.: 21 points, predict peak position; Simple-QNN wrapper — linear embed → VQC → linear+sigmoid) | direct comparability with the mentors' paper; cheap | 150 train / 250 val / 500 test (larger test set than theirs: use 2,000 to cut metric noise — **[PROPOSAL]**) |
| **T2** | Electron-vs-photon ECAL binary classification (ML4SCI QMLHEP standard dataset), PCA → 8–16 features, angle encoding | HEP relevance; the QMLHEP program's canonical benchmark | subsample ~2k train / 1k val / 4k test **[ASSUMPTION: dataset accessible via ML4SCI/CERN opendata; verify in Phase 2 and fall back to quark-gluon or scikit-learn digits if blocked]** |
| **T3** | Certified-optimum discrete task: prepare a target stabilizer state (chosen at known BFS depth d*) using ≤ G gates from the discrete set; score = fidelity, cost = gates; optimum known exactly | unique calibration: exact regret measurable; reuses fixed BFS engine | trivial compute |
| T4 *(P2, only if ahead of schedule)* | VQE on H₂/LiH (STO-3G) | broadens claim to non-QML | small |

### 6.2 Search methods (arms)

| Arm | Description |
|---|---|
| `random` | uniform samples from the IR grammar (with sane size priors matched to grammar limits) |
| `evolutionary` | (μ+λ) evolution over IR with hand-coded mutation/crossover operators (add/remove/perturb layer, change pattern, change measurement, grow/shrink qubits) |
| `greedy` | layer-wise growth: start minimal, add the best of k sampled single-layer extensions each round |
| `llm_iter` | Knipfer-style conversational agent (full history in context) emitting IR |
| `llm_evo` | LLM as mutation operator: prompt contains top-k archive (IR + scores + diversity note), LLM proposes offspring; FunSearch-style |
| `fixed refs` | not search arms; reference points: HEA/`EfficientSU2`, `StronglyEntanglingLayers`, `RealAmplitudes` (+ data re-uploading variant), each at 2–3 sizes, trained identically |

LLM models: primary = one mid-tier model (e.g., `gpt-5.4-mini` per the repo's `.env`, or Sonnet-class); **[PROPOSAL]** one flagship model on a subset (T1, both LLM arms) as a model-tier ablation. Temperature fixed (e.g., 0.7 for proposals), logged.

### 6.3 Protocol

- **Budget:** B = 60 candidate evaluations per run (with anytime curves reported at 10/25/60). A candidate evaluation = one full training of one circuit. Invalid IR proposals are rejected by the validator without consuming simulation budget, but are counted and reported separately (LLM-call budget and invalid-rate are secondary metrics). Duplicate proposals (same structural hash) consume budget by returning the cached result — this preserves the incentive structure while making collapse visible.
- **Repetitions:** non-LLM arms: 10 seeds per (arm × task). LLM arms: 5 seeds per (arm × task × model) — cost-bounded. Each seed controls: search RNG, candidate train-init RNG stream, and (for LLM arms) a fresh conversation.
- **Training pipeline (fixed for all arms):** AdamW, lr 0.05 with decay at fixed epochs, 20 epochs, batch 16, early metric = validation loss/AUC. **Feedback to searchers uses validation only. Test metrics are computed once and quarantined until analysis** — this corrects the test-leakage flaw in Knipfer et al.
- **Primary endpoint:** best-so-far *validation* metric curve vs evaluations spent; final comparison on *test* metric of each run's selected circuit (selected on validation).
- **Statistics:** per (task, budget checkpoint): Kruskal–Wallis across arms, then pairwise Mann–Whitney U with Holm correction; report medians, IQRs, bootstrap 95% CIs, and Cliff's δ effect sizes. Significance threshold α = 0.05. With n = 5–10 per cell, only report pairwise conclusions when effect sizes are large — otherwise report estimates with CIs and say so.
- **Stopping criteria:** experiments end when the matrix in Section 10 is complete; no optional extensions unless all P0/P1 rows are done. Compute sanity: **[ASSUMPTION]** ~1–3 min/candidate training on CPU ⇒ full matrix ≈ 3,000–4,500 trainings ≈ 2–5 CPU-days serial, < 1 day with 8-way parallelism. If a pilot shows >5 min/candidate, shrink epochs or feature dims (decision gate G2).

### 6.4 Controls and confounders

- **Search-space parity:** all arms draw from the identical IR grammar (the free-form-code ablation is explicitly labeled as non-parity).
- **Training noise:** per-candidate training seed drawn deterministically from run seed + structural hash, so the *same* circuit gets the *same* training result in every arm (also enables global caching).
- **LLM contamination:** LLMs know textbook ansätze; that is the "prior" under test, not a confound — but document model checkpoints and note that reference ansätze are in-distribution for the LLM.
- **Cherry-picking:** all runs registered in configs before analysis; no run deletion; failed runs reported.
- **Metric noise:** enlarge test sets (cheap) so that inter-arm differences aren't swamped by split noise.

---

## 7. Baselines and Ablations

**Baselines (establish the yardstick):** `random`, `evolutionary`, `greedy`, fixed reference ansätze. These isolate: does *any* search beat fixed ansätze? does *informed* search beat random? — before asking whether *LLM* search beats informed search.

**Ablations (isolate the mechanism):**

| ID | Ablation | Question | Arms |
|---|---|---|---|
| A1 | Feedback channels: (i) val loss only; (ii) + cost stats (gates/params/depth); (iii) + diagnostics (grad-var, expressibility, entanglement) | RQ3 / H4 | `llm_evo` on T1, T2 |
| A2 | Scaffold: `llm_iter` vs `llm_evo` (same model, same budget) | RQ2 / H3 | both tasks |
| A3 | Model tier: mini vs flagship | does capability matter more than scaffold? | T1 |
| A4 | Representation: IR-constrained vs free-form PennyLane code (validator sandbox) | what does the grammar cost/buy? comparability with Knipfer | T1, `llm_iter` |
| A5 | No-feedback control: LLM proposes B circuits blind (open-loop, no scores returned) | is the loop doing anything beyond prior sampling? | T1, T2 |
| A6 | Motif transplant: insert star-topology / data-compute-split / ring motifs into HEA-style templates; compare against motif-free size-matched controls | RQ4 / H5 | T1, T2 (no search; direct training study) |

A5 is the cheapest and most incisive control in the study: if closed-loop ≈ open-loop, the "agentic feedback" narrative of the prior literature is hollow at these budgets — a publishable finding on its own.

---

## 8. Claim-to-Evidence Map

| # | Potential paper claim | Required evidence |
|---|---|---|
| C1 | "We provide the first budget-matched, statistically controlled comparison of LLM-guided vs classical QAS" | the framework + main matrix (Section 10) executed with protocol of §6.3; released code + result store |
| C2 | "LLM-guided search does/does not outperform random and evolutionary search at budget B on tasks T1–T3" | main matrix, primary endpoint + stats; both directions of the claim are supported by the same experiment |
| C3 | "Early-budget advantage exists but decays" (H1/H2) | anytime curves with CIs at checkpoints 10/25/60; interaction effect across checkpoints |
| C4 | "Archive scaffolding mitigates exploration collapse" | A2 + diversity metrics (distinct structural hashes, duplicate-proposal rate) |
| C5 | "Feedback content X matters / doesn't" | A1 (+ A5 for the existence of any loop effect) |
| C6 | "Motifs found by agents are (not) causally beneficial" | A6 with size-matched controls, ≥10 train seeds per template, effect sizes |
| C7 | "LLM search approaches certified optima on discrete tasks with regret R(B)" | T3 with exact optima from corrected BFS |
| C8 | "Agent-discovered circuits transfer across tasks" *(only if observed)* | top circuits from T1 retrained on T2 and vice versa; compare vs references |
| — | Any claim of *quantum advantage*, *novel physics*, or *superhuman design* | **not supportable by this design — do not make.** |

---

## 9. Implementation Roadmap for Sonnet 5

Ordered phases; each has a gate. Do not start phase N+1 before the gate of phase N passes.

**Phase 0 — Repairs and scaffolding (0.5 wk)**
- Objective: correct the existing engine; set up experiment skeleton.
- Deliverables: `state_key` fix (−0.0 canonicalization); regression tests asserting N=1→4 states, N=2→60, N=3→1080 (saturation); depth-count regression test; docs note about the missing-S-gate asymmetry; repo hygiene (ruff/pytest CI script); config system (pydantic + YAML) and run-directory manifest writer.
- Acceptance: all tests pass; a dummy config produces a run dir with manifest, seeds, git SHA, env freeze.
- Risks: none material.

**Phase 1 — Circuit IR + compiler (1 wk)**
- Deliverables: IR schema, validators, canonicalizer, structural hash, IR→PennyLane compiler, IR→Qiskit exporter, random-IR sampler (this *is* the `random` arm's generator); re-encode Knipfer et al.'s published best circuits in IR as fixtures.
- Acceptance: round-trip and determinism tests; compiled QNode trains on toy data; Knipfer fixtures compile and match expected gate/param counts; sampler produces 100% valid IRs.
- Dependency: Phase 0. Risk: grammar too narrow — mitigated by the fixture requirement.

**Phase 2 — Tasks + evaluation harness (1 wk)**
- Deliverables: T1 generator (with frozen splits + hashes); T2 loader (PCA pipeline, frozen splits) **with fallback dataset decision documented**; fixed training pipeline; result store (SQLite or parquet); candidate cache keyed by (task, structural hash, train seed); budget ledger with resume.
- Acceptance: training a fixed reference ansatz on T1 reproduces Knipfer-magnitude RMSE (~0.02–0.06) **[ASSUMPTION to verify]**; kill-and-resume test passes; cache hit returns identical result.
- Gate **G1**: T2 data verified accessible, else fallback selected. Risk: dataset access; PennyLane/torch version friction.

**Phase 3 — Diagnostics (0.5 wk)**
- Deliverables: expressibility (KL vs Haar, fixed sample count), Meyer–Wallach entanglement, gradient variance at k inits, cost stats.
- Acceptance: unit tests against known values (e.g., idle circuit → expressibility worst; HEA depth scan reproduces Sim et al. qualitative ordering).

**Phase 4 — Non-LLM search arms (1 wk)**
- Deliverables: `random`, `evolutionary` (μ+λ, operators documented), `greedy`; runner executes (arm × task × seed) grids in parallel.
- Acceptance: 3-seed smoke grid on T1 at B=15 completes unattended; best-so-far curves monotone; evolutionary ≥ random on median (sanity, not a result).
- Gate **G2 (compute sizing)**: measured minutes/candidate ⇒ confirm full-matrix feasibility; shrink epochs/features if needed.

**Phase 5 — LLM arms (1 wk)**
- Deliverables: provider layer (OpenAI + Anthropic, cached, logged, cost-tracked); `llm_iter` and `llm_evo` drivers emitting IR (JSON schema-constrained decoding where available); prompt templates versioned in-repo; invalid-proposal and duplicate handling per §6.3.
- Acceptance: 2-seed smoke on T1 at B=15 for both arms; invalid-IR rate < 20% after retry; transcripts fully persisted; cost projection for full matrix ≤ agreed cap (**[PROPOSAL]** cap ≈ $150 mini-tier; flagship subset ≈ $100 — confirm with user before Phase 6).
- Risk: schema-constrained decoding unavailable for a provider → fall back to JSON extraction + one repair retry.

**Phase 6 — Pilot + decision gate (0.5 wk)**
- Deliverables: pilot: all arms, T1, B=25, 3 seeds; analysis notebook-free script producing curves + preliminary stats.
- Gate **G3**: (i) inter-seed variance small enough that arm differences are potentially resolvable at n=5–10 (if not: increase test-set size, seeds, or budget; re-pilot); (ii) no arm is degenerate (e.g., LLM always proposing duplicates → fix prompts first). Explicit go/no-go recorded in a DECISIONS.md.

**Phase 7 — Main matrix + ablations (1.5–2 wk, mostly compute)**
- Deliverables: full Section 10 matrix; A1–A6; T3 regret study; all runs in the result store.
- Acceptance: matrix complete; every cell has its planned n; stats script runs cleanly.

**Phase 8 — Analysis, figures, paper assets (1 wk, overlaps 7)**
- Deliverables: all figures/tables of Section 12 generated by script; results narrative; DECISIONS.md finalized; repo made reproducible end-to-end (`make reproduce-pilot` on a laptop).

Total: ~7 weeks — fits the remaining GSoC window **[ASSUMPTION: ~30 h/wk as committed in the proposal]**.

---

## 10. Experiment Matrix

| Block | Arms | Tasks | Budget | Seeds | Metrics | Artifacts |
|---|---|---|---|---|---|---|
| M0 references | HEA, SEL, RealAmp, re-upload variant × 2–3 sizes | T1, T2 | n/a (each trained) | 10 | val/test metric, cost, diagnostics | reference table |
| M1 main | random, evolutionary, greedy, llm_iter, llm_evo | T1, T2 | 60 (checkpoints 10/25/60) | 10 non-LLM / 5 LLM | best-so-far val; final test; diversity; invalid rate | curves, stats table |
| M2 regret | random, evolutionary, llm_evo | T3 (3 target states at d*=4,6,8) | 40 | 10/5 | exact regret vs optimum | regret curves |
| A1 feedback | llm_evo × {loss, +cost, +diag} | T1, T2 | 60 | 5 | as M1 | ablation figure |
| A2 scaffold | llm_iter vs llm_evo | T1, T2 | 60 | 5 | + duplicate rate, diversity | collapse analysis |
| A3 model tier | llm_iter, llm_evo × {mini, flagship} | T1 | 60 | 3–5 | as M1 | tier table |
| A4 representation | llm_iter × {IR, free-code} | T1 | 60 | 5 | as M1 + failure taxonomy | repr. figure |
| A5 open-loop | llm_evo vs blind LLM sampling | T1, T2 | 60 | 5 | as M1 | loop-effect figure |
| A6 motifs | motif-transplant templates vs controls | T1, T2 | n/a | 10 train seeds | test metric, effect size | motif table |

Priority if compute/cost forces cuts: M1 > A5 > A2 > M0 > A1 > M2 > A6 > A4 > A3.

---

## 11. Reproducibility and Research Engineering Requirements

- **Config:** every run fully specified by one YAML (task, arm, budget, seeds, model, temperature, prompts version); configs archived in-repo under `configs/experiments/`.
- **Seeds:** three independent, logged streams — data split, search, per-candidate training (derived as `hash(run_seed, structural_hash)`).
- **Environment:** `pip freeze` + Python version in every manifest; pinned `requirements.lock`; PennyLane/Qiskit versions pinned (Knipfer et al. hit a PennyLane deprecation mid-study — pin to avoid).
- **Logging:** per-run JSONL event log (proposal → validation outcome → evaluation result), full LLM transcripts, token/cost ledger.
- **Storage:** single result store (SQLite recommended) + per-run artifact dirs; nothing scientific lives only in stdout.
- **Checkpoint/recovery:** budget ledger checkpointed after every evaluation; `--resume` mandatory feature; kill-tested in CI.
- **Provenance:** git SHA + dirty flag recorded; refuse to launch matrix runs from a dirty tree.
- **Figures/tables:** one `analysis/make_figures.py` regenerates everything from the store; figures carry the config hash in metadata.
- **Cost control:** LLM cache keyed by (model, prompt hash, temperature, seed); hard per-run cost cap that aborts gracefully.

---

## 12. Publication Plan

- **Contribution statement:** "We introduce the first budget-matched, statistically controlled evaluation of LLM-guided quantum architecture search, disentangling the contributions of the LLM prior, the feedback loop, and the search scaffold on HEP-relevant QML tasks — including a discrete calibration task with certified optima — and causally test the circuit motifs that LLM agents discover."
- **Structure:** Intro → Related work (QAS; LLM-agents-for-quantum; Knipfer et al. as direct predecessor) → Framework (IR, arms, budget protocol) → Tasks → Main results (anytime curves, final table) → Ablations (loop, scaffold, feedback, representation, motifs) → Regret study → Limitations (simulator-only, ≤10 qubits, model snapshot dependence, no noise) → Conclusions.
- **Required figures/tables:** (F1) anytime best-so-far curves with CIs, per task; (F2) final test-metric box plots vs references; (F3) diversity/collapse analysis; (F4) feedback ablation; (F5) open-loop vs closed-loop; (F6) regret curves on T3; (F7) motif-transplant effects; (T1) main stats table with effect sizes; (T2) cost table ($, tokens, CPU-hrs per arm).
- **Venues (in order):** NeurIPS *Machine Learning and the Physical Sciences* workshop (deadline ~Sept — fits GSoC end); then full paper to *Machine Learning: Science and Technology* or *Quantum Machine Intelligence*; QTML as conference option. **[ASSUMPTION on deadlines — verify in August.]**
- **Minimum evidence threshold for submission:** M1 complete on both tasks with planned n; A5 and A2 complete; stats pipeline run; limitations honestly stated. Ablations A1/A6 strengthen but are not gating for a workshop submission.
- **Posture on outcomes:** a null result for RQ1 is written as "LLM priors do not (yet) buy sample efficiency in QAS at realistic budgets — and here is what does" — with the mentors as natural co-authors, this remains a solid contribution.

---

## 13. Risks, Decision Gates, and Pivot Conditions

1. **T2 dataset inaccessible** → Gate G1 (Phase 2): fall back to quark-gluon subsample or a standard non-HEP dataset; document. *Continue regardless.*
2. **Candidate training too slow** → Gate G2 (Phase 4): cut epochs/features/qubits until ≤ 2 min/candidate; if impossible, drop T2 sample size. *Continue.*
3. **Seed variance swamps arm differences** → Gate G3 (Phase 6): first enlarge test sets and add seeds; if still unresolvable, reframe primary endpoint as estimation-with-CIs rather than hypothesis testing (explicitly allowed; the paper becomes a calibrated benchmark rather than a horse race). *Continue with reframing.*
4. **LLM arms degenerate (duplicate storms, invalid-IR storms)** → fix prompts/retry policy once; if persistent, this *is* a result (report as failure-mode finding; A2/A5 become the paper's core). *Continue with re-weighted narrative.*
5. **Budget/cost overrun on LLM calls** → drop A3 (model tier), reduce LLM seeds 5→3, keep M1+A5+A2 intact. *Continue.*
6. **A genuinely positive LLM result appears early** → do not expand scope; deepen replication of that cell (more seeds) before believing it.
7. **Abandon conditions:** none for the framework itself (it is publishable with either outcome). The *only* abandonable element is the **[SPECULATIVE]** magic/entanglement side-track below.

**13.4 [SPECULATIVE] Secondary track — discrete search for magic/entanglement frontiers (Clifford+T, 3 qubits).** Extend the corrected BFS engine with T gates and track (stabilizer-Rényi-2 magic, entanglement) per state to numerically map the 3-qubit Pareto frontier, connecting to Roman et al.'s 2-qubit analytics. Pursue **only after** the main matrix is complete, or if the mentors explicitly request it. Value: possible standalone note with the group; Risk: analytics may overtake numerics; unbounded state space with T gates requires depth/T-count truncation and careful numerics.

---

## 14. Prioritized Task Backlog

**P0 — essential for scientific validity**
1. Fix `state_key` −0.0 bug + regression tests (60 / 1080 assertions).
2. Circuit IR, validators, canonicalizer, structural hash, compiler, random sampler.
3. Evaluation harness: fixed trainer, frozen splits, val-only feedback / test quarantine, result store, budget ledger with resume.
4. Arms: `random`, `evolutionary`, `llm_evo`, `llm_iter`; reference ansätze (M0).
5. M1 main matrix on T1 + one real-data task, with seeds and stats pipeline.
6. Provenance: manifests, seeds, transcripts, cost ledger.

**P1 — important for a strong paper**
7. A5 open-loop control; A2 scaffold ablation; diversity/collapse metrics.
8. Diagnostics module + A1 feedback ablation.
9. T3 certified-optimum regret study (repurposed BFS).
10. A6 motif-transplant study.
11. `greedy` arm; enlarged test sets; Knipfer-circuit IR fixtures.

**P2 — useful but optional**
12. A4 free-form-code ablation; A3 model-tier ablation.
13. T4 VQE task; hardware-topology constraint mode (CNOT-count objectives).
14. Transfer study (C8); public dataset cards + polished docs site.

**P3 — future work**
15. Clifford+T magic/entanglement frontier mapping (13.4).
16. Real-hardware or noise-model evaluation of top circuits.
17. Multi-agent / RAG-augmented proposal generation; RL baseline arm.

---

## 15. Sonnet 5 Handoff Prompt

Copy everything between the lines into a fresh Sonnet 5 session started in the `llm-vqc` repository.

---

You are the implementation engineer for the llm-vqc project (GSoC 2026, ML4SCI QMLHEP). The approved research design is in `LLM-VQC_MASTER_PLAN.md` (this file). Your job is to execute its Implementation Roadmap (Section 9) exactly, phase by phase. You are not authorized to change the research design: the circuit IR approach, the budget-matched protocol, the arm definitions, the validation-feedback/test-quarantine rule, and the statistics plan are fixed. If you believe a design element is wrong or infeasible, stop, write the issue and your proposed alternative into `DECISIONS.md`, and ask the user — do not silently substitute your own design.

Operating rules:

1. Work through phases 0 → 8 in order. Do not begin a phase until the previous phase's acceptance criteria are demonstrably met (run the tests/smoke runs and show the output). Record each phase completion, with evidence, in `DECISIONS.md`.
2. Phase 0 starts with a known bug: `state_key` in `llm_vqc/circuit_explorer.py` produces distinct hashes for identical states because rounded `-0.0` and `+0.0` differ at byte level. After fixing, these regression tests must pass: N=1 exploration saturates at 4 states; N=2 at 60; N=3 at 1080 (saturation depth 8); N=3 per-depth new-state counts up to depth 5 are {1, 6, 21, 74, 191, 373}.
3. Every experiment run must be launched from a config file, write a manifest (git SHA, seeds, env freeze, resolved config), persist all LLM transcripts and costs, and be resumable. If you find yourself producing a scientific number that is not in the result store, you are off-plan.
4. Search feedback uses validation metrics only; test metrics are computed once, stored, and never shown to any search arm or used for mid-study decisions.
5. All search arms must propose circuits in the shared IR grammar. When implementing the evolutionary arm and the LLM-mutation arm, the mutation operator definitions must be identical in both (hand-coded in one, described in the prompt of the other).
6. Before Phase 6 (pilot), present the user with the projected LLM API cost for the full matrix and get explicit approval.
7. At gates G1 (dataset), G2 (compute sizing), G3 (pilot variance), apply the decision rules in Section 13 of the plan and record the outcome in `DECISIONS.md` before proceeding.
8. Never fabricate, smooth, or drop experimental results. Failed runs are reported as failed. If results contradict the hypotheses (H1–H5), that is a valid outcome — record it.
9. Keep commits small and phase-scoped; do not commit secrets; pin dependency versions in a lock file at Phase 0.
10. Scope discipline: the P2/P3 backlog items and the speculative Clifford+T track (Section 13.4) are off-limits until the P0 and P1 items are complete and the user approves.

Begin with Phase 0 now: read Section 9 of the plan, restate Phase 0's deliverables as a checklist, and implement them.

---

*End of master plan.*
