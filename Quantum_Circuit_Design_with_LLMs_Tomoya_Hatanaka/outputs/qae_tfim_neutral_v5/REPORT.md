# QAE-TFIM v5 — incumbent-based free-form redesign report

Protocol: `docs/research/QAE_PROTOCOL_V5.md`, pre-registered at commit
`33740dc` BEFORE the matrix ran. v3 and v4 are preserved unchanged.
Metric: **held-out test trash fidelity** `F_trash = ⟨00|ρ_trash|00⟩ =
P(q2q3=00)`, higher is better ("held-out/protected" = the data split only).

## Headline

With the one-mutation restriction removed, **closed-loop semantic redesign
beats the open-loop batch for the first time**: LLM-Closed − LLM-Open =
**+0.0066** (95% CI [+0.0023, +0.0112], dz = 0.81, **11/12 seeds,
p = 0.012). The LLM used its freedom: redesign proposals changed a median
of 11/16 gate slots — far outside Greedy's one-change neighbourhood.

| Method | Design | Mean | Median | SD |
|---|---|---:|---:|---:|
| **LLM-Closed** | 4 semantic warm starts + 4 free-form redesigns | **0.9671** | 0.9701 | 0.0066 |
| LLM-Open | open-loop semantic batch of 8 | 0.9605 | 0.9602 | 0.0056 |
| Greedy | 4 random warm starts + 4 one-change refinements | 0.8696 | 0.8727 | 0.0752 |
| Random | 8 independent draws | 0.8520 | 0.8658 | 0.0675 |

Paired statistics (12 paired seeds, exact two-sided Wilcoxon, bootstrap
95% CI, Cohen's dz):

- **LLM-Closed − LLM-Open: +0.0066 [0.0023, 0.0112], dz = 0.81, 11/12,
  p = 0.0122** — the first detected closed-loop advantage in this series.
- LLM-Closed − Random: +0.1150 [0.0780, 0.1535], dz = 1.64, 11/12, p = 0.00098.
- LLM-Closed − Greedy: +0.0975 [0.0621, 0.1415], dz = 1.31, 12/12, p = 0.00049.
- LLM-Open − Random: +0.1084 [0.0726, 0.1471], dz = 1.58, 11/12, p = 0.00098.
- LLM-Open − Greedy: +0.0909 [0.0526, 0.1379], dz = 1.17, 11/12, p = 0.00146.
- Greedy − Random: +0.0175 [−0.0249, +0.0625], dz = 0.22, 6/12, p = 0.52 —
  parity, consistent with v4.

## Refinement gains (Δ_refine = final − best warm start, held-out test)

- Greedy: **+0.0159** (95% CI [+0.0044, +0.0288]), 10/12 improved,
  Wilcoxon p = 0.0020.
- LLM-Closed: **+0.0143** (95% CI [+0.0032, +0.0287]), 6/12 improved,
  6/12 unchanged, Wilcoxon (nonzero diffs) p = 0.031.

Both refinement phases add value after their warm starts. The decisive
difference is the *level*: LLM-Closed refines an already-strong semantic
incumbent to 0.9671, while Greedy refines a random incumbent to 0.8696.

## Redesign behaviour (pre-declared diagnostics)

- 48 refinement evaluations: 40 genuine LLM proposals + 8 flagged random
  fallbacks. **All 8 failures were capacity-invalid gate counts; there were
  ZERO duplicate proposals** (v4 closed loop: 5 duplicates + 3 invalid;
  v3: 25 duplicates + 1 invalid) — free redesign plus the do-not-repeat
  instruction eliminated the duplicate failure mode entirely.
- Edit distance from the incumbent: mean 0.63, median 0.69 of 16 slots
  (mean **10.1 slots changed**); range 1–16 slots. Only 4/40 proposals were
  local (≤4 slots; 1 accepted); 36/40 were global (>4 slots; 6 accepted).
  The successful redesigns were mostly **global**, i.e. the freedom was
  used and paid off — this is not one-mutation hill climbing in disguise.
- Self-reported strategy: topology_redesign 32/40 (6 accepted),
  global_redesign 7/40 (1 accepted), local_adjustment 1/40 (0).
  Metadata was logged only; scoring used the circuit alone.
- Acceptance: 7/40 LLM proposals became the new incumbent.
- Diversity (mean pairwise structural distance): warm batch 0.819,
  refinement proposals 0.698.

## Interpretation (within scope)

1. Conditioned on the current best architecture and its validation score,
   the pinned model performs **coordinated global redesign**, and that
   redesign detectably improves on its own open-loop batch — the v3/v4
   "closed ≈ open" finding was at least partly an artifact of the
   sequential-history prompt style, not of feedback per se.
2. Greedy vs LLM-Closed is deliberately NOT information-matched: they
   differ in search expressivity (one-change neighbourhood vs free
   redesign) as well as semantics. The fair equalities are capacity, gate
   set, parameter count, optimizer, data, B=8, 4+4 allocation, and the
   accept/reject rule.
3. Greedy again matches Random (p = 0.52), replicating v4's parity in a
   third independent draw of the non-semantic arms.

## API usage

`gpt-5.4-mini-2026-03-17` (single snapshot recorded per call), temperature
0.7. 64 benchmark calls + 1 smoke; 50,338 input + 22,530 output tokens;
hard `LLM_API_BUDGET_USD=2.00` cap (conservative pre-call estimates; token
counts are the authoritative usage record). Raw calls in `llm_calls/`.

## Reproduction

```bash
export LLM_API_BUDGET_USD=2.00
python scripts/qae/run_qae_tfim_neutral_v5.py            # resumable
python scripts/qae/run_qae_tfim_neutral_v5.py --no-api   # Random + Greedy only
python scripts/qae/build_qae_v5_figures.py               # figures from CSVs
```
