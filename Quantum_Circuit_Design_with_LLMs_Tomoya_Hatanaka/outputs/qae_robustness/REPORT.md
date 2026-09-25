# QAE robustness study — controlled one-factor-at-a-time report

Protocol: `docs/research/QAE_ROBUSTNESS_PROTOCOL.md`, frozen before any
robustness result was produced. Metric: **held-out test trash fidelity**
`F_trash = P(all trash qubits = 0)`, higher is better; selection and
feedback use validation only and the test set is read once.

The previous baseline cell is **reused** from the committed artifacts of
the earlier study (`outputs/qae_tfim_neutral_v5/`);
`tests/test_qae_robustness_reference.py` proves this code reproduces its
data, architecture streams, prompts and trainer bit-for-bit.

## Selected-architecture means (12 paired seeds per cell)

| Condition | Random | Greedy | LLM-Open | LLM-Closed | status |
|---|---:|---:|---:|---:|---|
| Previous baseline (4 qubits, TFIM, B=8, reference model) | 0.8520 | 0.8696 | 0.9605 | **0.9671** | complete |
| Budget B = 4 | 0.8001 | 0.7520 | 0.8682 | **0.9494** | complete |
| Budget B = 16 | 0.8893 | 0.9069 | 0.9681 | **0.9659** | complete |
| 6 qubits | 0.7254 | 0.6936 | 0.9222 | **0.8728** | complete |
| 8 qubits | 0.5924 | 0.5245 | 0.8765 | **0.8013** | complete |
| XXZ Heisenberg chain | 0.8094 | 0.7338 | 0.9215 | **0.9563** | complete |
| Alternative model | 0.8520 | 0.8696 | 0.9503 | **0.9166** | complete |

## Open − Random

| Condition | mean paired gain | 95% CI | dz | wins | Wilcoxon p | verdict |
|---|---:|---|---:|---:|---:|---|
| Previous baseline (4 qubits, TFIM, B=8, reference model) | +0.1084 | [+0.0726, +0.1471] | 1.58 | 11/12 | <0.001 | holds |
| Budget B = 4 | +0.0681 | [+0.0096, +0.1408] | 0.56 | 8/12 | 0.064 | not detected |
| Budget B = 16 | +0.0789 | [+0.0558, +0.1013] | 1.87 | 11/12 | <0.001 | holds |
| 6 qubits | +0.1968 | [+0.1798, +0.2135] | 6.26 | 12/12 | <0.001 | holds |
| 8 qubits | +0.2841 | [+0.2572, +0.3125] | 5.55 | 12/12 | <0.001 | holds |
| XXZ Heisenberg chain | +0.1121 | [+0.0152, +0.2208] | 0.59 | 5/12 | 0.791 | not detected |
| Alternative model | +0.0982 | [+0.0623, +0.1366] | 1.44 | 11/12 | <0.001 | holds |

## Closed − Open

| Condition | mean paired gain | 95% CI | dz | wins | Wilcoxon p | verdict |
|---|---:|---|---:|---:|---:|---|
| Previous baseline (4 qubits, TFIM, B=8, reference model) | +0.0066 | [+0.0023, +0.0112] | 0.81 | 11/12 | 0.012 | holds |
| Budget B = 4 | +0.0812 | [+0.0602, +0.0987] | 2.25 | 11/12 | <0.001 | holds |
| Budget B = 16 | -0.0023 | [-0.0078, +0.0017] | -0.25 | 9/12 | 0.677 | not detected |
| 6 qubits | -0.0494 | [-0.0783, -0.0239] | -0.97 | 0/12 | <0.001 | **reverses** |
| 8 qubits | -0.0752 | [-0.1320, -0.0271] | -0.78 | 1/12 | 0.007 | **reverses** |
| XXZ Heisenberg chain | +0.0348 | [+0.0297, +0.0406] | 3.45 | 12/12 | <0.001 | holds |
| Alternative model | -0.0336 | [-0.0551, -0.0114] | -0.83 | 4/12 | 0.052 | not detected |

## Closed − Random

| Condition | mean paired gain | 95% CI | dz | wins | Wilcoxon p | verdict |
|---|---:|---|---:|---:|---:|---|
| Previous baseline (4 qubits, TFIM, B=8, reference model) | +0.1150 | [+0.0780, +0.1535] | 1.64 | 11/12 | <0.001 | holds |
| Budget B = 4 | +0.1493 | [+0.0848, +0.2303] | 1.12 | 12/12 | <0.001 | holds |
| Budget B = 16 | +0.0766 | [+0.0541, +0.0988] | 1.85 | 11/12 | <0.001 | holds |
| 6 qubits | +0.1474 | [+0.1178, +0.1749] | 2.82 | 12/12 | <0.001 | holds |
| 8 qubits | +0.2089 | [+0.1502, +0.2592] | 2.08 | 11/12 | <0.001 | holds |
| XXZ Heisenberg chain | +0.1469 | [+0.0540, +0.2531] | 0.80 | 11/12 | <0.001 | holds |
| Alternative model | +0.0646 | [+0.0276, +0.1084] | 0.86 | 9/12 | 0.007 | holds |

## Greedy − Random

| Condition | mean paired gain | 95% CI | dz | wins | Wilcoxon p | verdict |
|---|---:|---|---:|---:|---:|---|
| Previous baseline (4 qubits, TFIM, B=8, reference model) | +0.0175 | [-0.0249, +0.0625] | 0.22 | 6/12 | 0.519 | not detected |
| Budget B = 4 | -0.0481 | [-0.0946, -0.0096] | -0.61 | 2/12 | 0.016 | **reverses** |
| Budget B = 16 | +0.0177 | [-0.0118, +0.0467] | 0.33 | 8/12 | 0.301 | not detected |
| 6 qubits | -0.0318 | [-0.0689, +0.0020] | -0.48 | 3/12 | 0.176 | not detected |
| 8 qubits | -0.0679 | [-0.1502, +0.0124] | -0.45 | 5/12 | 0.204 | not detected |
| XXZ Heisenberg chain | -0.0756 | [-0.1937, +0.0397] | -0.35 | 7/12 | 0.677 | not detected |
| Alternative model | +0.0175 | [-0.0249, +0.0625] | 0.22 | 6/12 | 0.519 | not detected |

## Robustness verdict

- **Open − Random**: sign preserved in 6/6 varied conditions; significant at p<0.05 in 4/6.
- **Closed − Open**: sign preserved in 2/6 varied conditions; significant at p<0.05 in 2/6; significant with the opposite sign in 6 qubits, 8 qubits.
- **Closed − Random**: sign preserved in 6/6 varied conditions; significant at p<0.05 in 6/6.
- **Greedy − Random**: sign preserved in 2/6 varied conditions; significant at p<0.05 in 0/6; significant with the opposite sign in Budget B = 4.

## LLM proposal validity (resource-contract compliance)

| Condition | evaluated LLM candidates | flagged random fallbacks | valid share |
|---|---:|---:|---:|
| Previous baseline (4 qubits, TFIM, B=8, reference model) | 192 | 8 | 95.8% |
| Budget B = 4 | 96 | 3 | 96.9% |
| Budget B = 16 | 384 | 15 | 96.1% |
| 6 qubits | 192 | 29 | 84.9% |
| 8 qubits | 192 | 22 | 88.5% |
| XXZ Heisenberg chain | 192 | 8 | 95.8% |
| Alternative model | 192 | 67 | 65.1% |

## API usage

- 511 recorded model calls across the study (the previous baseline's calls are included via its reused artifacts).
- 428,651 input tokens, 225,577 output tokens. Token counts are the authoritative usage record; the OpenAI chat API returns no dollar figure and none is fabricated.
- Model snapshots recorded: gpt-4.1-mini-2025-04-14, gpt-5.4-mini-2026-03-17.
- Reference model `gpt-5.4-mini-2026-03-17`, alternative `gpt-4.1-mini-2025-04-14`.
- Every paid call ran under a hard `LLM_API_BUDGET_USD` cap.

## Manifest verification

Each condition stores `manifest.json` with a `frozen` fingerprint of the
LLM workflow, prompt template, output schema, trainer, splits, selection
rule, gate set, method logic, RNG streams and analysis conventions; a
`factors` block; and a `derived` block recomputed from the factors alone.

| Condition | changed factors | frozen block matches baseline | violations |
|---|---|---|---|
| Previous baseline (4 qubits, TFIM, B=8, reference model) | (none — the baseline) | yes | none |
| Budget B = 4 | budget | yes | none |
| Budget B = 16 | budget | yes | none |
| 6 qubits | n_qubits | yes | none |
| 8 qubits | n_qubits | yes | none |
| XXZ Heisenberg chain | family | yes | none |
| Alternative model | model | yes | none |

All cells have complete 12-seed paired sets; nothing is preliminary and nothing was imputed.

## Reproduction

```bash
export LLM_API_BUDGET_USD=<cap>
python scripts/qae/run_qae_robustness.py --estimate --conditions all
python scripts/qae/run_qae_robustness.py --conditions all   # resumable
python scripts/qae/build_qae_robustness_figures.py
python scripts/qae/build_qae_robustness_report.py
python scripts/qae/build_qae_robustness_slice_figures.py
python scripts/qae/build_qae_robustness_slides.py
```
