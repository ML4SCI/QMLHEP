# Untrained QKAN Initialization Comparison: SineKAN vs. Chebyshev vs. Random

**Scope:** untrained QKAN circuit, ideal backend (`lightning.qubit`), top tagging with mass cut, `n5` partition,
seeds 10–14.
**Status:** complete. Post-training comparison of the sine warm start not covered (see Section 6).

## 1. Summary

Before any quantum fine-tuning, on the ideal simulator (`lightning.qubit`), the
QKAN circuit was evaluated with three initializations of the **same pruned
graph** on the **same test set** (top tagging, mass cut, `n5` partition, 9466
test jets per seed, seeds 10-14):

| init | mean AUC ± std (5 seeds) |
|---|---|
| Chebyshev warm start | **0.698 ± 0.004** |
| SineKAN warm start | 0.644 ± 0.019 |
| Random (N(0,1), untrained) | 0.482 ± 0.029 |

- Both warm starts carry real signal (AUC well above 0.5); random init is at chance.
- The SineKAN warm start is **~0.054 AUC below Chebyshev** on every seed (0.623-0.667 vs 0.692-0.701) and is more seed-dependent, but it is clearly better than random (~+0.16 AUC).
- Accuracy is ~0.5-0.54 for both warm starts, so the bias toward predicting "positive" documented in `reports/06_baseline_collapse_investigation.md` is present for sine as well (recall 0.87-0.99, precision 0.51-0.52). **AUC is the meaningful comparison here**, not accuracy/F1.
- The most likely cause of the sine gap is the poor quality of the sine edge fit (§4) rather than the circuit. This has not been isolated experimentally.

## 2. Method

**Sine extraction** (`src/architectures/extractor_sine.py`, `sine_basis.py`), derived from `tests/sine_edge_fit_test.py` (removed from `tests/`; available in git history at commit `0b7c53b`):
each edge's isolated response is fit with `y = Σ_k A_k sin(freq_k·x + phase_k)`, `k = 1..4` (same fixed degree as Chebyshev, `chebyshev_max_degree = 4`). freq/phase come from the fixed SineKAN grid (`build_sine_grid`, freq = 1..4), only `A_k` is fit, by `lstsq` (no gate, no bias term). Pruning, wires, sum/mult grouping and padding are inherited unchanged from `SymbolicWarmStartExtractor`; the graph is saved with `basis="sine"` (separate file `quantum_weights_sine.pt`).

**Sine circuit edge** (`QKANModel._qkan_edge_sine`): per harmonic `k`: `RY(A_k)` then `RZ(freq_k·x + phase_k)`, followed by a final `RY(0)` (padding slot). The Chebyshev edge is unchanged (`RY(c_i)`, `RZ(acos x)`). Stage 2 and the IsingZZ multiplication structure are identical for both.

**Evaluated (all untrained, ideal backend, no fine-tuning):**
1. `sine`: SineKAN warm start.
2. `chebyshev`: re-extracted from the same `03_retrained` checkpoint through the identical pipeline. Its confusion matrices and AUC match the previously stored baseline JSONs exactly for all 5 seeds, which validates the setup.
3. `random`: Chebyshev graph structure, weights ~ N(0,1), seeded with the run seed, **not trained** (the `random` results already stored in `outputs/` are post-training and therefore not directly comparable).

Reproduce: `python scripts/eval_sine_baseline.py --seeds 10 11 12 13 14`.

## 3. Results (per seed)

AUC (primary metric):

| seed | sine | chebyshev | random (untrained) |
|---:|---:|---:|---:|
| 10 | 0.667 | 0.700 | 0.444 |
| 11 | 0.626 | 0.699 | 0.456 |
| 12 | 0.623 | 0.699 | 0.503 |
| 13 | 0.656 | 0.701 | 0.499 |
| 14 | 0.649 | 0.692 | 0.507 |
| **mean ± std** | **0.644 ± 0.019** | **0.698 ± 0.004** | **0.482 ± 0.029** |

Other metrics, mean ± std over 5 seeds (threshold 0.5):

| init | accuracy | F1 | precision | recall |
|---|---:|---:|---:|---:|
| sine | 0.528 ± 0.012 | 0.665 ± 0.012 | 0.516 ± 0.008 | 0.940 ± 0.064 |
| chebyshev | 0.541 ± 0.027 | 0.680 ± 0.008 | 0.523 ± 0.016 | 0.976 ± 0.023 |
| random | 0.490 ± 0.016 | 0.482 ± 0.049 | 0.488 ± 0.018 | 0.478 ± 0.075 |

Confusion matrices `[[TN, FP], [FN, TP]]`:

| seed | sine | chebyshev | random |
|---:|---|---|---|
| 10 | [[132,4601],[33,4700]] | [[0,4733],[0,4733]] | [[2772,1961],[3036,1697]] |
| 11 | [[879,3854],[593,4140]] | [[863,3870],[224,4509]] | [[2284,2449],[2502,2231]] |
| 12 | [[1015,3718],[634,4099]] | [[847,3886],[234,4499]] | [[2344,2389],[2429,2304]] |
| 13 | [[443,4290],[96,4637]] | [[345,4388],[36,4697]] | [[2322,2411],[2309,2424]] |
| 14 | [[257,4476],[64,4669]] | [[446,4287],[66,4667]] | [[2166,2567],[2069,2664]] |

For reference, the *trained* random-init results already in `outputs/top/seed_*/results/qkan/ideal/random/` reach only AUC 0.527 ± 0.049, i.e. still far below both untrained warm starts.

## 4. Analysis

- **The warm start carries signal**: both extracted initializations outperform random initialization by a large margin, before any training. The AUC of the random circuit (0.48) is at chance level, as expected.
- **Sine < Chebyshev by ~0.05 AUC, consistently**, and with larger variance across seeds (std 0.019 vs 0.004).
- **Edge-fit quality explains a lot of this.** Fit R² over the extracted edges (20-23 edges/seed):

  | basis | mean R² | worst edge R² |
  |---|---:|---:|
  | Chebyshev | 0.997-0.999 | 0.983-0.995 |
  | sine | 0.49-0.61 | -0.66 to -0.50 |

  With 4 fixed-frequency, fixed-phase sine terms and no constant/bias term, the basis cannot represent offsets or monotone trends on [-1, 1] that the learned splines have; Chebyshev can. Several sine edges are worse than predicting the mean (negative R²). This is a property of the fit, before the circuit, and is a hypothesis for the AUC gap rather than a proven cause.
- **Calibration bias remains** for both warm starts (recall ≥ 0.87, precision ≈ 0.52); the sine start does not fix or worsen the collapse pattern described in `reports/06_baseline_collapse_investigation.md`.

## 5. Caveats

- Untrained circuits only; whether the sine start recovers after fine-tuning is not tested here.
- The circuit mapping of the sine coefficients (RY(A_k)/RZ(freq·x+phase) with the Chebyshev-style stage 2) is one possible design choice; it is not derived to reproduce the fitted function exactly. Chebyshev's circuit is likewise only loosely faithful to its polynomial.
- Random init is a single draw per seed (seeded with the run seed); its variability across draws was not measured.
- 5 seeds, each on a different disjoint `n5` subset, so seed effects and subset effects are mixed.
- Checkpoints for seeds 10-14 come from the pre-refactor layout (`outputs/top/seed_N/models/03_retrained`); test data is the canonical `n5` subset `seed % 5` (same size and matching baseline results as the original runs).

## 6. Possible next steps

- Add a constant term (or free per-edge frequencies/phases, as in the full SineKAN) to improve the sine fit, then re-run this comparison.
- Fine-tune the sine start and compare post-training AUC against Chebyshev.
- Evaluate multiple random-init draws per seed.

## 7. Artifacts

- Code: `src/architectures/{sine_basis,extractor_sine}.py`, `qkan_model.py` (sine edge), `scripts/eval_sine_baseline.py`, `tests/test_extractor_sine.py`.
- Metrics: `outputs/top/mass_cut/n5_subset{k}/seed_{N}/results/qkan/ideal/{sine_baseline,baseline,baseline_random}/metrics_*.json` (+ ROC/PR/CM plots under `plots/`).
