# HIGGS archive synthesis

The HIGGS studies are kept as scientific archives because their branch histories were
disconnected from `main`. Those branches have since been **deleted**; their full artifacts
are preserved on `main` under [`archive/pre-consolidation/`](../../archive/pre-consolidation)
with provenance SHAs recorded in [`archive/README.md`](../../archive/README.md).
This file records the findings needed for the unified research story.

## HIGGS-v1 — from the deleted branch `experiment/capacity-controlled-higgs-v1` (tip `932610b`)

Artifacts: [`archive/pre-consolidation/outputs/capacity_controlled_higgs_v1/`](../../archive/pre-consolidation/outputs/capacity_controlled_higgs_v1)

On the official HIGGS task with 500 training examples per block and PCA(8):
- search-arm differences were small and the strict equivalence test was underpowered,
- frozen quantum parameters and product circuits slightly outperformed trainable/entangled counterparts,
- the VQC stayed near or below trivial/classical baselines.

The study explicitly concluded that it was not yet a good setting for an LLM search comparison.

## Data-scale follow-up — from the deleted branch `experiment/higgs-data-scale-qualification-v1` (tip `24d6f1e`)

Artifacts: [`archive/pre-consolidation/outputs/higgs_data_scale_qualification_v1/`](../../archive/pre-consolidation/outputs/higgs_data_scale_qualification_v1)

The follow-up isolated the causes:
- **data scarcity** at n=500,
- **mis-tuned inherited optimizer settings** (learning rate 0.05),
- **PCA(8) information loss**.

With raw 21 features, n=10,000, and an audited protocol (lr 0.005, 50 epochs, weight decay 1e-3, early stopping), the entangled VQC reached approximately AUROC 0.635 and log-loss 0.664 in that study. The signs of the earlier frozen-vs-trainable and product-vs-entangled comparisons flipped at larger data scale.

## What this contributes to the QAE transition

The HIGGS result is not used as evidence that an LLM is good or bad. It contributes a design rule:

> Do not evaluate a search strategy until the underlying learning task is demonstrably non-degenerate.

The QAE benchmark therefore starts from a task with a direct quantum objective (state compression fidelity) and an exactly controlled circuit budget.
