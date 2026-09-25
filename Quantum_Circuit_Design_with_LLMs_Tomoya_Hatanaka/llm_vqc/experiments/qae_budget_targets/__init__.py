"""Minimum-budget-to-target study (protocol:
`docs/research/QAE_BUDGET_TARGET_PROTOCOL.md`).

Two things distinguish this package from `qae_robustness`, which it reuses
for every method, prompt, trainer and RNG stream:

1. **Arbitrary even budgets.** A condition here may sit at any even `B` on
   the frozen admissible grid, and it declares the *anchor* it is one
   factor away from, so a budget probe at XXZ can be verified without
   weakening the original one-factor rule.
2. **Validation only.** The runner never evaluates a candidate on the test
   set and never writes a test column.
"""
