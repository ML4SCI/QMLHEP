"""Controlled robustness study of the previous QAE architecture-search result.

One factor at a time: candidate-evaluation budget B, qubit count n,
Hamiltonian family, and the underlying LLM model. Every non-target
setting is held at the reference condition (n=4, TFIM, B=8, reference
model), and `manifest.py` proves it.
"""
