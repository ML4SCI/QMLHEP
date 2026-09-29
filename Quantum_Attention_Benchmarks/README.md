# Quantum Attention Benchmarks

This folder contains initial experiments for the **Quantum Particle Transformer**
(QMLHEP7) project. We focus on integrating quantum layers into attention-based
architectures using the JAX/Flax ecosystem.

## Contents

- `JAX_Quantum_Attention_Baseline.ipynb`: a minimal notebook demonstrating how to
  replace classical dense layers in a simple attention block with a PennyLane
  `FlaxLayer` quantum circuit.

## Requirements

- Python packages: `jax`, `jaxlib`, `flax`, `pennylane`, `optax`.
- A quantum device backend supported by PennyLane (e.g., `default.qubit`).

This documentation improves visibility for researchers exploring Transformer-style
benchmarks within the QMLHEP repository.