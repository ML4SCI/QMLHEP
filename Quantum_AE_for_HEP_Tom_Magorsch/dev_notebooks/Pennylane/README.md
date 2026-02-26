# PennyLane Quantum Autoencoders for HEP & ML

This folder contains a comprehensive exploration of **quantum autoencoder designs** implemented using **PennyLane**, including hybrid classical-quantum architectures and advanced optimization techniques.

## Objective

These notebooks showcase the evolution and optimization of quantum autoencoders (QAE, SQAE) for high-energy physics tasks and general machine learning benchmarks. A key focus is comparing different optimization strategies (gradient descent, QNG, parameter initialization) to improve convergence and reconstruction quality.

## Architecture Families

### Hybrid Autoencoders (HAE)
- **pennylane-HAE.ipynb**: Basic Hybrid AutoEncoder combining classical CNN encoder with quantum bottleneck.
- **pennylane-HAE-QG.ipynb**: HAE tuned for Quark-Gluon jet data.
- **pennylane-HAE (2) (2).ipynb**: Variant with improved circuit depth control.

### Single-Qubit Quantum Autoencoders (SQAE)
- **pennylane-SQAE.ipynb**: Fundamental SQAE architecture.
- **pennylane-SQAE-keras.ipynb**: SQAE integrated with Keras/TensorFlow training.
- **pennylane-hybrid.ipynb**: Hybrid training combining classical and quantum layers.
- **pennylane-hybridSQAE.ipynb**: Full hybrid workflow.

### Advanced Optimization Variants
- **pennylane-SQAE-QNG.ipynb**: Quantum Natural Gradient (QNG) optimizer for improved convergence.
- **pennylane-SQAE-ampenc.ipynb**: Amplitude encoding for efficient state preparation.
- **pennylane-SQAE-DRC.ipynb**: Dynamical Reconfigurable Circuits.
- **QNG_comparasion.ipynb**: Benchmarks QNG vs. standard gradient descent.

### Physics-Specific Models
- **pennylane-SQAE-convDRC-QG.ipynb**: Convolutional DRC for Quark-Gluon classification.
- **pennylane-SQAE-convDRC-QG-jax.ipynb**: Same architecture in JAX/Flax backend.
- **pennylane-SQAE-convDRC-QG-model.ipynb**: Production-ready configuration.
- **pennylane-SQAE-convDRC-EP.ipynb**: Electron-Photon variant.
- **pennylane-SQAE-convDRC-Qconv.ipynb**: Quantum convolutional layers.

### Supervised Learning with Quantum Circuits
- **pennylane-supervised-qg.ipynb**: Quantum circuit for supervised Quark-Gluon classification.
- **pennylane-supervised-qg-Copy1.ipynb**: Variant of above.

### Debugging & Development
- **Batched_hybrid_debug.ipynb**: Troubleshooting batched quantum-classical hybrid training.
- **animation.gif, animation2.gif, animation3.gif**: Training visualizations (Bloch sphere rotations, loss curves).
- **bloch.gif, bloch2.gif, bloch3.gif**: Animated Bloch sphere representations of learned quantum states.

## Key Innovations

- **Quantum Natural Gradient (QNG)**: Faster convergence than vanilla gradient descent (see `QNG_comparasion.ipynb`).
- **Dynamical Reconfigurable Circuits**: Adapting quantum circuit structure during training.
- **JAX Backend**: Several notebooks use JAX for automatic differentiation and GPU acceleration.
- **Batch Training**: Hybrid architectures supporting mini-batch updates (see `Batched_hybrid_debug.ipynb`).

## Typical Performance

- Hybrid Autoencoders: ~0.75–0.85 AUC on Quark-Gluon; ~50% latent dimension reduction.
- SQAE Models: Scalable to 200+ layers with careful initialization and learning rate tuning.
- QNG Optimization: 2–5× faster convergence vs. standard SGD on small circuits.

## Dependencies

Install via the parent directory's `requirements.txt`:

```bash
pip install -r ../requirements.txt
```

Essential packages:
- `pennylane` (quantum circuits & autoencoders)
- `pennylane-qiskit` or `pennylane-lightning` (simulator backends)
- `jax` / `jaxlib` (for JAX-based notebooks)
- `tensorflow` / `tensorflow-quantum` (for Keras integration)

## Usage

1. **Start with basics**: Run `pennylane-HAE.ipynb` or `pennylane-SQAE.ipynb` for fundamentals.
2. **Compare optimizers**: Use `QNG_comparasion.ipynb` to understand QNG benefits.
3. **Physics applications**: Jump to `pennylane-SQAE-convDRC-QG*.ipynb` for Quark-Gluon tasks.
4. **Visualize training**: Check `.gif` files for animated Bloch sphere and loss curves.

## Future Directions

- Port all models to full JAX/Flax ecosystem for seamless integration with QMLHEP7.
- Implement error mitigation techniques for hardware execution.
- Extend to multi-qubit entangling circuits for increased expressivity.
- Auto-tune hyperparameters using Bayesian optimization or RL.

---

**Contributing**: These notebooks form a foundation for quantum autoencoder research in ML4SCI/QMLHEP, enabling rapid prototyping and physics-aware optimization.
