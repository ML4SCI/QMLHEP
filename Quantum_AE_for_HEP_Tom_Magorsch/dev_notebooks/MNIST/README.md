# Quantum & Classical Autoencoders on MNIST

This folder contains a comprehensive benchmark suite of **autoencoder architectures** (both classical and quantum-enhanced) evaluated on the MNIST handwritten digit dataset.

## Objective

The MNIST dataset serves as a **controlled benchmark** for developing and validating autoencoder designs before applying them to high-energy physics data. This folder explores how quantum autoencoders compare to classical baselines in terms of compression efficiency, reconstruction fidelity, and trainability on a well-understood task.

## Dataset

- **MNIST**: 28×28 grayscale images of handwritten digits (0–9).
- **Source**: Built-in TensorFlow/Keras dataset.
- **Typical Shapes**: 60k training samples, 10k test samples, normalized to [0, 1].

## Notebooks Overview

### Classical Baselines
- **MNIST_CAE.ipynb**: Standard Convolutional AutoEncoder for image compression.
- **MNIST_convolutional_CAE.ipynb**: Advanced convolutional variant with detailed architecture.
- **MNIST_PCA_CAE.ipynb**: Hybrid approach combining PCA with convolutional layers.

### Quantum-Enhanced Models
- **MNIST_QAE.ipynb**: Quantum AutoEncoder with quantum latent space for digit representation.
- **MNIST_SQAE.ipynb**: Single-Qubit Quantum AutoEncoder variant for minimal quantum resource usage.

### Hyperparameter Optimization
- **MNIST_QAE_hyperparameters.ipynb**: Systematic tuning of quantum circuit depth, learning rates, and layer configurations.
- **MNIST_QAE_model.ipynb**: Best-performing QAE configuration discovered through tuning.
- **MNIST_SQAE_hyperparameters.ipynb**: Hyperparameter search for single-qubit variant.

## Key Metrics

- **Reconstruction MSE**: Measures how well the autoencoder reconstructs digits.
- **Latent Dimensionality**: Compression ratio achieved (original size vs. bottleneck).
- **Training Efficiency**: Wall-clock time and convergence speed for classical vs. quantum variants.

## Typical Results

- Classical CAE: MSE ~0.01–0.02 on test set; ~10× compression ratio.
- Quantum AE: Competitive reconstruction with fewer latent qubits; scalability limited by circuit depth.
- Hyperparameter tuning improves both classical and quantum variants by ~5–15%.

## Dependencies

Install via the parent directory's `requirements.txt`:

```bash
pip install -r ../requirements.txt
```

Key packages:
- `tensorflow` / `tensorflow-quantum`
- `pennylane` for quantum circuits
- `cirq` for quantum circuit visualization
- `scikit-learn` for PCA and preprocessing

## Usage

1. **Run a notebook**: Open any `.ipynb` file in Jupyter/Colab and execute cells from top to bottom.
2. **Compare architectures**: Each notebook produces training curves and reconstruction visualizations.
3. **Tune hyperparameters**: Start with `*_hyperparameters.ipynb` notebooks to explore the parameter space.
4. **Visualize results**: Check digit reconstruction plots and latent space embeddings.

## Future Directions

- Extend quantum circuits to deeper, more expressive topologies.
- Implement variational quantum autoencoders (VQAE) with Barren Plateau mitigation.
- Port to JAX/Flax for improved numerical stability and GPU acceleration.
- Apply learned representations as pretrained features for downstream HEP tasks (Quark-Gluon, Electron-Photon).

---

**Contributing**: This folder is part of the ML4SCI/QMLHEP initiative advancing quantum machine learning benchmarks and techniques.
