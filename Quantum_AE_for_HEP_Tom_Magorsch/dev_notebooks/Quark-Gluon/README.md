# Quantum & Classical Autoencoders for Quark-Gluon Jet Classification

This folder contains a collection of notebooks exploring **autoencoder architectures** (both classical and quantum-enhanced) for distinguishing quark-initiated jets from gluon-initiated jets in high-energy physics.

## Physics Objective

Quark-Gluon jet discrimination is a fundamental classification task in particle physics. Distinguishing quarks from gluons helps identify the underlying hard process in collider experiments. This folder explores how classical autoencoders (CAE, VAE, DAE) and quantum-enhanced autoencoders (QAE, SQAE) can learn compressed representations of jet structures for improved classification.

## Dataset

- **LHC Quark-Gluon Jet Dataset**: Multi-dimensional jet feature data from simulated particle collisions.
- **File Format**: HDF5 (.hdf5) containing jet features (X) and binary labels (y: 0=gluon, 1=quark).
- **Typical Shapes**: Training sets of ~793k samples, test sets of 10k–139k samples with 40×40 spatial jet representations.

## Notebooks Overview

### Classical Baselines
- **Quark-Gluon-CAE.ipynb**: Convolutional AutoEncoder baseline for unsupervised feature learning.
- **Quark-Gluon-VAE.ipynb**: Variational AutoEncoder for probabilistic jet representation.
- **Quark-Gluon-DAE.ipynb**: Denoising AutoEncoder for robust feature extraction.

### Quantum-Enhanced Models
- **Quark-Gluon-QAE.ipynb**: Quantum AutoEncoder combining classical encoders with quantum latent layers.
- **Quark-Gluon-SQAE.ipynb**: Single-qubit Quantum AutoEncoder variant.
- **ViT-QSAL.ipynb**: Vision Transformer with Quantum Self-Attention Layers for attention-based jet classification.

### Data Utilities
- **Explore_Data.ipynb**: EDA of jet structure, feature distributions, and class balance.
- **Quark-Gluon-data-scaling.ipynb**: Normalization and preprocessing pipelines.
- **Compress_Data.ipynb**: Methods for reducing data size for efficient training.

### Advanced Models
- **Coatnet.ipynb**: Hybrid convolutional-attention architecture for jet classification.
- **swae.ipynb**: Sliced Wasserstein AutoEncoder variant.

## Key Results

- Classical autoencoders (CAE/VAE) achieve ~0.75–0.80 AUC on test sets.
- Quantum-enhanced variants show competitive or improved performance with reduced latent dimensionality.
- Vision Transformer + Quantum Attention demonstrates the feasibility of attention-based quantum circuits for jet data.

## Dependencies

Install via the parent directory's `requirements.txt`:

```bash
pip install -r ../requirements.txt
```

Key packages:
- `tensorflow` / `tensorflow-quantum`
- `pennylane` for quantum circuits
- `h5py` for data loading
- `scikit-learn` for preprocessing

## Usage

1. **Download data**: Place HDF5 files in `../../data/quark-gluon/` (adjust paths as needed).
2. **Run a notebook**: Open any `.ipynb` file in Jupyter and execute cells from top to bottom.
3. **Compare models**: Each notebook can be run independently; visualizations compare reconstruction quality and classification metrics.

## Future Directions

- Extend to full JAX/Flax ecosystem for improved scalability.
- Implement distributed training for larger datasets.
- Explore error mitigation techniques for NISQ hardware simulation.

---

**Contributing**: This folder is part of the broader ML4SCI/QMLHEP effort to advance quantum machine learning in high-energy physics research.
