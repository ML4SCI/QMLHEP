# Quantum & Classical Autoencoders for Electron-Photon Classification

This folder contains a collection of notebooks exploring **autoencoder architectures** (both classical and quantum-enhanced) for distinguishing electrons from photons in high-energy physics calorimeter data.

## Physics Objective

Electron-Photon discrimination is a critical classification task in particle physics. Both particles leave similar electromagnetic (EM) signatures in calorimeters, making their separation challenging. This folder explores how classical autoencoders (CAE, VAE) and quantum-enhanced autoencoders (QAE, SQAE) can learn discriminative representations of calorimeter shower patterns for robust electron-photon tagging.

## Dataset

- **LHC Electron-Photon Dataset**: Calorimeter images from simulated electron and photon events.
- **File Format**: HDF5 (.hdf5) containing EM calorimeter shower images (X) and binary labels (y: 0=photon, 1=electron).
- **Typical Shapes**: Multi-thousand samples of 2D/3D calorimeter energy deposits.

## Notebooks Overview

### Classical Baselines
- **Electron-Photon_CAE_fulldata.ipynb**: Convolutional AutoEncoder for full dataset feature learning.
- **Electron-Photon_CAE_keras_tuner.ipynb**: Hyperparameter-tuned CAE using Keras Tuner.
- **Electron-Photon_VAE_fulldata.ipynb**: Variational AutoEncoder for probabilistic shower representation.

### Quantum-Enhanced Models
- **Electron-Photon_QAE.ipynb**: Quantum AutoEncoder with quantum latent space.
- **Electron-Photon_QAE_fulldata.ipynb**: Full-dataset variant of QAE.
- **Electron-Photon_SQAE.ipynb**: Single-qubit Quantum AutoEncoder.
- **Electron-Photon_SQAE_fulldata.ipynb**: Full-dataset training variant.

### Data Utilities
- **EMD_experiment.ipynb**: Earth Mover Distance (EMD) analysis of shower patternsfor model comparison.

### Advanced Models
- **gammaetune/**: Hyperparameter optimization workflows for gamma/electron discrimination.

## Key Results

- Classical autoencoders (CAE/VAE) achieve competitive classification metrics on EM calorimeter data.
- Quantum autoencoders show promise in learning compressed shower representations.
- Full-data training variants improve generalization compared to subset-trained models.

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
- `keras-tuner` for hyperparameter optimization

## Usage

1. **Download data**: Place HDF5 files (`electron.hdf5`, `photon.hdf5`) in `../../data/` (adjust paths as needed).
2. **Run a notebook**: Open any `.ipynb` file in Jupyter and execute cells from top to bottom.
3. **Compare models**: Each architecture can be independently evaluated; check training curves and ROC/AUC metrics.

## Future Directions

- Scale to full calorimeter geometries (e.g., 100×100 pixel grids).
- Implement quantum error mitigation for accurate circuit simulation.
- Explore attention-based variants (ViT-style) for multi-scale feature learning.
- Extend to JAX/Flax for improved computational efficiency.

---

**Contributing**: This folder is part of the ML4SCI/QMLHEP initiative to advance quantum machine learning in high-energy physics research.
