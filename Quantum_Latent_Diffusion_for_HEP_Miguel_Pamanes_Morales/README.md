# Hybrid/Quantum Latent Diffusion Models for High-Resolution Simulation

**Miguel Pámanes Morales · ML4SCI / QMLHEP · Google Summer of Code 2026**

[Project report](https://mpm-cvr.github.io/gsoc-2026-report/) · [Jet-image dataset](https://drive.google.com/file/d/1WO2K-SfU2dntGU4Bb3IYBp9Rh7rtTYEr/view?usp=sharing) · [Original development repository](https://github.com/ReyGuadarrama/Quantum-Latent-Diffusion-Models-for-High-Resolution-Simulation)

## Overview

This project investigates conditional generation of **quark and gluon jet images** with classical and quantum–classical latent diffusion models. A variational autoencoder (VAE) compresses the images; a denoiser then learns the reverse diffusion process in the smaller latent space. The research asks where parameterized quantum circuits can contribute to denoising and how they compare with carefully matched classical models.

The recorded comparisons do **not** establish a quantum advantage. The classical latent U-Net remains the strongest generative reference in the completed experiments; the quantum studies identify viable architectures, training behavior, and limitations for further work.

## Data and workflow

The jet images have **Tracker, ECAL, and HCAL** channels. The notebooks use `64 × 64 × 3` images cropped from the original `125 × 125` data. In the main latent-diffusion experiments, the VAE produces a standardized latent tensor of shape `4 × 16 × 16`.

`Jet image → VAE latent → conditional diffusion denoiser → VAE decoder → generated jet image`

The notebooks evaluate both denoising loss and image-space observables, including integrated intensity, active-pixel fraction, radial profiles, and Wasserstein-1 distances. The [full report](https://mpm-cvr.github.io/gsoc-2026-report/) explains the methods, comparisons, and results in detail.

## Repository guide

### Classical foundations

| Directory | Contents |
| --- | --- |
| [VAE](Classical/VAE/) | Autoencoder and VAE development on MNIST, Cats vs. Dogs, and jet images. |
| [Diffusion](Classical/DIFFUSION/) | Image-space DDPM, DDIM, and Flow Matching experiments. |
| [Latent diffusion](Classical/LDM/) | Initial and final classical LDMs, plus the [simplified baseline](Classical/LDM/LDM_simplfied.ipynb) used for quantum comparisons. |

### Quantum and hybrid experiments

Each experiment is documented in its own directory. Quantum notebooks include the classical VAE and diffusion components needed for their particular comparison.

| Experiment | Main idea |
| --- | --- |
| [1. Simple quantum bottleneck](Quantum/First-Quantum-Bottleneck/) | Insert an eight-qubit module into the U-Net bottleneck. |
| [2. Three-circuit bottleneck](Quantum/Second-Structured-Mandatory-Quantum-Bottleneck/) | Use separate circuits for latent features, conditioning, and fusion. |
| [3. Pure patchwise denoiser](Quantum/Third-Pure%20Patchwise%20Quantum%20Denoiser/) | Replace the U-Net denoiser with a shared circuit applied patch by patch. |
| [4. Patchwise denoiser with spatial mixer](Quantum/Fourth-Patchwise-PQC%2BSpatial-Mixer/) | Add local spatial communication between quantum-processed patches. |
| [5. Hybrid multiscale U-Net](Quantum/Fifth_Hybrid_Quantum_UNET/) | Apply quantum blocks at `16 × 16` and `8 × 8`; keep the `4 × 4` stages and central bottleneck classical. |
| [6. Final three-core denoiser](Quantum/Final_Quantum_Latent_Diffusion_Experiment/) | Study content, condition, and fusion circuits at reduced and full latent resolution, then compare entangled, non-entangled, frozen, and classical controls. |

The final experiment contains three stages: **Q10** (three-core denoising on an `8 × 8` grid), **Q11** (the same design on the full `16 × 16` grid), and **Q12** (extended training and entanglement controls).

## Running the notebooks

1. Download the [jet-image dataset](https://drive.google.com/file/d/1WO2K-SfU2dntGU4Bb3IYBp9Rh7rtTYEr/view?usp=sharing) and set the dataset path used by the selected notebook.
2. Install the packages imported by that notebook. The experiments use Jupyter, PyTorch, NumPy, pandas, SciPy, Matplotlib, h5py, and other notebook-specific packages; some circuit implementations also use PennyLane.
3. Run the notebook **from top to bottom**. Later quantum sections reuse the trained VAE, data loaders, scheduler, and other objects from preceding cells. In particular, Q11 and Q12 continue from objects created earlier in the final notebook.

The dataset and trained model checkpoints are not stored in this directory. Saved notebook outputs document the recorded runs; reproducing them requires the dataset and a fresh execution of the relevant cells.

## Project status

The **final three-core experiment** was the closing experiment of GSoC 2026 and remains the current research focus. Its methods and results may be updated as additional comparisons and repeat runs are completed.

##
🚧 Note: This repository is under active development and continuously updated.
