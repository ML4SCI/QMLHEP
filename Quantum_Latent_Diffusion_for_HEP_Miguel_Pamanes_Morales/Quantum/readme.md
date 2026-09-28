# Quantum Experiments

This folder contains the quantum and hybrid experiments developed for the **Google Summer of Code 2026** project *Hybrid/Quantum Latent Diffusion Models for High-Resolution Simulation*.

As in the previous implementations, all experiments reuse the classical baseline pipeline, including the VAE, latent preprocessing, diffusion scheduler, training objective, and evaluation metrics.

## Experiments

* **[Simple Quantum Bottleneck](LDM_quantum_simple_bottleneck.ipynb):** replaces only the U-Net bottleneck with a single eight-qubit quantum module.

* **[Complex Quantum Bottleneck](LDM_quantum_complex_bottleneck.ipynb):** uses three VQCs for latent processing, conditioning, and feature fusion.

* **[Pure Patchwise Quantum Denoiser](LDM_pure_patchwise_quantum_denoiser.ipynb):** divides the latent representation into patches processed by a shared PQC without a U-Net or classical denoising bypass.

* **[Patchwise Quantum Denoiser with Spatial Mixer](LDM_patchwise_quantum_spatial_mixer.ipynb):** adds restricted classical spatial communication between neighboring quantum-processed patches.

* **[Hybrid Multiscale Quantum U-Net](LDM_hybrid_quantum_unet.ipynb):** uses quantum blocks at the `16×16` and `8×8` resolutions while keeping the `4×4` level and bottleneck classical.

* **[Final Quantum Experiment](LDM_final_quantum_experiment.ipynb):** studies a three-core quantum denoiser through Q10, Q11, and Q12, progressing from reduced-resolution processing to full-resolution processing and entanglement controls.

## Current Work

The **Final Quantum Experiment** is the closing experiment of the GSoC project and the one currently under development. We will continue refining its methodology and evaluation, so this notebook and its results will be updated.
