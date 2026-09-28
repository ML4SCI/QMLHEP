# Classical Models

This folder contains the classical models developed during the project. These implementations establish the data pipeline, architectures, and reference results used as baselines for the quantum experiments.

## Contents

* **[VAE](VAE/):** preliminary autoencoder and variational autoencoder implementations evaluated on MNIST, Cats vs. Dogs, and jet images.

* **[Diffusion Models](DIFFUSION/):** implementations of DDPM, DDIM, and Flow Matching tested on MNIST and jet datasets.

* **[Latent Diffusion Models](LDM/):** models combining the VAE and diffusion components, including the initial LDM, the final classical version, and the simplified baseline used in the quantum experiments.

Together, these models provide the classical foundation for the hybrid and quantum latent-diffusion architectures developed later in the project.

