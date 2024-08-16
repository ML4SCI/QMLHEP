# GSOC Quantum Diffusion Model

This is a research project for ML4Sci with Google Summer of Code aimed at building a quantum denoising diffusion model.

## Description

The mid-term update blog for this project can be accessed here: [text](https://medium.com/@mashapotatoes/gsoc-2024-quantum-diffusion-models-for-high-energy-physics-892e59ddcd3e).

## Getting Started

### Dependencies and Installation

* ``` pip install requirements.txt ``` to download all dependencies in your own environment
* All code and model training can run locally on cropped data (16x16) and smaller sample sizes (eg. 1k instead of 100k samples)
* Quantum or hybrid models use Pennylane for simulations, which can be much slower than the classical variant, so experimenting with the device could be helpful depending on hardware availability

### Executing program

* Most of the code is in Jupyter notebooks with preloaded examples
* Run each notebook separately cell by cell or export to python scripts
