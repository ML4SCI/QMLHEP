# Quantum Contrastive Representation Learning for HEP Analysis

Welcome to the official repository for the QMLHEP project, conducted during ML4Sci's GSOC24. 

## Contents

- [Contents](#contents)
- [Setup](#setup)
- [Project Structure](#project-structure)
- [Datasets](#datasets)
  - [MNIST](#mnist)
  - [Electron Photon](#electron-photon)
  - [Quark Gluon](#quark-gluon)
- [Models](#models)
  - [Classical](#classical)
  - [Hybrid Quantum](#hybrid-quantum)
  - [Fully Quantum](#fully-quantum)
- [Results](#results)
- [References](#references)

## Setup

To begin, create a new conda environment and install the necessary dependencies:

```bash
conda create --name qml
conda activate qml
pip install -r requirements.txt
python -m pip install -e .
```

## Project Structure

```
.
├── .gitignore
├── README.md
├── demos
│   ├── demo_classical.ipynb
│   ├── demo_hybrid.ipynb
│   └── demo_qcnn_cons_MNIST.ipynb
├── scripts
│   ├── runner_classical.py
│   ├── runner_hybrid.py
│   └── runner_quantum.py
├── setup.py
├── src
│   └── qml_contrastive
│       ├── __init__.py
│       ├── data_mnist.py
│       ├── data_pe.py
│       ├── models
│       │   ├── __init__.py
│       │   ├── classical.py
│       │   ├── hybrid
│       │   │   ├── __init__.py
│       │   │   └── contrastive.py
│       │   └── quantum
│       │       ├── __init__.py
│       │       ├── qcnn.py
│       │       └── qcnn_reupload_contrastive.py
│       └── utils.py
└── tests
```

## Datasets

Several datasets are used to validate and debug the models:

### MNIST

Rescaled 12x12 classic MNIST dataset, for initial validations and debugging.

### Electron Photon 

Rescaled 12x12 ECAL images of electrons and photons.

### Quark Gluon

Rescaled 12x12 ECAL images of Quark and Gluon-initiated jets. 

## Models

### Classical

### Hybrid Quantum

### Fully Quantum

## Results


## References

