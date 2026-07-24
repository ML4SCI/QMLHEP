# Quantum Transformers Alessandro Tesi
*Note: Some circuits printed with SVG may not display correctly on GitHub. Please try viewing them with VS Code or Jupyter Notebook.*

This project aims to implement the  Orthogonal patch wise neural network outlined in the paper [Quantum Vision Transformers](https://arxiv.org/abs/2209.08167) by El Amine Cherrat, Iordanis Kerenidis, Natansh Mathur, Jonas Landman, Martin Strahm, Yun Yvonna Li, harnessing the power of quantum computing to enhance the capabilities of vision transformers. 

Furthermore, I draw inspiration and insights from last year's Google Summer of Code (GSoC) projects focused on Quantum Vision Transformers. These projects, documented in detail on [Sal's GSoC blog](https://salcc.github.io/blog/gsoc23/) and [Eyüp B. Ünlü's Medium post](https://medium.com/@eyupb.unlu/gsoc-2023-with-ml4sci-quantum-transformer-for-high-energy-physics-analysis-final-report-cd9ed594e4a2).

## Datasets

The architectures have been evaluated on the following datasets:

- [MNIST Digits](http://yann.lecun.com/exdb/mnist/), as a toy dataset for rapid prototyping
- [Quark-Gluon](https://arxiv.org/abs/1902.08276), one of the main high-energy physics datasets used in the project, which contains images of the recordings of the CMS detector of quark and gluon jets.
- [Electron-Photon](https://arxiv.org/abs/1807.11916), the other main high-energy physics dataset used in the project, which contains images of the recordings of the CMS detector of electron and photon showers.

## Structure

The project is organized as follows:

### Main Directory:
- **LICENSE**: Contains the licensing information for the project.
- **README**: Provides an overview of the project, its objectives, and structure.
- **Pennylane/**: A folder dedicated to implementations using Pennylane.
  - **MNIST.ipynb**: Contains a classical transformer, a quantum neural network, a quantum self-attention transformer, and a full quantum transformer architecture tested on the MNIST dataset.
  - **Photon Electron.ipynb**: Implements the same models as above but evaluated on the Electron-Photon dataset.
  - **Quark Gluon.ipynb**: Implements the same models as above but evaluated on the Quark-Gluon dataset.
  - **QViT Circuits Pennylane/**: Contains the implementation of circuits from the *Quantum Vision Transformers* paper using Pennylane.
  - **requirements.txt**: Lists the dependencies and libraries required for running the Pennylane notebooks.

- **Tensorflow Quantum/**: A folder focused on implementations using Tensorflow Quantum.
  - **Compound Transformer.ipynb**: Implements a compound transformer architecture, tested on the MNIST dataset (binary classification of digits 3 and 6).
  - **Quantum Orthogonal Neural Network.ipynb**: Implements a quantum orthogonal neural network, also tested on the MNIST dataset (binary classification of digits 3 and 6).
  - **Quantum Circuits.py**: Contains Python code for implementing the quantum circuits from the paper.
  - **QViT Paper Circuits.ipynb**: Contains the implementation of circuits from the *Quantum Vision Transformers* paper using Tensorflow Quantum (Cirq).
  - **requirements.txt**: Lists the dependencies and libraries required for running the Tensorflow Quantum notebooks. 

This structure allows for the exploration of quantum neural network architectures across different datasets and frameworks (Pennylane and Tensorflow Quantum). Each folder contains both classical and quantum models, as well as circuit implementations from the referenced papers.

### Additional Resources

- **`README.md`**: This file provides an overview of the project, its objectives, and its structure.
- **`requirements.txt`**: Lists the dependencies and libraries required to run the notebooks and scripts in this project.
- **`LICENSE`**: Contains the licensing information for the project.

## Contact

If you have any questions, feel free to email me at tesi.alessandro88 in gmail
