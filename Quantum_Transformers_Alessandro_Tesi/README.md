# Quantum Transformers Alessandro Tesi

This project aims to implement the  Orthogonal patch wise neural network outlined in the paper [Quantum Vision Transformers](https://arxiv.org/abs/2209.08167) by El Amine Cherrat, Iordanis Kerenidis, Natansh Mathur, Jonas Landman, Martin Strahm, Yun Yvonna Li, harnessing the power of quantum computing to enhance the capabilities of vision transformers. 

My work is primarily based on TensorFlow Quantum, a robust software framework that facilitates the construction, simulation, and training of quantum machine learning models. [TensorFlow Quantum: A Software Framework for Quantum Machine Learning](https://arxiv.org/abs/2003.02989) by Michael Broughton, Guillaume Verdon, Trevor McCourt, Antonio J. Martinez, Jae Hyeon Yoo, Sergei V. Isakov, Philip Massey, Ramin Halavati, Murphy Yuezhen Niu, Alexander Zlokapa, Evan Peters, Owen Lockwood, Andrea Skolik, Sofiene Jerbi, Vedran Dunjko, Martin Leib, Michael Streif, David Von Dollen, Hongxiang Chen, Shuxiang Cao, Roeland Wiersema, Hsin-Yuan Huang, Jarrod R. McClean, Ryan Babbush, Sergio Boixo, Dave Bacon, Alan K. Ho, Hartmut Neven, Masoud Mohseni.

Furthermore, I draw inspiration and insights from last year's Google Summer of Code (GSoC) projects focused on Quantum Vision Transformers. These projects, documented in detail on [Sal's GSoC blog](https://salcc.github.io/blog/gsoc23/) and [Eyüp B. Ünlü's Medium post](https://medium.com/@eyupb.unlu/gsoc-2023-with-ml4sci-quantum-transformer-for-high-energy-physics-analysis-final-report-cd9ed594e4a2).

## Datasets

The architectures have been evaluated on the following datasets:

- [MNIST Digits](http://yann.lecun.com/exdb/mnist/), as a toy dataset for rapid prototyping
- [Quark-Gluon](https://arxiv.org/abs/1902.08276), one of the main high-energy physics datasets used in the project, which contains images of the recordings of the CMS detector of quark and gluon jets.
- [Electron-Photon](https://arxiv.org/abs/1807.11916), the other main high-energy physics dataset used in the project, which contains images of the recordings of the CMS detector of electron and photon showers.

## Structure

Below is an overview of the project's structure:

### `QViT_paper_circuits.ipynb`

This notebook contains examples and implementations of the quantum circuits proposed in the foundational papers relevant to this project.

- **Kerenidis, I., Landman, J., & Mathur, N. (2021).** *Classical and Quantum Algorithms for Orthogonal Neural Networks.*
- **Landman, J., Mathur, N., Li, Y. Y., Strahm, M., Kazdaghli, S., Prakash, A., & Kerenidis, I. (2022).** *Quantum Methods for Neural Networks and Application to Medical Image Classification.*
- **Cherrat, E. A., Kerenidis, I., Mathur, N., Landman, J., Strahm, M., & Li, Y. Y. (2022).** *Quantum Vision Transformers.*


### `classic/`

The `classic` directory contains notebooks and scripts where a traditional vision transformer model is trained and evaluated. These classic models serve as benchmarks for comparison against quantum counterparts. This directory includes:


### `quantum/`

The `quantum` directory is subdivided into folders representing different quantum neural network architectures:

- **`Orthogonal NN/`**: Contains notebooks and code for implementing orthogonal neural networks as described in the referenced papers.
  
- **`Transformer/`**: Focuses on quantum transformers, incorporating the architecture and training methodologies specific to quantum versions of vision transformers. 
  
- **`Compound Transformer/`**: This folder investigates a novel architecture introduced in the paper *Quantum Vision Transformers*, which combines elements of both Transformers and MLP mixers leveraging quantum NNs.

### Additional Resources

- **`README.md`**: This file provides an overview of the project, its objectives, and its structure.
- **`requirements.txt`**: Lists the dependencies and libraries required to run the notebooks and scripts in this project.
- **`LICENSE`**: Contains the licensing information for the project.

## Contact

If you have any questions, feel free to email me at tesi.alessandro88 in gmail