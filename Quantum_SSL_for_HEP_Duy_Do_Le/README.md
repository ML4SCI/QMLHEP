# Quantum Contrastive Representation Learning for HEP Analysis

Welcome to my repository for the ML4Sci project, conducted during GSOC24.

## Contents

- [Quantum Contrastive Representation Learning for HEP Analysis](#quantum-contrastive-representation-learning-for-hep-analysis)
  - [Contents](#contents)
  - [Setup](#setup)
  - [Project Structure](#project-structure)
  - [Datasets](#datasets)
    - [MNIST](#mnist)
    - [Electron Photon](#electron-photon)
    - [Quark Gluon Image](#quark-gluon-image)
    - [Quark Gluon Jet](#quark-gluon-jet)
    - [Quantum Machine 7](#quantum-machine-7)
  - [References](#references)

## Setup

To begin, create a new conda environment and install the necessary dependencies:

```bash
conda create --name py310 python==3.10
conda activate py310 
pip install -e .
```

Some dependencies of `torch-geometric` have to be installed separately:

```bash
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.0+12.1.html
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.4.0+12.1.html
```

## Project Structure

The project is structured as follows:

```
.
├── poetry.lock
├── pyproject.toml
├── README.md
├── setup.py
├── data
├── demos
│   ├── eda
│   │   ├── image_datasets.ipynb
│   │   ├── jets_graph.ipynb
│   │   └── jets_qg.ipynb
│   ├── losses.ipynb
│   ├── self-supervised
│   │   ├── images_quantum_qcnn_cons_MNIST.ipynb
│   │   ├── img_mnist.ipynb
│   │   └── logs
│   └── supervised
│       ├── images_classical.ipynb
│       ├── images_hybrid.ipynb
│       ├── img_mnist.ipynb
│       ├── img_pe.ipynb
│       ├── img_qg.ipynb
│       ├── jets_img_graph.ipynb
│       ├── jets_img_graph_.ipynb
│       ├── jets_qg_graph.ipynb
│       ├── logs
│       ├── models.py
│       └── utils.py
├── scripts
│   ├── runner_classical.py
│   ├── runner_hybrid.py
│   └── runner_quantum.py
└── src
    └── qml_ssl
        ├── data
        │   ├── graph_syn_qg.py
        │   ├── graph_transform.py
        │   ├── img_mnist.py
        │   ├── img_pe.py
        │   ├── img_qg.py
        │   └── __init__.py
        ├── __init__.py
        ├── losses
        │   └── __init__.py
        ├── models
        │   ├── img_classical.py
        │   ├── img_hybrid.py
        │   ├── img_qae.py
        │   ├── img_qcnn.py
        │   ├── img_qcnn_siamese.py
        │   ├── __init__.py
        │   └── mods.py
        ├── rgcl
        │   ├── aug.py
        │   ├── evaluate_embedding.py
        │   ├── gan_losses.py
        │   ├── gin.py
        │   ├── __init__.py
        │   ├── losses.py
        │   ├── mi_networks.py
        │   ├── model.py
        │   └── rgcl.py
        └── utils
            ├── __init__.py
            ├── plotting.py
            └── training.py
```

The key components include:
- **demos/**: Contains notebooks for both supervised and self-supervised learning pipelines. Experiments are logged locally as well as to [Comet](https://www.comet.com/duydl/quantum-contrastive-representation-learning/view/new/panels).
- **src/qml_ssl/**: Source code for models, data loading, and utilities.

## Datasets

Several datasets are used to validate and debug the models:

### MNIST

Classic **MNIST** dataset used for initial validations and debugging.

### Electron Photon 

**ECAL** images of electrons and photons from the CMS experiment.

- Dataset Link: [Photon HDF5](https://cernbox.cern.ch/remote.php/dav/public-files/AtBT8y4MiQYFcgc/SinglePhotonPt50_IMGCROPS_n249k_RHv1.hdf5), [Electron HDF5](https://cernbox.cern.ch/remote.php/dav/public-files/FbXw3V4XNyYB3oA/SingleElectronPt50_IMGCROPS_n249k_RHv1.hdf5)

### Quark Gluon Image

Images of Quark and Gluon-initiated jets for classification tasks.

- Dataset link: [Quark-Gluon Dataset (Kaggle)](https://www.kaggle.com/datasets/prabhashkumarjha/quark-gluon-data-set-n139306-hdf5)

### Quark Gluon Jet

Synthetic **Quark-Gluon Particle Jets** generated using the **PYTHIA** event generator.

- Dataset link: [Quark-Gluon Particle Jets](https://zenodo.org/records/3164691)

### Quantum Machine 7

The **Quantum Machine 7 (QM7)** dataset consists of molecules with associated formation energies, used for regression tasks.

- Dataset link: [Quantum Machine 7](http://quantum-machine.org/datasets/)


## References


<span id="ref1">[1]</span> P. H. Le-Khac, G. Healy, and A. F. Smeaton, “Contrastive Representation Learning: A Framework and Review,” *IEEE Access*, vol. 8, pp. 193907–193934, 2020, doi: [10.1109/ACCESS.2020.3031549](https://doi.org/10.1109/ACCESS.2020.3031549).

<span id="ref2">[2]</span> P. Khosla *et al.*, “Supervised Contrastive Learning,” *arXiv*, Mar. 10, 2021, doi: [10.48550/arXiv.2004.11362](https://doi.org/10.48550/arXiv.2004.11362).

<span id="ref3">[3]</span> B. Jaderberg, L. W. Anderson, W. Xie, S. Albanie, M. Kiffner, and D. Jaksch, “Quantum Self-Supervised Learning,” *arXiv*, Apr. 04, 2022, doi: [10.48550/arXiv.2103.14653](https://doi.org/10.48550/arXiv.2103.14653).

<span id="ref4">[4]</span> F. Wang and H. Liu, “Understanding the Behaviour of Contrastive Loss,” *arXiv*, Mar. 20, 2021, doi: [10.48550/arXiv.2012.09740](https://doi.org/10.48550/arXiv.2012.09740).

<span id="ref5">[5]</span> T. Wang and P. Isola, “Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere,” *arXiv*, Aug. 15, 2022, doi: [10.48550/arXiv.2005.10242](https://doi.org/10.48550/arXiv.2005.10242).

<span id="ref6">[6]</span> R. Dangovski *et al.*, “Equivariant Contrastive Learning,” *arXiv*, Mar. 14, 2022, doi: [10.48550/arXiv.2111.00899](https://doi.org/10.48550/arXiv.2111.00899).

<span id="ref7">[7]</span> W. Ju *et al.*, “Towards Graph Contrastive Learning: A Survey and Beyond,” *arXiv*, May 20, 2024, doi: [10.48550/arXiv.2405.11868](https://doi.org/10.48550/arXiv.2405.11868).

<span id="ref8">[8]</span> A. Pérez-Salinas, A. Cervera-Lierta, E. Gil-Fuster, and J. I. Latorre, “Data re-uploading for a universal quantum classifier,” *Quantum*, vol. 4, p. 226, Feb. 2020, doi: [10.22331/q-2020-02-06-226](https://doi.org/10.22331/q-2020-02-06-226).

<span id="ref9">[9]</span> “[2112.05261] Equivariant Quantum Graph Circuits.” Accessed: Sep. 06, 2024. Available: [https://arxiv.org/abs/2112.05261](https://arxiv.org/abs/2112.05261)

<span id="ref10">[10]</span> S. Li, X. Wang, A. Zhang, Y. Wu, X. He, and T.-S. Chua, “Let Invariant Rationale Discovery Inspire Graph Contrastive Learning,” *arXiv*, Jun. 15, 2022, doi: [10.48550/arXiv.2206.07869](https://doi.org/10.48550/arXiv.2206.07869).

<span id="ref11">[11]</span> B. M. Dillon, G. Kasieczka, H. Olischlager, T. Plehn, P. Sorrenson, and L. Vogel, “Symmetries, Safety, and Self-Supervision,” *SciPost Phys.*, vol. 12, no. 6, p. 188, Jun. 2022, doi: [10.21468/SciPostPhys.12.6.188](https://doi.org/10.21468/SciPostPhys.12.6.188).

