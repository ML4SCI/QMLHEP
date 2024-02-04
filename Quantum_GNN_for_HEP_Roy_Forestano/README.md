# Invariant and Equivariant Classical and Quantum Graph Neural Networks

![gsoc_ml4sci](https://github.com/royforestano/2023_gsoc_ml4sci_qmlhep_gnn/assets/96851867/3ed6ecda-bbe2-4e80-8e97-fa3e3b6647bf)

2023 Google Summer of Code (GSOC) Machine Learning for Science (ML4SCI) Quantum Machine Learning for High Energy Physics (QMLHEP).
This work is currently under review in the 2023 NeurIPS Proceedings.

Machine learning algorithms are heavily relied on to understand the vast amounts of data from high-energy particle collisions at the CERN Large Hadron Collider (LHC). The data from such collision events can naturally be represented with graph structures. Therefore, deep geometric methods, such as graph neural networks (GNNs), have been leveraged for various data analysis tasks in high-energy physics. One typical task is jet tagging, where jets are viewed as point clouds with distinct features and edge connections between their constituent particles. The increasing size and complexity of the LHC particle datasets, as well as the computational models used for their analysis, greatly motivate the development of alternative fast and efficient computational paradigms such as quantum computation. In addition, to enhance the validity and robustness of deep networks, one can leverage the fundamental symmetries present in the data through the use of invariant inputs and equivariant layers. In this paper, we perform a fair and comprehensive comparison between classical graph neural networks (GNNs) and equivariant graph neural networks (EGNNs) and their quantum counterparts: quantum graph neural networks (QGNNs) and equivariant quantum graph neural networks (EQGNN). The four architectures were benchmarked on a binary classification task to classify the parton-level particle initiating the jet. Based on their AUC scores, the quantum networks were shown to outperform the classical networks. However, seeing the computational advantage of the quantum networks in practice may have to wait for the further development of quantum technology and its associated APIs. 

Original Project Proposal: Invariant and Equivariant Quantum Graph Attention Transformers for HEP Analysis at the LHC

See public proposal here: [https://summerofcode.withgoogle.com/programs/2023/projects/1ERZ3hp2]

See final blog post here: [https://royforestano.github.io/blog/2023/2023-gsoc-ml4sci-qmlhep/]

There are five notebooks in this repository based on:

1. Loading and Sorting the Data  (notebook in models/auxiliary_notebooks/)
2. GNN and EGNN Models           (notebook in models/)
3. QGNN Model                    (notebook in models/)
4. EQGNN Model                   (notebook in models/)
5. ROC Curves for all models     (notebook in models/auxiliary_notebooks/)

There are also

6. Preprocessing (python file in utils/)
7. Saved sorted numpy arrays for testing (2.-4.)  (numpy arrays in data/).
8. Saved sorted numpy arrays for testing (5.)  (numpy arrays in roc_data/).

To run the first notebook (1.), you need to download the 20 Pythia 8 [https://zenodo.org/records/3164691] files and uncomment the first few lines reading in the 20 data files.

The model notebooks (2.-4.) can be run using (6.-7.).

The roc notebook (5.) can be run using (8.).
