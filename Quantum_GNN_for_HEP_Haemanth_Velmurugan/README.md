# Quantum-enhanced Graph Neural Networks
### A Google Summer of Code (GSoC) 2024 project with the ML4SCI organization

![image](https://github.com/user-attachments/assets/a3f77bf5-bf09-4759-b770-ab8cf2208ea0)

Jet tagging is a particularly important task at CERN in order to identify useful signals from the billions of data generated every second. Graph Neural Networks are particularly suitable for the task due to the sparse and heterogenous nature of the data produced. This project aims to develop scalable quantum-enhanced graph neural networks and analyze their performance for the jet tagging task. 

For a detailed explanation of the proposed approach and the dataset, refer to the [blog](https://medium.com/@haemanth10/quantum-enhanced-graph-neural-networks-4c1270c2d094)

## Dataset
We are primarily working with the Pythia8 Quark and Gluon jets dataset [[1]](#1)

## Libraries Used
We use the [PyTorch geometric](https://pytorch-geometric.readthedocs.io/en/latest/) package to implement Graph Neural Networks and [Pennylane](https://docs.pennylane.ai/en/stable/) to implement the Quantum Circuits. 

## Structure

* `code` folder contains all the codes used in the project and is organized into subdirectories as follows:
    * `data` folder contains all the python files related to preparing the datasets used
        * The `config.json` file specifies the hyperparameters to be used while loading the dataset such as train-val-test ratio, batch size, etc.
        * The `load_jets.py` file loads the Quark and Gloun jets and pre-processes them, transforms the particle clouds into graphs and generated the PyTorch dataloaders
    * `models` folder contains all the GNN models implemented
        * `GCNConv_Layers` folder contains the custom implementations of the graph convolution operations built using PyTorch Geometric's Message Passing interface
            * `Custom_GCN_Conv.py` implements the classical graph convolution operation
            * `QGCN_Conv.py` implemented the quantum-enhanced graph convolution operation
        * `Quantum_classifiers` folder contains all the quantum classifier architectures we have used
            * `MPS.py` is the implementation of the Matrix Product State quantum classifier [[2]](#2)
            * `TTN.py` is the implementation of the Tensor Tree Network quantum classifier [[3]](#3)
        * `PyTorch_GCN.py` is the traditional implementation of GCN in PyTorch Geometric using the inbuilt graph convolution layers
        * `Custom_GCN.py` is a classical graph convolutional neural network built using the custom graph convolution layers we have implemented
        * `QNN_Node_Embedding.py` is a typical Quantum Neural Network circuit
        * `Quantum_GCN.py` is the quantum-enhanced graph convolutional network
    * `training` folder contains the codes required to train and evaluate the models
        * `config.json` contains the model-specific details like dimensions of the layers to instantiate the GNN models.
        * `train.py`contains the training, evaluation and testing functions to optimize the model parameters
        * `utils.py` contains helper functions to visualize the loss, accuracy, etc.
* `notebooks` folder contains the model training experiments and visualizations
    * `Dataset Experiments.ipynb` contains some experiments that have been tried out on the Quark and Gluon jets to better understand the dataset
    * `QGNN - Setup and Sample Experiments.ipynb` contains the training results, loss and accuracy curves, AUC plots for all the models we have implemented

## References
<a id="1">[1]</a>
Komiske, Patrick T., Eric M. Metodiev, and Jesse Thaler. “Energy flow networks: deep sets for particle jets.” Journal of High Energy Physics 2019, no. 1 (2019): 1–46.

<a id="2">[2]</a>
Bhatia, Amandeep Singh, Mandeep Kaur Saggi, Ajay Kumar, and Sushma Jain. “Matrix product state–based quantum classifier.” Neural computation 31, no. 7 (2019): 1499–151.

<a id="3">[3]</a>
Grant, Edward, Marcello Benedetti, Shuxiang Cao, Andrew Hallam, Joshua Lockhart, Vid Stojevic, Andrew G. Green, and Simone Severini. “Hierarchical quantum classifiers.” npj Quantum Information 4, no. 1 (2018): 65

<a id="4">[4]</a>
Tüysüz, Cenk, Carla Rieger, Kristiane Novotny, Bilge Demirköz, Daniel Dobos, Karolos Potamianos, Sofia Vallecorsa, Jean-Roch Vlimant, and Richard Forster. “Hybrid quantum classical graph neural networks for particle track reconstruction.” Quantum Machine Intelligence 3 (2021): 1–20
