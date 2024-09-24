# Quantum Contrastive Learning<br>for High-Energy Physics Analysis at the LHC | GSoC 2024
## Learning quantum representations of classical high energy physics data with contrastive learning


![ML4Sci@GSoC2024](https://miro.medium.com/v2/resize:fit:1100/format:webp/0*8KAp7eW2atsaRwdS.jpeg)

</hr>

### What was Accomplished?
As part of GSoC 2024, I contributed in Machine Learning for Science (ML4Sci). It is an open-source organization that brings together modern machine learning techniques and applies them to cutting edge problems in Science, Technology, Engineering, and Math (STEM). Over the summer, I worked on Quantum Machine Learning applied on High Energy Physics data (QMLHEP) to contrastively train models to output embeddings that can be used for other downstream tasks like classification.

- Code on QMLHEP GitHub Repository: [ML4SCI/QMLHEP/tree/main/Quantum_SSL_for_HEP_Sanya_Nanda](https://github.com/ML4SCI/QMLHEP/tree/main/Quantum_SSL_for_HEP_Sanya_Nanda)
- Code on my GitHub Repository: [SanyaNanda/ML4Sci_QuantumContrastiveLearning](https://github.com/SanyaNanda/ML4Sci_QuantumContrastiveLearning) (For latest updates post GSoC)
- Project Documentation: [Technical Documentation/Final blog](https://sanyananda.github.io/ML4Sci_QuantumContrastiveLearning/)


<b>[Project description: Learning quantum representations of classical high energy physics data with contrastive learning](https://ml4sci.org/gsoc/2024/proposal_QMLHEP3.html)</b>
- Implemented multiple trainable embedding functions to encode HEP data onto contrastive learning models.
- Developed numerous computer vision, graph-based and quantum hybrid models for contrastive learning framework.
- Experimented with different approaches for embedding functions and contrastive losses for training.
- Demonstrated an effort to prove quantum advantage using Quantum ML-based hybrid model.
- Benchmarked the trained embeddings of classical and quantum models.

Following are important documents pertaining to the project:
- Proposal submitted for GSoC 2024: [Proposal](slides/sanya-ml4sci-proposal.pdf)
- Iest Tasks solved for GSoC 2024: [Test Tasks](https://github.com/SanyaNanda/ML4Sci-QMLHEP-2024)
- GSoC Abstract: [GSoC Abstract](https://summerofcode.withgoogle.com/programs/2024/projects/IDScJm9Z)
- Mid Term Blog elucidating the work accomplished by the mid-term evaluation: [Blog](https://medium.com/@sanya.nanda/quantum-contrastive-learning-on-lhc-hep-dataset-1b3084a0b141)
- Mid-term lightning talk given to an audience of all the contributors and mentors of ML4Sci: [Mid-term Lightning Talk Presentation](slides/ML4Sci-MidTerm.pdf)
- Complete technical documentation or final blog, describing the whole project in depth, submitted for final evaluation: [Blog](https://sanyananda.github.io/ML4Sci_QuantumContrastiveLearning/)
- Final-term lightning talk given to an audience of all the contributors and mentors of ML4Sci as part of final evaluation: [Final Lightning Talk Presentation](slides/ML4Sci-FinalEvaluation.pdf)
- GSoC Lightning Talk, presented to all the contributors of GSoC 2024 on 25th September: [GSoC Presentation](slides/GSoC2024-LightningTalk-Contributor-SanyaNanda.pdf)
- Weights and Biases Reports on the experiments conducted: [Benchmarking](https://sanyananda.github.io/ML4Sci_QuantumContrastiveLearning/#benchmarking) section of the project documentation


<hr>

### My Contribution:
To begin with, the proposal I submitted ML4Sci GSoC Project can be found [here](slides/sanya-ml4sci-proposal.pdf).
Following is the code designed and developed during the course of the project:

- Code Repository: https://github.com/ML4SCI/QMLHEP/tree/main/Quantum_SSL_for_HEP_Sanya_Nanda
- Project Documentation: https://sanyananda.github.io/ML4Sci_QuantumContrastiveLearning/

During the program, I worked on couple of blogs to document and communicate the project in a clear, concise and compact form, following are some relevant work products:
- Midterm Blog: https://medium.com/@sanya.nanda/quantum-contrastive-learning-on-lhc-hep-dataset-1b3084a0b141
- Final Evaluation Blog (Technical Documentation): https://sanyananda.github.io/ML4Sci_QuantumContrastiveLearning/
- A blog on the whole program from the selection process to the very end (coming soon)

At ML4Sci, presenting our work and ideas is just as important as working towards developing novel solutions, if not more. All the contributors and mentors of the organization met in a weekly call to present our findings and discuss our work. These calls were full of great learnings and amazing experiences. During the mid-term evaluation, we internally presented a 3 min lightning talk on our milestones in the first half of the program. Similarly, we presented at the end of the program as well to capture the essence of the whole project and our accomplishments. I am also selected to give a lightning talk at GSoC level to share my experience. Following are the links for the same:
- Midterm Lightning Talk: slides/ML4Sci-MidTerm.pdf
- Final Evaluation Lightning Talk: slides/ML4Sci-FinalEvaluation.pdf
- GSoC Lightning Talk: slides/GSoC2024-LightningTalk-Contributor-SanyaNanda.pdf

<hr>

### Project Structure:


#### Jupyter Notebooks
A comprehensive order to go through the jupyter notebooks:

It is advisable to walthrough these notebooks along with the project documentation.
- [Classical GNN Model on Quark-Gluon](notebooks/gnn_cl_lct.ipynb)
- [Quantum GNN vs Classical on Quark-Gluon](notebooks/qgnn_cl_lct.ipynb)
- Different QGNNs on Quark-Gluon: [QC1](notebooks/qgnn_cl_lct.ipynb), [QC2](notebooks/qgnns1.ipynb), [QC3](notebooks/qgnns2.ipynb)
  
- [Classical CNN Base Model on MNIST](notebooks/Experiment_MNIST_0_1/2_classical_base_model_mnist-wandb.ipynb)
- [Quantum Hybrid CNN on MNIST](notebooks/Experiment_MNIST_0_1/3_hybrid_base_model_mnist.ipynb)
- CNN on quark-gluon: [Experiments on quark-gluon](notebooks/Experiment_quark_gluon)
  
- [Quark-Gluon data visualisation](notebooks/Experiment_quark_gluon/1_1_data_visualisation_preprocessing_qg.ipynb)
- [Quark-Gluon preprocessing and augmentation](notebooks/Experiment_quark_gluon/2_data_preprocessing_augmentation.ipynb)
- [QCNN on quark-gluon](notebooks/Experiment_quark_gluon/3_exp3_base_hybrid.ipynb)
- [Resnet on quark-gluon 1](notebooks/qg_resnet.ipynb), [Resnet on quark-gluon 2](notebooks/Experiment_quark_gluon/3_exp2_resnet18.ipynb)


#### Description of Directories and files

- `qssl/`: Main package directory.
  - `__init__.py`: Makes `qssl` a Python package.
  - `data/`: Directory for data-related scripts consisting of dataloaders, pair-creation, preprocessing, augmentation and visualization.
  - `loss/`: Directory for loss function scripts consisting of contrastive pair loss, infoNCE, NT-Xent and fidelity loss functions.
  - `models/`: Directory for model definitions consisting of cnn, resnet18 and gnn encoders along with quantum circuits and hybrids.
  - `training/`: Directory for training scripts consisting of trainers for the models
  - `evaluation/`: Directory for evaluation scripts consisting of helper functions like confusion matrix, auc, lct etc
  - `utils/`: Directory for utility scripts consisiting of functions for extracting and plotting embeddings
  - `config.py`: Configuration file for the project.
- `scripts/`: Directory for scripts to run training and evaluation.
- `notebooks/`: Directory for Jupyter notebooks. [Comprehensive Guide to navigate notebooks](#Jupyter-Notebooks)
- `slides/`: Directory for presentation slides, lightning talks, proposal etc
- `docs/`: Directory for html and assets for the project documentation github.io page
- `requirements.txt`: File listing the dependencies for the project.
- `setup.py`: Setup script for the package.
- `README.md`: This file.
- `LICENSE`: License for the project.
- `.gitignore`: Git ignore file.

<hr>

### Experimentation Outcomes:

Experiments in a tabular format can be found in the [Benchmarking](https://sanyananda.github.io/ML4Sci_QuantumContrastiveLearning/#benchmarking) section of the project documentation. Wandb reports can be found in the same section.

<hr>

### How to Run
- Create a virtualenv
- Pip install the libraries mentioned in requirements.txt
- Try out the jupyter notebooks for better understanding
- To train the a model for example qcnn, use this command "python scripts/run_training.py"