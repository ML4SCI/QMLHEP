<div align="center">

  # Quantum Graph Neural Networks for High Energy Physics Analysis at the LHC

<img src="https://img.shields.io/badge/Google%20Summer%20of%20Code-2023-fbbc05?style=flat&logo=data%3Aimage%2Fpng%3Bbase64%2CiVBORw0KGgoAAAANSUhEUgAAAGQAAABkCAMAAABHPGVmAAAALVBMVEVHcEz7vQD7vQD8vQD7vQD8vQD7vQD8vQD8vQD7vQD7vQD8vQD7vQD7vQD7vQAgxtLpAAAADnRSTlMAZvVQ6QrVPhl6oSmHvzL6LQUAAASGSURBVHjatdnZdusgDAVQELMY%2Fv9zb2%2Bwc%2BIKDzQLvTXB3gYBFqmaDVeKU4sCBlFyy43WqLjlBpR1BpR1BpR1xjoFxmIFBpSVBpSVBpSVBpSVBpQ1xvdK1oPgblhfOWltjNaJq7ddYT2IfImYJqMDrENUChGDZn%2FWQ%2FMHxBcD4BMyBc5XCHkNQTq60vfIgXAx5xByju6T8V8itsT3%2FUPi6r39Ce8rp%2FCWYrHfIDXs95FZJs%2FvTob6Z4T2buQE4eikvHeG%2FoZY7TpRfDsNWzrjtP0L4s12NYhh%2BO1ZjJ9HfOjdYGo3QZx7YvwEAgOPdx3eQJlArMFA3wXSZ%2BwMQvplJGoPY6sqNU0gxcGYUVx5jtSIx3oS6HysTxEbMMDPAmkM9iFSXnPXt8nwuQ%2FYI8TH%2F425TQe7%2FnBPEH2bECI6T4t%2Bgvh4N1istR50FJdeIX1Ek%2FqJdGGQOWmAa4u7rn18vuuIzUq52gbxvpiSuzIau%2BuO9FUUfTvvCjcoQ4MMltRnEOqF0pdD%2FwiBZWxoqGCn8r2VGKIUCHOoTyHK2g7y1bsJRRqNe3%2FlXv5GbNhWEWXxbsf1UITRF4kYcM4KiI%2FbeFIevNNq7P2EIg0bVL%2BfqCcyYV2rbDdExWSPjUPPGBRh9JTowTscW0Dqf%2BwLXGmPthgKKMJo1f1OSQ29hf1Mbdlmg5NFV1H7KoICA3mruIQ4vl4TTFhvuAlxxrdb1J55KMJoBatEPCv6mr3sJzK%2F9RQKDAx49Ji5ctSLwsxAxgyuiduOAeVtIG14zppPKtAka9lcMZz71IHyNoAcCpvIx6UfxGLleCim3ggUpe0dQhe7I86mWvQERZmCIocryAqPsdYOSQlVIjCgyMRbLSaXxi3GD4LEw4AipzCyyvS5a5ThMpJTGAYUuQljhiWL53R11FN5BxhQsK0UWbE747E7evGV2FaEAUWmDave0H4LQxg6nErl1IEBBRdmOzjkBPpdqFB%2BpUtUGb0tDKloZP44hQLthQoDwXYiXlowpMJIymExdARL8SViYzymhGEMFR%2FR3cOyNoRCpQcZFu1s6AsNhlQuSiJP%2B1Kk90dNRHW9BYyhwlszhNgdb05CjmGcKDb3DotAoYIYV9wWxjDSZcHNmN%2Fj0KpPm3R7dMjq7HlrSokvjIqjww3SEhb4XJDpg3CLvM9%2BPG%2FMHOcaOwzYRFScNe8QHJb9nOEDhvkGwV48eZC3BgfzWwSHZaXthKEVMvkMaQnKhKESzSCkJ37uQqlJ7RmCIcbr%2By5qUEjiIwQK3q4yZKHqYDxEUIo4U6%2BNahxKr0kEZwv8HC%2BDqo69UaI2ieBAujN2RNhOoPybQjBr9oNSKNXSoQ%2B2luCUQuk1iSCIg9oiZl24Vv8TtXLROaotAtO3%2F9ooWSFcjDnH6BQio2SZQSRz%2FpsPfsifQ2RY1tmNBM3oxQRCbRjkOZn%2FEACT2J%2B1vkZiGESyG1SZS%2FqJ1wTogE1hEFHNh9yNCbvvREwqCwwoawwoKw0oKw0oKw0oKw0oKw0oKw0oMFYqMFYqMFYqMBYq88Y%2FxB7wiOJRvWkAAAAASUVORK5CYII%3D" />

[![Open Source Love](https://firstcontributions.github.io/open-source-badges/badges/open-source-v2/open-source.svg)](https://github.com/firstcontributions/open-source-badges)

**A Google Summer of Code 2023 Project Repository.**<br>The main purpose of this project is to explore the use of Quantum Graph Neural Netowrks (QGNNs) in the domain of High Energy Physics<br>

</div>

## Introduction

The LHC at CERN contains large detectors which are made up of numerous small detectors that capture the hundreds of particles produced during collisions. It’s one of the most difficult tasks in High Energy Physics (HEP) to determine whether the jet particles correspond to the signal or background. Graph Neural Networks (GNNs) have recently gained popularity and shown to exhibit higher AUC scores for jet tagging. Quantum Machine learning has shown interesting applications in HEP. The project aims to explore Quantum Graph Neural Networks (QGNNs) for event classification. QGNNs can leverage the power of quantum computing to perform more efficient and accurate analyses of large datasets in HEP. This project has the potential to significantly advance our understanding of the fundamental particles and forces that govern our universe.

## Setup

Tested on Ubuntu 22.04.1 LTS

```
git clone https://github.com/Gopal-Dahale/qgnn-hep.git
cd qgnn-hep
python -m venv qenv
source qenv/bin/activate
export PYTHONPATH=.
pip install -r requirements.txt
```

## Data

For initial analysis, we used the [MUTAG](https://paperswithcode.com/dataset/mutag) dataset and for HEP, Quark Gluon [[1]](#1) dataset was used.

## Structure and Usage

The `qgnn_hep` contains code for hybrid QGNN which is based on the following paper [[2]](#2) which can be used as follows:

First, we need to build the tfds mutag dataset with the following command:

```
tfds build qgnn_hep/data/mutag --config_idx 0
```
then the model can be trained with the following command:

```
python training/run_experiment.py --config=training/configs/default_graph_conv_net.py --workdir=logs/
```
For a quantum gnn, use the config `default_qgraph_conv_net.py`.

<hr>

`ego_net` has the approximate implementation of the paper [[3]](#3) and can be executed using

```
python ego_net/train.py
```
<hr>

Based on the 2021 paper [[4]](#4) which presents a simple and hyperparameter-free whole graph embedding method based on the DHC (Degree, H-index, and Coreness) theorem and Shannon Entropy (E), abbreviated as DHC-E. We encode this in a single qubit to perform the classification of data in `dhce` directory and can be executed using the following command:

```
python dhce/qnn.py
```

<hr>

The `notebooks` directory contains development notebooks used during the period of GSoC.

## Experiments

[Wandb](https://wandb.ai/) was used for train logs and reports. Reports can be found [here](https://wandb.ai/gopald/qgnn-hep/reportlist). In the course of our project, we encountered challenges in achieving desired results on the Quark-Gluon dataset during training. While the initial outcomes did not meet our expectations, we are committed to further research and development with the intent to enhance our results and continue advancing this project.

## Acknowledgement

I extend my gratitude to Dr. Sergei Gleyzer for his invaluable guidance throughout the project, as well as the tremendous support received from the ML4SCI community.

## References

<a id="1">[1]</a> Andrews, M., Alison, J., An, S., Burkle, B., Gleyzer, S., Narain, M., ... & Usai, E. (2020). End-to-end jet classification of quarks and gluons with the CMS Open Data. Nuclear instruments and methods in physics research section A: accelerators, spectrometers, detectors and associated equipment, 977, 164304.

<a id="2">[2]</a> Tüysüz, C., Rieger, C., Novotny, K. et al. Hybrid quantum classical graph neural networks for particle track reconstruction. Quantum Mach. Intell. 3, 29 (2021). https://doi.org/10.1007/s42484-021-00055-9

<a id="3">[3]</a> Ai, Xing & Zhang, Zhihong & Sun, Luzhe & Yan, Junchi & Hancock, Edwin. (2023). Decompositional Quantum Graph Neural Network.

<a id="4">[4]</a> Wang, Hao & Deng, Yue & Lü, Linyuan & Chen, Guanrong. (2021). Hyperparameter-free and Explainable Whole Graph Embedding.