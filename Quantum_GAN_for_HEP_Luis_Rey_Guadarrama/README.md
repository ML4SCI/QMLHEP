<div align="center">

# **Quantum Generative Adversarial Networks for Monte Carlo Simulations**


<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/GSoC-logo-horizontal.svg/800px-GSoC-logo-horizontal.svg.png?20190922122222" />

Google Summer of Code 2024.
---

<div align="justify">

## **Introduction**
---
Monte Carlo simulations rely on accurate probability distributions to generate meaningful results, but selecting incorrect distributions can lead to
inaccuracies and require extensive computational resources. QGANs use variational quantum circuits to learn implicit probability distributions from 
datasets and encode them as quantum states. The quantum generator then produces synthetic data following the learned distribution, which is combined
with real datasets and fed into a classical discriminator to distinguish between real and generated data. By exploiting the randomness of quantum 
systems, the project seeks implementing QGANs, training them on various datasets relevant to Monte Carlo simulations, exploring different QGAN
architectures for optimal performance, and benchmarking against standard Monte Carlo simulations to assess improvements. Finally, this works aims to 
use the implemented architectures to generate quark-versus-gluon-initiated jets events from the dataset constructed by 
[Andrews et.al.](https://doi.org/10.1016/j.nima.2020.164304) [1]

## **Setup**
---

### **Prerequisites**
To download and run this code, the following software is required:
* Git
* pip
* python3

### **Installation**

```
git clone git@github.com:ReyGuadarrama/QGAN_for_MC_Simulations.git
cd QGAN_for_MC_Simulations
python3 -m venv env
source env/bin/activate
pip install -r docs/requierements.txt

```

## References
[1] Andrews, M., Alison, J., An, S., Bryant, P., Burkle, B., Gleyzer, S., Narain, M., Paulini, M., Poczos, B. & Usai, E. (2019). End-to-End Jet Classification 
of Quarks and Gluons with the CMS Open Data. Nucl. Instrum. Methods Phys. Res. A 977, 164304 (2020). https://doi.org/10.1016/j.nima.2020.164304