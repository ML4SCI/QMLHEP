# GSoC-QGAN
Quantum Generative Adversarial Networks

Project repository for the 2023 GSoC project [Quantum Generative Adversarial Networks for HEP event generation the LHC](https://summerofcode.withgoogle.com/programs/2023/projects/ggoiGDQ5) with the [ML4SCI](https://ml4sci.org/) organization. 
For more information see also the [blog post](https://www.tommago.com/posts/gsoc23/).

I thank the ML4SCI QML group for the support and collaboration and [NERSC](https://www.nersc.gov/) for the access to perlmutter for training.

## Code

Currently the main code for training and evaluating the QGANs is located in the notebooks folder. There are different notebooks for

1. Fully quantum GAN (SWAP test)
2. Hybrid QGAN (classical discriminator)
3. CV QGAN (Modeling continous PDFs by data embedding and expecation value measurement) 

### Installation

The notebooks can be run after installing the `requirements.txt`.
Alternatively, since I have been performing some training on [NERSCs perlmutter](https://www.nersc.gov/systems/perlmutter/), I provide a docker image for GPU execution under [tommago
/
culane](https://hub.docker.com/repository/docker/tommago/culane/general).
It is based on [NVIDIAs cuQuantum appliance](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuquantum-appliance) and can be used for simple GPU execution by enabling

``` python
dev = qml.device('lightning.gpu', wires=9)
```

In principle the appliance also works for multi gpu, however the scaling for smaller number of qubits is not very good (See [this blog post](https://www.tommago.com/posts/nersc/)).
It would be helpful to have a multi GPU implementation with MPI which parallelizes the circuit executions within in single batch.
