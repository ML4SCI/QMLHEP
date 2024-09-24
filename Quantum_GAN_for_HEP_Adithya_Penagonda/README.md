
# Quantum Generative Adverarial network for High energy physics

Google summer of code 2024 for ML4SCI/QMLHEP

#### Description of Directories and files

- `code/`: Contains all the parts of the code
  - `Classical_gans`: Contains code of all classical models with jupyter notebooks
    - `Hybrid_gans`: Contains code for hybrid models with jupyter notebooks
    - `Quantum_gans`: Contains code for fully hybrid GAN models

- `image_outputs`: Contains samples of generated images during training
- `notebooks/`: Contains the jupyter notebooks used during the project
  - `Classical_gans`: Contains code regarding classical models and different loss tests
  - `Hybrid_gans`: Contains code for hybrid models with FIDs
  - `IQGAN_replication`: Contains code for replicating the results of the model IQGAN_replication
  - `Quantum_gans`: Contains code for fully quantum models

#### Project Documentation: 
- [Proposal](https://github.com/Userfound404/GSoC-QMLHEP-Tasks/blob/main/GSOC_2024_Proposal.pdf)
- [Test Tasks](https://github.com/Userfound404/GSoC-QMLHEP-Tasks)
- [Mid-term Blog](https://medium.com/@swheatdreamz/gsoc-24-quantum-generative-adversarial-networks-for-hep-event-generation-the-lhc-4bb1fb50faba)
- [GSoC Abstract](https://summerofcode.withgoogle.com/programs/2024/projects/OKkyXUkV)
- [Project slides](https://www.figma.com/proto/iMmkVXKyAnJbFkPxtYhEpi/Gsoc-meeting-slides?node-id=109-2&t=WXEJ2L9zZ7Iyztez-1&scaling=contain&content-scaling=fixed&page-id=0%3A1)

#### Proposed Loss function: Preceptual Quantum loss

The idea is use a loss function that favors both quantum and classical scenarios. Since we are working with quantum data, we want to balance two key objectives during training:

1. **Image Quality**: Ensuring that the generated images are perceptually similar to real images.
2. **Quantum Fidelity**: Ensuring that the quantum states of the generated data resemble those of the real data.

To achieve this balance, we introduce a combined loss function known as **Perceptual Quantum Loss**.

$$
L_total = L_adversarial + alpha * L_perceptual - beta * L_fidelity 
$$

3. **Combined Loss Function**

The **Perceptual Quantum Loss** combines these two components:



- **Adversarial Loss (`L_adversarial`)**: This is the standard GAN loss that drives the generator to produce images that fool the discriminator.
- **Perceptual Loss (`L_perceptual`)**: Encourages the generator to produce images that are perceptually similar to real images.
- **Quantum Fidelity (`L_fidelity`)**: Ensures that the quantum states of generated data match those of the real data.

Here, `alpha` controls the importance of the perceptual loss, and `beta` controls the importance of quantum fidelity. By adjusting these hyperparameters, we can fine-tune the balance between visual similarity and quantum state similarity.

##### Why Use Perceptual Quantum Loss?

In Quantum GANs, we aim to generate data that is accurate in two domains:
- **Visual domain**: Ensuring that the images look like real images.
- **Quantum domain**: Ensuring that the quantum data respects the underlying quantum structure.

#### All results will be uploaded to the blog at [medium](https://medium.com/@swheatdreamz)

#### How to use
here is an example of implementing quantum GANs:
1. **Clone the Repository**

   ```bash
   git clone https://github.com//ML4SCI/QMLHEP/Quantum_GAN_for_HEP_Adithya_Penagonda
   cd code/Quantum_gans/Quantum_gan_with_PerceptualQuantumloss
   ```

2. **Set Up a Virtual Environment**
- Using `venv`:
  ```
  python3 -m venv env
  source env/bin/activate
  ```
- using `conda`:
  ```
  conda create -n quantum_gan_env python=3.8
  conda activate quantum_gan_env
  ```

3. **Install dependencies**
` pip install -r requirements.txt `

4. **Prepare the Dataset**
Create a dir `data` and place you dataset there. you can find the dataset [here](https://data.mendeley.com/datasets/4r4v785rgx/1).

5. **run the training script**
- Using the Provided Shell Script
  ```
  chmod +x scripts/run_training.sh
  bash scripts/run_training.sh
  ```
- Running Directly via Python
  ```
  python src/train.py \
    --data_path data/jet-images_Mass60-100_pT250-300_R1.25_Pix25.hdf5 \
    --num_samples 10000 \
    --pca_components 8 \
    --n_layers 3 \
    --epochs 30 \
    --batch_size 64 \
    --g_lr 0.01 \
    --d_lr 0.01 \
    --alpha 1.0 \
    --beta 1.0 \
    --log_interval 100
  ```

  
