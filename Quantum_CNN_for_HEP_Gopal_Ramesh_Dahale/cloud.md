
## Create a Cloud TPU VM or Node with gcloud

gcloud alpha compute tpus tpu-vm create qcnn-tpu \
--zone=europe-west4-a \
--accelerator-type=v3-8 \
--version=tpu-vm-tf-2.7.3

### My Cloud TPU quota:

5 on-demand Cloud TPU v2-8 device(s) in zone us-central1-f
5 on-demand Cloud TPU v3-8 device(s) in zone europe-west4-a
100 preemptible Cloud TPU v2-8 device(s) in zone us-central1-f

## Set default zone

gcloud config set compute/zone your-zone-here

gcloud config set compute/zone us-central1-f

## Connect to your Cloud TPU VM

gcloud alpha compute tpus tpu-vm ssh qcnn-tpu2 \
  --zone us-central1-f
  
## Clean up

exit

Delete your Cloud TPU.

gcloud alpha compute tpus tpu-vm delete qcnn-tpu \
--zone=europe-west4-a

Verify the resources have been deleted 

gcloud alpha compute tpus tpu-vm list --zone=europe-west4-a

## Create venv

sudo apt install python3.8-venv
python3 -m venv qenv
source qenv/bin/activate

## Installing Jupyter Notebook

sudo apt-get update
sudo apt install python3-pip # optional
pip3 install Jupyter
echo "export PATH=$PATH:~/.local/bin" >> ~/.bashrc
source ~/.bashrc

pip install markupsafe==2.0.1
jupyter notebook --generate-config
nano ~/.jupyter/jupyter_notebook_config.py

Add below lines

c = get_config()
c.NotebookApp.ip = '*'
c.NotebookApp.open_browser = False
c.NotebookApp.port = 8888

jupyter notebook

notebook will run on http://<external-ip-address>:<port-number>

### Add firewall rule
VPC Network -> Firewall -> Create a firewall rule

Targets: All instances in the network
Source IPV4 ranges: 0.0.0.0/0
Specified protocols and ports: tcp 8888

Everything else default

