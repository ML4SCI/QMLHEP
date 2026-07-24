# Classical
from .classical.resnet.v1 import ResnetV1
from .classical.resnet.v2 import ResnetV2
from .classical.resnet.resnet50 import Resnet50
from .classical.cnn import CNN
from .classical.mlp import MLP

# Quantum
from .quantum.qcnn import QCNN
from .quantum.fqcnn import FQCNN
from .quantum.qcnn_hybrid import QCNNHybrid
from .quantum.qcnn_chen import QCNNChen
from .quantum.qcnn_cong import QCNNCong
from .quantum.qcnn_sandwich import QCNNSandwich
from .quantum.vqc import VQC
from .quantum.resnetq50 import ResnetQ50

from .base_model import BaseModel