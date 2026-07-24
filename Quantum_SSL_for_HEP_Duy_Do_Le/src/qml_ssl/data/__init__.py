from torch.utils.data import DataLoader, TensorDataset

from .img_mnist import load_mnist_img, visualize_data as visualize_mnist
from .img_pe import load_pe_img, inspect_h5py_file, visualize_data as visualize_pe
from .img_qg import load_qg_img, visualize_average_images as visualize_qg, visualize_diff_average_images as visualize_qg_diff

from .graph_syn_qg import QG_Jets
from .graph_transform import TopKMomentum, ToTopMomentum, KNNGroup

__all__ = [
    "load_mnist_img",
    "load_pe_img",
    "load_qg_img",
    "QG_Jets",
    "TopKMomentum",
    "ToTopMomentum",
    "KNNGroup"
]

def create_data_loader(data, labels, batch_size=64, shuffle=True, num_workers=4):
    dataset = TensorDataset(data, labels)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return data_loader