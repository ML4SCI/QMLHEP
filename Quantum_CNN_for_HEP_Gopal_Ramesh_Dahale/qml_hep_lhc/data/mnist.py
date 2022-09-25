from tensorflow.keras.datasets import mnist
from qml_hep_lhc.data.base_data_module import BaseDataModule
import numpy as np


class MNIST(BaseDataModule):
    """
    MNIST Data module
    """
    def __init__(self, args=None) -> None:
        super().__init__(args)

        self.classes = list(range(10))

        self.dims = (28, 28, 1)
        self.output_dims = (1, )
        self.mapping = list(range(10))

        # Parse args
        self.args['is_binary_data'] = False
        self.filename = self.data_dir / 'mnist.npz'

    def prepare_data(self):
        # Load the data

        if self.processed:
            # Extract the data
            data = np.load(self.filename, allow_pickle=True)
            self.x_train, self.y_train = data['x_train'], data['y_train']
            self.x_test, self.y_test = data['x_test'], data['y_test']
        else:
            (self.x_train,
             self.y_train), (self.x_test,
                             self.y_test) = mnist.load_data(self.filename)

    def __repr__(self) -> str:
        return super().__repr__("MNIST")

    @staticmethod
    def add_to_argparse(parser):
        return parser
