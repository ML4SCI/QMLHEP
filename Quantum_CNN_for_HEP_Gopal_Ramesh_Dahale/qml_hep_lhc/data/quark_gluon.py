from qml_hep_lhc.data.base_data_module import BaseDataModule
import numpy as np


class QuarkGluon(BaseDataModule):
    def __init__(self, args=None) -> None:
        super().__init__(args)

        self.dims = (40, 40, 1)
        self.output_dims = (1, )
        self.mapping = list(range(2))

        self.classes = ['Quark', 'Gluon']

        # Parse args
        self.args['is_binary_data'] = True
        self.filename = self.data_dir / f"quark_gluon_{self.dataset_type}.npz"

    def prepare_data(self):
        # Load the data
        if self.dataset_type == 0:
            raise ValueError("Small dataset not available")
        elif not self.filename.exists():
            raise ValueError(
                "Specify the dataset dir for medium/large dataset")

        # Extract the data
        data = np.load(self.filename, allow_pickle=True)
        self.x_train, self.y_train = data['x_train'], data['y_train']
        self.x_test, self.y_test = data['x_test'], data['y_test']

    def __repr__(self) -> str:
        return super().__repr__(f"Quark Gluon {self.dataset_type}")

    @staticmethod
    def add_to_argparse(parser):
        return parser
