from qml_hep_lhc.data.base_data_module import BaseDataModule, _download_raw_dataset
import numpy as np
from qml_hep_lhc.data.constants import ELECTRON_PHOTON_SMALL_DATASET_URL


class ElectronPhoton(BaseDataModule):
    """
    Electron Photon Data module
    """
    def __init__(self, args=None) -> None:
        super().__init__(args)

        self.dims = (32, 32, 1)
        self.output_dims = (1, )
        self.mapping = list(range(2))

        self.classes = ['Photon', 'Electron']

        # Parse args
        self.args['is_binary_data'] = True
        self.filename = self.data_dir / f"electron_photon_{self.dataset_type}.npz"

    def prepare_data(self):
        # Load the data
        if not self.filename.exists():
            if self.dataset_type == 0:
                # Downloads the small data
                _download_raw_dataset(ELECTRON_PHOTON_SMALL_DATASET_URL,
                                      self.filename)
            else:
                raise ValueError(
                    "Specify the dataset dir for medium/large dataset")

        # Extract the data
        data = np.load(self.filename, allow_pickle=True)
        self.x_train, self.y_train = data['x_train'], data['y_train']
        self.x_test, self.y_test = data['x_test'], data['y_test']

    def __repr__(self) -> str:
        return super().__repr__(f"Electron Photon {self.dataset_type}")

    @staticmethod
    def add_to_argparse(parser):
        return parser
