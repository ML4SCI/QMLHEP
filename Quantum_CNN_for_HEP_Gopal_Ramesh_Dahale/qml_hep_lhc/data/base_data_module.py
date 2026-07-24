from pathlib import Path
from tqdm import tqdm
from urllib.request import urlretrieve
from tabulate import tabulate
import os
from tensorflow import argmax
import numpy as np
from qml_hep_lhc.data.utils import extract_samples, create_tf_ds
from sklearn.utils import shuffle
from qml_hep_lhc.data.preprocessor import DataPreprocessor
from sklearn.model_selection import train_test_split


class BaseDataModule():
    """
    The BaseDataModule class is a base class for all the datasets. It contains the basic functions that
    are common to all the datasets
    """
    def __init__(self, args=None) -> None:
        self.args = vars(args) if args is not None else {}

        # Set the data directories
        self.data_dir = self.data_dirname() / "downloaded"
        self.processed_data_dir = self.data_dirname() / "processed"

        if self.args.get("data_dir") is not None:
            self.data_dir = Path(self.args.get("data_dir")) / "downloaded"
            self.processed_data_dir = Path(
                self.args.get("data_dir")) / "processed"

        # Create data directories if does not exist
        if not self.data_dir.exists():
            os.makedirs(self.data_dir)
        if not self.processed_data_dir.exists():
            os.makedirs(self.processed_data_dir)

        # Set the data files
        self.dims = None
        self.output_dims = None
        self.mapping = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.train_ds = None
        self.test_ds = None
        self.val_ds = None

        # Parse arguments
        self.batch_size = self.args.get("batch_size", 128)
        self.validation_split = self.args.get("validation_split", 0.2)
        self.processed = self.args.get("processed", False)
        self.dataset_type = self.args.get("dataset_type", 1)

        if self.processed:
            self.data_dir = self.processed_data_dir

        # Percent of data to use for training and testing
        self.percent_samples = self.args.get("percent_samples", 1.0)

    @classmethod
    def data_dirname(cls):
        """
        It returns the path to the directory containing the datasets
        
        Args:
          cls: the class of the dataset.
        
        Returns:
          The path to the datasets folder.
        """
        return Path(__file__).resolve().parents[2] / "datasets"

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--batch-size", "-batch", type=int, default=128)
        parser.add_argument("--percent-samples",
                            "-per-samp",
                            type=float,
                            default=1.0)
        parser.add_argument("--data-dir", "-data-dir", type=str, default=None)
        parser.add_argument("--validation-split",
                            "-val-split",
                            type=float,
                            default=0.1)
        parser.add_argument("--processed",
                            "-processed",
                            action="store_true",
                            default=False)
        parser.add_argument("--dataset-type",
                            type=int,
                            default=1,
                            choices=[0, 1, 2, 3, 4])
        return parser

    def config(self):
        """
        Return important settings of the classical dataset, which will be passed to instantiate models.
        """
        return {
            "input_dims": self.dims,
            "output_dims": self.output_dims,
            "mapping": self.mapping
        }

    def prepare_data(self):
        """
        Use this method to do things that might write to disk or that need to be done only from a single GPU
        in distributed settings (so don't set state `self.x = y`).
        """
        pass

    def setup(self):
        """
        Split into train, val, test, and set dims and other tasks.
        """
        # Extract percent_samples of data from x_train and x_test
        if self.percent_samples != 1.0:
            self.x_train, self.y_train = extract_samples(
                self.x_train, self.y_train, self.mapping, self.percent_samples)
            self.x_test, self.y_test = extract_samples(self.x_test,
                                                       self.y_test,
                                                       self.mapping,
                                                       self.percent_samples)

        # Shuffle the data
        self.x_train, self.y_train = shuffle(self.x_train, self.y_train)
        self.x_test, self.y_test = shuffle(self.x_test, self.y_test)

        # Preprocess the data
        preprocessor = DataPreprocessor(self.args)
        self.x_train, self.y_train, self.x_test, self.y_test = preprocessor.process(
            self.x_train, self.y_train, self.x_test, self.y_test,
            self.config(), self.classes)

        # Set the configuration
        self.dims = preprocessor.dims
        self.output_dims = preprocessor.output_dims
        self.mapping = preprocessor.mapping
        self.classes = preprocessor.classes

        # Get validation data from training data
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(
            self.x_train,
            self.y_train,
            test_size=self.validation_split,
            random_state=42,
            stratify=self.y_train)

        # Set the data
        self.train_ds = create_tf_ds(self.x_train, self.y_train,
                                     self.batch_size)
        del self.x_train, self.y_train

        self.val_ds = create_tf_ds(self.x_val, self.y_val, self.batch_size)
        del self.x_val, self.y_val

        self.test_ds = create_tf_ds(self.x_test, self.y_test, self.batch_size)
        del self.x_test, self.y_test

    def __repr__(self, name) -> str:
        """
        Print info about the dataset.
        
        Args:
          name: The name of the dataset.
        """
        train_stats = get_stats(self.train_ds, self.mapping)
        test_stats = get_stats(self.test_ds, self.mapping)
        val_stats = get_stats(self.val_ds, self.mapping)

        headers = ["Data", "Train size", "Val size", "Test size", "Dims"]

        r1 = [
            "X", train_stats['x_size'], val_stats['x_size'],
            test_stats['x_size'], self.dims
        ]
        r2 = [
            "y", train_stats['y_size'], val_stats['y_size'],
            test_stats['y_size'], self.output_dims
        ]
        rows = [r1, r2]

        data = f"\nDataset :{name}\n"
        data += tabulate(rows, headers, tablefmt="fancy_grid") + "\n\n"

        headers = [
            "Type", "Min", "Max", "Mean", "Std", "Samples for each class"
        ]

        r1 = [
            "Train Images", f"{train_stats['min']:.2f}",
            f"{train_stats['max']:.2f}", f"{train_stats['mean']:.2f}",
            f"{train_stats['std']:.2f}", train_stats['n_samples_per_class']
        ]
        r2 = [
            "Val Images", f"{val_stats['min']:.2f}", f"{val_stats['max']:.2f}",
            f"{val_stats['mean']:.2f}", f"{val_stats['std']:.2f}",
            val_stats['n_samples_per_class']
        ]
        r3 = [
            "Test Images", f"{test_stats['min']:.2f}",
            f"{test_stats['max']:.2f}", f"{test_stats['mean']:.2f}",
            f"{test_stats['std']:.2f}", test_stats['n_samples_per_class']
        ]
        rows = [r1, r2, r3]

        data += tabulate(rows, headers, tablefmt="fancy_grid") + "\n\n"

        return data


class TqdmUpTo(tqdm):
    """From https://github.com/tqdm/tqdm/blob/master/examples/tqdm_wget.py"""
    def update_to(self, blocks=1, bsize=1, tsize=None):
        """
        Parametersy_train
        ----------
        blocks: int, optional
            Number of blocks transferred so far [default: 1].
        bsize: int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize: int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize  # pylint: disable=attribute-defined-outside-init
        self.update(blocks * bsize -
                    self.n)  # will also set self.n = b * bsize


def _download_raw_dataset(url, filename):
    """
    It downloads a file from a URL to a local file
    
    Args:
      url: The URL of the file to download.
      filename: The name of the file to download to.
    """
    print(f"Downloading raw dataset from {url} to {filename}")
    with TqdmUpTo(unit="B", unit_scale=True, unit_divisor=1024,
                  miniters=1) as t:
        urlretrieve(url, filename, reporthook=t.update_to, data=None)  # nosec


def get_stats(ds, mapping):
    ds = ds.unbatch()
    ds = ds.as_numpy_iterator()
    ds = [element for element in ds]
    x = np.array([element[0] for element in ds])
    y = np.array([element[1] for element in ds])
    x_size = x.shape
    y_size = y.shape
    max = x.max()
    min = x.min()
    mean = x.mean()
    std = x.std()

    if len(y_size) == 2:
        n_samples_per_class = [
            np.sum(argmax(y, axis=-1) == i) for i in (mapping)
        ]
    else:
        n_samples_per_class = [np.sum(y == i) for i in (mapping)]

    return {
        "x_size": x_size,
        "y_size": y_size,
        "max": max,
        "min": min,
        "mean": mean,
        "std": std,
        "n_samples_per_class": n_samples_per_class
    }
