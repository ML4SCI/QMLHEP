"""mutag dataset."""

from pathlib import Path
import pickle

from mutag_dataset_config import MutagConfig
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_datasets as tfds
import toml
from tqdm import tqdm

from qgnn_hep import util


class Builder(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for mutag dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }
    BUILDER_CONFIGS = [
        # `name` (and optionally `description`) are required for each config
        MutagConfig(name="mutag", description="Classic Mutag dataset"),
    ]

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        # Specifies the tfds.core.DatasetInfo object
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(
                {
                    # These are the features of your dataset like images, labels ...
                    "num_nodes": tfds.features.Tensor(shape=(), dtype=tf.int32),
                    "num_edges": tfds.features.Tensor(shape=(), dtype=tf.int32),
                    "node_feats": tfds.features.Tensor(shape=(None, 7), dtype=tf.float32),
                    "edge_feats": tfds.features.Tensor(shape=(None, 4), dtype=tf.float32),
                    "edge_index": tfds.features.Tensor(shape=(2, None), dtype=tf.int32),
                    "label": tfds.features.ClassLabel(num_classes=2),  # Here, 'label' can be 0 or 1.
                }
            ),
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            supervised_keys=("image", "label"),  # Set to `None` to disable
            homepage="https://dataset-homepage/",
        )

    def _download_raw_dataset(self, metadata, data_path):
        data_path.mkdir(parents=True, exist_ok=True)
        filename = data_path / metadata["filename"]
        if filename.exists():
            return filename
        print(f"Downloading raw dataset from {metadata['url']} to {filename}...")
        util.download_url(metadata["url"], filename)
        return filename

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        # Downloads the data and defines the splits
        metadata_file_path = Path(__file__).parents[3] / "data/raw/mutag/metadata.toml"
        data_path = Path(__file__).parents[3] / "data/downloaded/"
        metadata = toml.load(metadata_file_path)
        self._download_raw_dataset(metadata, data_path)

        with open(data_path / metadata["filename"], "rb") as f:
            mutag_ds = pickle.load(f)

        train_size = int(len(mutag_ds) * 0.8)
        test_size = len(mutag_ds) - train_size

        x = [mol["input_graph"] for mol in mutag_ds]
        y = [mol["target"][0] for mol in mutag_ds]

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42, stratify=y)

        # Returns the Dict[split names, Iterator[Key, Example]]
        return {
            "train": self._generate_examples(x_train, y_train),
            "test": self._generate_examples(x_test, y_test),
        }

    def _generate_examples(self, x, y):
        """Yields examples."""
        # Yields (key, example) tuples from the dataset
        for idx, (mol, label) in enumerate(tqdm(zip(x, y))):
            yield idx, {
                "num_nodes": mol.n_node[0],
                "num_edges": mol.n_edge[0],
                "node_feats": mol.nodes,
                "edge_feats": mol.edges,
                "edge_index": np.stack((mol.senders, mol.receivers)),
                "label": label,
            }
