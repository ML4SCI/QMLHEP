"""mutag dataset config"""

import dataclasses
from typing import Tuple

import tensorflow_datasets as tfds


@dataclasses.dataclass
class MutagConfig(tfds.core.BuilderConfig):
    """BuilderConfig for mutag."""

    img_size: Tuple[int, int] = (0, 0)
