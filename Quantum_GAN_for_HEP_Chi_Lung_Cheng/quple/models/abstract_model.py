from typing import Optional, Union, Dict
from collections.abc import Iterable
from abc import ABC, abstractmethod

import tensorflow as tf
import numpy as np

class AbstractModel(ABC):
    """Abstract class for building quantum models"""
    def __init__(self, name:Optional[str]=None,
                 random_state:Optional[int]=None, 
                 checkpoint_dir:Optional[str]=None,
                 checkpoint_interval:int=10,
                 checkpoint_max_to_keep:Optional[int]=None):
        """Instantiate the quantum model
        
            Arguments:
                name: (Optional) string
                    Name given to the model.
                random_state: (Optional) int
                    The random state used at the beginning of a training
                    for reproducible result.
                checkpoint_dir: Optional[string]
                    The path to a directory in which to write checkpoints. If None,
                    no checkpoint will be saved.
                checkpoint_interval: int, default=10
                    Number of epochs between each checkpoint.
                checkpoint_max_to_keep: (Optional) int
                    Number of checkpoints to keep. If None, all checkpoints are kept.
        """
        # for restoring epoch at checkpoint
        self.start_epoch = 0
        
        self.name = name
        self.random_state = random_state
        
        self.set_random_state(self.random_state)
        self._setup_checkpoint(checkpoint_dir, checkpoint_interval,
                               checkpoint_max_to_keep)
        
        self._validate_init()
        
    def _validate_init(self):
        pass
    
    @abstractmethod
    def _create_checkpoint(self, *args, **kwargs):
        pass
    
    def _setup_checkpoint(self, directory:Optional[str]=None, interval:int=10,
                          max_to_keep:Optional[int]=None, *args, **kwargs):
        """Setup training checkpoints.
            
            Note that checkpoints are created only when training starts.
            
            Arguments:
                directory: string
                    The path to a directory in which to write checkpoints.
                interval: int, default=10
                    Number of epochs between each checkpoint.
                max_to_keep: (Optional) int
                    Number of checkpoints to keep. If None, all checkpoints are kept.
        """
        self.checkpoint_dir = directory
        self.checkpoint_interval = interval
        self.checkpoint_max_to_keep = max_to_keep
        
        if self.checkpoint_dir is not None:
            self.checkpoint = self._create_checkpoint(*args, **kwargs)
            self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, 
                                                                 self.checkpoint_dir,
                                                                 self.checkpoint_max_to_keep)
        else:
            self.checkpoint = None
            self.checkpoint_manager = None
    
    @staticmethod
    def set_random_state(random_state:Optional[int]=None):
        """Set the global random states for the tensorflow and the numpy libraries
            
            Arguments:
                random_state: (Optional) int
                    Random state to set.
        """
        tf.random.set_seed(random_state)
        np.random.seed(random_state)
        
    @staticmethod
    def prepare_dataset(*data, batch_size:int, seed=None, drop_remainder=True, buffer_size=10000):
        """Create batched dataset with shuffling
        
            Arguments:
                *data: iterable of list/np.ndarray/tf.Tensor
                    Iterable of data that make up the dataset.
                batchsize: int
                    Number of consecutive elements of the dataset to combine in a single batch.
                seed: (Optional.) int
                    Random seed used in shuffling the dataset.
                drop_remainder: boolean, default=True
                    Whether the last batch should be dropped in the case it has fewer than batch_size elements.
                buffer_size: int
                    Buffer size used in shuffling the dataset.
            Returns:
                tf.Dataset containing the shuffled dataset.
        """
        buffer_size = len(data[0])
        ds = tf.data.Dataset.from_tensor_slices(data)
        ds = ds.shuffle(buffer_size=buffer_size, seed=seed, reshuffle_each_iteration=True)
        ds = ds.batch(batch_size, drop_remainder=drop_remainder)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds
    
    @tf.function
    def to_prob(self, x):
        """Convert quantum circuit output to probabilities"""
        return tf.divide(tf.add(x, 1), 2)
    
    def restore_checkpoint(self, checkpoint:tf.train.Checkpoint=None):
        if (self.checkpoint is None) or (self.checkpoint_manager is None):
            raise RuntimeError("check point not initialized")
        if checkpoint is None:
            checkpoint = self.checkpoint_manager.latest_checkpoint
        self.checkpoint.restore(checkpoint)
        self.start_epoch = int(self.checkpoint.step) + 1
    
    def _train_preprocess(self, *args, **kwargs):
        self.set_random_state(self.random_state)
        
    def _train_post_epoch(self, epoch:int, *args, **kwargs):
        if (self.checkpoint is not None) and (self.checkpoint_manager is not None):
            self.checkpoint.step.assign_add(1)
            if int(self.checkpoint.step) % self.checkpoint_interval == 0:
                self.checkpoint_manager.save()
        
    def _train_postprocess(self, *args, **kwargs):
        pass