from typing import Tuple
import operator

import numpy as np
from sklearn.model_selection import train_test_split

def _validate_train_val_test_sizes(train_size, val_size, test_size):
    sizes = [train_size, val_size, test_size]
    dtypes = [np.asarray(size).dtype.kind for size in sizes]
    if all(size==None for size in sizes):
        # set default ratios
        train_size = 0.7
        test_size = 0.15
        train_size = 0.15
    elif all(dtype == 'f' for dtype in dtypes):
        sizes_sum = np.sum(sizes)
        if sizes_sum < 0 or sizes_sum > 1:
            raise ValueError(
            'The sum of train_size, test_size and train_size = {}, '
            'should be within (0,1)'.format(sizes_sum))
    elif not all(dtype == 'i' for dtype in dtypes):
        raise ValueError('train_size={}, val_size={}, test_size={} must '
                         'be all integers, all floats or all None'.format(
                             train_size, val_size, test_size))
        
    return tuple(sizes)
    

def train_val_test_split(x:np.ndarray, y:np.ndarray, train_size=None, val_size=None,
                         test_size=None, stratify=None, shuffle=True, 
                         random_state=None, weight:np.ndarray=None):
    """Split inputs into the training, validation and test set based on sklearn train_test_split
    
    Allowed inputs are lists, numpy arrays, scipy-sparse matrices or pandas dataframes.
    
    Args:
        x, y: sequence of indexables with same length / shape[0]
        train_size: float or int, default=None
            If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the train split. If int, represents the absolute number of train samples. If both train_size, val_size and test_size are None, it will be set to 0.7.
        val_size: float or int, default=None
            If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the validation split. If int, represents the absolute number of validation samples. If both train_size, val_size and test_size are None, it will be set to 0.15.
        test_size: float or int, default=None
            If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split. If int, represents the absolute number of test samples. If both train_size, val_size and test_size are None, it will be set to 0.15. 
        stratify: array-like, default=None
            If not None, data is split in a stratified fashion, using this as the class labels.
        shuffle: bool, default=True
            Whether or not to shuffle the data before splitting. If shuffle=False then stratify must be None.            
        random_state: int or RandomState instance, default=None
            Controls the shuffling applied to the data before applying the split. Pass an int for reproducible output across multiple function calls. 
            
    Returns:
        List containing train-val-test split of inputs.            
    """
    train_size, val_size, test_size = _validate_train_val_test_sizes(train_size, val_size, test_size)
    
    arrays = (x, y) if weight is None else (x, y, weight)

    splitting = train_test_split(*arrays, train_size=train_size,
                                          test_size=val_size+test_size,
                                          shuffle=shuffle,
                                          stratify=stratify,
                                          random_state=random_state)
    if np.asarray(val_size).dtype.kind == 'f' and np.asarray(test_size).dtype.kind == 'f':
        normalizer = 1/(val_size+test_size)
        val_size *=normalizer
        test_size *=normalizer
    if stratify is not None:
        stratify = splitting[3]
    arrays = (splitting[1], splitting[3]) if weight is None else (splitting[1], splitting[3], splitting[5])
    new_splitting = train_test_split(*arrays, train_size=val_size,
                                              test_size=test_size,
                                              shuffle=shuffle,
                                              stratify=stratify,
                                              random_state=random_state)                                                    
    if weight is None:
        return (splitting[0], new_splitting[0], new_splitting[1], splitting[2], new_splitting[2], new_splitting[3])
    else:
        return (splitting[0], new_splitting[0], new_splitting[1], splitting[2], new_splitting[2], new_splitting[3], 
                splitting[4], new_splitting[4], new_splitting[5])

def prepare_train_val_test(x:np.ndarray, y:np.ndarray, train_size=None, val_size=None,
                           test_size=None, preprocessors=None, shuffle=True, stratify=None,
                           random_state=None, weight=None):
    """Prepares training, validation and test datasets from inputs with proprocessing 
    
    Allowed inputs are lists, numpy arrays, scipy-sparse matrices or pandas dataframes.
    
    Args:
        x, y: sequence of indexables with same length / shape[0]
        train_size: float or int, default=None
            If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the train split. If int, represents the absolute number of train samples. If both train_size, val_size and test_size are None, it will be set to 0.7.
        val_size: float or int, default=None
            If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the validation split. If int, represents the absolute number of validation samples. If both train_size, val_size and test_size are None, it will be set to 0.15.
        test_size: float or int, default=None
            If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split. If int, represents the absolute number of test samples. If both train_size, val_size and test_size are None, it will be set to 0.15. 
        stratify: array-like, default=None
            If not None, data is split in a stratified fashion, using this as the class labels.
        shuffle: bool, default=True
            Whether or not to shuffle the data before splitting. If shuffle=False then stratify must be None.            
        random_state: int or RandomState instance, default=None
            Controls the shuffling applied to the data before applying the split. Pass an int for reproducible output across multiple function calls. 
        preprocessors: sequence of sklearn transformers 
            Preprocess data with the given preprocessors.
            
    Returns:
        List containing train-val-test split of preprocessed inputs.
    """
    
    splitting = \
    train_val_test_split(x, y, train_size, val_size, test_size, 
                         shuffle=shuffle, 
                         stratify=stratify,
                         random_state=random_state,
                         weight=weight)
    x_train, x_val, x_test = splitting[0], splitting[1], splitting[2]
    y_train, y_val, y_test = splitting[3], splitting[4], splitting[5]
    if weight is not None:
        weight_train, weight_val, weight_test = splitting[6], splitting[7], splitting[8]
    if preprocessors:
        import sklearn
        for preprocessor in preprocessors:
            if not isinstance(preprocessor, sklearn.base.TransformerMixin):
                raise ValueError('Data preprocessor must be an instance of '
                                 'sklearn.base.TransformerMixin')
            transformer = preprocessor.fit(x_train)
            x_train = transformer.transform(x_train)
            x_val = transformer.transform(x_val)
            x_test = transformer.transform(x_test)
    if weight is None:                                       
        return x_train, x_val, x_test, y_train, y_val, y_test  
    else:
        return x_train, x_val, x_test, y_train, y_val, y_test, weight_train, weight_val, weight_test
    
def rescale_data(*data, val_range:Tuple=(0, 1.)):
    min_value = np.min([np.min(d) for d in data])
    max_value = np.max([np.max(d) for d in data])
    range_min = val_range[0]
    range_max = val_range[1]
    rescaled_data = tuple([(((d-min_value)/(max_value-min_value))*(range_max-range_min))+range_min for d in data])
    if len(rescaled_data) == 1:
        return rescaled_data[0]
    return rescaled_data

def crop_images(images:np.ndarray, dimension:Tuple[int]):
    """Crop the central region of an image (2D array) or multiple images (3D array) to the required dimension
    
    Arguments:
        image: numpy.ndarray
            In case of a 2D array, it represents the pixel values of an image with shape (rows, cols)
            In case of a 3D array, it represents a batch of images with shape (batchsize, rows, cols)
        dimension: Tuple[int]
            A 2-tuple specifying the (width, height) of the cropped frame
    Returns:
        Image(s) cropped to the specified dimension
    """
    if images.ndim == 3:
        dimension = (images.shape[0],) + dimension
    elif images.ndim != 2:
        raise ValueError("image must be a 2D or 3D array")
    start = tuple(map(lambda a, da: a//2-da//2, images.shape, dimension))
    end = tuple(map(operator.add, start, dimension))
    slices = tuple(map(slice, start, end))
    return images[slices]    