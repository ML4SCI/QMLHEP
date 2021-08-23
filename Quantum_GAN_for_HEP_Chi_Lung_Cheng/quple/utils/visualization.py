from typing import Optional, Dict

import matplotlib.pyplot as plt

def visualize_images(images:"numpy.ndarray", columns:int=8, subplots_options:Optional[Dict]=None,
                     labels:Optional["numpy.ndarray"]=None, label_map:Optional[Dict]=None):
    """Visualize a set of 2D images in a grid
    
    Arguments:
        images: numpy.ndarray
            A 3D numpy array representing batches of images of shape (batchsize, rows, cols).
        colunms: int
            Number of coumns in the grid.
        subplots_options: (Optional) dict
            A dictionary containing subplots options.
        labels: (Optional) numpy.ndarray
            A 1D numpy array representing the object labels to the images of shape (batchsize).
        label_map: (Optional) dict
            A dictionary containing the map from an object label to a string representation of
            that label which will be displayed in the image title. If None, images are displayed
            without titles.
        save_as: (Optional) str
            The path from which output image is saved. If None, the output image will not be saved.
    Returns:
        matplotlib.pyplot object
    """
    plt.clf()
    size = images.shape[0]
    rows = ( size // columns ) + 1
    fig = plt.figure(figsize=(20, rows*3))
    if (labels is not None) and (label_map is not None):
        assert labels.shape[0] == images.shape[0]
        titles = ["Image {}: {}".format(i, label_map[labels[i]]) for i in range(size)]
    else:
        titles = [None]*size
    for i in range(images.shape[0]):
        ax = fig.add_subplot(rows, columns, i+1)
        ax.set_title(titles[i])
        ax.imshow(images[i])
    if subplots_options is not None:
        plt.subplots_adjust(**subplots_options)
    return plt