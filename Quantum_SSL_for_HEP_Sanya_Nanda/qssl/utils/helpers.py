def filter_labels(images, labels, label):
    filter_mask = np.isin(labels, [label])
    return images[filter_mask]