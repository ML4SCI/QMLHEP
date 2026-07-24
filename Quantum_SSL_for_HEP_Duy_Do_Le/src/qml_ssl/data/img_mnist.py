import torch
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
    
def load_mnist_img(classes=None, reduced_dim=None, dataset_size=None, data_dir="../../data"):
    """
    Load and preprocess MNIST data.

    Args:
        classes (tuple): Tuple of classes to filter (default is (3, 6)).
        reduced_dim (int): Size to resize the images to (default is None).
        dataset_size (tuple): Custom dataset size (train_size, test_size) (default is None).

    Returns:
        dict: A dictionary with the preprocessed training and test datasets.
    """

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    # Load MNIST dataset
    train_dataset = datasets.MNIST(root=data_dir, train=True, download=False, transform=data_transform)
    test_dataset = datasets.MNIST(root=data_dir, train=False, download=False, transform=data_transform)

    def filter_classes(dataset, classes):
        mask = torch.zeros_like(dataset.targets, dtype=torch.bool)
        for cls in classes:
            mask |= (dataset.targets == cls)
        dataset.data = dataset.data[mask]
        dataset.targets = dataset.targets[mask]
        return dataset
    
    # Filter out only the specified classes
    if classes:
        train_dataset = filter_classes(train_dataset, classes)
        test_dataset = filter_classes(test_dataset, classes)

    # # Create binary labels for the specified classes
    # def create_binary_labels(dataset, positive_class):
    #     dataset.targets = (dataset.targets == positive_class).long()
    #     return dataset

    # train_dataset = create_binary_labels(train_dataset, classes[0])
    # test_dataset = create_binary_labels(test_dataset, classes[0])

     # Resize images
    if reduced_dim:
        resize_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((reduced_dim, reduced_dim), interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.ToTensor()
        ])

        train_dataset.data = torch.stack([resize_transform(img) for img in train_dataset.data])
        test_dataset.data = torch.stack([resize_transform(img) for img in test_dataset.data])

    # Reduce dataset size for faster training if specified
    if dataset_size:
        train_dataset.data = train_dataset.data[:dataset_size[0]]
        test_dataset.data = test_dataset.data[:dataset_size[1]]
        train_dataset.targets = train_dataset.targets[:dataset_size[0]]
        test_dataset.targets = test_dataset.targets[:dataset_size[1]]

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    return {
        "train_data": train_dataset.data,
        "train_labels": train_dataset.targets,
        "test_data": test_dataset.data,
        "test_labels": test_dataset.targets
    }


def visualize_data(data, labels, classes, title=""):
    """
    Visualize the dataset.

    Args:
        data (torch.Tensor): Dataset images.
        labels (torch.Tensor): Corresponding labels.
        classes (tuple): Tuple of class labels to visualize.
        title (str): Title of the plot (default is "").
    """
    fig, axs = plt.subplots(1, len(classes), figsize=(8, 4))

    for i, cls in enumerate(classes):
        class_data = data[labels == cls]
        if len(class_data) > 0:
            axs[i].imshow(class_data[0].squeeze(), cmap='binary')
            axs[i].set_title(f"Class {cls}")
        else:
            axs[i].set_title(f"Class {cls} (No data)")
    fig.suptitle(title)
    plt.show()

if __name__ == "__main__":
    # Define the classes and size
    classes = (3, 6, 8)
    reduced_dim = 16
    dataset_size = (1000, 300)

    # Load and preprocess the MNIST data
    mnist_data_original = load_mnist_img(classes=classes, dataset_size=dataset_size)
    mnist_data_resized = load_mnist_img(classes=classes, reduced_dim=reduced_dim, dataset_size=dataset_size)
    
    # Visualize the preprocessed images
    visualize_data(mnist_data_original["train_data"], mnist_data_original["train_labels"], classes, title="Original MNIST Data")
    visualize_data(mnist_data_resized["train_data"], mnist_data_resized["train_labels"], classes, title="Preprocessed MNIST Data")
