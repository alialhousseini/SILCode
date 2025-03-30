"""
Utility functions for data processing and manipulation.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


def tensor_to_numpy(tensor):
    """
    Convert a PyTorch tensor to a numpy array.

    Args:
        tensor (torch.Tensor): Input tensor

    Returns:
        numpy.ndarray: Numpy array
    """
    if tensor.requires_grad:
        tensor = tensor.detach()
    if tensor.is_cuda:
        tensor = tensor.cpu()
    return tensor.numpy()


def get_class_distribution(dataset):
    """
    Get the class distribution of a dataset.

    Args:
        dataset (torch.utils.data.Dataset): Dataset

    Returns:
        dict: Dictionary mapping class indices to counts
    """
    if hasattr(dataset, 'targets'):
        targets = dataset.targets
    else:
        # For subset datasets, we need to extract targets differently
        if isinstance(dataset, torch.utils.data.Subset):
            dataset_targets = dataset.dataset.targets
            if isinstance(dataset_targets, torch.Tensor):
                targets = dataset_targets[dataset.indices]
            else:
                targets = [dataset_targets[i] for i in dataset.indices]
        else:
            # Extract targets by iterating through the dataset
            targets = []
            for _, target in dataset:
                targets.append(target)
            targets = torch.tensor(targets)

    # Count occurrences of each class
    unique_classes, counts = torch.unique(
        torch.tensor(targets), return_counts=True)
    class_distribution = {int(cls.item()): int(count.item())
                          for cls, count in zip(unique_classes, counts)}

    return class_distribution


def visualize_batch(batch, num_images=16, title=None, save_path=None):
    """
    Visualize a batch of images.

    Args:
        batch (torch.Tensor): Batch of images
        num_images (int): Number of images to visualize
        title (str, optional): Title for the plot
        save_path (str, optional): Path to save the visualization

    Returns:
        matplotlib.figure.Figure: Figure object
    """
    # Ensure batch is a tensor
    if not isinstance(batch, torch.Tensor):
        batch = batch[0]  # Assume batch is a tuple (images, labels)

    # Limit to num_images
    batch = batch[:num_images]

    # Create grid of images
    grid = make_grid(batch, nrow=int(np.sqrt(num_images)))

    # Convert to numpy and transpose
    grid = tensor_to_numpy(grid)
    grid = np.transpose(grid, (1, 2, 0))

    # Normalize if needed
    if grid.max() > 1:
        grid = grid / 255.0

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(grid)
    ax.axis('off')

    if title:
        ax.set_title(title)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    return fig
