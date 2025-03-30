"""
Common utility functions for CNN-LSX.

This module contains utility functions used throughout the codebase,
including device setup, seed setting, and other helper functions.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Union, Any


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility across all libraries.

    Args:
        seed: Random seed
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # For CUDA operations
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(no_cuda: bool = False) -> torch.device:
    """
    Get the device to use (CPU or CUDA).

    Args:
        no_cuda: If True, use CPU even if CUDA is available

    Returns:
        torch.device: Device to use
    """
    if not torch.cuda.is_available() or no_cuda:
        print("Using CPU")
        return torch.device("cpu")
    else:
        print("Using CUDA")
        return torch.device("cuda")


def colored_text(r: int, g: int, b: int, text: str) -> str:
    """
    Return colored text for terminal output.

    Args:
        r: Red component (0-255)
        g: Green component (0-255)
        b: Blue component (0-255)
        text: Text to color

    Returns:
        Colored text string
    """
    return f"\033[38;2;{r};{g};{b}m{text}\033[0m"


def normalize_saliency(saliency: Tensor, positive_only: bool = True) -> Tensor:
    """
    Normalize saliency map to [0, 1] range.

    Args:
        saliency: Saliency map tensor
        positive_only: If True, set negative values to zero

    Returns:
        Normalized saliency map
    """
    shape = saliency.shape
    saliency = saliency.view(saliency.size(0), -1)

    if positive_only:
        saliency[saliency < 0] = 0.

    # Min-max normalization
    saliency_min = saliency.min(1, keepdim=True)[0]
    saliency_max = saliency.max(1, keepdim=True)[0]

    # Avoid division by zero
    denominator = saliency_max - saliency_min
    denominator[denominator == 0] = 1.0

    saliency = (saliency - saliency_min) / denominator

    return saliency.view(shape)


def compute_accuracy(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None
) -> float:
    """
    Compute classification accuracy.

    Args:
        model: Neural network model
        dataloader: Data loader
        device: Device to use
        max_batches: Maximum number of batches to evaluate (None for all)

    Returns:
        Classification accuracy (0.0-1.0)
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            if max_batches is not None and i >= max_batches:
                break

            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    model.train()

    return accuracy


def get_last_conv_layer(model: nn.Module) -> nn.Module:
    """
    Find the last convolutional layer in a model.

    Args:
        model: Neural network model

    Returns:
        Last convolutional layer

    Raises:
        Exception: If no convolutional layer is found
    """
    conv_layers = []

    # Recursively find all conv layers
    def find_conv_layers(module):
        for child in module.children():
            if isinstance(child, nn.Conv2d):
                conv_layers.append(child)
            find_conv_layers(child)

    find_conv_layers(model)

    if not conv_layers:
        raise Exception("Model has no convolutional layers")

    return conv_layers[-1]


def simple_plot(data: np.ndarray, grid_size: Tuple[int, int] = (10, 10), figsize: Tuple[int, int] = (10, 10)) -> plt.Figure:
    """
    Create a simple grid plot of images.

    Args:
        data: Image data (N, H, W)
        grid_size: Grid dimensions (rows, cols)
        figsize: Figure size

    Returns:
        matplotlib figure
    """
    fig, axs = plt.subplots(*grid_size, figsize=figsize)

    for i, ax in enumerate(axs.reshape(-1)):
        if i < len(data):
            ax.imshow(data[i], cmap='gray')
        ax.axis('off')

    plt.tight_layout()
    return fig
