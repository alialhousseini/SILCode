"""
Neural network architectures for CNN-LSX.

This module defines various neural network architectures that can be used as
the learner or critic models in the CNN-LSX framework.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Optional


class Net(nn.Module):
    """
    Standard CNN architecture used as the primary model in CNN-LSX.

    This model has a structure similar to LeNet but adapted for MNIST.
    It also stores the intermediate representations for possible use.
    """

    def __init__(self, num_classes: int = 10):
        """
        Initialize the network.

        Args:
            num_classes: Number of output classes
        """
        super().__init__()
        # First convolutional block
        self.conv1 = nn.Conv2d(1, 32, 3, 1)

        # Second convolutional block
        self.conv2 = nn.Conv2d(32, 64, 3, 1)

        # Dropout for regularization
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

        # Fully connected layers
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, num_classes)

        # Store intermediate representation
        self.enc = None

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, 1, height, width)

        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # First convolutional block
        x = self.conv1(x)
        x = F.relu(x)

        # Second convolutional block
        x = self.conv2(x)
        x = F.relu(x)
        x = F.avg_pool2d(x, 2)
        # x = self.dropout1(x)  # Dropout disabled by default

        # Flatten and store intermediate representation
        x = torch.flatten(x, 1)
        self.enc = x

        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        # x = self.dropout2(x)  # Dropout disabled by default
        x = self.fc2(x)

        # Log softmax for numerical stability
        return F.log_softmax(x, dim=1)

    @property
    def n_parameters(self) -> int:
        """Get the total number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())


class SimpleConvNet(nn.Module):
    """
    A simpler CNN architecture that can be used as an alternative model.
    """

    def __init__(self, num_classes: int = 10):
        """
        Initialize the network.

        Args:
            num_classes: Number of output classes
        """
        super().__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.max_pool1 = nn.MaxPool2d(2, 2)
        # Named for GradCAM compatibility
        self.last_conv = nn.Conv2d(20, 50, 5, 1)
        self.max_pool2 = nn.MaxPool2d(2, 2)

        # Fully connected layers
        self.fc1 = nn.Linear(4*4*50, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, 1, height, width)

        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # First convolutional block
        x = F.relu(self.conv1(x))
        x = self.max_pool1(x)

        # Second convolutional block
        x = F.relu(self.last_conv(x))
        x = self.max_pool2(x)

        # Flatten
        x = x.view(-1, 4*4*50)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)

    @property
    def n_parameters(self) -> int:
        """Get the total number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())


class MLP(nn.Module):
    """
    Multi-layer perceptron (fully connected network).

    This can be used for simpler datasets or as the critic model.
    """

    def __init__(self, num_classes: int = 10):
        """
        Initialize the network.

        Args:
            num_classes: Number of output classes
        """
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, 1, height, width)

        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Flatten input
        x = x.view(-1, 28 * 28)

        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return F.log_softmax(x, dim=1)

    @property
    def n_parameters(self) -> int:
        """Get the total number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())


def get_network(model_name: str, num_classes: int = 10) -> nn.Module:
    """
    Factory function to create a network based on its name.

    Args:
        model_name: Name of the model architecture
        num_classes: Number of output classes

    Returns:
        Neural network model

    Raises:
        ValueError: If model_name is not recognized
    """
    if model_name == "Net1":
        return Net(num_classes)
    elif model_name == "SimpleConvNet":
        return SimpleConvNet(num_classes)
    elif model_name == "MLP":
        return MLP(num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
