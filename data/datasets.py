"""
Custom datasets for CNN-LSX.

This module contains custom dataset classes that extend PyTorch's datasets
with functionalities needed for CNN-LSX, such as normalization and
handling specific data transformations.
"""

import os
import torch
import numpy as np
import torchvision
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.datasets import MNIST
from typing import Tuple, Optional, List, Dict, Any, Union, Callable


class NormalizedMNIST(MNIST):
    """
    MNIST dataset with normalization and subset selection.

    This dataset extends PyTorch's MNIST dataset with:
    - Automatic normalization
    - Support for few-shot learning
    - Support for critic samples
    """

    # MNIST mean and standard deviation for normalization
    MEAN: float = 0.1307
    STD: float = 0.3081

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        few_shot_percent: float = 1.0,
        critic_samples: Optional[int] = None
    ):
        """
        Initialize NormalizedMNIST dataset.

        Args:
            root: Root directory of dataset
            train: If True, creates dataset from training set, otherwise from test set
            transform: A function/transform for inputs
            target_transform: A function/transform for targets
            download: If True, downloads the dataset
            few_shot_percent: Percentage of data to use (for small-data experiments)
            critic_samples: Number of samples to use for critic (if None, use all)
        """
        super().__init__(root, train, transform, target_transform, download)

        # Calculate how many samples to use
        self.few_shot_percent = few_shot_percent
        n_samples = int(self.few_shot_percent * len(self.targets))

        # Calculate class weights for loss function
        # self.class_weights = torch.bincount(self.targets).float()
        # self.class_weights = self.class_weights.sum() / (self.class_weights *
        #                                                  len(self.class_weights))

        # Take only the subset of data
        self.data = self.data[:n_samples]
        self.targets = self.targets[:n_samples]

        # Scale data to [0,1] and add channel dimension
        self.data = self.data.unsqueeze(1).float().div(255)

        # Normalize with MNIST mean and std
        self.data = self.data.sub_(self.MEAN).div_(self.STD)

        # If critic_samples is specified, take only that many samples
        if critic_samples is not None and critic_samples > 0:
            assert n_samples >= critic_samples, "Not enough samples for critic"
            indices = np.random.choice(
                len(self.targets), critic_samples, replace=False)
            self.data = self.data[indices]
            self.targets = self.targets[indices]

    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        """
        Get a sample from the dataset.

        Args:
            index: Index of the sample

        Returns:
            Tuple of (image, target)
        """
        # Skip the usual transform pipeline since we've already processed the data
        img, target = self.data[index], self.targets[index]

        # Apply transforms if specified
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class DecoyMNIST(NormalizedMNIST):
    """
    DecoyMNIST dataset with normalization and subset selection.

    DecoyMNIST adds artificial correlations (decoys) to MNIST digits
    to test the ability of models to focus on relevant features.
    """

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        few_shot_percent: float = 1.0,
        critic_samples: Optional[int] = None
    ):
        """
        Initialize DecoyMNIST dataset.

        Args:
            root: Root directory of dataset
            train: If True, creates dataset from training set, otherwise from test set
            transform: A function/transform for inputs
            target_transform: A function/transform for targets
            download: If True, downloads the dataset
            few_shot_percent: Percentage of data to use (for small-data experiments)
            critic_samples: Number of samples to use for critic (if None, use all)
        """
        # Initialize the parent class
        super().__init__(root, train, transform, target_transform, download,
                         few_shot_percent, critic_samples)

        # Add decoys to the images
        self._add_decoys()

    def _add_decoys(self) -> None:
        """
        Add decoy features to the images.

        This method adds class-specific decoys (e.g., patterns in corners)
        to create artificial correlations with the target class.
        """
        # Get a copy of the data
        data_with_decoys = self.data.clone()

        # For each image, add a decoy based on its class
        for i in range(len(self.data)):
            digit = self.targets[i].item()

            # Example decoy: Add a pattern in the top-left corner based on digit value
            # In a real implementation, this would be more sophisticated
            intensity = 1.0 - 0.1 * digit  # Intensity based on digit

            # Add the decoy to the top-left 4x4 corner
            data_with_decoys[i, :, :4, :4] = intensity

        # Replace the data with the decoy version
        self.data = data_with_decoys


class ExplanationDataset(Dataset):
    """
    Dataset of images paired with their explanations.

    This dataset is used for finetuning models with pre-computed explanations.
    """

    def __init__(
        self,
        inputs: List[Tensor],
        explanations: List[Tensor],
        labels: List[Tensor]
    ):
        """
        Initialize ExplanationDataset.

        Args:
            inputs: List of input tensors
            explanations: List of explanation tensors
            labels: List of label tensors
        """
        # Check that all lists have the same length
        assert len(inputs) == len(explanations) == len(
            labels), "Inputs, explanations, and labels must have the same length"

        self.inputs = inputs
        self.explanations = explanations
        self.labels = labels

    def __len__(self) -> int:
        """Get the number of samples in the dataset."""
        return len(self.inputs)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Get a sample from the dataset.

        Args:
            index: Index of the sample

        Returns:
            Tuple of (input, explanation, label)
        """
        return self.inputs[index], self.explanations[index], self.labels[index]


# Dictionary of available datasets
DATASETS = {
    "mnist": NormalizedMNIST,
    "decoymnist": DecoyMNIST,
}


def get_dataset(
    name: str,
    root: str = "data",
    **kwargs
) -> Dataset:
    """
    Get a dataset by name.

    Args:
        name: Name of the dataset
        root: Root directory for the dataset
        **kwargs: Additional arguments to pass to the dataset constructor

    Returns:
        Dataset instance

    Raises:
        ValueError: If the dataset name is not recognized
    """
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}")

    return DATASETS[name](root=root, **kwargs)
