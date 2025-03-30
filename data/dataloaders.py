"""
Data loading utilities for CNN-LSX.

This module provides functions for creating data loaders from datasets,
with support for splitting data into train, test, and critic sets.
"""

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from typing import Tuple, Optional, Dict, Any, List, NamedTuple

# Import datasets
from data.datasets import get_dataset, ExplanationDataset
from config.config import Config


class DataLoaders(NamedTuple):
    """
    Container for data loaders used in CNN-LSX.

    Attributes:
        train: DataLoader for training set
        critic: DataLoader for critic set
        test: DataLoader for test set
        visualization: DataLoader for visualization (small subset of test)
    """
    train: DataLoader
    critic: DataLoader
    test: DataLoader
    visualization: DataLoader


def create_dataloaders(
    config: Config,
) -> DataLoaders:
    """
    Create data loaders based on configuration.

    Args:
        config: Configuration object with dataset parameters

    Returns:
        DataLoaders object containing all necessary data loaders
    """
    # Create training dataset
    train_dataset = get_dataset(
        name=config.dataset,
        root="data",
        train=True,
        download=True,
        few_shot_percent=config.few_shot_train_percent,
        critic_samples=None
    )

    # Create critic dataset (either a subset of training or a separate dataset)
    critic_dataset = get_dataset(
        name=config.dataset,
        root="data",
        train=True,
        download=True,
        few_shot_percent=config.few_shot_train_percent,
        critic_samples=config.n_critic_batches * config.batch_size_critic
    )

    # Create test dataset
    test_dataset = get_dataset(
        name=config.dataset,
        root="data",
        train=False,
        download=True,
        few_shot_percent=config.few_shot_test_percent,
        critic_samples=None
    )

    # If using separate critic set, split the training set
    if config.sep_critic_set:
        n_critic_samples = config.n_critic_batches * config.batch_size_critic
        n_training_samples = len(train_dataset) - n_critic_samples
        assert n_training_samples > 0, "Not enough samples for training with separate critic"

        train_split = [n_training_samples, n_critic_samples]
        train_dataset, critic_dataset = random_split(
            train_dataset, train_split)

    # Print dataset info
    print(f"Dataset: {config.dataset}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Critic samples: {len(critic_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0
    )

    critic_loader = DataLoader(
        critic_dataset,
        batch_size=config.batch_size_critic,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.test_batch_size,
        shuffle=False,
        num_workers=0
    )

    # Create visualization dataset (small subset of test for visualization)
    viz_loader = create_visualization_loader(test_dataset, config.batch_size)

    # Update config with batch counts
    config.n_training_batches = len(train_loader)
    config.n_test_batches = len(test_loader)

    return DataLoaders(train_loader, critic_loader, test_loader, viz_loader)


def create_visualization_loader(
    dataset: Dataset,
    batch_size: int,
    samples_per_class: int = 4
) -> DataLoader:
    """
    Create a DataLoader for visualization purposes.

    This selects a small number of samples from each class for visualization.

    Args:
        dataset: Source dataset
        batch_size: Batch size
        samples_per_class: Number of samples to include per class

    Returns:
        DataLoader for visualization
    """
    # Get dataset targets
    try:
        targets = dataset.targets
    except AttributeError:
        # If dataset doesn't have targets attribute (e.g., if it's a Subset)
        try:
            targets = dataset.dataset.targets[dataset.indices]
        except:
            # Last resort, we'll create a simple loader with one batch
            return DataLoader(dataset, batch_size=min(10, len(dataset)), shuffle=True)

    # Find indices for each class
    indices = []
    classes = torch.unique(targets)

    for cls in classes:
        cls_indices = torch.where(targets == cls)[0][:samples_per_class]
        indices.extend(cls_indices.tolist())

    # Create subset and dataloader
    viz_dataset = Subset(dataset, indices)
    return DataLoader(viz_dataset, batch_size=len(viz_dataset), shuffle=False)


def create_explanation_loader(
    inputs: List[torch.Tensor],
    explanations: List[torch.Tensor],
    labels: List[torch.Tensor],
    batch_size: int
) -> DataLoader:
    """
    Create a DataLoader for input-explanation-label triplets.

    Used for finetuning stage where we use precomputed explanations.

    Args:
        inputs: List of input tensors
        explanations: List of explanation tensors
        labels: List of label tensors
        batch_size: Batch size

    Returns:
        DataLoader for explanation dataset
    """
    dataset = ExplanationDataset(inputs, explanations, labels)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
