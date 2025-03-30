"""
Learner model for CNN-LSX.

The Learner is the primary model that makes predictions and generates explanations.
It wraps a neural network and provides methods for generating and managing explanations.
"""

from explanations.methods import get_explanation, process_explanation
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from typing import Tuple, Optional, List, Dict, Any, Callable

# Import explanation methods
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Learner:
    """
    The Learner model makes predictions and generates explanations.

    It wraps a neural network and provides methods for generating explanations,
    saving and loading model states, and other utilities.
    """

    def __init__(
        self,
        classifier: nn.Module,
        device: torch.device,
        explanation_mode: str = "input_x_gradient",
        optimizer_type: str = "adadelta",
        learning_rate: float = 0.01
    ):
        """
        Initialize a Learner model.

        Args:
            classifier: Neural network model for classification
            device: Device to use
            explanation_mode: Explanation method to use
            optimizer_type: Type of optimizer to use
            learning_rate: Learning rate for the optimizer
        """
        self.classifier = classifier.to(device)
        self.device = device
        self.explanation_mode = explanation_mode
        self.optimizer_type = optimizer_type
        self.optimizer = None

        # Initialize optimizer
        self.initialize_optimizer(learning_rate)

    def initialize_optimizer(self, learning_rate: float) -> None:
        """
        Initialize the optimizer for the classifier.

        Args:
            learning_rate: Learning rate for the optimizer

        Raises:
            ValueError: If optimizer_type is not recognized
        """
        if self.optimizer_type == "adadelta":
            self.optimizer = optim.Adadelta(
                self.classifier.parameters(), lr=learning_rate)
        elif self.optimizer_type == "adam":
            self.optimizer = optim.Adam(
                self.classifier.parameters(), lr=learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_type}")

    def get_explanation_batch(self, inputs: Tensor, labels: Tensor) -> Tensor:
        """
        Generate explanations for a batch of inputs.

        Args:
            inputs: Input tensors of shape (batch_size, 1, height, width)
            labels: Target labels of shape (batch_size)

        Returns:
            Explanation tensors of shape (batch_size, 1, height, width)
        """
        # Generate raw explanations
        explanations = get_explanation(
            model=self.classifier,
            inputs=inputs,
            labels=labels,
            explanation_mode=self.explanation_mode,
            device=self.device
        )

        # Process explanations
        processed_explanations = process_explanation(
            explanation=explanations,
            clip_negative=True,
            normalize=True
        )

        return processed_explanations

    def get_labeled_explanation_batches(self, dataloader: torch.utils.data.DataLoader) -> List[Tuple[Tensor, Tensor]]:
        """
        Generate explanations for all batches in a dataloader, paired with their labels.

        Args:
            dataloader: DataLoader containing input-label pairs

        Returns:
            List of (explanation, label) pairs
        """
        explanation_label_pairs = []

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            explanations = self.get_explanation_batch(inputs, labels)
            explanation_label_pairs.append((explanations, labels))

        return explanation_label_pairs

    def get_detached_explanations(self, dataloader: torch.utils.data.DataLoader) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
        """
        Generate explanations for all batches in a dataloader and detach them from the graph.

        Args:
            dataloader: DataLoader containing input-label pairs

        Returns:
            Tuple of (inputs, explanations, labels) as detached CPU tensors
        """
        inputs_all = []
        explanations_all = []
        labels_all = []

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            explanations = self.get_explanation_batch(inputs, labels)

            # Detach and move to CPU
            inputs_all.append(inputs.detach().cpu())
            explanations_all.append(explanations.detach().cpu())
            labels_all.append(labels.detach().cpu())

        return inputs_all, explanations_all, labels_all

    def save_state(self, path: str, epoch: int, loss: float) -> None:
        """
        Save the model state to a file.

        Args:
            path: Path to save the model
            epoch: Current epoch
            loss: Current loss
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)

        state_dict = {
            'epoch': epoch,
            'model_state_dict': self.classifier.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'loss': loss
        }

        torch.save(state_dict, path)
        print(f"Model saved to {path}")

    def load_state(self, path: str) -> None:
        """
        Load the model state from a file.

        Args:
            path: Path to the saved model
        """
        # Make sure path exists
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Model file not found: {path}")

        # Load the state dict
        state_dict = torch.load(path, map_location=self.device)

        # Load model state
        self.classifier.load_state_dict(state_dict['model_state_dict'])

        # Load optimizer state if available and optimizer exists
        if 'optimizer_state_dict' in state_dict and state_dict['optimizer_state_dict'] and self.optimizer:
            self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])

        # Set model to training mode
        self.classifier.train()

        print(f"Model loaded from {path}")

    def predict(self, inputs: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Make predictions for the inputs.

        Args:
            inputs: Input tensors of shape (batch_size, 1, height, width)

        Returns:
            Tuple of (probabilities, predicted_classes)
        """
        self.classifier.eval()
        with torch.no_grad():
            outputs = self.classifier(inputs)
            # Convert log softmax to probabilities
            probabilities = torch.exp(outputs)
            _, predicted = torch.max(outputs, 1)
        self.classifier.train()

        return probabilities, predicted
