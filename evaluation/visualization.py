"""
Visualization utilities for CNN-LSX.

This module contains functions for visualizing explanations, model predictions,
and other aspects of CNN-LSX training and evaluation.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from typing import Tuple, List, Dict, Any, Optional, Callable

# Import constants for visualization
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


class Visualizer:
    """
    Visualization utility for CNN-LSX.

    This class provides methods for visualizing inputs, explanations,
    and model predictions, and for logging these visualizations to TensorBoard.
    """

    def __init__(
        self,
        writer: Optional[SummaryWriter] = None,
        dataset_mean: float = MNIST_MEAN,
        dataset_std: float = MNIST_STD
    ):
        """
        Initialize the visualizer.

        Args:
            writer: TensorBoard SummaryWriter (optional)
            dataset_mean: Mean value for dataset normalization
            dataset_std: Standard deviation for dataset normalization
        """
        self.writer = writer
        self.dataset_mean = dataset_mean
        self.dataset_std = dataset_std

    def visualize_inputs(
        self,
        inputs: Tensor,
        caption: str = "Input Images",
        global_step: Optional[int] = None
    ) -> None:
        """
        Visualize input images.

        Args:
            inputs: Input tensor of shape (batch_size, channels, height, width)
            caption: Caption for the visualization
            global_step: Global step for TensorBoard logging
        """
        # Un-normalize inputs for visualization
        normalized_inputs = self._un_normalize(inputs)

        # Create grid of images
        grid = torchvision.utils.make_grid(
            normalized_inputs, nrow=8, padding=2, normalize=True)

        # Log to TensorBoard if writer is available
        if self.writer is not None:
            self.writer.add_image(caption, grid, global_step=global_step)

    def visualize_explanations(
        self,
        explanations: Tensor,
        caption: str = "Explanations",
        global_step: Optional[int] = None
    ) -> None:
        """
        Visualize explanation heatmaps.

        Args:
            explanations: Explanation tensor of shape (batch_size, channels, height, width)
            caption: Caption for the visualization
            global_step: Global step for TensorBoard logging
        """
        # Create grid of explanations
        grid = torchvision.utils.make_grid(
            explanations, nrow=8, padding=2, normalize=True)

        # Log to TensorBoard if writer is available
        if self.writer is not None:
            self.writer.add_image(caption, grid, global_step=global_step)

    def visualize_overlay(
        self,
        inputs: Tensor,
        explanations: Tensor,
        caption: str = "Explanations Overlay",
        alpha: float = 0.7,
        colormap: str = "jet",
        global_step: Optional[int] = None
    ) -> None:
        """
        Visualize explanations overlaid on input images.

        Args:
            inputs: Input tensor of shape (batch_size, channels, height, width)
            explanations: Explanation tensor of shape (batch_size, channels, height, width)
            caption: Caption for the visualization
            alpha: Alpha value for the overlay
            colormap: Colormap for the explanations
            global_step: Global step for TensorBoard logging
        """
        # Un-normalize inputs for visualization
        normalized_inputs = self._un_normalize(inputs)

        # Convert to numpy for matplotlib
        inputs_np = normalized_inputs.detach().cpu().numpy()
        explanations_np = explanations.detach().cpu().numpy()

        batch_size = inputs.shape[0]
        height = inputs.shape[2]
        width = inputs.shape[3]

        # Create a figure to hold the overlay images
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        axes = axes.flatten()

        # Create overlay for a subset of images
        for i in range(min(batch_size, 8)):
            # Get a single image and explanation
            img = inputs_np[i, 0]  # Assuming grayscale images
            expl = explanations_np[i, 0]

            # Display the image
            axes[i].imshow(img, cmap='gray')

            # Overlay the explanation
            heatmap = axes[i].imshow(expl, cmap=colormap, alpha=alpha)
            axes[i].axis('off')

            # Add colorbar
            if i == 7:
                fig.colorbar(heatmap, ax=axes[i], fraction=0.046, pad=0.04)

        # Remove empty subplots
        for i in range(batch_size, 8):
            fig.delaxes(axes[i])

        fig.tight_layout()

        # Convert the figure to a tensor for TensorBoard
        canvas = fig.canvas
        canvas.draw()
        overlay_array = np.array(canvas.renderer.buffer_rgba())
        plt.close(fig)

        # Convert RGBA to RGB
        overlay_array = overlay_array[:, :, :3]

        # Transpose to PyTorch expected format (C, H, W)
        overlay_tensor = torch.from_numpy(
            overlay_array).permute(2, 0, 1).float() / 255.0

        # Log to TensorBoard if writer is available
        if self.writer is not None:
            self.writer.add_image(caption, overlay_tensor,
                                  global_step=global_step)

    def visualize_comparison(
        self,
        inputs_a: Tensor,
        explanations_a: Tensor,
        inputs_b: Tensor,
        explanations_b: Tensor,
        caption_a: str = "Model A",
        caption_b: str = "Model B",
        caption: str = "Comparison",
        global_step: Optional[int] = None
    ) -> None:
        """
        Visualize a comparison of explanations from two models.

        Args:
            inputs_a: Input tensor for model A
            explanations_a: Explanation tensor for model A
            inputs_b: Input tensor for model B
            explanations_b: Explanation tensor for model B
            caption_a: Caption for model A
            caption_b: Caption for model B
            caption: Overall caption
            global_step: Global step for TensorBoard logging
        """
        # Un-normalize inputs for visualization
        normalized_inputs_a = self._un_normalize(inputs_a)
        normalized_inputs_b = self._un_normalize(inputs_b)

        # Convert to numpy for matplotlib
        inputs_a_np = normalized_inputs_a.detach().cpu().numpy()
        explanations_a_np = explanations_a.detach().cpu().numpy()
        inputs_b_np = normalized_inputs_b.detach().cpu().numpy()
        explanations_b_np = explanations_b.detach().cpu().numpy()

        batch_size = min(inputs_a.shape[0], inputs_b.shape[0])

        # Create a figure for comparison
        fig, axes = plt.subplots(batch_size, 4, figsize=(12, 3 * batch_size))

        if batch_size == 1:
            axes = axes.reshape(1, -1)

        # Column titles
        cols = ["Input", "Explanation", "Input", "Explanation"]
        for ax, col in zip(axes[0], cols):
            ax.set_title(col)

        # Row titles
        for i in range(min(batch_size, 4)):
            axes[i, 0].set_ylabel(f"Example {i+1}")

            # Model A
            axes[i, 0].imshow(inputs_a_np[i, 0], cmap='gray')
            axes[i, 1].imshow(explanations_a_np[i, 0], cmap='jet')
            axes[i, 0].axis('off')
            axes[i, 1].axis('off')

            # Model B
            axes[i, 2].imshow(inputs_b_np[i, 0], cmap='gray')
            axes[i, 3].imshow(explanations_b_np[i, 0], cmap='jet')
            axes[i, 2].axis('off')
            axes[i, 3].axis('off')

        # Add super titles
        fig.text(0.25, 0.95, caption_a, ha='center', fontsize=14)
        fig.text(0.75, 0.95, caption_b, ha='center', fontsize=14)

        # Overall title
        fig.suptitle(caption, fontsize=16)

        fig.tight_layout(rect=[0, 0, 1, 0.95])

        # Convert the figure to a tensor for TensorBoard
        canvas = fig.canvas
        canvas.draw()
        comparison_array = np.array(canvas.renderer.buffer_rgba())
        plt.close(fig)

        # Convert RGBA to RGB
        comparison_array = comparison_array[:, :, :3]

        # Transpose to PyTorch expected format (C, H, W)
        comparison_tensor = torch.from_numpy(
            comparison_array).permute(2, 0, 1).float() / 255.0

        # Log to TensorBoard if writer is available
        if self.writer is not None:
            self.writer.add_image(
                caption, comparison_tensor, global_step=global_step)

    def _un_normalize(self, tensor: Tensor) -> Tensor:
        """
        Un-normalize a tensor using dataset mean and std.

        Args:
            tensor: Normalized tensor

        Returns:
            Un-normalized tensor
        """
        return tensor.mul(self.dataset_std).add(self.dataset_mean)

    def log_metrics(
        self,
        metrics: Dict[str, float],
        prefix: str = "",
        global_step: Optional[int] = None
    ) -> None:
        """
        Log metrics to TensorBoard.

        Args:
            metrics: Dictionary of metric names and values
            prefix: Prefix for metric names
            global_step: Global step for TensorBoard logging
        """
        if self.writer is not None:
            for name, value in metrics.items():
                metric_name = f"{prefix}/{name}" if prefix else name
                self.writer.add_scalar(
                    metric_name, value, global_step=global_step)
