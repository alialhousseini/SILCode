"""
Explanation methods for CNN-LSX.

This module contains implementations of various explanation methods
that generate saliency maps for neural network predictions.
"""

from utils.common import normalize_saliency, get_last_conv_layer
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from captum.attr import InputXGradient, IntegratedGradients, LayerGradCam, LayerAttribution
from typing import Tuple, Optional, Union, List, Dict, Any

# Import utility functions
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def input_gradient(
    model: nn.Module,
    inputs: Tensor,
    labels: Tensor,
    device: torch.device
) -> Tensor:
    """
    Compute input × gradient explanation.

    This method computes the gradient of the output with respect to the input,
    and multiplies it elementwise with the input to generate the explanation.

    Args:
        model: Neural network model
        inputs: Input tensor of shape (batch_size, 1, height, width)
        labels: Target labels of shape (batch_size)
        device: Device to use

    Returns:
        Explanation tensor of shape (batch_size, 1, height, width)
    """
    # Use captum's implementation of InputXGradient
    inputs = inputs.to(device).requires_grad_(True)

    # Create the attributor
    input_x_gradient = InputXGradient(model)

    # Compute attributions
    attributions = input_x_gradient.attribute(inputs=inputs, target=labels)

    return attributions


def integrated_gradient(
    model: nn.Module,
    inputs: Tensor,
    labels: Tensor,
    device: torch.device,
    n_steps: int = 50,
    baseline: Optional[Tensor] = None
) -> Tensor:
    """
    Compute integrated gradients explanation.

    This method integrates the gradient of the output with respect to the input
    along a path from a baseline to the input, providing a more robust explanation.

    Args:
        model: Neural network model
        inputs: Input tensor of shape (batch_size, 1, height, width)
        labels: Target labels of shape (batch_size)
        device: Device to use
        n_steps: Number of steps for the integration
        baseline: Baseline tensor (default: all zeros)

    Returns:
        Explanation tensor of shape (batch_size, 1, height, width)
    """
    inputs = inputs.to(device).requires_grad_(True)

    # Create default baseline if none provided
    if baseline is None:
        baseline = torch.zeros_like(inputs).to(device)

    # Create the attributor
    integrated_gradients = IntegratedGradients(model)

    # Compute attributions
    attributions = integrated_gradients.attribute(
        inputs=inputs,
        target=labels,
        baselines=baseline,
        n_steps=n_steps
    )

    return attributions.float()


def gradcam(
    model: nn.Module,
    inputs: Tensor,
    labels: Tensor,
    device: torch.device,
    target_layer: Optional[nn.Module] = None
) -> Tensor:
    """
    Compute GradCAM explanation.

    GradCAM uses the gradients flowing into the final convolutional layer to
    produce a coarse localization map highlighting important regions.

    Args:
        model: Neural network model
        inputs: Input tensor of shape (batch_size, 1, height, width)
        labels: Target labels of shape (batch_size)
        device: Device to use
        target_layer: Target layer for GradCAM (default: last conv layer)

    Returns:
        Explanation tensor of shape (batch_size, 1, height, width)
    """
    inputs = inputs.to(device).requires_grad_(True)

    # Find the target layer if not specified
    if target_layer is None:
        target_layer = get_last_conv_layer(model)

    # Create the attributor
    grad_cam = LayerGradCam(model, target_layer)

    # Compute attributions
    attributions = grad_cam.attribute(
        inputs=inputs, target=labels, relu_attributions=False)

    # Normalize the attributions
    norm_attrs = normalize_saliency(attributions)

    # Interpolate to input size
    return LayerAttribution.interpolate(norm_attrs, (inputs.shape[2], inputs.shape[3]))


def get_explanation(
    model: nn.Module,
    inputs: Tensor,
    labels: Tensor,
    explanation_mode: str,
    device: torch.device
) -> Tensor:
    """
    Generate explanations using the specified method.

    Args:
        model: Neural network model
        inputs: Input tensor of shape (batch_size, 1, height, width)
        labels: Target labels of shape (batch_size)
        explanation_mode: Explanation method to use
        device: Device to use

    Returns:
        Explanation tensor of shape (batch_size, 1, height, width)

    Raises:
        ValueError: If explanation_mode is not recognized
    """
    if explanation_mode == "input_x_gradient":
        return input_gradient(model, inputs, labels, device)
    elif explanation_mode == "integrated_gradient":
        return integrated_gradient(model, inputs, labels, device)
    elif explanation_mode == "input_x_integrated_gradient":
        ig_attrs = integrated_gradient(model, inputs, labels, device)
        return ig_attrs * inputs
    elif explanation_mode == "gradcam":
        return gradcam(model, inputs, labels, device)
    elif explanation_mode == "input":
        return inputs.clone()
    else:
        raise ValueError(f"Unknown explanation mode: {explanation_mode}")


def process_explanation(
    explanation: Tensor,
    clip_negative: bool = True,
    normalize: bool = True
) -> Tensor:
    """
    Process the raw explanation to make it more interpretable.

    Args:
        explanation: Raw explanation tensor
        clip_negative: Whether to clip negative values
        normalize: Whether to normalize the explanation

    Returns:
        Processed explanation tensor
    """
    # Clone to avoid modifying the original tensor
    processed = explanation.clone()

    # Clip negative values if requested
    if clip_negative:
        processed[processed < 0] = 0

    # Normalize if requested
    if normalize:
        processed = normalize_saliency(processed)

    return processed
