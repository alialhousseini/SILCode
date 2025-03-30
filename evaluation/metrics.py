"""
Evaluation metrics for CNN-LSX explanations.

This module contains implementations of various metrics for evaluating
the quality and faithfulness of explanations.
"""

from utils.common import normalize_saliency
import torch
import numpy as np
from torch import Tensor
import torch.nn as nn
from typing import Tuple, List, Dict, Any, Optional, Union, Callable

# Import utility functions
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def compute_comprehensiveness(
    model: nn.Module,
    inputs: Tensor,
    explanations: Tensor,
    labels: Tensor,
    percentiles: List[float] = [1, 5, 10, 20, 50],
    replacement_value: Union[str, float] = "median"
) -> Tensor:
    """
    Compute comprehensiveness metric for explanations.

    Comprehensiveness measures how much the prediction changes when the most
    important features (according to the explanation) are removed.

    Args:
        model: Neural network model
        inputs: Input tensors
        explanations: Explanation tensors
        labels: Target labels
        percentiles: Percentiles of most important features to remove
        replacement_value: Value to replace removed features with

    Returns:
        Tensor of comprehensiveness scores for each percentile
    """
    model.eval()

    batch_size = inputs.shape[0]
    input_shape = inputs.shape[2:]

    # Flatten explanations for sorting
    explanations_flat = explanations.view(batch_size, -1)

    # Normalize explanations
    explanations_norm = normalize_saliency(explanations_flat)

    # Original predictions
    with torch.no_grad():
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, dim=1)

    # Get original probabilities for predicted classes
    orig_probs = probs[torch.arange(batch_size), preds]

    # Determine replacement value
    if replacement_value == "median":
        replace_val = torch.median(inputs)
    elif replacement_value == "mean":
        replace_val = torch.mean(inputs)
    elif replacement_value == "min":
        replace_val = torch.min(inputs)
    elif replacement_value == "zero":
        replace_val = 0.0
    else:
        replace_val = float(replacement_value)

    # Compute comprehensiveness for each percentile
    comp_scores = []

    for p in percentiles:
        # Threshold for this percentile
        threshold = 1.0 - p/100.0

        # Create modified inputs
        modified_inputs = inputs.clone().view(batch_size, -1)

        # Find indices of values to remove (above threshold)
        indices = torch.nonzero(explanations_norm >= threshold)

        # Replace these values
        modified_inputs[indices[:, 0], indices[:, 1]] = replace_val

        # Reshape back
        modified_inputs = modified_inputs.view(batch_size, *inputs.shape[1:])

        # Get new predictions
        with torch.no_grad():
            outputs = model(modified_inputs)
            probs = torch.softmax(outputs, dim=1)

        # Get new probabilities for originally predicted classes
        new_probs = probs[torch.arange(batch_size), preds]

        # Compute drop in probability
        comp_score = torch.mean(orig_probs - new_probs)
        comp_scores.append(comp_score.item())

    model.train()
    return torch.tensor(comp_scores)


def compute_sufficiency(
    model: nn.Module,
    inputs: Tensor,
    explanations: Tensor,
    labels: Tensor,
    percentiles: List[float] = [1, 5, 10, 20, 50],
    replacement_value: Union[str, float] = "median"
) -> Tensor:
    """
    Compute sufficiency metric for explanations.

    Sufficiency measures how much the prediction changes when only the most
    important features (according to the explanation) are retained.

    Args:
        model: Neural network model
        inputs: Input tensors
        explanations: Explanation tensors
        labels: Target labels
        percentiles: Percentiles of most important features to retain
        replacement_value: Value to replace non-important features with

    Returns:
        Tensor of sufficiency scores for each percentile
    """
    model.eval()

    batch_size = inputs.shape[0]
    input_shape = inputs.shape[2:]

    # Flatten explanations for sorting
    explanations_flat = explanations.view(batch_size, -1)

    # Normalize explanations
    explanations_norm = normalize_saliency(explanations_flat)

    # Original predictions
    with torch.no_grad():
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, dim=1)

    # Get original probabilities for predicted classes
    orig_probs = probs[torch.arange(batch_size), preds]

    # Determine replacement value
    if replacement_value == "median":
        replace_val = torch.median(inputs)
    elif replacement_value == "mean":
        replace_val = torch.mean(inputs)
    elif replacement_value == "min":
        replace_val = torch.min(inputs)
    elif replacement_value == "zero":
        replace_val = 0.0
    else:
        replace_val = float(replacement_value)

    # Compute sufficiency for each percentile
    suff_scores = []

    for p in percentiles:
        # Threshold for this percentile
        threshold = 1.0 - p/100.0

        # Create modified inputs
        modified_inputs = inputs.clone().view(batch_size, -1)

        # Find indices of values to keep (above threshold)
        indices = torch.nonzero(explanations_norm < threshold)

        # Replace non-important values
        modified_inputs[indices[:, 0], indices[:, 1]] = replace_val

        # Reshape back
        modified_inputs = modified_inputs.view(batch_size, *inputs.shape[1:])

        # Get new predictions
        with torch.no_grad():
            outputs = model(modified_inputs)
            probs = torch.softmax(outputs, dim=1)

        # Get new probabilities for originally predicted classes
        new_probs = probs[torch.arange(batch_size), preds]

        # Compute drop in probability (higher is better for sufficiency)
        suff_score = torch.mean(orig_probs - new_probs)
        suff_scores.append(suff_score.item())

    model.train()
    return torch.tensor(suff_scores)


def compute_faithfulness_correlation(
    model: nn.Module,
    inputs: Tensor,
    explanations: Tensor,
    labels: Tensor,
    n_samples: int = 10
) -> float:
    """
    Compute faithfulness correlation for explanations.

    This metric measures the correlation between feature importance and
    the effect of perturbing those features on the prediction.

    Args:
        model: Neural network model
        inputs: Input tensors
        explanations: Explanation tensors
        labels: Target labels
        n_samples: Number of perturbation samples

    Returns:
        Faithfulness correlation score
    """
    model.eval()

    batch_size = inputs.shape[0]
    corr_scores = []

    for i in range(batch_size):
        input_i = inputs[i:i+1]
        expl_i = explanations[i:i+1].view(-1)
        label_i = labels[i:i+1]

        # Original prediction
        with torch.no_grad():
            outputs = model(input_i)
            orig_prob = torch.softmax(outputs, dim=1)[0, label_i[0]]

        # Compute feature importance and perturbation effect vectors
        importances = []
        effects = []

        # Create a vector of indices sorted by explanation magnitude
        indices = torch.argsort(expl_i, descending=True)

        # Sample n_samples indices
        sampled_indices = indices[:n_samples]

        for j in sampled_indices:
            # Feature importance
            importance = expl_i[j].item()
            importances.append(importance)

            # Perturb feature and compute effect
            perturbed_input = input_i.clone().view(-1)
            # Replace with mean value
            perturbed_input[j] = torch.mean(input_i)
            perturbed_input = perturbed_input.view_as(input_i)

            # Compute new prediction
            with torch.no_grad():
                outputs = model(perturbed_input)
                new_prob = torch.softmax(outputs, dim=1)[0, label_i[0]]

            # Effect is the drop in probability
            effect = (orig_prob - new_prob).item()
            effects.append(effect)

        # Compute correlation
        if len(importances) > 1:  # Need at least 2 points for correlation
            corr = np.corrcoef(importances, effects)[0, 1]
            if not np.isnan(corr):
                corr_scores.append(corr)

    model.train()
    # Average correlation across batch
    return np.mean(corr_scores) if corr_scores else 0.0


def evaluate_explanations(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    explanation_function: Callable,
    device: torch.device,
    max_batches: Optional[int] = None
) -> Dict[str, float]:
    """
    Comprehensive evaluation of explanations.

    This function computes multiple metrics to evaluate the quality of
    explanations generated by the given explanation function.

    Args:
        model: Neural network model
        dataloader: DataLoader for test data
        explanation_function: Function that generates explanations
        device: Device to use
        max_batches: Maximum number of batches to evaluate

    Returns:
        Dictionary of metric names and values
    """
    # Initialize metrics
    comp_scores = []
    suff_scores = []
    faith_corr_scores = []

    # Process batches
    for i, (inputs, labels) in enumerate(dataloader):
        if max_batches is not None and i >= max_batches:
            break

        inputs, labels = inputs.to(device), labels.to(device)

        # Generate explanations
        explanations = explanation_function(inputs, labels)

        # Compute metrics
        comp = compute_comprehensiveness(model, inputs, explanations, labels)
        suff = compute_sufficiency(model, inputs, explanations, labels)
        faith_corr = compute_faithfulness_correlation(
            model, inputs, explanations, labels)

        comp_scores.append(comp)
        suff_scores.append(suff)
        faith_corr_scores.append(faith_corr)

    # Average scores
    avg_comp = torch.stack(comp_scores).mean(dim=0)
    avg_suff = torch.stack(suff_scores).mean(dim=0)
    avg_faith_corr = np.mean(faith_corr_scores)

    # Create results dictionary
    percentiles = [1, 5, 10, 20, 50]
    results = {
        "faithfulness_correlation": avg_faith_corr
    }

    for i, p in enumerate(percentiles):
        results[f"comprehensiveness_{p}"] = avg_comp[i].item()
        results[f"sufficiency_{p}"] = avg_suff[i].item()

    # Average across percentiles
    results["comprehensiveness_avg"] = avg_comp.mean().item()
    results["sufficiency_avg"] = avg_suff.mean().item()

    return results
