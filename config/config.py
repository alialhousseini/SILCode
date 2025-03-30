"""
Configuration system for CNN-LSX.

This module defines a Configuration class that holds all parameters for training and evaluation.
It also provides utilities for loading and saving configurations.
"""

import argparse
import dataclasses
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Literal, List, Dict, Any, Union


@dataclass
class Config:
    """Configuration parameters for CNN-LSX."""

    # Training mode
    # Training mode: joint, pretrain_and_joint, etc.
    training_mode: str = "pretrain_and_joint"

    # General settings
    random_seed: int = 42                # Random seed for reproducibility
    dataset: str = "mnist"               # Dataset: mnist, decoymnist, etc.
    # Model architecture: Net1, SimpleConvNet, etc.
    model: str = "Net1"
    model_pt: Optional[str] = 'pt_models/'       # Path to pretrained model
    # Path to vanilla model for comparison
    vanilla_model_pt: Optional[str] = 'vanilla_models/'
    no_cuda: bool = False                # Disable CUDA

    # Training parameters
    batch_size: int = 64                # Batch size for training
    batch_size_critic: int = 32         # Batch size for critic
    test_batch_size: int = 64           # Batch size for testing
    learning_rate: float = 0.001          # Learning rate
    learning_rate_finetune: float = 0.001  # Learning rate for finetuning
    learning_rate_critic: float = 0.2    # Learning rate for critic
    pretrain_learning_rate: float = 0.05  # Learning rate for pretraining
    classification_loss_weight: float = 50.0  # Weight for classification loss
    explanation_loss_weight: float = 50.0  # Weight for explanation loss
    # Weight for explanation loss in finetuning
    explanation_loss_weight_finetune: float = 50.0
    optimizer: str = "adam"          # Optimizer: adadelta, adam
    explanation_mode: str = "input_x_gradient"  # Explanation method

    # Scheduler parameters
    # scheduler #TODO: add scheduler
    lr_scheduling: bool = False  # Use learning rate scheduling
    lr_scheduling_step: int = 5  # Step size for learning rate scheduling
    learning_rate_step: float = 0.7      # Learning rate decay factor

    # Dataset parameters
    # Number of critic batches (total_data_size = n_critic_batches * batch_size_critic)
    n_critic_batches: int = 68
    # Use separate critic set (True) or use training data for critic (False)
    sep_critic_set: bool = False         # Use separate critic set
    n_epochs: int = 15  # 40                 # Number of joint training epochs
    n_pretraining_epochs: int = 10       # Number of pretraining epochs
    n_finetuning_epochs: int = 15  # 50       # Number of finetuning epochs
    few_shot_train_percent: float = 0.5  # Percentage of training data to use
    # Percentage of test data to use (with respect to training data)
    few_shot_test_percent: float = 0.2

    # Logging parameters
    logging_disabled: bool = False       # Disable logging
    log_interval: int = 12             # Log interval
    log_interval_critic: int = 12         # Log interval for critic
    log_interval_pretraining: int = 12    # Log interval for pretraining
    # Log interval for accuracy (recommended = #batches)
    log_interval_accuracy: int = 64
    run_name: str = ""                   # Name for the run

    @property
    def joint_iterations(self) -> int:
        """Calculate number of joint training iterations."""
        return self.n_epochs * self.n_training_batches * self.n_critic_batches

    @property
    def pretraining_iterations(self) -> int:
        """Calculate number of pretraining iterations."""
        return self.n_pretraining_epochs * self.n_training_batches

    @property
    def finetuning_iterations(self) -> int:
        """Calculate number of finetuning iterations."""
        return self.n_finetuning_epochs * self.n_training_batches

    @property
    def n_iterations(self) -> int:
        """Calculate total number of iterations based on training mode."""
        if self.training_mode == 'joint' or self.training_mode == "pretrained":
            return self.joint_iterations
        elif self.training_mode == 'pretrain_and_joint':
            return self.pretraining_iterations + self.joint_iterations
        elif self.training_mode == 'pretrain_and_joint_and_finetuning':
            return self.pretraining_iterations + self.joint_iterations + self.finetuning_iterations
        elif self.training_mode == 'finetuning':
            return self.finetuning_iterations
        elif self.training_mode == 'only_classification':
            return self.pretraining_iterations
        else:
            raise ValueError(f"Invalid training mode: {self.training_mode}")


def parse_args() -> Config:
    """Parse command-line arguments and create a Config object."""
    parser = argparse.ArgumentParser(
        description="CNN-LSX: Training neural networks with learnable saliency explanations")

    # Training mode
    parser.add_argument("--training_mode", type=str, default="pretrain_and_joint",
                        choices=["joint", "pretrain_and_joint", "pretrain_and_joint_and_finetuning",
                                 "finetuning", "pretrained", "only_classification", "test",
                                 "faithfulness", "save_expls"],
                        help="Training mode")

    # General settings
    parser.add_argument("--random_seed", type=int,
                        default=42, help="Random seed")
    parser.add_argument("--dataset", type=str,
                        default="mnist", help="Dataset name")
    parser.add_argument("--model", type=str, default="Net1",
                        help="Model architecture")
    parser.add_argument("--model_pt", type=str, default=None,
                        help="Path to pretrained model")
    parser.add_argument("--vanilla_model_pt", type=str,
                        default=None, help="Path to vanilla model for comparison")
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA")

    # Training parameters
    parser.add_argument("--batch_size", type=int,
                        default=128, help="Batch size")
    parser.add_argument("--batch_size_critic", type=int,
                        default=128, help="Batch size for critic")
    parser.add_argument("--test_batch_size", type=int,
                        default=128, help="Test batch size")
    parser.add_argument("--learning_rate", type=float,
                        default=0.01, help="Learning rate")
    parser.add_argument("--learning_rate_finetune", type=float,
                        default=0.001, help="Learning rate for finetuning")
    parser.add_argument("--learning_rate_step", type=float,
                        default=0.7, help="Learning rate step")
    parser.add_argument("--learning_rate_critic", type=float,
                        default=0.2, help="Learning rate for critic")
    parser.add_argument("--pretrain_learning_rate", type=float,
                        default=0.05, help="Learning rate for pretraining")
    parser.add_argument("--classification_loss_weight", type=float,
                        default=50, help="Weight for classification loss")
    parser.add_argument("--explanation_loss_weight", type=float,
                        default=50, help="Weight for explanation loss")
    parser.add_argument("--explanation_loss_weight_finetune", type=float,
                        default=50, help="Weight for explanation loss in finetuning")
    parser.add_argument("--optimizer", type=str, default="adadelta",
                        choices=["adadelta", "adam"], help="Optimizer")
    parser.add_argument("--explanation_mode", type=str, default="input_x_gradient",
                        choices=["input_x_gradient", "integrated_gradient",
                                 "input_x_integrated_gradient", "gradcam", "input"],
                        help="Explanation method")

    # Dataset parameters
    parser.add_argument("--n_critic_batches", type=int,
                        default=64, help="Number of critic batches")
    parser.add_argument("--sep_critic_set", action="store_true",
                        help="Use separate critic set")
    parser.add_argument("--n_epochs", type=int, default=10,
                        help="Number of joint training epochs")
    parser.add_argument("--n_pretraining_epochs", type=int,
                        default=10, help="Number of pretraining epochs")
    parser.add_argument("--n_finetuning_epochs", type=int,
                        default=50, help="Number of finetuning epochs")
    parser.add_argument("--few_shot_train_percent", type=float,
                        default=0.5, help="Percentage of training data to use")
    parser.add_argument("--few_shot_test_percent", type=float,
                        default=0.2, help="Percentage of test data to use")

    # Logging parameters
    parser.add_argument("--logging_disabled",
                        action="store_true", help="Disable logging")
    parser.add_argument("--log_interval", type=int,
                        default=100, help="Log interval")
    parser.add_argument("--log_interval_critic", type=int,
                        default=100, help="Log interval for critic")
    parser.add_argument("--log_interval_pretraining", type=int,
                        default=100, help="Log interval for pretraining")
    parser.add_argument("--log_interval_accuracy", type=int,
                        default=100, help="Log interval for accuracy")
    parser.add_argument("--run_name", type=str, default="",
                        help="Name for the run")

    args = parser.parse_args()

    # Convert args to Config
    config_dict = vars(args)
    config = Config(**config_dict)

    return config


def save_config(config: Config, path: str) -> None:
    """Save configuration to a JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(dataclasses.asdict(config), f, indent=2)


def load_config(path: str) -> Config:
    """Load configuration from a JSON file."""
    with open(path, 'r') as f:
        config_dict = json.load(f)
    return Config(**config_dict)


def config_string(config: Config) -> str:
    """Create a string representation of the configuration for logging directories."""
    date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    lr_mode = "_sched" if getattr(config, 'lr_scheduling', False) else ""
    run_name = f"{config.run_name}_" if config.run_name else ""

    return (f"{run_name}"
            f"{config.model}_{config.explanation_mode}_"
            f"seed{config.random_seed}_dataset_{config.dataset}_"
            f"{config.training_mode}_cr{config.n_critic_batches}_"
            f"lr{config.learning_rate}{lr_mode}_"
            f"bs{config.batch_size}_ep{config.n_epochs}_p-ep{config.n_pretraining_epochs}_"
            f"lambda{config.explanation_loss_weight}_"
            f"lambdaft{config.explanation_loss_weight_finetune}_"
            f"lambdacls{config.classification_loss_weight}_"
            f"fs{config.few_shot_train_percent}_"
            f"{date_time}")
