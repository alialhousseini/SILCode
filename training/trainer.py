"""
Training pipeline for CNN-LSX.

This module contains the Trainer class that implements the full training
pipeline for CNN-LSX, including pretraining, joint training, and finetuning.
"""

from utils.common import compute_accuracy, colored_text
from evaluation.visualization import Visualizer
from data.dataloaders import DataLoaders, create_explanation_loader
from models.networks import get_network
from models.critic import Critic
from models.learner import Learner
import os
import time
from typing import Tuple, Dict, List, Optional, Any, Union, Callable

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR

# Import from other modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Trainer:
    """
    Trainer for CNN-LSX models.

    This class implements the full CNN-LSX training pipeline, including
    pretraining, joint training with a critic, and finetuning.
    """

    def __init__(
        self,
        config,
        dataloaders: DataLoaders,
        device: torch.device,
        writer: Optional[SummaryWriter] = None
    ):
        """
        Initialize the trainer.

        Args:
            config: Configuration object
            dataloaders: DataLoaders object containing all data loaders
            device: Device to use for training
            writer: TensorBoard SummaryWriter (optional)
        """
        self.config = config
        self.dataloaders = dataloaders
        self.device = device
        self.writer = writer

        # Create the learner model
        num_classes = self._get_num_classes()
        learner_network = get_network(config.model, num_classes)
        self.learner = Learner(
            classifier=learner_network,
            device=device,
            explanation_mode=config.explanation_mode,
            optimizer_type=config.optimizer,
            learning_rate=config.learning_rate
        )

        # Create the visualizer
        self.visualizer = Visualizer(writer=writer)

        # Initialize training state
        self.global_step = 0
        self.current_epoch = 0

        # Print model information
        print(f"Model: {config.model}")
        print(f"Parameters: {learner_network.n_parameters:,}")
        print(f"Explanation mode: {config.explanation_mode}")
        print(f"Device: {device}")

    def pretrain(self) -> Tuple[float, float]:
        """
        Pretrain the learner model for classification only.

        Returns:
            Tuple of (initial_loss, final_loss)
        """
        print(colored_text(
            0, 200, 0, f"=== Starting pretraining ({self.config.n_pretraining_epochs} epochs) ==="))

        # Set up optimizer and loss function
        optimizer = self.learner.optimizer
        criterion = nn.CrossEntropyLoss()

        # Set up learning rate scheduler if enabled
        if getattr(self.config, 'lr_scheduling', False):
            scheduler = StepLR(optimizer, step_size=1,
                               gamma=self.config.learning_rate_step)
        else:
            scheduler = None

        # Initialize loss tracking
        initial_loss = None
        final_loss = None

        # Initialize epoch progress
        start_time = time.time()

        # Pretraining loop
        for epoch in range(self.config.n_pretraining_epochs):
            self.current_epoch = epoch
            epoch_loss = 0.0

            # Training loop for one epoch
            self.learner.classifier.train()
            for batch_idx, (inputs, labels) in enumerate(self.dataloaders.train):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Zero gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = self.learner.classifier(inputs)

                # Calculate loss
                loss = criterion(outputs, labels)

                # Store initial loss
                if epoch == 0 and batch_idx == 0:
                    initial_loss = loss.item()

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

                # Update loss statistics
                epoch_loss += loss.item()

                # Update global step counter
                self.global_step += 1

                # Log batch statistics
                if batch_idx % self.config.log_interval_pretraining == 0:
                    current_loss = loss.item()
                    print(f"Pretrain Epoch: {epoch+1}/{self.config.n_pretraining_epochs} "
                          f"[{batch_idx+1}/{len(self.dataloaders.train)}] "
                          f"Loss: {current_loss:.6f}")

                    # Log to TensorBoard
                    if self.writer:
                        self.writer.add_scalar(
                            "Pretraining/Loss", current_loss, self.global_step)

                # Evaluate accuracy periodically
                if batch_idx % self.config.log_interval_accuracy == 0:
                    self._evaluate_accuracy("Pretraining")

                    # Visualize explanations
                    self._visualize_batch("Pretraining")

            # Update learning rate if scheduler is enabled
            if scheduler:
                scheduler.step()
                if self.writer:
                    self.writer.add_scalar("Pretraining/LearningRate",
                                           optimizer.param_groups[0]['lr'],
                                           self.global_step)

            # Calculate average epoch loss
            epoch_loss /= len(self.dataloaders.train)
            final_loss = epoch_loss

            # Log epoch statistics
            print(f"Pretrain Epoch: {epoch+1}/{self.config.n_pretraining_epochs} "
                  f"Average Loss: {epoch_loss:.6f}")

            # Log to TensorBoard
            if self.writer:
                self.writer.add_scalar(
                    "Pretraining/EpochLoss", epoch_loss, epoch)

        # Calculate training time
        training_time = time.time() - start_time
        print(f"Pretraining completed in {training_time:.2f} seconds")

        # Save pretrained model
        if not self.config.logging_disabled:
            model_path = os.path.join(
                "runs", self.config.run_name, "pretrained_model.pt")
            self.learner.save_state(
                model_path, self.config.n_pretraining_epochs, final_loss)

        return initial_loss, final_loss

    def joint_train(self) -> Tuple[float, float]:
        """
        Perform joint training with the learner and critic.

        Returns:
            Tuple of (initial_loss, final_loss)
        """
        print(colored_text(
            0, 200, 0, f"=== Starting joint training ({self.config.n_epochs} epochs) ==="))

        # Set up optimizer and loss function
        optimizer = self.learner.optimizer
        criterion = nn.CrossEntropyLoss()

        # Set up learning rate scheduler if enabled
        if getattr(self.config, 'lr_scheduling', False):
            scheduler = StepLR(optimizer, step_size=1,
                               gamma=self.config.learning_rate_step)
        else:
            scheduler = None

        # Initialize loss tracking
        initial_loss = None
        final_loss = None

        # Initialize epoch progress
        start_time = time.time()

        # Joint training loop
        for epoch in range(self.config.n_epochs):
            self.current_epoch = epoch
            epoch_loss = 0.0
            epoch_classification_loss = 0.0
            epoch_explanation_loss = 0.0

            # Training loop for one epoch
            self.learner.classifier.train()
            for batch_idx, (inputs, labels) in enumerate(self.dataloaders.train):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Zero gradients
                optimizer.zero_grad()

                # Forward pass for classification
                outputs = self.learner.classifier(inputs)

                # Calculate classification loss
                classification_loss = self.config.classification_loss_weight * \
                    criterion(outputs, labels)

                # Calculate explanation loss by training the critic
                critic_loss = self._train_critic()

                # Calculate total loss
                # For explanation loss, we normalize by number of critic batches
                explanation_loss = self.config.explanation_loss_weight * \
                    critic_loss / len(self.dataloaders.critic)

                total_loss = classification_loss + explanation_loss

                # Store initial loss
                if epoch == 0 and batch_idx == 0:
                    initial_loss = total_loss.item()

                # Backward pass for classification loss
                classification_loss.backward()

                # Update optimizer
                optimizer.step()

                # Update loss statistics
                epoch_loss += total_loss.item()
                epoch_classification_loss += classification_loss.item()
                epoch_explanation_loss += explanation_loss.item()

                # Update global step counter
                self.global_step += 1

                # Log batch statistics
                if batch_idx % self.config.log_interval == 0:
                    print(f"Joint Train Epoch: {epoch+1}/{self.config.n_epochs} "
                          f"[{batch_idx+1}/{len(self.dataloaders.train)}] "
                          f"Loss: {total_loss.item():.6f} "
                          f"(C: {classification_loss.item():.6f}, E: {explanation_loss.item():.6f})")

                    # Log to TensorBoard
                    if self.writer:
                        self.writer.add_scalar(
                            "JointTraining/TotalLoss", total_loss.item(), self.global_step)
                        self.writer.add_scalar(
                            "JointTraining/ClassificationLoss", classification_loss.item(), self.global_step)
                        self.writer.add_scalar(
                            "JointTraining/ExplanationLoss", explanation_loss.item(), self.global_step)

                # Evaluate accuracy periodically
                if batch_idx % self.config.log_interval_accuracy == 0:
                    self._evaluate_accuracy("JointTraining")

                    # Visualize explanations
                    self._visualize_batch("JointTraining")

            # Update learning rate if scheduler is enabled
            if scheduler:
                scheduler.step()
                if self.writer:
                    self.writer.add_scalar("JointTraining/LearningRate",
                                           optimizer.param_groups[0]['lr'],
                                           self.global_step)

            # Calculate average epoch losses
            epoch_loss /= len(self.dataloaders.train)
            epoch_classification_loss /= len(self.dataloaders.train)
            epoch_explanation_loss /= len(self.dataloaders.train)
            final_loss = epoch_loss

            # Log epoch statistics
            print(f"Joint Train Epoch: {epoch+1}/{self.config.n_epochs} "
                  f"Average Loss: {epoch_loss:.6f} "
                  f"(C: {epoch_classification_loss:.6f}, E: {epoch_explanation_loss:.6f})")

            # Log to TensorBoard
            if self.writer:
                self.writer.add_scalar(
                    "JointTraining/EpochTotalLoss", epoch_loss, epoch)
                self.writer.add_scalar(
                    "JointTraining/EpochClassificationLoss", epoch_classification_loss, epoch)
                self.writer.add_scalar(
                    "JointTraining/EpochExplanationLoss", epoch_explanation_loss, epoch)

        # Calculate training time
        training_time = time.time() - start_time
        print(f"Joint training completed in {training_time:.2f} seconds")

        # Save joint trained model
        if not self.config.logging_disabled:
            model_path = os.path.join(
                "runs", self.config.run_name, "joint_model.pt")
            self.learner.save_state(
                model_path, self.config.n_epochs, final_loss)

        return initial_loss, final_loss

    def finetune(self) -> Tuple[float, float]:
        """
        Finetune the model using precomputed explanations.

        Returns:
            Tuple of (initial_loss, final_loss)
        """
        print(colored_text(
            0, 200, 0, f"=== Starting finetuning ({self.config.n_finetuning_epochs} epochs) ==="))

        # Collect explanations for the training set
        print("Generating explanations for finetuning...")
        inputs, explanations, labels = self._get_explanation_dataset()

        # Create a DataLoader for finetuning
        finetuning_loader = create_explanation_loader(
            inputs=inputs,
            explanations=explanations,
            labels=labels,
            batch_size=self.config.batch_size
        )

        # Set up optimizer and loss functions
        self.learner.initialize_optimizer(self.config.learning_rate_finetune)
        optimizer = self.learner.optimizer
        classification_criterion = nn.CrossEntropyLoss()
        explanation_criterion = nn.MSELoss()

        # Set up learning rate scheduler if enabled
        if getattr(self.config, 'lr_scheduling', False):
            scheduler = StepLR(optimizer, step_size=1,
                               gamma=self.config.learning_rate_step)
        else:
            scheduler = None

        # Initialize loss tracking
        initial_loss = None
        final_loss = None

        # Initialize epoch progress
        start_time = time.time()

        # Finetuning loop
        for epoch in range(self.config.n_finetuning_epochs):
            self.current_epoch = epoch
            epoch_loss = 0.0
            epoch_classification_loss = 0.0
            epoch_explanation_loss = 0.0

            # Training loop for one epoch
            self.learner.classifier.train()
            for batch_idx, (inputs, target_explanations, labels) in enumerate(finetuning_loader):
                inputs = inputs.to(self.device)
                target_explanations = target_explanations.to(self.device)
                labels = labels.to(self.device)

                # Zero gradients
                optimizer.zero_grad()

                # Forward pass for classification
                outputs = self.learner.classifier(inputs)

                # Generate current explanations
                current_explanations = self.learner.get_explanation_batch(
                    inputs, labels)

                # Calculate losses
                classification_loss = classification_criterion(outputs, labels)
                explanation_loss = explanation_criterion(
                    current_explanations, target_explanations)

                # Combined loss with weighting
                total_loss = classification_loss + \
                    self.config.explanation_loss_weight_finetune * explanation_loss

                # Store initial loss
                if epoch == 0 and batch_idx == 0:
                    initial_loss = total_loss.item()

                # Backward pass and optimize
                total_loss.backward()
                optimizer.step()

                # Update loss statistics
                epoch_loss += total_loss.item()
                epoch_classification_loss += classification_loss.item()
                epoch_explanation_loss += explanation_loss.item()

                # Update global step counter
                self.global_step += 1

                # Log batch statistics
                if batch_idx % self.config.log_interval == 0:
                    print(f"Finetune Epoch: {epoch+1}/{self.config.n_finetuning_epochs} "
                          f"[{batch_idx+1}/{len(finetuning_loader)}] "
                          f"Loss: {total_loss.item():.6f} "
                          f"(C: {classification_loss.item():.6f}, E: {explanation_loss.item():.6f})")

                    # Log to TensorBoard
                    if self.writer:
                        self.writer.add_scalar(
                            "Finetuning/TotalLoss", total_loss.item(), self.global_step)
                        self.writer.add_scalar(
                            "Finetuning/ClassificationLoss", classification_loss.item(), self.global_step)
                        self.writer.add_scalar(
                            "Finetuning/ExplanationLoss", explanation_loss.item(), self.global_step)

                # Evaluate accuracy periodically
                if batch_idx % self.config.log_interval_accuracy == 0:
                    self._evaluate_accuracy("Finetuning")

                    # Visualize explanations
                    self._visualize_batch("Finetuning")

            # Update learning rate if scheduler is enabled
            if scheduler:
                scheduler.step()
                if self.writer:
                    self.writer.add_scalar("Finetuning/LearningRate",
                                           optimizer.param_groups[0]['lr'],
                                           self.global_step)

            # Calculate average epoch losses
            epoch_loss /= len(finetuning_loader)
            epoch_classification_loss /= len(finetuning_loader)
            epoch_explanation_loss /= len(finetuning_loader)
            final_loss = epoch_loss

            # Log epoch statistics
            print(f"Finetune Epoch: {epoch+1}/{self.config.n_finetuning_epochs} "
                  f"Average Loss: {epoch_loss:.6f} "
                  f"(C: {epoch_classification_loss:.6f}, E: {epoch_explanation_loss:.6f})")

            # Log to TensorBoard
            if self.writer:
                self.writer.add_scalar(
                    "Finetuning/EpochTotalLoss", epoch_loss, epoch)
                self.writer.add_scalar(
                    "Finetuning/EpochClassificationLoss", epoch_classification_loss, epoch)
                self.writer.add_scalar(
                    "Finetuning/EpochExplanationLoss", epoch_explanation_loss, epoch)

        # Calculate training time
        training_time = time.time() - start_time
        print(f"Finetuning completed in {training_time:.2f} seconds")

        # Save finetuned model
        if not self.config.logging_disabled:
            model_path = os.path.join(
                "runs", self.config.run_name, "finetuned_model.pt")
            self.learner.save_state(
                model_path, self.config.n_finetuning_epochs, final_loss)

        return initial_loss, final_loss

    def run_full_pipeline(self) -> None:
        """
        Run the full CNN-LSX training pipeline.

        This includes pretraining, joint training, and finetuning.
        """
        print(colored_text(0, 200, 0, "=== Starting full CNN-LSX pipeline ==="))

        # 1. Pretraining
        pretrain_init_loss, pretrain_final_loss = self.pretrain()
        print(
            f"Pretraining: initial loss={pretrain_init_loss:.6f}, final loss={pretrain_final_loss:.6f}")

        # Visualize pretrained explanations
        self._visualize_batch("AfterPretraining")

        # 2. Joint training
        joint_init_loss, joint_final_loss = self.joint_train()
        print(
            f"Joint training: initial loss={joint_init_loss:.6f}, final loss={joint_final_loss:.6f}")

        # Visualize joint-trained explanations
        self._visualize_batch("AfterJointTraining")

        # 3. Finetuning
        finetune_init_loss, finetune_final_loss = self.finetune()
        print(
            f"Finetuning: initial loss={finetune_init_loss:.6f}, final loss={finetune_final_loss:.6f}")

        # Visualize finetuned explanations
        self._visualize_batch("AfterFinetuning")

        print(colored_text(0, 200, 0, "=== CNN-LSX pipeline completed ==="))

    def test(self) -> float:
        """
        Test the model on the test set.

        Returns:
            Test accuracy
        """
        print(colored_text(0, 200, 0, "=== Testing model ==="))

        # Evaluate on test set
        test_accuracy = compute_accuracy(
            model=self.learner.classifier,
            dataloader=self.dataloaders.test,
            device=self.device
        )

        print(f"Test accuracy: {test_accuracy:.4f}")

        # Visualize test explanations
        self._visualize_batch("TestExplanations")

        return test_accuracy

    def save_model(self, path: str) -> None:
        """
        Save the learner model.

        Args:
            path: Path to save the model
        """
        self.learner.save_state(path, self.current_epoch, 0.0)
        print(f"Model saved to {path}")

    def load_model(self, path: str) -> None:
        """
        Load a learner model.

        Args:
            path: Path to the saved model
        """
        self.learner.load_state(path)
        print(f"Model loaded from {path}")

    def _train_critic(self) -> float:
        """
        Train the critic on explanations from the learner.

        Returns:
            Mean critic loss
        """
        # Create a critic network
        num_classes = self._get_num_classes()
        critic_network = get_network(self.config.model, num_classes)

        # Create a critic
        critic = Critic(
            model=critic_network,
            critic_loader=self.dataloaders.critic,
            device=self.device,
            log_interval=self.config.log_interval_critic
        )

        # Generate explanations for the critic
        explanations = []
        for inputs, labels in self.dataloaders.critic:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            explanations.append(
                self.learner.get_explanation_batch(inputs, labels))

        # Train the critic
        _, _, mean_loss = critic.train(
            explanations=explanations,
            learning_rate=self.config.learning_rate_critic
        )

        return mean_loss

    def _get_explanation_dataset(self) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
        """
        Generate explanations for the entire training set.

        Returns:
            Tuple of (inputs, explanations, labels)
        """
        return self.learner.get_detached_explanations(self.dataloaders.train)

    def _get_num_classes(self) -> int:
        """
        Get the number of classes in the dataset.

        Returns:
            Number of classes
        """
        # Try to get from the data directly
        train_dataset = self.dataloaders.train.dataset
        try:
            return len(torch.unique(train_dataset.targets))
        except:
            # Default to 10 for MNIST-like datasets
            return 10

    def _evaluate_accuracy(self, prefix: str = "") -> Tuple[float, float]:
        """
        Evaluate the model's accuracy on train and test sets.

        Args:
            prefix: Prefix for TensorBoard logging

        Returns:
            Tuple of (train_accuracy, test_accuracy)
        """
        # Compute training accuracy
        train_accuracy = compute_accuracy(
            model=self.learner.classifier,
            dataloader=self.dataloaders.train,
            device=self.device,
            max_batches=min(len(self.dataloaders.train),
                            10)  # Limit for efficiency
        )

        # Compute test accuracy
        test_accuracy = compute_accuracy(
            model=self.learner.classifier,
            dataloader=self.dataloaders.test,
            device=self.device
        )

        # Log to console
        print(
            f"Accuracy: train={train_accuracy:.4f}, test={test_accuracy:.4f}")

        # Log to TensorBoard
        if self.writer:
            self.writer.add_scalar(
                f"{prefix}/TrainAccuracy", train_accuracy, self.global_step)
            self.writer.add_scalar(
                f"{prefix}/TestAccuracy", test_accuracy, self.global_step)

        return train_accuracy, test_accuracy

    def _visualize_batch(self, caption_prefix: str = "") -> None:
        """
        Visualize explanations for a batch from the visualization set.

        Args:
            caption_prefix: Prefix for the visualization caption
        """
        # Get a batch from the visualization set
        viz_inputs, viz_labels = next(iter(self.dataloaders.visualization))
        viz_inputs, viz_labels = viz_inputs.to(
            self.device), viz_labels.to(self.device)

        # Generate explanations
        with torch.no_grad():
            explanations = self.learner.get_explanation_batch(
                viz_inputs, viz_labels)

        # Visualize inputs
        self.visualizer.visualize_inputs(
            inputs=viz_inputs,
            caption=f"{caption_prefix}/Inputs",
            global_step=self.global_step
        )

        # Visualize explanations
        self.visualizer.visualize_explanations(
            explanations=explanations,
            caption=f"{caption_prefix}/Explanations",
            global_step=self.global_step
        )

        # Visualize overlay
        self.visualizer.visualize_overlay(
            inputs=viz_inputs,
            explanations=explanations,
            caption=f"{caption_prefix}/Overlay",
            global_step=self.global_step
        )
