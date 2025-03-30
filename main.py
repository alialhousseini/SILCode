"""
CNN-LSX main script.

This script is the entry point for CNN-LSX training and evaluation.
It parses command-line arguments, sets up the environment, and runs
the requested training mode.
"""

import os
import time
import argparse
from torch.utils.tensorboard import SummaryWriter

# Import from modules
from config.config import parse_args, config_string
from data.dataloaders import create_dataloaders
from evaluation.metrics import evaluate_explanations
from training.trainer import Trainer
from utils.common import set_seed, get_device, colored_text


def main():
    """Main entry point for CNN-LSX."""
    # Parse command-line arguments
    config = parse_args()

    # Set random seed for reproducibility
    set_seed(config.random_seed)

    # Get device (CPU or CUDA)
    device = get_device(config.no_cuda)

    # Create log directory and TensorBoard writer if logging is enabled
    writer = None
    if not config.logging_disabled:
        run_name = config_string(config)
        log_dir = os.path.join("runs", run_name)
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir)
        config.run_name = run_name

        print(f"Logging to: {log_dir}")

    # Create data loaders
    dataloaders = create_dataloaders(config, device)

    # Create trainer
    trainer = Trainer(config, dataloaders, device, writer)

    # Run the requested training mode
    if config.training_mode == "pretrain_and_joint_and_finetuning":
        trainer.run_full_pipeline()
    elif config.training_mode == "pretrain_and_joint":
        trainer.pretrain()
        trainer.joint_train()
    elif config.training_mode == "joint":
        trainer.joint_train()
    elif config.training_mode == "only_classification":
        trainer.pretrain()
    elif config.training_mode == "finetuning":
        if config.model_pt is None:
            raise ValueError(
                "Model path (--model_pt) must be specified for finetuning")
        trainer.load_model(config.model_pt)
        trainer.finetune()
    elif config.training_mode == "test":
        if config.model_pt is None:
            raise ValueError(
                "Model path (--model_pt) must be specified for testing")
        trainer.load_model(config.model_pt)
        trainer.test()
    elif config.training_mode == "faithfulness":
        if config.model_pt is None:
            raise ValueError(
                "Model path (--model_pt) must be specified for faithfulness evaluation")
        trainer.load_model(config.model_pt)

        # Run faithfulness evaluation
        print(colored_text(0, 200, 0, "=== Evaluating explanation faithfulness ==="))
        metrics = evaluate_explanations(
            model=trainer.learner.classifier,
            dataloader=dataloaders.test,
            explanation_function=lambda x, y: trainer.learner.get_explanation_batch(
                x, y),
            device=device,
            max_batches=10
        )

        # Print metrics
        print("Faithfulness metrics:")
        for name, value in metrics.items():
            print(f"  {name}: {value:.4f}")
    else:
        raise ValueError(f"Unknown training mode: {config.training_mode}")

    # Close TensorBoard writer
    if writer:
        writer.close()

    print(colored_text(0, 200, 0, "Done!"))


if __name__ == "__main__":
    main()
