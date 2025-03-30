# CNN-LSX: CNNs with Learnable Saliency Explanations

This is a clean implementation of CNN-LSX, a framework for training neural networks with high-quality saliency explanations.

## Overview

CNN-LSX implements a novel approach for training neural networks that not only make accurate predictions but also produce meaningful explanations for those predictions. It uses a learner-critic framework:

- **Learner**: A neural network that makes predictions and generates explanations
- **Critic**: A separate network that evaluates the quality of those explanations

## Key Features

- Multiple explanation methods (input × gradient, integrated gradients, GradCAM)
- Flexible training pipeline with multiple stages (pretraining, joint training, finetuning)
- Support for different datasets (MNIST, DecoyMNIST, ColorMNIST)
- Comprehensive evaluation metrics for explanation quality

## Installation

```bash
git clone https://github.com/yourusername/cnn-lsx.git
cd cnn-lsx
pip install -r requirements.txt
```

## Quick Start

### Training a model

```bash
# Train on MNIST with default parameters
python main.py --dataset mnist --training_mode pretrain_and_joint_and_finetuning

# Train on a small portion of the data
python main.py --dataset mnist --few_shot_train_percent 0.02 --training_mode pretrain_and_joint_and_finetuning
```

### Different training modes

- **Classification only**: `--training_mode only_classification`
- **Joint training only**: `--training_mode joint`
- **Pretrain then joint train**: `--training_mode pretrain_and_joint`
- **Complete pipeline**: `--training_mode pretrain_and_joint_and_finetuning`
- **Finetuning only**: `--training_mode finetuning --model_pt path/to/model.pt`
- **Testing a model**: `--training_mode test --model_pt path/to/model.pt`

### Explanation parameters

Control the balance between classification accuracy and explanation quality:

```bash
python main.py --classification_loss_weight 1 --explanation_loss_weight 100 --explanation_loss_weight_finetune 100
```

## Understanding the Training Process

1. **Pretraining Phase**: The learner is trained for classification only.
2. **Joint Training Phase**: The learner is trained to optimize both classification performance and explanation quality, as judged by the critic.
3. **Finetuning Phase**: The model is refined to further improve explanation quality.

## Visualization

Training progress and model evaluations are logged to TensorBoard:

```bash
tensorboard --logdir runs
```

## Detailed Documentation

For more information about CNN-LSX, including the theoretical background and implementation details, please refer to the documentation in the `docs/` directory.