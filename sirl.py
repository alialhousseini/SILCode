import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score
from captum.attr import InputXGradient
from torchvision.datasets import MNIST

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

###############################################################################
# 1) SimpleCNN with conv1, conv2, and conv3
###############################################################################


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.AvgPool2d(kernel_size=2)

        # Two linear layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))   # 1st conv
        x = F.relu(self.conv2(x))   # 2nd conv
        x = self.pool(x)
        x = F.relu(self.conv3(x))   # 3rd conv
        x = self.pool(x)
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def predict(self, x):
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)


###############################################################################
# 2) Ensure GPU usage with model/data on `device`
###############################################################################
def input_x_gradient(model, inputs, targets, retain_graph=False):
    """
    Manually compute "Input × Gradient" for the batch.
    If `retain_graph=True`, you can backprop through these explanations.
    """
    # Ensure inputs are a tensor that requires grad
    # (Clone & detach so we don't modify the original in-place)
    inputs = inputs.clone().detach()
    inputs.requires_grad_(True)

    # Forward pass
    logits = model(inputs)  # shape [B, 10], for instance

    # Gather the logit for each sample's target
    # (assuming targets is shape [B])
    selected_logits = logits[torch.arange(logits.shape[0]), targets]

    # Backward pass to get gradients w.r.t. inputs
    grads = torch.autograd.grad(
        outputs=selected_logits,
        inputs=inputs,
        grad_outputs=torch.ones_like(selected_logits),
        create_graph=retain_graph
    )[0]  # shape same as `inputs`

    # "Input × Gradient" explanation
    explanation = inputs * grads
    return explanation


def train_vanilla_cnn(model, train_loader, val_loader, epochs=5, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            # Move data to device
            data, target = data.to(model.device), target.to(model.device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Validation
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(model.device), target.to(model.device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        accuracy = 100 * correct / total
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        val_accuracies.append(accuracy)

        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, '
              f'Validation Accuracy: {accuracy:.2f}%')

    return train_losses, val_accuracies


def train_lsx(learner, critic, train_loader, val_loader,
              critic_data_fraction=0.5, epochs=5, lsx_iterations=3,
              lambda_val=100, lr=0.001):
    """
    Train using the LSX framework:
    1. Fit: Train the learner on the base task
    2. Explain: Generate explanations
    3. Reflect: Train critic to classify explanations
    4. Revise: Update learner based on critic's feedback
    """
    optimizer = optim.Adam(learner.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    val_accuracies = []

    # Split data for critic
    critic_batch_count = int(len(train_loader) * critic_data_fraction)

    # Step 1: Fit - Initial training of learner
    print("Step 1: Initial fitting of learner model")
    for epoch in range(epochs):
        learner.train()
        epoch_loss = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(learner.device), target.to(learner.device)

            optimizer.zero_grad()
            output = learner(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)

        # Validation
        learner.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(
                    learner.device), target.to(learner.device)
                output = learner(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        accuracy = 100 * correct / total
        val_accuracies.append(accuracy)
        print(f'Initial Fit - Epoch {epoch+1}/{epochs}, '
              f'Loss: {avg_loss:.4f}, Validation Accuracy: {accuracy:.2f}%')

    # Steps 2-4: multiple LSX iterations
    for iteration in range(lsx_iterations):
        print(f"\nLSX Iteration {iteration+1}/{lsx_iterations}")

    # Step 2: Explain - Generate explanations for critic data
    print("Step 2: Generating explanations")
    learner.eval()
    explanations_list = []
    targets_list = []

    critic_data_loader = []
    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx < critic_batch_count:
            data, target = data.to(learner.device), target.to(learner.device)
            critic_data_loader.append((data, target))

            # Remove `with torch.no_grad():` here, to allow computing input gradients
            expl = input_x_gradient(learner, data, target, retain_graph=False)

            # Then detach so we don't backprop from the critic step into the learner
            expl = expl.detach()

            explanations_list.append(expl)
            targets_list.append(target)
        else:
            break

        # Concatenate all explanations and targets
        all_explanations = torch.cat(explanations_list)
        all_targets = torch.cat(targets_list)

        # Step 3: Reflect - Train critic on explanations
        print("Step 3: Training critic on explanations")
        critic = SimpleCNN().to(learner.device)  # reset critic
        critic_optimizer = optim.Adam(critic.parameters(), lr=lr)
        critic.train()
        critic_losses = []

        batch_size = 64
        num_samples = all_explanations.size(0)
        indices = list(range(num_samples))
        np.random.shuffle(indices)

        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i:min(i+batch_size, num_samples)]
            explanations_batch = all_explanations[batch_indices].to(
                learner.device)
            targets_batch = all_targets[batch_indices].to(learner.device)

            critic_optimizer.zero_grad()
            critic_output = critic(explanations_batch)
            critic_loss = criterion(critic_output, targets_batch)
            critic_loss.backward()
            critic_optimizer.step()

            critic_losses.append(critic_loss.item())

        avg_critic_loss = sum(critic_losses) / len(critic_losses)
        print(f"Critic Training Loss: {avg_critic_loss:.4f}")

        # Step 4: Revise - Update learner based on critic's feedback
        # Here we DO need to backprop from the critic into the learner's parameters
        print("Step 4: Revising learner based on critic's feedback")
        learner_optimizer = optim.Adam(learner.parameters(), lr=lr)

        for epoch in range(1):  # one epoch of revision
            learner.train()
            epoch_loss = 0

            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(
                    learner.device), target.to(learner.device)
                learner_optimizer.zero_grad()

                # Base task loss
                output = learner(data)
                base_loss = criterion(output, target)

                # (B) Get explanation with retain_graph=True so that
                #     the gradient from explanation_loss can flow back
                explanations = input_x_gradient(
                    learner, data, target, retain_graph=True
                )

                critic_output = critic(explanations)  # forward pass in critic
                explanation_loss = criterion(critic_output, target)

                # Combined loss
                combined_loss = base_loss + lambda_val * explanation_loss
                combined_loss.backward()
                learner_optimizer.step()

                epoch_loss += combined_loss.item()

            avg_loss = epoch_loss / len(train_loader)
            print(f'Revise - Epoch {epoch+1}, Loss: {avg_loss:.4f}')

        # Validation after revision
        learner.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(
                    learner.device), target.to(learner.device)
                output = learner(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        accuracy = 100 * correct / total
        val_accuracies.append(accuracy)
        print(
            f'Validation Accuracy after LSX iteration {iteration+1}: {accuracy:.2f}%')

    # Final explanation fine-tuning
    print("\nFinal explanation fine-tuning")
    learner.eval()
    all_explanations_final = []
    with torch.no_grad():
        for data, _ in train_loader:
            data = data.to(learner.device)
            targets = learner.predict(data)
            explanations = input_x_gradient(learner, data, targets)
            all_explanations_final.append(explanations.cpu())  # store on CPU

    learner_optimizer = optim.Adam(learner.parameters(), lr=lr * 0.1)
    for epoch in range(1):
        learner.train()
        epoch_loss = 0

        for batch_idx, ((data, target), stored_explanations) in enumerate(zip(train_loader, all_explanations_final)):
            data, target = data.to(learner.device), target.to(learner.device)
            stored_explanations = stored_explanations.to(learner.device)

            learner_optimizer.zero_grad()
            output = learner(data)
            base_loss = criterion(output, target)

            # Generate new explanations
            new_targets = learner.predict(data)
            new_explanations = input_x_gradient(learner, data, new_targets)

            # MSE loss to maintain explanations
            mse_loss = F.mse_loss(new_explanations, stored_explanations)

            combined_loss = base_loss + mse_loss
            combined_loss.backward()
            learner_optimizer.step()

            epoch_loss += combined_loss.item()

        print(f'Final Fine-tuning - Loss: {epoch_loss/len(train_loader):.4f}')

    # Final validation
    learner.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(learner.device), target.to(learner.device)
            output = learner(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    final_accuracy = 100 * correct / total
    val_accuracies.append(final_accuracy)
    print(f'Final Validation Accuracy: {final_accuracy:.2f}%')

    return train_losses, val_accuracies


def evaluate_faithfulness(model, test_loader, percentages=[1, 5, 10, 20, 50]):
    """
    Evaluate explanation faithfulness using comprehensiveness and sufficiency metrics.
    """
    model.eval()

    # Get baseline accuracy
    baseline_correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            baseline_correct += (predicted == target).sum().item()

    baseline_accuracy = baseline_correct / total
    print(f"Baseline accuracy: {baseline_accuracy:.4f}")

    # Calculate median value for replacement
    all_values = []
    for data, _ in test_loader:
        all_values.append(data.cpu().numpy().flatten())
    median_value = np.median(np.concatenate(all_values))

    comprehensiveness_scores = []
    sufficiency_scores = []

    # Random baseline scores
    random_compr_scores = []
    random_suff_scores = []

    for percentage in percentages:
        # Evaluate with important features removed (comprehensiveness)
        comprehensiveness_correct = 0
        random_compr_correct = 0

        # Evaluate with only important features (sufficiency)
        sufficiency_correct = 0
        random_suff_correct = 0

        total_evaluated = 0

        for data, target in test_loader:
            # Only evaluate a subset to save time
            if total_evaluated >= 1000:
                break

            # Get explanations
            explanations = input_x_gradient(model, data, target)

            # For each sample
            for i in range(data.size(0)):
                sample = data[i].unsqueeze(0)
                explanation = explanations[i].unsqueeze(0)
                true_label = target[i].item()

                # Calculate importance of each pixel
                importance = torch.abs(explanation).sum(
                    dim=1)  # Sum across channels

                # Flatten for sorting
                flat_importance = importance.view(-1)

                # Get indices of top k% important pixels
                k = int((percentage/100) * flat_importance.numel())
                _, top_indices = torch.topk(flat_importance, k)

                # Get random indices for baseline
                random_indices = torch.randperm(flat_importance.numel())[:k]

                # Create masks
                top_mask = torch.zeros_like(flat_importance)
                top_mask[top_indices] = 1
                top_mask = top_mask.view(importance.shape)

                random_mask = torch.zeros_like(flat_importance)
                random_mask[random_indices] = 1
                random_mask = random_mask.view(importance.shape)

                # Create modified images for comprehensiveness (remove important pixels)
                compr_sample = sample.clone()
                for c in range(compr_sample.size(1)):
                    compr_sample[0, c][top_mask.squeeze() > 0] = median_value

                random_compr_sample = sample.clone()
                for c in range(random_compr_sample.size(1)):
                    random_compr_sample[0, c][random_mask.squeeze(
                    ) > 0] = median_value

                # Create modified images for sufficiency (keep only important pixels)
                suff_sample = torch.ones_like(sample) * median_value
                for c in range(suff_sample.size(1)):
                    suff_sample[0, c][top_mask.squeeze(
                    ) > 0] = sample[0, c][top_mask.squeeze() > 0]

                random_suff_sample = torch.ones_like(sample) * median_value
                for c in range(random_suff_sample.size(1)):
                    random_suff_sample[0, c][random_mask.squeeze(
                    ) > 0] = sample[0, c][random_mask.squeeze() > 0]

                # Evaluate
                with torch.no_grad():
                    # Comprehensiveness
                    compr_output = model(compr_sample)
                    _, compr_pred = torch.max(compr_output, 1)
                    comprehensiveness_correct += (compr_pred.item()
                                                  == true_label)

                    random_compr_output = model(random_compr_sample)
                    _, random_compr_pred = torch.max(random_compr_output, 1)
                    random_compr_correct += (random_compr_pred.item()
                                             == true_label)

                    # Sufficiency
                    suff_output = model(suff_sample)
                    _, suff_pred = torch.max(suff_output, 1)
                    sufficiency_correct += (suff_pred.item() == true_label)

                    random_suff_output = model(random_suff_sample)
                    _, random_suff_pred = torch.max(random_suff_output, 1)
                    random_suff_correct += (random_suff_pred.item()
                                            == true_label)

                total_evaluated += 1
                if total_evaluated >= 1000:
                    break

        # Calculate scores
        comprehensiveness_acc = comprehensiveness_correct / total_evaluated
        random_compr_acc = random_compr_correct / total_evaluated

        sufficiency_acc = sufficiency_correct / total_evaluated
        random_suff_acc = random_suff_correct / total_evaluated

        # Adjusted scores as defined in the paper
        comprehensiveness = random_compr_acc - comprehensiveness_acc
        sufficiency = random_compr_acc - sufficiency_acc

        comprehensiveness_scores.append(comprehensiveness)
        sufficiency_scores.append(sufficiency)

        print(f"Percentage: {percentage}%")
        print(f"  Comprehensiveness: {comprehensiveness:.4f}")
        print(f"  Sufficiency: {sufficiency:.4f}")

    # Calculate average scores
    avg_comprehensiveness = sum(
        comprehensiveness_scores) / len(comprehensiveness_scores)
    avg_sufficiency = sum(sufficiency_scores) / len(sufficiency_scores)

    return avg_comprehensiveness, avg_sufficiency

# Function to evaluate explanation separability using Ridge Regression


def evaluate_explanation_separability(model, test_loader):
    """
    Evaluate how separable the explanations are by training a ridge regression model.
    """
    model.eval()

    # Generate explanations for test data
    explanations = []
    labels = []

    with torch.no_grad():
        for data, target in test_loader:
            batch_explanations = input_x_gradient(model, data, target)

            # Flatten explanations
            batch_explanations = batch_explanations.view(
                batch_explanations.size(0), -1).cpu().numpy()

            explanations.append(batch_explanations)
            labels.append(target.cpu().numpy())

    explanations = np.vstack(explanations)
    labels = np.concatenate(labels)

    # Split data for training and testing
    n_samples = len(explanations)
    n_train = int(0.8 * n_samples)

    indices = np.random.permutation(n_samples)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]

    X_train = explanations[train_indices]
    y_train = labels[train_indices]
    X_test = explanations[test_indices]
    y_test = labels[test_indices]

    # Train Ridge Regression model
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)

    # Evaluate
    y_pred = ridge.predict(X_test)
    y_pred_classes = np.round(y_pred).astype(int)

    # Handle predictions outside the valid range
    y_pred_classes = np.clip(y_pred_classes, 0, 9)

    accuracy = accuracy_score(y_test, y_pred_classes)
    print(f"Ridge Regression Accuracy on Explanations: {accuracy:.4f}")

    return accuracy

# Function to calculate IIES (Inter- vs Intraclass Explanation Similarity)


def calculate_iies(model, test_loader, n_classes=10):
    """
    Calculate the Inter- vs Intraclass Explanation Similarity metric.
    """
    model.eval()

    # Generate explanations and group by class
    class_explanations = [[] for _ in range(n_classes)]

    with torch.no_grad():
        for data, target in test_loader:
            batch_explanations = input_x_gradient(model, data, target)

            # Flatten explanations
            batch_explanations = batch_explanations.view(
                batch_explanations.size(0), -1).cpu().numpy()

            for i, label in enumerate(target):
                class_explanations[label.item()].append(batch_explanations[i])

    # Calculate class means
    class_means = []
    for class_expl in class_explanations:
        if len(class_expl) > 0:
            class_means.append(np.mean(np.array(class_expl), axis=0))
        else:
            class_means.append(np.zeros(batch_explanations.shape[1]))

    # Calculate IIES
    iies_values = []

    for k in range(n_classes):
        if len(class_explanations[k]) == 0:
            continue

        # Calculate intraclass distances (samples to class mean)
        intraclass_distances = []
        for expl in class_explanations[k]:
            dist = np.linalg.norm(expl - class_means[k])
            intraclass_distances.append(dist)

        avg_intraclass_distance = np.mean(
            intraclass_distances) if intraclass_distances else 0

        # Calculate interclass distances (between class means)
        interclass_distances = []
        for j in range(n_classes):
            if j != k and len(class_explanations[j]) > 0:
                dist = np.linalg.norm(class_means[k] - class_means[j])
                interclass_distances.append(dist)

        avg_interclass_distance = np.mean(
            interclass_distances) if interclass_distances else 0

        # Calculate class IIES (lower is better)
        if avg_interclass_distance > 0:
            class_iies = avg_intraclass_distance / avg_interclass_distance
            iies_values.append(class_iies)

    # Calculate overall IIES
    overall_iies = np.mean(iies_values) if iies_values else 0
    print(f"IIES Score (lower is better): {overall_iies:.4f}")

    return overall_iies

# Function to visualize explanations


def visualize_explanations(model, data_loader, num_samples=10):
    """
    Visualize explanations for random samples.
    """
    model.eval()

    # Get samples
    samples = []
    for data, target in data_loader:
        for i in range(min(num_samples, data.size(0))):
            samples.append((data[i], target[i]))
        if len(samples) >= num_samples:
            break

    # Generate explanations
    for i, (sample, target) in enumerate(samples):
        sample = sample.unsqueeze(0)
        target = target.unsqueeze(0)

        explanation = input_x_gradient(model, sample, target)

        # Plot
        plt.figure(figsize=(10, 4))

        # Original image
        plt.subplot(1, 2, 1)
        plt.imshow(sample.squeeze().cpu().numpy(), cmap='gray')
        plt.title(f"Original (Class: {target.item()})")
        plt.axis('off')

        # Explanation
        plt.subplot(1, 2, 2)
        plt.imshow(explanation.squeeze().abs().sum(
            dim=0).cpu().numpy(), cmap='hot')
        plt.title("Explanation (InputXGradient)")
        plt.axis('off')

        plt.tight_layout()
        plt.show()

# Main function to run the experiment


def run_experiment(data_fraction=0.2, epochs=5, lsx_iterations=3):
    """
    Run experiment comparing vanilla CNN with CNN-LSX.
    """
    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = datasets.MNIST(
        './data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    # If using a subset of data
    if data_fraction < 1.0:
        train_size = int(len(train_dataset) * data_fraction)
        train_indices = torch.randperm(len(train_dataset))[:train_size]
        train_dataset = torch.utils.data.Subset(train_dataset, train_indices)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    print(f"Training with {len(train_dataset)} samples")

    # Create models
    vanilla_model = SimpleCNN()
    lsx_model = SimpleCNN()
    critic_model = SimpleCNN()

    # Train vanilla CNN
    print("\n=== Training Vanilla CNN ===")
    vanilla_losses, vanilla_accuracies = train_vanilla_cnn(
        vanilla_model, train_loader, test_loader, epochs=epochs
    )

    # Train with LSX
    print("\n=== Training with CNN-LSX ===")
    lsx_losses, lsx_accuracies = train_lsx(
        lsx_model, critic_model, train_loader, test_loader,
        epochs=epochs, lsx_iterations=lsx_iterations
    )

    # Evaluate explanation quality metrics
    print("\n=== Evaluating Explanation Faithfulness for Vanilla CNN ===")
    vanilla_compr, vanilla_suff = evaluate_faithfulness(
        vanilla_model, test_loader)

    print("\n=== Evaluating Explanation Faithfulness for CNN-LSX ===")
    lsx_compr, lsx_suff = evaluate_faithfulness(lsx_model, test_loader)

    print("\n=== Evaluating Explanation Separability for Vanilla CNN ===")
    vanilla_separability = evaluate_explanation_separability(
        vanilla_model, test_loader)

    print("\n=== Evaluating Explanation Separability for CNN-LSX ===")
    lsx_separability = evaluate_explanation_separability(
        lsx_model, test_loader)

    print("\n=== Calculating IIES for Vanilla CNN ===")
    vanilla_iies = calculate_iies(vanilla_model, test_loader)

    print("\n=== Calculating IIES for CNN-LSX ===")
    lsx_iies = calculate_iies(lsx_model, test_loader)

    # Print summary of results
    print("\n=== Results Summary ===")
    print(f"Vanilla CNN Final Accuracy: {vanilla_accuracies[-1]:.2f}%")
    print(f"CNN-LSX Final Accuracy: {lsx_accuracies[-1]:.2f}%")
    print(
        f"Accuracy Improvement: {lsx_accuracies[-1] - vanilla_accuracies[-1]:.2f}%")

    print("\nExplanation Faithfulness:")
    print(
        f"Vanilla CNN - Comprehensiveness: {vanilla_compr:.4f}, Sufficiency: {vanilla_suff:.4f}")
    print(
        f"CNN-LSX - Comprehensiveness: {lsx_compr:.4f}, Sufficiency: {lsx_suff:.4f}")

    print("\nExplanation Separability (Ridge Regression Accuracy):")
    print(f"Vanilla CNN: {vanilla_separability:.4f}")
    print(f"CNN-LSX: {lsx_separability:.4f}")

    print("\nInter- vs Intraclass Explanation Similarity (IIES, lower is better):")
    print(f"Vanilla CNN: {vanilla_iies:.4f}")
    print(f"CNN-LSX: {lsx_iies:.4f}")

    # Visualize some explanations
    print("\n=== Visualizing Vanilla CNN Explanations ===")
    visualize_explanations(vanilla_model, test_loader, num_samples=3)

    print("\n=== Visualizing CNN-LSX Explanations ===")
    visualize_explanations(lsx_model, test_loader, num_samples=3)

    return {
        'vanilla_acc': vanilla_accuracies[-1],
        'lsx_acc': lsx_accuracies[-1],
        'vanilla_compr': vanilla_compr,
        'lsx_compr': lsx_compr,
        'vanilla_suff': vanilla_suff,
        'lsx_suff': lsx_suff,
        'vanilla_separability': vanilla_separability,
        'lsx_separability': lsx_separability,
        'vanilla_iies': vanilla_iies,
        'lsx_iies': lsx_iies
    }


# Example usage
if __name__ == "__main__":
    # Run with 20% of MNIST data (12,000 samples) to speed up training
    results = run_experiment(data_fraction=0.2, epochs=3, lsx_iterations=2)
