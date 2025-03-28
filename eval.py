import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from captum.attr import InputXGradient
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the CNN architecture for both learner and critic models


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Two convolutional layers as described in the paper
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.AvgPool2d(kernel_size=2)
        # Two linear layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    # Function to get the predicted class
    def predict(self, x):
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)

# Function to compute InputXGradient explanations using captum


def compute_explanations(model, inputs, targets):
    """
    Compute InputXGradient explanations using captum.
    """
    # Initialize InputXGradient explainer
    input_x_grad = InputXGradient(model)

    # Compute attributions for each input-target pair
    explanations = []

    # Process one sample at a time to handle different target classes
    for i in range(inputs.size(0)):
        input_sample = inputs[i:i+1]  # Keep batch dimension
        target_class = targets[i].item()

        # Get attribution
        attribution = input_x_grad.attribute(input_sample, target=target_class)
        explanations.append(attribution[0])  # Remove batch dimension

    return torch.stack(explanations)

# Function to train a vanilla CNN


def train_vanilla_cnn(model, train_loader, val_loader, epochs=5, lr=0.001):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
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
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        accuracy = 100 * correct / total
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        val_accuracies.append(accuracy)

        print(
            f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Validation Accuracy: {accuracy:.2f}%')

    return train_losses, val_accuracies

# Main LSX training loop


def train_lsx(learner, critic, train_loader, val_loader, critic_data_fraction=0.5,
              epochs=5, lsx_iterations=3, lambda_val=100, lr=0.001):
    """
    Train using the LSX framework:
    1. Fit: Train the learner on the base task
    2. Explain: Generate explanations
    3. Reflect: Train critic to classify explanations
    4. Revise: Update learner based on critic's feedback
    """
    learner.to(device)
    critic.to(device)
    optimizer = optim.Adam(learner.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    val_accuracies = []

    # Split data for critic
    critic_batch_count = int(len(train_loader) * critic_data_fraction)

    # Step 1: Fit - Initial training of learner
    print("Step 1: Initial fitting of learner model")
    for epoch in range(epochs):
        learner.train()
        epoch_loss = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = learner(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Validation
        learner.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = learner(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        accuracy = 100 * correct / total
        val_accuracies.append(accuracy)

        print(f'Initial Fit - Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader):.4f}, '
              f'Validation Accuracy: {accuracy:.2f}%')

    # Steps 2-4: Explain, Reflect, Revise for multiple iterations
    for iteration in range(lsx_iterations):
        print(f"\nLSX Iteration {iteration+1}/{lsx_iterations}")

        # Step 2: Explain - Generate explanations for critic data
        print("Step 2: Generating explanations")
        learner.eval()
        explanations_list = []
        targets_list = []

        # Collect a subset of data for the critic
        critic_data_loader = []
        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_idx < critic_batch_count:
                data, target = data.to(device), target.to(device)
                critic_data_loader.append((data, target))

                # Generate explanations
                explanations = compute_explanations(learner, data, target)
                explanations_list.append(explanations)
                targets_list.append(target)
            else:
                break

        # Concatenate all explanations and targets
        all_explanations = torch.cat(explanations_list)
        all_targets = torch.cat(targets_list)

        # Step 3: Reflect - Train critic to classify explanations
        print("Step 3: Training critic on explanations")

        # Reset critic
        critic = SimpleCNN().to(device)
        critic_optimizer = optim.Adam(critic.parameters(), lr=lr)

        # Train critic for one epoch
        critic.train()
        critic_losses = []

        # Create dataset from explanations
        batch_size = 64
        num_samples = all_explanations.size(0)
        indices = list(range(num_samples))
        np.random.shuffle(indices)

        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i:min(i + batch_size, num_samples)]
            explanations_batch = all_explanations[batch_indices]
            targets_batch = all_targets[batch_indices]

            critic_optimizer.zero_grad()
            critic_output = critic(explanations_batch)
            critic_loss = criterion(critic_output, targets_batch)
            critic_loss.backward()
            critic_optimizer.step()

            critic_losses.append(critic_loss.item())

        avg_critic_loss = sum(critic_losses) / len(critic_losses)
        print(f"Critic Training Loss: {avg_critic_loss:.4f}")

        # Step 4: Revise - Update learner based on critic's feedback
        print("Step 4: Revising learner based on critic's feedback")

        learner_optimizer = optim.Adam(learner.parameters(), lr=lr)

        for epoch in range(1):  # One epoch of revision
            learner.train()
            epoch_loss = 0

            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                learner_optimizer.zero_grad()

                # Base task loss
                output = learner(data)
                base_loss = criterion(output, target)

                # Explanation loss with critic
                explanations = compute_explanations(learner, data, target)
                critic_output = critic(explanations)
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
                data, target = data.to(device), target.to(device)
                output = learner(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        accuracy = 100 * correct / total
        val_accuracies.append(accuracy)

        print(
            f'Validation Accuracy after LSX iteration {iteration+1}: {accuracy:.2f}%')

    # Final validation
    learner.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = learner(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    final_accuracy = 100 * correct / total
    val_accuracies.append(final_accuracy)

    print(f'Final Validation Accuracy: {final_accuracy:.2f}%')

    return val_accuracies

# Function to evaluate explanation separability (Ridge Regression)


def evaluate_explanation_separability(model, test_loader):
    """
    Evaluate how separable the explanations are by training a ridge regression model.
    """
    model.eval()
    model.to(device)

    # Generate explanations for test data
    explanations = []
    labels = []

    # Limit to 1000 samples for speed
    sample_count = 0
    max_samples = 1000

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            # Generate explanations
            batch_explanations = compute_explanations(model, data, target)

            # Flatten explanations
            batch_explanations = batch_explanations.view(
                batch_explanations.size(0), -1).cpu().numpy()

            explanations.append(batch_explanations)
            labels.append(target.cpu().numpy())

            sample_count += data.size(0)
            if sample_count >= max_samples:
                break

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

# Function to visualize explanations


def visualize_explanations(model, data_loader, title, num_samples=5):
    """
    Visualize explanations for random samples.
    """
    model.eval()
    model.to(device)

    # Get samples
    samples = []
    for data, target in data_loader:
        data, target = data.to(device), target.to(device)
        for i in range(min(num_samples, data.size(0))):
            samples.append((data[i:i+1], target[i:i+1]))
        if len(samples) >= num_samples:
            break

    # Set up plot
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 3*num_samples))

    # Generate and plot explanations
    for i, (sample, target) in enumerate(samples):
        # Generate explanation
        explanation = compute_explanations(model, sample, target)

        # Original image
        axes[i, 0].imshow(sample.squeeze().cpu().numpy(), cmap='gray')
        axes[i, 0].set_title(f"Class: {target.item()}")
        axes[i, 0].axis('off')

        # Explanation
        # Sum attribution across channels and take absolute value for visualization
        attr_vis = explanation.squeeze().abs().sum(dim=0).cpu().numpy()
        im = axes[i, 1].imshow(attr_vis, cmap='hot')
        axes[i, 1].set_title("Attribution Map")
        axes[i, 1].axis('off')

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    return fig

# Main function to run comparison


def run_comparison(data_fraction=0.05):
    """
    Run experiment comparing vanilla CNN with CNN-LSX.
    """
    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = datasets.MNIST(
        './data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    # Use a subset of data
    if data_fraction < 1.0:
        train_size = int(len(train_dataset) * data_fraction)
        indices = torch.randperm(len(train_dataset))[:train_size]
        train_dataset = torch.utils.data.Subset(train_dataset, indices)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    print(
        f"Training with {len(train_dataset)} samples (approx. {data_fraction*60000:.0f} images)")

    # Create models
    vanilla_model = SimpleCNN()
    lsx_model = SimpleCNN()
    critic_model = SimpleCNN()

    # Training parameters
    epochs = 3
    lsx_iterations = 2

    # Train vanilla CNN
    print("\n=== Training Vanilla CNN ===")
    vanilla_losses, vanilla_accuracies = train_vanilla_cnn(
        vanilla_model, train_loader, test_loader, epochs=epochs
    )

    # Train with LSX
    print("\n=== Training with CNN-LSX ===")
    lsx_accuracies = train_lsx(
        lsx_model, critic_model, train_loader, test_loader,
        epochs=epochs, lsx_iterations=lsx_iterations
    )

    # Evaluate explanation separability
    print("\n=== Evaluating Explanation Separability ===")
    vanilla_separability = evaluate_explanation_separability(
        vanilla_model, test_loader)
    lsx_separability = evaluate_explanation_separability(
        lsx_model, test_loader)

    # Visualize explanations
    print("\n=== Visualizing Explanations ===")
    vanilla_vis = visualize_explanations(vanilla_model, test_loader,
                                         "Vanilla CNN Explanations", num_samples=3)
    lsx_vis = visualize_explanations(lsx_model, test_loader,
                                     "CNN-LSX Explanations", num_samples=3)

    # Print summary
    print("\n=== Results Summary ===")
    print(f"Vanilla CNN Final Accuracy: {vanilla_accuracies[-1]:.2f}%")
    print(f"CNN-LSX Final Accuracy: {lsx_accuracies[-1]:.2f}%")
    print(
        f"Accuracy Improvement: {lsx_accuracies[-1] - vanilla_accuracies[-1]:.2f}%")

    print("\nExplanation Separability (Ridge Regression Accuracy):")
    print(f"Vanilla CNN: {vanilla_separability:.4f}")
    print(f"CNN-LSX: {lsx_separability:.4f}")
    print(
        f"Separability Improvement: {lsx_separability - vanilla_separability:.4f}")

    # Plot accuracy comparison
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(vanilla_accuracies) + 1),
             vanilla_accuracies, 'bo-', label='Vanilla CNN')

    # For LSX, we need to map the epochs correctly
    # Initial epochs + epochs after each LSX iteration
    lsx_epochs = list(range(1, epochs + 1))
    for i in range(lsx_iterations):
        lsx_epochs.append(epochs + i + 1)
    lsx_epochs.append(epochs + lsx_iterations + 1)  # Final point

    plt.plot(lsx_epochs, lsx_accuracies, 'ro-', label='CNN-LSX')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Accuracy (%)')
    plt.title(
        f'Accuracy Comparison with {data_fraction*100:.0f}% of Training Data')
    plt.legend()
    plt.grid(True)
    plt.savefig('accuracy_comparison.png')

    # Save visualization figures
    vanilla_vis.savefig('vanilla_explanations.png')
    lsx_vis.savefig('lsx_explanations.png')

    plt.show()

    return {
        'vanilla_acc': vanilla_accuracies[-1],
        'lsx_acc': lsx_accuracies[-1],
        'vanilla_separability': vanilla_separability,
        'lsx_separability': lsx_separability
    }


if __name__ == "__main__":
    # Run with 5% of MNIST data (3,000 samples)
    results = run_comparison(data_fraction=0.05)
