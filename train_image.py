import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from data_loader import load_mnist, load_cifar10
from models.mnist_net import MNISTNet
from models.cifar_net import CIFARNet


def get_image_dataset_and_model(name, batch_size):
    if name == "mnist":
        return load_mnist(batch_size), MNISTNet()
    elif name == "cifar":
        return load_cifar10(batch_size), CIFARNet()
    else:
        raise ValueError(f"Unknown dataset: {name}")


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    pbar = tqdm(loader, desc="Training", leave=False)
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == y).sum().item()
        total += y.size(0)
        pbar.set_postfix({"loss": loss.item()})

    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device, desc="Evaluating"):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        pbar = tqdm(loader, desc=desc, leave=False)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)
            total_loss += loss.item() * x.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            pbar.set_postfix({"loss": loss.item()})

    return total_loss / total, correct / total


def save_model(model, path):
    torch.save(model.state_dict(), path)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (train_loader, val_loader, test_loader), model = get_image_dataset_and_model(
        args.dataset, args.batch_size)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = getattr(optim, args.optimizer)(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3)

    best_val_acc = 0
    log_dir = Path("logs")
    model_dir = Path("models")
    log_dir.mkdir(exist_ok=True)
    model_dir.mkdir(exist_ok=True)

    log_file = log_dir / \
        f"{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    with open(log_file, "w") as f:
        for epoch in range(args.epochs):
            print(f"\nEpoch {epoch + 1}/{args.epochs}")
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = evaluate(
                model, val_loader, criterion, device, desc="Validation")
            scheduler.step(val_loss)

            log = f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}"
            print(log)
            f.write(log + "\n")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_model(model, model_dir / f"{args.dataset}_best.pt")

    print("\nTesting best model...")
    model.load_state_dict(torch.load(model_dir / f"{args.dataset}_best.pt"))
    test_loss, test_acc = evaluate(
        model, test_loader, criterion, device, desc="Testing")
    print(f"Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['mnist', 'cifar'])
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--optimizer', type=str, default='Adam',
                        choices=['Adam', 'SGD', 'RMSprop'])
    args = parser.parse_args()

    main(args)
