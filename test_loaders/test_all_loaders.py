# fmt: off
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data_loader import (
    load_mnist,
    load_cifar10,
    load_titanic,
    load_wine_dataset,
    load_adult_income,
    load_compas,
    load_vehicle_dataset,
    load_breast_cancer
)


def test_loader(name, loader_fn):
    print(f"\n--- Testing {name} ---")
    try:
        train_loader, val_loader, test_loader = loader_fn(batch_size=32)
        for x_batch, y_batch in train_loader:
            print(f"Train batch - X: {x_batch.shape}, Y: {y_batch.shape}")
            break
        for x_batch, y_batch in val_loader:
            print(f"Val batch - X: {x_batch.shape}, Y: {y_batch.shape}")
            break
        for x_batch, y_batch in test_loader:
            print(f"Test batch - X: {x_batch.shape}, Y: {y_batch.shape}")
            break
    except Exception as e:
        print(f"Error loading {name}: {e}")


def main():
    datasets = {
        "MNIST": load_mnist,
        "CIFAR-10": load_cifar10,
        "Titanic": load_titanic,
        "Wine": load_wine_dataset,
        "Adult Income": load_adult_income,
        "COMPAS": load_compas,
        "Vehicle": load_vehicle_dataset,
        "BreastCancer": load_breast_cancer
    }

    for name, loader_fn in datasets.items():
        test_loader(name, loader_fn)


if __name__ == "__main__":
    main()