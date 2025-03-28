from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


def load_cifar10(batch_size=64, val_factor=0.15):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_data = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform)
    test_data = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform)

    val_size = int(val_factor * len(train_data))
    train_size = len(train_data) - val_size
    train_data, val_data = random_split(train_data, [train_size, val_size])

    return (
        DataLoader(train_data, batch_size=batch_size, shuffle=True),
        DataLoader(val_data, batch_size=batch_size),
        DataLoader(test_data, batch_size=batch_size)
    )
