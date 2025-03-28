# models/cifar_net.py
import torch.nn as nn
import torch.nn.functional as F


class CIFARNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)       # 3x32x32 -> 6x28x28
        self.pool = nn.MaxPool2d(2, 2)        # -> 6x14x14
        self.conv2 = nn.Conv2d(6, 16, 5)      # -> 16x10x10 -> 16x5x5

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.out = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)
