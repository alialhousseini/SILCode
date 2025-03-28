# models/adult_net.py
import torch.nn as nn
import torch.nn.functional as F


class AdultNet(nn.Module):
    def __init__(self, input_dim=108, hidden_dims=(128, 64), num_classes=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.bn1 = nn.BatchNorm1d(hidden_dims[0])
        self.drop1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.bn2 = nn.BatchNorm1d(hidden_dims[1])
        self.drop2 = nn.Dropout(0.3)

        self.out = nn.Linear(hidden_dims[1], num_classes)

    def forward(self, x):
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        return self.out(x)
