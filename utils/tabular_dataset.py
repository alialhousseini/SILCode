import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd


class TabularDataset(Dataset):
    def __init__(self, X, y):
        if isinstance(X, pd.DataFrame):
            self.X = torch.tensor(X.values, dtype=torch.float32)
        elif isinstance(X, np.ndarray):
            self.X = torch.tensor(X, dtype=torch.float32)
        else:
            raise TypeError("Unsupported X type")

        if isinstance(y, pd.Series):
            self.y = torch.tensor(y.values, dtype=torch.long)
        elif isinstance(y, np.ndarray):
            self.y = torch.tensor(y, dtype=torch.long)
        else:
            raise TypeError("Unsupported y type")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
