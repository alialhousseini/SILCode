from sklearn.datasets import load_wine
import pandas as pd
from utils.dataloader_utils import to_dataloaders


def load_wine_dataset(batch_size=64):
    data = load_wine()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    return to_dataloaders(X, y, batch_size)
