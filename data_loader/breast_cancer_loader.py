from sklearn.datasets import load_breast_cancer
import pandas as pd
from utils.dataloader_utils import to_dataloaders


def load_breast_cancer_dataset(batch_size=64):
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    return to_dataloaders(X, y, batch_size)
