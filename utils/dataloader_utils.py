from .tabular_dataset import TabularDataset
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, random_split
import pandas as pd


def smart_normalize(df, threshold=2):
    df_copy = df.copy()
    numeric_cols = df_copy.select_dtypes(
        include=['float64', 'float32', 'int']).columns

    cols_to_normalize = []
    for col in numeric_cols:
        unique_vals = df_copy[col].nunique()
        if unique_vals > threshold:  # Avoid normalizing binary or near-binary columns
            cols_to_normalize.append(col)

    scaler = StandardScaler()
    df_copy[cols_to_normalize] = scaler.fit_transform(
        df_copy[cols_to_normalize])
    return df_copy


def to_dataloaders(X, y, batch_size=64, val_ratio=0.12, test_ratio=0.12, normalize=False):
    if normalize:
        X = smart_normalize(X)

    dataset = TabularDataset(X, y)
    total = len(dataset)
    val_size = int(val_ratio * total)
    test_size = int(test_ratio * total)
    train_size = total - val_size - test_size

    train_ds, val_ds, test_ds = random_split(
        dataset, [train_size, val_size, test_size])

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size),
        DataLoader(test_ds, batch_size=batch_size)
    )
