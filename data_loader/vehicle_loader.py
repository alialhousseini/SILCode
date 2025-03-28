import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder
from utils.dataloader_utils import to_dataloaders


def load_vehicle_dataset(batch_size=64):
    df = fetch_openml(name="vehicle", version=1, as_frame=True).frame
    label_col = [col for col in df.columns if 'class' in col.lower()][0]
    y = LabelEncoder().fit_transform(df[label_col])
    X = df.drop(columns=[label_col])
    return to_dataloaders(X, y, batch_size)
