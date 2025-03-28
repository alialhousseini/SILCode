import pandas as pd
from sklearn.preprocessing import LabelEncoder
from .base_loader import encode_categoricals
from utils.dataloader_utils import to_dataloaders


def load_adult_income(batch_size=64):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    columns = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation",
               "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]
    df = pd.read_csv(url, names=columns, na_values=" ?", skipinitialspace=True)
    df.dropna(inplace=True)
    y = LabelEncoder().fit_transform(df['income'])
    X = encode_categoricals(df.drop(columns=['income']))
    return to_dataloaders(X, y, batch_size)
