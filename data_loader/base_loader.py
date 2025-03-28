import pandas as pd
from sklearn.preprocessing import StandardScaler


def encode_categoricals(df):
    return pd.get_dummies(df).astype('float32')


def normalize(df):
    scaler = StandardScaler()
    return pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
