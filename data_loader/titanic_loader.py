import seaborn as sns
from .base_loader import encode_categoricals
from utils.dataloader_utils import to_dataloaders


def load_titanic(batch_size=64):
    df = sns.load_dataset('titanic').dropna()
    y = df['survived']
    X = encode_categoricals(df.drop(columns=['survived']))
    return to_dataloaders(X, y, batch_size)
