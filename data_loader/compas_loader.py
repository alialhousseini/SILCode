import pandas as pd
from sklearn.preprocessing import LabelEncoder
from .base_loader import encode_categoricals
from utils.dataloader_utils import to_dataloaders


def load_compas(batch_size=64):
    url = "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"
    df = pd.read_csv(url)
    df = df[(df['days_b_screening_arrest'] <= 30) & (df['days_b_screening_arrest'] >= -30) &
            (df['is_recid'] != -1) & (df['c_charge_degree'] != 'O') & (df['score_text'] != 'N/A')]
    y = df['is_recid']
    drop_cols = [
        'id', 'name', 'first', 'last', 'compas_screening_date', 'dob',
        'c_jail_in', 'c_jail_out', 'c_case_number', 'screening_date'
    ]
    df.drop(
        columns=[col for col in drop_cols if col in df.columns], inplace=True)
    selected_features = ['sex', 'age', 'age_cat', 'race', 'juv_fel_count',
                         'juv_misd_count', 'juv_other_count', 'priors_count',
                         'c_charge_degree', 'two_year_recid', 'is_recid']
    df = df[selected_features]
    X = encode_categoricals(df.drop(columns=['is_recid']))
    return to_dataloaders(X, y, batch_size)
