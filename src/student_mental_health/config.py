import pandas as pd


TARGET_COLUMN = 'Mental_Health_Risk'

RANDOM_SEED = 42
def print_title(text='not provided'):
    print('=' * 50)
    print(text)
    print('=' * 50)

def load_raw():
    return pd.read_csv('../data/raw/student-mental-health.csv')

def get_columns(df, include):
    return df.select_dtypes(include=include).columns.to_list()