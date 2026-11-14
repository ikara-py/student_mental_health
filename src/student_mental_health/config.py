
import pandas as pd
from pathlib import Path


TARGET_COLUMN = 'Mental_Health_Risk'

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DATA_DIR = PROJECT_ROOT / 'data' / 'processed'


RANDOM_SEED = 42


def print_title(text='not provided'):
    print('=' * 50)
    print(text)
    print('=' * 50)


def load_raw():
    file_path = PROJECT_ROOT / 'data' / 'raw' / 'student-mental-health.csv'
    return pd.read_csv(file_path)


def get_columns(df, include):
    return df.select_dtypes(include=include).columns.to_list()
