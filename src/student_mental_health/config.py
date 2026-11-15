'''Imports pandas for DataFrame operations and Path from pathlib 
   for platform-independent filesystem path management.'''

import pandas as pd
from pathlib import Path

'''Defines global project configurations, including the target column name (Mental_Health_Risk), 
   the dynamically resolved project root directory, the processed data output path, 
   and a fixed RANDOM_SEED (42) for reproducibility.'''

TARGET_COLUMN = 'Mental_Health_Risk'

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DATA_DIR = PROJECT_ROOT / 'data' / 'processed'


RANDOM_SEED = 42

'''Defines print_title() to format and print standardized console banners 
   wrapped in = dividers for clearer log and notebook output readability.'''

def print_title(text='not provided'):
    print('=' * 50)
    print(text)
    print('=' * 50)

'''Defines load_raw() to read the source CSV file (student-mental-health.csv) 
   from data/raw/ into a pandas DataFrame.'''

def load_raw():
    file_path = PROJECT_ROOT / 'data' / 'raw' / 'student-mental-health.csv'
    return pd.read_csv(file_path)

'''Defines get_columns() to filter a DataFrame by specified data types (include) and 
   return the matching column names as a Python list.'''

def get_columns(df, include):
    return df.select_dtypes(include=include).columns.to_list()
