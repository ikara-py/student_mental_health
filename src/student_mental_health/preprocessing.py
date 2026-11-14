
import pandas as pd

from sklearn.model_selection import train_test_split

from src.student_mental_health import config



def clean_data(df):

    # Clean inconsistent values in the dataset.
    df = df.copy()

    # Ensure Social Media Hours does not exceed Screen Time Hours
    mask = df['Social_Media_Hours'] > df['Screen_Time_Hours']

    df.loc[mask, 'Social_Media_Hours'] = (df.loc[mask, 'Screen_Time_Hours'] )

    return df


def split_data(df, test_size=0.2):
    # Split the dataset into stratified training and testing sets.

    X = df.drop(columns=[config.TARGET_COLUMN, 'Student_ID'] )

    y = df[config.TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=config.RANDOM_SEED, stratify=y )

    return X_train, X_test, y_train, y_test


def lower_upper(df, col):
    # Calculate IQR lower and upper boundaries.

    q1, q3 = df[col].quantile([0.25, 0.75])

    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    return lower_bound, upper_bound


def check_outliers(df, numerical_columns):
    # Identify outliers in numerical columns.

    result = {}

    for col in numerical_columns:

        lower_bound, upper_bound = lower_upper(df, col)

        outliers = df[
            (df[col] < lower_bound) |
            (df[col] > upper_bound)
        ]

        result[col] = outliers

    return result


def cap_outliers(X_train, X_test, numerical_columns):
    # Cap outliers using boundaries calculated from training data.

    X_train = X_train.copy()
    X_test = X_test.copy()

    bounds = {}

    for col in numerical_columns:

        lower_bound, upper_bound = lower_upper(X_train, col)

        bounds[col] = (lower_bound, upper_bound)

        X_train[col] = X_train[col].clip(lower=lower_bound, upper=upper_bound )

        X_test[col] = X_test[col].clip(lower=lower_bound, upper=upper_bound )

    return X_train, X_test


def prepare_data():
    # Load raw dataset
    df = config.load_raw()

    # Clean inconsistent values
    df = clean_data(df)

    # Split dataset
    X_train, X_test, y_train, y_test = split_data(df)

    # Identify numerical columns
    numerical_columns = config.get_columns(X_train, 'number')

    # Detect columns containing outliers
    outliers = check_outliers(X_train, numerical_columns)

    columns_containing_outliers = [
        col for col, values in outliers.items()
        if not values.empty
    ]

    # Cap outliers
    X_train, X_test = cap_outliers(
        X_train,
        X_test,
        columns_containing_outliers
    )

    # Rebuild training and testing DataFrames
    train_df = X_train.copy()
    train_df[config.TARGET_COLUMN] = y_train

    test_df = X_test.copy()
    test_df[config.TARGET_COLUMN] = y_test

    # Save processed datasets
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    train_path = config.PROCESSED_DATA_DIR / 'train.csv'
    test_path = config.PROCESSED_DATA_DIR / 'test.csv'

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print('Data preparation completed successfully.')

    return str(train_path), str(test_path)