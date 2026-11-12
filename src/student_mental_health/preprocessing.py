# imports

from collections import defaultdict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



def lower_upper(df, col):
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    return lower_bound, upper_bound

# Function to identify the outliers

def check_outliers(df, numerical_columns):
    result = defaultdict(list)
    for col in numerical_columns:

        lower_bound, upper_bound = lower_upper(df, col)
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]

        result[col] = outliers
    return result


def metrics(y_test, y_pred):
    {'Accuracy': accuracy_score(y_test, y_pred), 
    'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0), 
    'Recall': recall_score(y_test, y_pred, average='weighted', zero_division=0), 
    'F1-Score': f1_score(y_test, y_pred, average='weighted', zero_division=0) }