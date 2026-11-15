'''Imports pandas for tabular data handling, joblib for loading serialized model artifacts, 
   Scikit-learn classification metrics (accuracy_score, precision_score, recall_score, f1_score), 
   and the project's config module.'''

import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.student_mental_health import config

def metrics(y_test, y_pred):
    return {
            'Accuracy': accuracy_score(y_test, y_pred ),

            'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0 ),

            'Recall': recall_score( y_test, y_pred, average='weighted', zero_division=0 ),

            'F1-Score': f1_score( y_test, y_pred, average='weighted', zero_division=0) }

'''Defines metrics() to compute Accuracy along with weighted Precision, Recall, and F1-Score (
   zero_division=0 prevents warnings when a class has no predicted samples), 
   returning the results as a dictionary.'''

def metrics(y_test, y_pred):
    return {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(
            y_test, y_pred, average='weighted', zero_division=0
        ),
        'Recall': recall_score(
            y_test, y_pred, average='weighted', zero_division=0
        ),
        'F1-Score': f1_score(
            y_test, y_pred, average='weighted', zero_division=0
        )
    }




'''Defines evaluate_model() to load the processed test dataset (test.csv) and the saved 
   logistic_regression.joblib model, generate predictions, compute and print evaluation metrics,
   export the scores to reports/metrics.csv, and return the output file path.'''

def evaluate_model():

    # Load test data
    test_path = config.PROCESSED_DATA_DIR / 'test.csv'
    test_df = pd.read_csv(test_path)

    # Separate features and target
    X_test = test_df.drop(columns=[config.TARGET_COLUMN])
    y_test = test_df[config.TARGET_COLUMN]

    # Load the trained model
    model_path = config.PROJECT_ROOT / 'models' / 'logistic_regression.joblib'
    model = joblib.load(model_path)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    results = metrics(y_test, y_pred)

    # Display results
    print('Model Evaluation Results')
    for metric, value in results.items():
        print(f'{metric}: {value:.4f}')

    # Save results
    reports_dir = config.PROJECT_ROOT / 'reports'
    reports_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame([results])
    results_path = reports_dir / 'metrics.csv'
    results_df.to_csv(results_path, index=False)

    print(f'Metrics saved to: {results_path}')

    return str(results_path)
