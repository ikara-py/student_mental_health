from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def metrics(y_test, y_pred):
    return {
            'Accuracy': accuracy_score(y_test, y_pred ),

            'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0 ),

            'Recall': recall_score( y_test, y_pred, average='weighted', zero_division=0 ),

            'F1-Score': f1_score( y_test, y_pred, average='weighted', zero_division=0) }




import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from src.student_mental_health import config


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




# airflow function

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
