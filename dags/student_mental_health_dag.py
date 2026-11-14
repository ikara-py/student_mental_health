
import sys
from datetime import datetime

from airflow import DAG
from airflow.decorators import task

sys.path.insert(0, '/opt/airflow')

from src.student_mental_health import preprocessing, training, evaluation


with DAG(
    dag_id='student_mental_health_pipeline',
    start_date=datetime(2026, 11, 13),
    schedule=None,
    catchup=False,
    tags=['student-mental-health', 'machine-learning']
) as dag:

    @task
    def prepare_data():
        return preprocessing.prepare_data()

    @task
    def train_model():
        return training.train_model()

    @task
    def evaluate_model():
        return evaluation.evaluate_model()

    # Define workflow
    prepare = prepare_data()
    train = train_model()
    evaluate = evaluate_model()

    prepare >> train >> evaluate
