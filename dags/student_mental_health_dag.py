'''Imports sys, datetime, and Airflow's DAG and @task TaskFlow decorators,
   then inserts /opt/airflow into the system path so the DAG can import the
   src.student_mental_health modules (preprocessing, training, and evaluation).'''

import sys
from datetime import datetime

from airflow import DAG
from airflow.decorators import task

sys.path.insert(0, '/opt/airflow')

from src.student_mental_health import preprocessing, training, evaluation

'''Initializes the student_mental_health_pipeline DAG with a manual trigger (schedule=None),
   disables historical backfilling (catchup=False), and applies organizational tags.'''

with DAG(dag_id='student_mental_health_pipeline', 
         start_date=datetime(2026, 11, 13), 
         schedule=None, 
         catchup=False, 
         tags=['student-mental-health', 'machine-learning']
) as dag:
    
    # Wraps each stage of the machine learning workflow into an Airflow 
    # TaskFlow task: data preparation, model training, and model evaluation.

    @task
    def prepare_data():
        return preprocessing.prepare_data()

    @task
    def train_model():
        return training.train_model()

    @task
    def evaluate_model():
        return evaluation.evaluate_model()

    # Instantiates the three tasks and chains them sequentially 
    # (prepare >> train >> evaluate) so each step runs only after the preceding step succeeds.

    prepare = prepare_data()
    train = train_model()
    evaluate = evaluate_model()

    prepare >> train >> evaluate
