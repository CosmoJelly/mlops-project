from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
import sys
import os

# Set correct working directory for Python imports if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# === Import your local functions ===
from collectData import main as collect_data_main
from processData import main as process_data_main
from main import main as train_model_main

default_args = {
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
}

with DAG(
    dag_id='mlops_pipeline',
    default_args=default_args,
    schedule_interval='@daily',
    catchup=False,
    tags=['mlops'],
) as dag:

    collect_data = PythonOperator(
        task_id='collect_data',
        python_callable=collect_data_main
    )

    process_data = PythonOperator(
        task_id='process_data',
        python_callable=process_data_main
    )

    train_model = PythonOperator(
        task_id='train_model',
        python_callable=train_model_main
    )

    # === Task Order ===
    collect_data >> process_data >> train_model
