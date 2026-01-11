"""
Example ML Pipeline DAG
Demonstrates breast cancer detection workflow with MLflow tracking
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.providers.docker.operators.docker import DockerOperator
import os

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def load_data_from_s3(**context):
    """Load training data from S3/Spaces"""
    s3_hook = S3Hook(aws_conn_id='s3_default')
    
    # Example: Download dataset
    bucket_name = 'breast-cancer-detection-ml'
    key = 'datasets/training_data.csv'
    
    print(f"Loading data from s3://{bucket_name}/{key}")
    # obj = s3_hook.get_key(key, bucket_name)
    print("Data loaded successfully!")
    
    # Push to XCom for next task
    context['ti'].xcom_push(key='data_path', value=f's3://{bucket_name}/{key}')

def preprocess_data(**context):
    """Preprocess the training data"""
    data_path = context['ti'].xcom_pull(key='data_path', task_ids='load_data')
    print(f"Preprocessing data from {data_path}")
    
    # Your preprocessing logic here
    print("Preprocessing complete!")
    context['ti'].xcom_push(key='processed_data_path', value='s3://bucket/processed_data.csv')

def train_model(**context):
    """Train model with MLflow tracking"""
    mlflow_uri = os.getenv('MLFLOW_TRACKING_URI')
    mlflow_experiment = os.getenv('MLFLOW_EXPERIMENT_NAME')
    
    print(f"Training model with MLflow: {mlflow_uri}")
    print(f"Experiment: {mlflow_experiment}")
    
    # Your training logic with MLflow here
    # import mlflow
    # mlflow.set_tracking_uri(mlflow_uri)
    # mlflow.set_experiment(mlflow_experiment)
    # with mlflow.start_run():
    #     # Training code
    #     mlflow.log_metric("accuracy", 0.95)
    
    print("Model training complete!")
    context['ti'].xcom_push(key='model_uri', value='s3://bucket/models/model.pkl')

def evaluate_model(**context):
    """Evaluate model performance"""
    model_uri = context['ti'].xcom_pull(key='model_uri', task_ids='train_model')
    print(f"Evaluating model: {model_uri}")
    
    # Your evaluation logic here
    print("Evaluation complete! Accuracy: 0.95")

def deploy_model(**context):
    """Deploy model to production"""
    model_uri = context['ti'].xcom_pull(key='model_uri', task_ids='train_model')
    print(f"Deploying model: {model_uri}")
    
    # Your deployment logic here
    print("Model deployed successfully!")

with DAG(
    'example_ml_pipeline',
    default_args=default_args,
    description='Example ML pipeline for breast cancer detection',
    schedule='@daily',
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['ml', 'example', 'breast-cancer'],
) as dag:
    
    load_task = PythonOperator(
        task_id='load_data',
        python_callable=load_data_from_s3,
    )
    
    preprocess_task = PythonOperator(
        task_id='preprocess_data',
        python_callable=preprocess_data,
    )
    
    train_task = PythonOperator(
        task_id='train_model',
        python_callable=train_model,
    )
    
    evaluate_task = PythonOperator(
        task_id='evaluate_model',
        python_callable=evaluate_model,
    )
    
    deploy_task = PythonOperator(
        task_id='deploy_model',
        python_callable=deploy_model,
    )
    
    # Docker task to run ML base container
    docker_task = DockerOperator(
        task_id='run_ml_base_container',
        image='pauloyuncha/ml-base:latest',
        command='sh -c "echo Running ML base container && python --version"',
        docker_url='unix://var/run/docker.sock',  # Use the Docker daemon from the host
        network_mode='bridge',
        auto_remove='success',  # Options: 'never', 'success', or 'force'
        mount_tmp_dir=False,
    )
    
    # Define task dependencies
    load_task >> preprocess_task >> train_task >> evaluate_task >> docker_task >> deploy_task
