"""
Blood Cell Classification Training DAG (API Version)

LIGHTWEIGHT DAG - No PyTorch dependencies.
Calls FastAPI for all ML tasks.

Orchestrates the training pipeline:
1. Check API health
2. Trigger training via API
3. Wait for training completion
4. Check metrics and decide on promotion
5. Optionally promote to Production

Schedule: Manual trigger or weekly
"""

import os
from datetime import datetime, timedelta

from airflow import DAG, Dataset
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.utils.trigger_rule import TriggerRule


# ============================================================
# Configuration
# ============================================================

FASTAPI_URL = os.getenv("FASTAPI_URL", "http://api:8000")
MIN_ACCURACY_THRESHOLD = 0.90

# Airflow Datasets — same URIs as in data_validation DAG
DATASET_RAW = Dataset("file:///app/data/raw/bloodcells_dataset")
DATASET_PROCESSED = Dataset("file:///app/data/processed/bloodcells_dataset")

default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


# ============================================================
# Task Functions
# ============================================================

def check_api_health(**context) -> bool:
    """Check if FastAPI is healthy and ready."""
    from dags.utils.api_client import wait_for_api
    
    print(f"🔍 Checking API health at {FASTAPI_URL}")
    wait_for_api(max_retries=30, retry_interval=2.0)
    
    print("✅ API is healthy and ready")
    return True


def trigger_training(**context) -> dict:
    """Trigger training via FastAPI and wait for completion."""
    from dags.utils.api_client import run_training_and_wait

    dag_conf = context.get("dag_run", {}) and context["dag_run"].conf or {}

    # Extract all supported training overrides from DAG conf
    training_kwargs = {}
    for key in ("dataset_path", "model_name", "epochs", "learning_rate", "batch_size"):
        value = dag_conf.get(key)
        if value is not None:
            training_kwargs[key] = value

    if training_kwargs:
        print(f"🔧 Training overrides from DAG conf: {training_kwargs}")
    else:
        print("🗂️  Using all defaults from conf.yaml")

    print("🚀 Starting training via API...")

    result = run_training_and_wait(**training_kwargs)
    
    # Extract results
    task_result = result.get("result", {})
    test_accuracy = task_result.get("test_accuracy", 0)
    best_val_acc = task_result.get("best_val_acc", 0)
    model_version = task_result.get("model_version")
    mlflow_run_id = task_result.get("mlflow_run_id")
    
    print("\n✅ Training completed!")
    print(f"   Test Accuracy: {test_accuracy:.4f}")
    print(f"   Best Val Accuracy: {best_val_acc:.4f}")
    print(f"   Model Version: {model_version}")
    print(f"   MLflow Run ID: {mlflow_run_id}")
    
    # Push to XCom
    context["ti"].xcom_push(key="test_accuracy", value=test_accuracy)
    context["ti"].xcom_push(key="best_val_acc", value=best_val_acc)
    context["ti"].xcom_push(key="model_version", value=model_version)
    context["ti"].xcom_push(key="mlflow_run_id", value=mlflow_run_id)
    
    return task_result


def decide_promotion(**context) -> str:
    """Decide whether to promote model to Production."""
    ti = context["ti"]
    test_accuracy = ti.xcom_pull(task_ids="trigger_training", key="test_accuracy")
    
    print(f"📊 Test accuracy: {test_accuracy:.4f}")
    print(f"📊 Threshold: {MIN_ACCURACY_THRESHOLD}")
    
    if test_accuracy and test_accuracy >= MIN_ACCURACY_THRESHOLD:
        print(f"✅ Accuracy {test_accuracy:.4f} >= {MIN_ACCURACY_THRESHOLD}, promoting to Staging")
        return "promote_model"
    else:
        print(f"⚠️ Accuracy {test_accuracy:.4f} < {MIN_ACCURACY_THRESHOLD}, skipping promotion")
        return "skip_promotion"


def promote_model(**context) -> dict:
    """Promote model to Production via API."""
    import httpx
    
    ti = context["ti"]
    model_version = ti.xcom_pull(task_ids="trigger_training", key="model_version")
    
    if not model_version:
        print("⚠️ No model version found, skipping promotion")
        return {"status": "skipped"}
    
    print(f"🚀 Promoting model version {model_version} to Staging...")

    with httpx.Client(timeout=30.0) as client:
        response = client.post(
            f"{FASTAPI_URL}/mlflow/promote/{model_version}",
            params={"stage": "Staging"},
        )

        if response.status_code == 200:
            print(f"✅ Model version {model_version} promoted to Staging")
            return response.json()
        else:
            print(f"❌ Promotion failed: {response.text}")
            raise Exception(f"Promotion failed: {response.text}")


def skip_promotion(**context):
    """Log that promotion was skipped."""
    ti = context["ti"]
    test_accuracy = ti.xcom_pull(task_ids="trigger_training", key="test_accuracy")
    print(f"⏭️ Promotion skipped. Accuracy {test_accuracy:.4f} below threshold {MIN_ACCURACY_THRESHOLD}")


def training_complete(**context):
    """Final task to mark training as complete."""
    ti = context["ti"]
    test_accuracy = ti.xcom_pull(task_ids="trigger_training", key="test_accuracy")
    model_version = ti.xcom_pull(task_ids="trigger_training", key="model_version")
    
    print("\n" + "=" * 50)
    print("🎉 Training Pipeline Complete!")
    print("=" * 50)
    print(f"   Test Accuracy: {test_accuracy:.4f}")
    print(f"   Model Version: {model_version}")
    print("=" * 50)


# ============================================================
# DAG Definition
# ============================================================

with DAG(
    dag_id="bloodcells_train_model_api",
    default_args=default_args,
    description="Train blood cell classifier via FastAPI (lightweight)",
    schedule_interval=None,  # Manual trigger
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["mlops", "training", "bloodcells", "api"],
    doc_md=__doc__,
) as dag:
    
    # Task: Check API health
    check_api = PythonOperator(
        task_id="check_api_health",
        python_callable=check_api_health,
        provide_context=True,
    )
    
    # Task: Trigger training (consumes datasets)
    train = PythonOperator(
        task_id="trigger_training",
        python_callable=trigger_training,
        provide_context=True,
        inlets=[DATASET_RAW, DATASET_PROCESSED],
    )
    
    # Task: Decide promotion
    decide = BranchPythonOperator(
        task_id="decide_promotion",
        python_callable=decide_promotion,
        provide_context=True,
    )
    
    # Task: Promote model
    promote = PythonOperator(
        task_id="promote_model",
        python_callable=promote_model,
        provide_context=True,
    )
    
    # Task: Skip promotion
    skip = PythonOperator(
        task_id="skip_promotion",
        python_callable=skip_promotion,
        provide_context=True,
    )
    
    # Task: Training complete
    complete = PythonOperator(
        task_id="training_complete",
        python_callable=training_complete,
        provide_context=True,
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )
    
    # Define task dependencies
    check_api >> train >> decide
    decide >> [promote, skip]
    [promote, skip] >> complete
