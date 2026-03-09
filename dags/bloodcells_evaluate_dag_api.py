"""
Blood Cell Model Evaluation DAG (API Version)

LIGHTWEIGHT DAG - No PyTorch dependencies.
Calls FastAPI for all ML tasks.

Orchestrates model evaluation:
1. Check API health
2. Trigger evaluation via API
3. Wait for completion
4. Report results

Schedule: Manual trigger
"""

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator


# ============================================================
# Configuration
# ============================================================

FASTAPI_URL = os.getenv("FASTAPI_URL", "http://api:8000")

default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
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


def trigger_evaluation(**context) -> dict:
    """Trigger evaluation via FastAPI and wait for completion."""
    from dags.utils.api_client import run_evaluation_and_wait
    
    # Get config from DAG run conf
    dag_run = context.get("dag_run")
    conf = dag_run.conf if dag_run else {}
    
    checkpoint_path = conf.get("checkpoint_path")
    
    print("🚀 Starting evaluation via API...")
    if checkpoint_path:
        print(f"   Checkpoint: {checkpoint_path}")
    
    result = run_evaluation_and_wait(checkpoint_path=checkpoint_path)
    
    # Extract results
    task_result = result.get("result", {})
    test_accuracy = task_result.get("test_accuracy", 0)
    train_accuracy = task_result.get("train_accuracy", 0)
    val_accuracy = task_result.get("val_accuracy", 0)
    
    print("\n✅ Evaluation completed!")
    print(f"   Train Accuracy: {train_accuracy:.4f}")
    print(f"   Val Accuracy: {val_accuracy:.4f}")
    print(f"   Test Accuracy: {test_accuracy:.4f}")
    
    # Push to XCom
    context["ti"].xcom_push(key="test_accuracy", value=test_accuracy)
    context["ti"].xcom_push(key="train_accuracy", value=train_accuracy)
    context["ti"].xcom_push(key="val_accuracy", value=val_accuracy)
    
    return task_result


def report_results(**context):
    """Report final results."""
    ti = context["ti"]
    
    test_accuracy = ti.xcom_pull(task_ids="trigger_evaluation", key="test_accuracy")
    train_accuracy = ti.xcom_pull(task_ids="trigger_evaluation", key="train_accuracy")
    val_accuracy = ti.xcom_pull(task_ids="trigger_evaluation", key="val_accuracy")
    
    print("\n" + "=" * 50)
    print("🎉 Evaluation Pipeline Complete!")
    print("=" * 50)
    print(f"   Train Accuracy: {train_accuracy:.4f}")
    print(f"   Val Accuracy: {val_accuracy:.4f}")
    print(f"   Test Accuracy: {test_accuracy:.4f}")
    print("=" * 50)


# ============================================================
# DAG Definition
# ============================================================

with DAG(
    dag_id="bloodcells_evaluate_model_api",
    default_args=default_args,
    description="Evaluate blood cell classifier via FastAPI (lightweight)",
    schedule_interval=None,  # Manual trigger
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["mlops", "evaluation", "bloodcells", "api"],
    doc_md=__doc__,
) as dag:
    
    # Task: Check API health
    check_api = PythonOperator(
        task_id="check_api_health",
        python_callable=check_api_health,
        provide_context=True,
    )
    
    # Task: Trigger evaluation
    evaluate = PythonOperator(
        task_id="trigger_evaluation",
        python_callable=trigger_evaluation,
        provide_context=True,
    )
    
    # Task: Report results
    report = PythonOperator(
        task_id="report_results",
        python_callable=report_results,
        provide_context=True,
    )
    
    # Define task dependencies
    check_api >> evaluate >> report
