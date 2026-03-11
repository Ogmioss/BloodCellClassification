"""
Blood Cell Data Validation DAG (API Version)

LIGHTWEIGHT DAG - No ML dependencies.
Calls FastAPI for data validation.

Orchestrates data validation:
1. Check API health
2. Trigger data validation via API
3. Wait for completion
4. Report results

Schedule: Manual trigger
"""

import os
from datetime import datetime, timedelta

from airflow import DAG, Dataset
from airflow.operators.python import PythonOperator


# ============================================================
# Configuration
# ============================================================

FASTAPI_URL = os.getenv("FASTAPI_URL", "http://api:8000")

# Airflow Datasets — visible in the Datasets UI tab
DATASET_RAW = Dataset("file:///app/data/raw/bloodcells_dataset")
DATASET_PROCESSED = Dataset("file:///app/data/processed/bloodcells_dataset")

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

    print(f"Checking API health at {FASTAPI_URL}")
    wait_for_api(max_retries=30, retry_interval=2.0)

    print("API is healthy and ready")
    return True


def trigger_data_validation(**context) -> dict:
    """Trigger data validation via FastAPI and wait for completion."""
    from dags.utils.api_client import run_data_validation_and_wait

    # Get config from DAG run conf
    dag_run = context.get("dag_run")
    conf = dag_run.conf if dag_run else {}

    dataset_path = conf.get("dataset_path")
    check_images = conf.get("check_images", False)
    max_images_to_check = conf.get("max_images_to_check")

    print("Starting data validation via API...")
    if dataset_path:
        print(f"   Dataset path: {dataset_path}")
    print(f"   Check images: {check_images}")

    result = run_data_validation_and_wait(
        dataset_path=dataset_path,
        check_images=check_images,
        max_images_to_check=max_images_to_check,
    )

    # Extract results
    task_result = result.get("result", {})
    is_valid = task_result.get("is_valid", False)
    total_images = task_result.get("total_images", 0)
    class_distribution = task_result.get("class_distribution", {})
    warnings = task_result.get("warnings", [])
    errors = task_result.get("errors", [])

    # Push to XCom
    ti = context["ti"]
    ti.xcom_push(key="is_valid", value=is_valid)
    ti.xcom_push(key="total_images", value=total_images)
    ti.xcom_push(key="class_distribution", value=class_distribution)
    ti.xcom_push(key="warnings", value=warnings)
    ti.xcom_push(key="errors", value=errors)

    return task_result


def report_results(**context):
    """Report final validation results."""
    ti = context["ti"]

    is_valid = ti.xcom_pull(task_ids="trigger_data_validation", key="is_valid")
    total_images = ti.xcom_pull(task_ids="trigger_data_validation", key="total_images")
    class_distribution = ti.xcom_pull(task_ids="trigger_data_validation", key="class_distribution")
    warnings = ti.xcom_pull(task_ids="trigger_data_validation", key="warnings")
    errors = ti.xcom_pull(task_ids="trigger_data_validation", key="errors")

    print("\n" + "=" * 50)
    print("Data Validation Report")
    print("=" * 50)
    print(f"   Valid: {is_valid}")
    print(f"   Total images: {total_images}")

    if class_distribution:
        print("\n   Class distribution:")
        for class_name, count in class_distribution.items():
            print(f"     {class_name}: {count}")

    if warnings:
        print(f"\n   Warnings ({len(warnings)}):")
        for w in warnings:
            print(f"     - {w}")

    if errors:
        print(f"\n   Errors ({len(errors)}):")
        for e in errors:
            print(f"     - {e}")

    print("=" * 50)

    if not is_valid:
        raise ValueError(f"Data validation failed with {len(errors)} error(s)")


# ============================================================
# DAG Definition
# ============================================================

with DAG(
    dag_id="bloodcells_data_validation_api",
    default_args=default_args,
    description="Validate dataset quality via FastAPI (lightweight)",
    schedule_interval=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["mlops", "validation", "bloodcells", "api"],
    doc_md=__doc__,
) as dag:

    check_api = PythonOperator(
        task_id="check_api_health",
        python_callable=check_api_health,
        provide_context=True,
    )

    validate = PythonOperator(
        task_id="trigger_data_validation",
        python_callable=trigger_data_validation,
        provide_context=True,
    )

    report = PythonOperator(
        task_id="report_results",
        python_callable=report_results,
        provide_context=True,
        outlets=[DATASET_RAW, DATASET_PROCESSED],
    )

    check_api >> validate >> report
