"""
Blood Cell Batch Inference DAG (API Version)

LIGHTWEIGHT DAG - No PyTorch dependencies.
Calls FastAPI for all ML tasks.

Orchestrates batch inference:
1. Check API health
2. Trigger batch inference via API
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


def trigger_batch_inference(**context) -> dict:
    """Trigger batch inference via FastAPI and wait for completion."""
    from dags.utils.api_client import run_batch_inference_and_wait
    
    # Get config from DAG run conf
    dag_run = context.get("dag_run")
    conf = dag_run.conf if dag_run else {}
    
    input_dir = conf.get("input_dir")
    output_dir = conf.get("output_dir")
    max_images = conf.get("max_images")
    
    print("🚀 Starting batch inference via API...")
    if input_dir:
        print(f"   Input directory: {input_dir}")
    if output_dir:
        print(f"   Output directory: {output_dir}")
    if max_images:
        print(f"   Max images: {max_images}")
    
    result = run_batch_inference_and_wait(
        input_dir=input_dir,
        output_dir=output_dir,
        max_images=max_images,
    )
    
    # Extract results
    task_result = result.get("result", {})
    total_images = task_result.get("total_images", 0)
    successful = task_result.get("successful", 0)
    failed = task_result.get("failed", 0)
    output_file = task_result.get("output_file")
    summary = task_result.get("summary", {})
    
    print("\n✅ Batch inference completed!")
    print(f"   Total images: {total_images}")
    print(f"   Successful: {successful}")
    print(f"   Failed: {failed}")
    print(f"   Output file: {output_file}")
    
    if summary:
        print("\n📊 Summary by class:")
        for class_name, count in summary.items():
            print(f"   {class_name}: {count}")
    
    # Push to XCom
    context["ti"].xcom_push(key="total_images", value=total_images)
    context["ti"].xcom_push(key="successful", value=successful)
    context["ti"].xcom_push(key="failed", value=failed)
    context["ti"].xcom_push(key="output_file", value=output_file)
    context["ti"].xcom_push(key="summary", value=summary)
    
    return task_result


def report_results(**context):
    """Report final results."""
    ti = context["ti"]
    
    total_images = ti.xcom_pull(task_ids="trigger_batch_inference", key="total_images")
    successful = ti.xcom_pull(task_ids="trigger_batch_inference", key="successful")
    failed = ti.xcom_pull(task_ids="trigger_batch_inference", key="failed")
    output_file = ti.xcom_pull(task_ids="trigger_batch_inference", key="output_file")
    ti.xcom_pull(task_ids="trigger_batch_inference", key="summary")

    print("\n" + "=" * 50)
    print("🎉 Batch Inference Pipeline Complete!")
    print("=" * 50)
    print(f"   Total images processed: {total_images}")
    print(f"   Successful: {successful}")
    print(f"   Failed: {failed}")
    print(f"   Success rate: {successful / total_images * 100:.1f}%" if total_images else "N/A")
    print(f"   Results saved to: {output_file}")
    print("=" * 50)


# ============================================================
# DAG Definition
# ============================================================

with DAG(
    dag_id="bloodcells_batch_inference_api",
    default_args=default_args,
    description="Run batch inference via FastAPI (lightweight)",
    schedule_interval=None,  # Manual trigger
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["mlops", "inference", "bloodcells", "api"],
    doc_md=__doc__,
) as dag:
    
    # Task: Check API health
    check_api = PythonOperator(
        task_id="check_api_health",
        python_callable=check_api_health,
        provide_context=True,
    )
    
    # Task: Trigger batch inference
    inference = PythonOperator(
        task_id="trigger_batch_inference",
        python_callable=trigger_batch_inference,
        provide_context=True,
    )
    
    # Task: Report results
    report = PythonOperator(
        task_id="report_results",
        python_callable=report_results,
        provide_context=True,
    )
    
    # Define task dependencies
    check_api >> inference >> report
