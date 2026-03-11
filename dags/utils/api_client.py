"""
FastAPI Client for Airflow DAGs

Lightweight client to call FastAPI ML endpoints from Airflow.
This allows Airflow to trigger ML tasks without having PyTorch installed.
"""

import os
import time
from typing import Optional

import httpx


# API Configuration
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://api:8000")
API_TIMEOUT = 30.0  # seconds for initial request
POLL_INTERVAL = 5.0  # seconds between status checks
MAX_WAIT_TIME = 3600  # 1 hour max wait for task completion


class APIError(Exception):
    """Exception raised when API call fails."""
    pass


class TaskTimeoutError(Exception):
    """Exception raised when task takes too long."""
    pass


def _make_request(method: str, endpoint: str, **kwargs) -> dict:
    """Make HTTP request to FastAPI."""
    url = f"{FASTAPI_URL}{endpoint}"
    
    with httpx.Client(timeout=API_TIMEOUT) as client:
        if method == "GET":
            response = client.get(url, **kwargs)
        elif method == "POST":
            response = client.post(url, **kwargs)
        elif method == "DELETE":
            response = client.delete(url, **kwargs)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        if response.status_code >= 400:
            raise APIError(f"API error {response.status_code}: {response.text}")
        
        return response.json()


def check_api_health() -> bool:
    """Check if FastAPI is healthy."""
    try:
        result = _make_request("GET", "/health")
        return result.get("status") == "healthy"
    except Exception as e:
        print(f"API health check failed: {e}")
        return False


def wait_for_api(max_retries: int = 30, retry_interval: float = 2.0) -> bool:
    """Wait for FastAPI to be ready."""
    for i in range(max_retries):
        if check_api_health():
            print(f"✅ API is ready after {i * retry_interval}s")
            return True
        print(f"⏳ Waiting for API... ({i + 1}/{max_retries})")
        time.sleep(retry_interval)
    
    raise APIError(f"API not ready after {max_retries * retry_interval}s")


def list_datasets() -> list[dict]:
    """
    List available datasets from API.

    Returns:
        List of dataset info dicts (name, path, num_classes, classes, total_images)
    """
    return _make_request("GET", "/ml/datasets")


def start_training(
    dataset_path: Optional[str] = None,
    epochs: Optional[int] = None,
    learning_rate: Optional[float] = None,
    batch_size: Optional[int] = None,
) -> str:
    """
    Start a training task via API.

    Returns:
        task_id: ID of the started task
    """
    payload = {}
    if dataset_path is not None:
        payload["dataset_path"] = dataset_path
    if epochs is not None:
        payload["epochs"] = epochs
    if learning_rate is not None:
        payload["learning_rate"] = learning_rate
    if batch_size is not None:
        payload["batch_size"] = batch_size

    result = _make_request("POST", "/ml/train", json=payload)
    return result["task_id"]


def start_evaluation(checkpoint_path: Optional[str] = None) -> str:
    """
    Start an evaluation task via API.
    
    Returns:
        task_id: ID of the started task
    """
    payload = {}
    if checkpoint_path:
        payload["checkpoint_path"] = checkpoint_path
    
    result = _make_request("POST", "/ml/evaluate", json=payload)
    return result["task_id"]


def start_batch_inference(
    input_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    max_images: Optional[int] = None,
) -> str:
    """
    Start a batch inference task via API.
    
    Returns:
        task_id: ID of the started task
    """
    payload = {}
    if input_dir:
        payload["input_dir"] = input_dir
    if output_dir:
        payload["output_dir"] = output_dir
    if max_images:
        payload["max_images"] = max_images
    
    result = _make_request("POST", "/ml/batch-inference", json=payload)
    return result["task_id"]


def get_task_status(task_id: str) -> dict:
    """Get the status of a task."""
    return _make_request("GET", f"/ml/tasks/{task_id}")


def wait_for_task(
    task_id: str,
    poll_interval: float = POLL_INTERVAL,
    max_wait_time: float = MAX_WAIT_TIME,
) -> dict:
    """
    Wait for a task to complete.
    
    Args:
        task_id: ID of the task to wait for
        poll_interval: Seconds between status checks
        max_wait_time: Maximum seconds to wait
        
    Returns:
        Task result dict
        
    Raises:
        TaskTimeoutError: If task takes too long
        APIError: If task fails
    """
    start_time = time.time()
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > max_wait_time:
            raise TaskTimeoutError(f"Task {task_id} timed out after {max_wait_time}s")
        
        status = get_task_status(task_id)
        task_status = status.get("status")
        
        if task_status == "completed":
            print(f"✅ Task {task_id} completed in {elapsed:.1f}s")
            return status
        
        if task_status == "failed":
            error = status.get("error", "Unknown error")
            raise APIError(f"Task {task_id} failed: {error}")
        
        print(f"⏳ Task {task_id} status: {task_status} ({elapsed:.1f}s elapsed)")
        time.sleep(poll_interval)


def run_training_and_wait(**kwargs) -> dict:
    """Start training and wait for completion."""
    wait_for_api()
    task_id = start_training(**kwargs)
    print(f"🚀 Training started: task_id={task_id}")
    return wait_for_task(task_id)


def run_evaluation_and_wait(**kwargs) -> dict:
    """Start evaluation and wait for completion."""
    wait_for_api()
    task_id = start_evaluation(**kwargs)
    print(f"🚀 Evaluation started: task_id={task_id}")
    return wait_for_task(task_id)


def run_batch_inference_and_wait(**kwargs) -> dict:
    """Start batch inference and wait for completion."""
    wait_for_api()
    task_id = start_batch_inference(**kwargs)
    print(f"🚀 Batch inference started: task_id={task_id}")
    return wait_for_task(task_id)


def start_data_validation(
    dataset_path: Optional[str] = None,
    check_images: bool = False,
    max_images_to_check: Optional[int] = None,
) -> str:
    """
    Start a data validation task via API.

    Returns:
        task_id: ID of the started task
    """
    payload: dict = {"check_images": check_images}
    if dataset_path:
        payload["dataset_path"] = dataset_path
    if max_images_to_check:
        payload["max_images_to_check"] = max_images_to_check

    result = _make_request("POST", "/ml/validate-data", json=payload)
    return result["task_id"]


def run_data_validation_and_wait(**kwargs) -> dict:
    """Start data validation and wait for completion."""
    wait_for_api()
    task_id = start_data_validation(**kwargs)
    print(f"🚀 Data validation started: task_id={task_id}")
    return wait_for_task(task_id)


def predict_image(image_base64: str) -> dict:
    """
    Make a single prediction via API.
    
    Args:
        image_base64: Base64 encoded image
        
    Returns:
        Prediction result
    """
    result = _make_request("POST", "/predict", json={"image_base64": image_base64})
    return result


def get_model_info() -> dict:
    """Get model information from API."""
    return _make_request("GET", "/model/info")
