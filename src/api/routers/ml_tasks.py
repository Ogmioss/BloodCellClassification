"""
ML Tasks Router

Endpoints for executing ML tasks (train, evaluate, batch inference).
These endpoints run the actual ML workloads with PyTorch.
Airflow DAGs call these endpoints instead of running PyTorch directly.

Architecture note: This module uses deferred imports from src.pipe/ inside
background task functions (_run_training, _run_evaluation, _run_batch_inference).
This is intentional — pipe/ modules pull in heavy ML dependencies (torch, etc.)
and should only be loaded when a task is actually executed, not at API startup.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from src.api.task_store import TaskResult, TaskStatus, TaskStore


router = APIRouter(prefix="/ml", tags=["ML Tasks"])


# Persistent task storage (survives API restarts)
_store = TaskStore()


# ============================================================
# Request/Response Schemas
# ============================================================

class TrainRequest(BaseModel):
    """Request to start training."""
    epochs: Optional[int] = Field(None, description="Override number of epochs")
    learning_rate: Optional[float] = Field(None, description="Override learning rate")
    batch_size: Optional[int] = Field(None, description="Override batch size")


class EvaluateRequest(BaseModel):
    """Request to evaluate a model."""
    checkpoint_path: Optional[str] = Field(None, description="Path to checkpoint (default: best_model.pth)")


class BatchInferenceRequest(BaseModel):
    """Request for batch inference."""
    input_dir: Optional[str] = Field(None, description="Input directory with images")
    output_dir: Optional[str] = Field(None, description="Output directory for results")
    max_images: Optional[int] = Field(None, description="Maximum images to process")


class DataValidationRequest(BaseModel):
    """Request for data validation."""
    dataset_path: Optional[str] = Field(None, description="Path to dataset (default: bloodcells_dataset)")
    check_images: bool = Field(False, description="Also validate image integrity (slower)")
    max_images_to_check: Optional[int] = Field(None, description="Max images to check for integrity")


class TaskResponse(BaseModel):
    """Response when starting a task."""
    task_id: str
    task_type: str
    status: TaskStatus
    message: str


# ============================================================
# Background Task Runners
# ============================================================

def _run_training(task_id: str, config_overrides: dict):
    """Run training in background."""
    import torch
    torch.backends.mkldnn.enabled = False

    _store.update_task(
        task_id,
        status=TaskStatus.RUNNING,
        started_at=datetime.now().isoformat(),
    )

    try:
        from src.pipe.train_model import main as train_main

        # Note: config_overrides (epochs, lr, batch_size) are not yet passed to
        # train_main() because it reads from conf.yaml directly. To implement,
        # train_main() needs to accept an optional overrides dict.
        results = train_main()

        _store.update_task(
            task_id,
            status=TaskStatus.COMPLETED,
            completed_at=datetime.now().isoformat(),
            result={
                "test_accuracy": results["test_results"]["accuracy"],
                "best_val_acc": results["training_metrics"]["best_val_acc"],
                "model_version": results.get("model_version"),
                "mlflow_run_id": results.get("mlflow_run_id"),
            },
        )
    except Exception as e:
        _store.update_task(
            task_id,
            status=TaskStatus.FAILED,
            completed_at=datetime.now().isoformat(),
            error=str(e),
        )


def _run_evaluation(task_id: str, checkpoint_path: Optional[str]):
    """Run evaluation in background."""
    import torch
    torch.backends.mkldnn.enabled = False

    _store.update_task(
        task_id,
        status=TaskStatus.RUNNING,
        started_at=datetime.now().isoformat(),
    )

    try:
        from src.pipe.evaluate_model import main as evaluate_main

        results = evaluate_main()

        _store.update_task(
            task_id,
            status=TaskStatus.COMPLETED,
            completed_at=datetime.now().isoformat(),
            result={
                "test_accuracy": results["test_results"]["accuracy"],
                "train_accuracy": results["train_results"]["accuracy"],
                "val_accuracy": results["val_results"]["accuracy"],
            },
        )
    except Exception as e:
        _store.update_task(
            task_id,
            status=TaskStatus.FAILED,
            completed_at=datetime.now().isoformat(),
            error=str(e),
        )


def _run_batch_inference(task_id: str, input_dir: Optional[str], output_dir: Optional[str], max_images: Optional[int]):
    """Run batch inference in background."""
    import torch
    torch.backends.mkldnn.enabled = False

    _store.update_task(
        task_id,
        status=TaskStatus.RUNNING,
        started_at=datetime.now().isoformat(),
    )

    try:
        from src.pipe.batch_inference_pipeline import run_batch_inference
        from src.services.yaml_loader import YamlLoader

        loader = YamlLoader()

        # Use defaults if not provided
        if input_dir is None:
            input_dir = str(loader.data_dir / "raw" / "bloodcells_dataset")
        if output_dir is None:
            output_dir = str(loader.project_root / "models" / "inference_results")

        result = run_batch_inference(
            input_dir=Path(input_dir),
            output_dir=Path(output_dir),
            max_images=max_images,
            save_results=True,
            show_progress=False,
        )

        _store.update_task(
            task_id,
            status=TaskStatus.COMPLETED,
            completed_at=datetime.now().isoformat(),
            result={
                "total_images": result.total_images,
                "successful": result.successful,
                "failed": result.failed,
                "output_file": str(result.output_file) if result.output_file else None,
                "summary": f"{result.successful}/{result.total_images} images processed, {result.failed} failed",
            },
        )
    except Exception as e:
        _store.update_task(
            task_id,
            status=TaskStatus.FAILED,
            completed_at=datetime.now().isoformat(),
            error=str(e),
        )


def _run_data_validation(task_id: str, dataset_path: Optional[str], check_images: bool, max_images_to_check: Optional[int]):
    """Run data validation in background."""
    _store.update_task(
        task_id,
        status=TaskStatus.RUNNING,
        started_at=datetime.now().isoformat(),
    )

    try:
        from src.services.data_validation_service import DataValidationService
        from src.services.yaml_loader import YamlLoader

        loader = YamlLoader()

        if dataset_path is None:
            dataset_path = str(loader.data_dir / "raw" / "bloodcells_dataset")

        validator = DataValidationService()
        report = validator.validate_dataset(Path(dataset_path))

        result = report.to_dict()

        if check_images:
            image_report = validator.validate_images(
                Path(dataset_path),
                max_images_to_check=max_images_to_check,
            )
            result["image_validation"] = {
                "corrupted_images": image_report.corrupted_images,
                "invalid_size_images": image_report.invalid_size_images,
                "total_checked": image_report.total_images,
            }

        _store.update_task(
            task_id,
            status=TaskStatus.COMPLETED,
            completed_at=datetime.now().isoformat(),
            result=result,
        )
    except Exception as e:
        _store.update_task(
            task_id,
            status=TaskStatus.FAILED,
            completed_at=datetime.now().isoformat(),
            error=str(e),
        )


# ============================================================
# Endpoints
# ============================================================

@router.post(
    "/train",
    response_model=TaskResponse,
    status_code=202,
    summary="Start model training",
    description="Start a training task in the background. Returns immediately with a task ID."
)
async def start_training(
    request: TrainRequest,
    background_tasks: BackgroundTasks,
) -> TaskResponse:
    """Start model training."""
    # GPU mutex: reject if another GPU task (train/evaluate) is active
    if _store.has_running_gpu_task():
        raise HTTPException(
            status_code=409,
            detail="A GPU task (training or evaluation) is already running. "
            "Only one GPU task can run at a time.",
        )

    task_id = str(uuid4())

    task = TaskResult(
        task_id=task_id,
        task_type="train",
        status=TaskStatus.PENDING,
        created_at=datetime.now().isoformat(),
    )
    _store.put(task_id, task)

    # Start training in background
    config_overrides = {
        k: v for k, v in request.model_dump().items() if v is not None
    }
    background_tasks.add_task(_run_training, task_id, config_overrides)

    return TaskResponse(
        task_id=task_id,
        task_type="train",
        status=TaskStatus.PENDING,
        message="Training task started. Use /ml/tasks/{task_id} to check status.",
    )


@router.post(
    "/evaluate",
    response_model=TaskResponse,
    status_code=202,
    summary="Start model evaluation",
    description="Start an evaluation task in the background."
)
async def start_evaluation(
    request: EvaluateRequest,
    background_tasks: BackgroundTasks,
) -> TaskResponse:
    """Start model evaluation."""
    # GPU mutex: reject if another GPU task (train/evaluate) is active
    if _store.has_running_gpu_task():
        raise HTTPException(
            status_code=409,
            detail="A GPU task (training or evaluation) is already running. "
            "Only one GPU task can run at a time.",
        )

    task_id = str(uuid4())

    task = TaskResult(
        task_id=task_id,
        task_type="evaluate",
        status=TaskStatus.PENDING,
        created_at=datetime.now().isoformat(),
    )
    _store.put(task_id, task)

    background_tasks.add_task(_run_evaluation, task_id, request.checkpoint_path)

    return TaskResponse(
        task_id=task_id,
        task_type="evaluate",
        status=TaskStatus.PENDING,
        message="Evaluation task started. Use /ml/tasks/{task_id} to check status.",
    )


@router.post(
    "/batch-inference",
    response_model=TaskResponse,
    status_code=202,
    summary="Start batch inference",
    description="Start a batch inference task in the background."
)
async def start_batch_inference(
    request: BatchInferenceRequest,
    background_tasks: BackgroundTasks,
) -> TaskResponse:
    """Start batch inference."""
    task_id = str(uuid4())

    task = TaskResult(
        task_id=task_id,
        task_type="batch_inference",
        status=TaskStatus.PENDING,
        created_at=datetime.now().isoformat(),
    )
    _store.put(task_id, task)

    background_tasks.add_task(
        _run_batch_inference,
        task_id,
        request.input_dir,
        request.output_dir,
        request.max_images,
    )

    return TaskResponse(
        task_id=task_id,
        task_type="batch_inference",
        status=TaskStatus.PENDING,
        message="Batch inference task started. Use /ml/tasks/{task_id} to check status.",
    )


@router.post(
    "/validate-data",
    response_model=TaskResponse,
    status_code=202,
    summary="Start data validation",
    description="Validate dataset quality and integrity before training."
)
async def start_data_validation(
    request: DataValidationRequest,
    background_tasks: BackgroundTasks,
) -> TaskResponse:
    """Start data validation."""
    task_id = str(uuid4())

    task = TaskResult(
        task_id=task_id,
        task_type="data_validation",
        status=TaskStatus.PENDING,
        created_at=datetime.now().isoformat(),
    )
    _store.put(task_id, task)

    background_tasks.add_task(
        _run_data_validation,
        task_id,
        request.dataset_path,
        request.check_images,
        request.max_images_to_check,
    )

    return TaskResponse(
        task_id=task_id,
        task_type="data_validation",
        status=TaskStatus.PENDING,
        message="Data validation task started. Use /ml/tasks/{task_id} to check status.",
    )


@router.get(
    "/tasks/{task_id}",
    response_model=TaskResult,
    summary="Get task status",
    description="Get the status and result of a task."
)
async def get_task_status(task_id: str) -> TaskResult:
    """Get task status."""
    task = _store.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    return task


@router.get(
    "/tasks",
    summary="List all tasks",
    description="List all tasks with their status."
)
async def list_tasks(
    status: Optional[TaskStatus] = None,
    task_type: Optional[str] = None,
) -> list[TaskResult]:
    """List all tasks."""
    return _store.list_all(status=status, task_type=task_type)


@router.delete(
    "/tasks/{task_id}",
    summary="Delete a task",
    description="Delete a completed or failed task from the list."
)
async def delete_task(task_id: str) -> dict:
    """Delete a task."""
    task = _store.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    if task.status == TaskStatus.RUNNING:
        raise HTTPException(status_code=400, detail="Cannot delete a running task")

    _store.delete(task_id)
    return {"message": f"Task {task_id} deleted"}
