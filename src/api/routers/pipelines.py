"""
Pipelines Router

Endpoints for triggering and monitoring Airflow DAGs.
Provides API integration with Airflow for pipeline orchestration.
"""

import os
from typing import Optional

import httpx
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field


# ============================================================
# Configuration
# ============================================================

AIRFLOW_API_URL = os.getenv("AIRFLOW_API_URL", "http://localhost:8080/api/v1")
AIRFLOW_USERNAME = os.getenv("AIRFLOW_USERNAME", "admin")
AIRFLOW_PASSWORD = os.getenv("AIRFLOW_PASSWORD", "admin")

# DAG IDs — suffix configurable via AIRFLOW_DAG_SUFFIX env var
# Default: "_api" (lightweight DAGs that call FastAPI for ML execution)
_DAG_SUFFIX = os.getenv("AIRFLOW_DAG_SUFFIX", "_api")
DAG_TRAIN = f"bloodcells_train_model{_DAG_SUFFIX}"
DAG_EVALUATE = f"bloodcells_evaluate_model{_DAG_SUFFIX}"
DAG_BATCH_INFERENCE = f"bloodcells_batch_inference{_DAG_SUFFIX}"
DAG_DATA_VALIDATION = f"bloodcells_data_validation{_DAG_SUFFIX}"


# ============================================================
# Schemas
# ============================================================

class TriggerResponse(BaseModel):
    """Response for DAG trigger endpoints."""
    status: str = Field(..., description="Status of the trigger request")
    dag_id: str = Field(..., description="DAG identifier")
    dag_run_id: Optional[str] = Field(None, description="DAG run identifier")
    message: str = Field(..., description="Human-readable message")
    execution_date: Optional[str] = Field(None, description="Execution date of the DAG run")


class DAGRunStatus(BaseModel):
    """Status of a DAG run."""
    dag_id: str
    dag_run_id: str
    state: str
    execution_date: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class BatchInferenceConfig(BaseModel):
    """Configuration for batch inference trigger."""
    input_dir: Optional[str] = Field(None, description="Input directory for images")
    output_dir: Optional[str] = Field(None, description="Output directory for results")
    max_images: Optional[int] = Field(None, description="Maximum number of images to process")


# ============================================================
# Router
# ============================================================

router = APIRouter(prefix="/pipelines", tags=["Pipelines"])


def _get_airflow_auth() -> tuple[str, str]:
    """Get Airflow authentication credentials."""
    return (AIRFLOW_USERNAME, AIRFLOW_PASSWORD)


async def _trigger_dag(
    dag_id: str,
    conf: Optional[dict] = None,
    logical_date: Optional[str] = None,
) -> dict:
    """
    Trigger an Airflow DAG.
    
    Args:
        dag_id: DAG identifier
        conf: Configuration to pass to the DAG
        logical_date: Optional logical date for the run
        
    Returns:
        Airflow API response
        
    Raises:
        HTTPException: If trigger fails
    """
    url = f"{AIRFLOW_API_URL}/dags/{dag_id}/dagRuns"
    
    payload = {
        "conf": conf or {},
    }
    
    if logical_date:
        payload["logical_date"] = logical_date
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                url,
                json=payload,
                auth=_get_airflow_auth(),
            )
            
            if response.status_code == 401:
                raise HTTPException(
                    status_code=503,
                    detail="Airflow authentication failed. Check AIRFLOW_USERNAME and AIRFLOW_PASSWORD."
                )
            
            if response.status_code == 404:
                raise HTTPException(
                    status_code=404,
                    detail=f"DAG '{dag_id}' not found in Airflow."
                )
            
            if response.status_code not in (200, 201):
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Airflow API error: {response.text}"
                )
            
            return response.json()
            
    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail=f"Cannot connect to Airflow at {AIRFLOW_API_URL}. Is Airflow running?"
        )
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail="Airflow API request timed out."
        )


async def _get_dag_run_status(dag_id: str, dag_run_id: str) -> dict:
    """
    Get status of a DAG run.
    
    Args:
        dag_id: DAG identifier
        dag_run_id: DAG run identifier
        
    Returns:
        DAG run status
    """
    url = f"{AIRFLOW_API_URL}/dags/{dag_id}/dagRuns/{dag_run_id}"
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                url,
                auth=_get_airflow_auth(),
            )
            
            if response.status_code == 404:
                raise HTTPException(
                    status_code=404,
                    detail=f"DAG run '{dag_run_id}' not found."
                )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Airflow API error: {response.text}"
                )
            
            return response.json()
            
    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail=f"Cannot connect to Airflow at {AIRFLOW_API_URL}."
        )


# ============================================================
# Endpoints
# ============================================================

@router.post(
    "/train",
    response_model=TriggerResponse,
    summary="Trigger training pipeline",
    description="Trigger the blood cell classification training DAG in Airflow."
)
async def trigger_training() -> TriggerResponse:
    """Trigger the training pipeline."""
    result = await _trigger_dag(DAG_TRAIN)
    
    return TriggerResponse(
        status="triggered",
        dag_id=DAG_TRAIN,
        dag_run_id=result.get("dag_run_id"),
        message="Training pipeline triggered successfully.",
        execution_date=result.get("execution_date"),
    )


@router.post(
    "/evaluate",
    response_model=TriggerResponse,
    summary="Trigger evaluation pipeline",
    description="Trigger the model evaluation DAG in Airflow."
)
async def trigger_evaluation() -> TriggerResponse:
    """Trigger the evaluation pipeline."""
    result = await _trigger_dag(DAG_EVALUATE)
    
    return TriggerResponse(
        status="triggered",
        dag_id=DAG_EVALUATE,
        dag_run_id=result.get("dag_run_id"),
        message="Evaluation pipeline triggered successfully.",
        execution_date=result.get("execution_date"),
    )


@router.post(
    "/batch-inference",
    response_model=TriggerResponse,
    summary="Trigger batch inference pipeline",
    description="Trigger the batch inference DAG in Airflow with optional configuration."
)
async def trigger_batch_inference(
    config: Optional[BatchInferenceConfig] = None,
) -> TriggerResponse:
    """
    Trigger the batch inference pipeline.
    
    Optionally pass configuration for input/output directories and max images.
    """
    conf = {}
    if config:
        if config.input_dir:
            conf["input_dir"] = config.input_dir
        if config.output_dir:
            conf["output_dir"] = config.output_dir
        if config.max_images:
            conf["max_images"] = config.max_images
    
    result = await _trigger_dag(DAG_BATCH_INFERENCE, conf=conf)
    
    return TriggerResponse(
        status="triggered",
        dag_id=DAG_BATCH_INFERENCE,
        dag_run_id=result.get("dag_run_id"),
        message="Batch inference pipeline triggered successfully.",
        execution_date=result.get("execution_date"),
    )


@router.post(
    "/validate-data",
    response_model=TriggerResponse,
    summary="Trigger data validation pipeline",
    description="Trigger the data validation DAG to check dataset quality before training."
)
async def trigger_data_validation() -> TriggerResponse:
    """Trigger the data validation pipeline."""
    result = await _trigger_dag(DAG_DATA_VALIDATION)
    
    return TriggerResponse(
        status="triggered",
        dag_id=DAG_DATA_VALIDATION,
        dag_run_id=result.get("dag_run_id"),
        message="Data validation pipeline triggered successfully.",
        execution_date=result.get("execution_date"),
    )


@router.get(
    "/status/{dag_id}/{dag_run_id}",
    response_model=DAGRunStatus,
    summary="Get DAG run status",
    description="Get the status of a specific DAG run."
)
async def get_dag_run_status(
    dag_id: str,
    dag_run_id: str,
) -> DAGRunStatus:
    """Get the status of a DAG run."""
    result = await _get_dag_run_status(dag_id, dag_run_id)
    
    return DAGRunStatus(
        dag_id=result.get("dag_id"),
        dag_run_id=result.get("dag_run_id"),
        state=result.get("state"),
        execution_date=result.get("execution_date"),
        start_date=result.get("start_date"),
        end_date=result.get("end_date"),
    )


@router.get(
    "/runs/{dag_id}",
    summary="List DAG runs",
    description="List recent runs for a specific DAG."
)
async def list_dag_runs(
    dag_id: str,
    limit: int = Query(default=10, ge=1, le=100, description="Maximum number of runs to return"),
) -> list[dict]:
    """List recent DAG runs."""
    url = f"{AIRFLOW_API_URL}/dags/{dag_id}/dagRuns"
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                url,
                params={"limit": limit, "order_by": "-execution_date"},
                auth=_get_airflow_auth(),
            )
            
            if response.status_code == 404:
                raise HTTPException(
                    status_code=404,
                    detail=f"DAG '{dag_id}' not found."
                )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Airflow API error: {response.text}"
                )
            
            data = response.json()
            runs = data.get("dag_runs", [])
            
            return [
                {
                    "dag_run_id": run.get("dag_run_id"),
                    "state": run.get("state"),
                    "execution_date": run.get("execution_date"),
                    "start_date": run.get("start_date"),
                    "end_date": run.get("end_date"),
                }
                for run in runs
            ]
            
    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail=f"Cannot connect to Airflow at {AIRFLOW_API_URL}."
        )


@router.get(
    "/dags",
    summary="List available DAGs",
    description="List all blood cell classification DAGs."
)
async def list_dags() -> list[dict]:
    """List available DAGs for this project."""
    dags = [
        {
            "dag_id": DAG_DATA_VALIDATION,
            "description": "Validate dataset quality before training",
            "trigger_endpoint": "/pipelines/validate-data",
        },
        {
            "dag_id": DAG_TRAIN,
            "description": "Train blood cell classification model",
            "trigger_endpoint": "/pipelines/train",
        },
        {
            "dag_id": DAG_EVALUATE,
            "description": "Evaluate existing model checkpoint",
            "trigger_endpoint": "/pipelines/evaluate",
        },
        {
            "dag_id": DAG_BATCH_INFERENCE,
            "description": "Run batch inference on images",
            "trigger_endpoint": "/pipelines/batch-inference",
        },
    ]
    
    # Try to get status from Airflow
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            for dag in dags:
                url = f"{AIRFLOW_API_URL}/dags/{dag['dag_id']}"
                try:
                    response = await client.get(url, auth=_get_airflow_auth())
                    if response.status_code == 200:
                        data = response.json()
                        dag["is_paused"] = data.get("is_paused", True)
                        dag["airflow_status"] = "available"
                    else:
                        dag["airflow_status"] = "not_found"
                except Exception:
                    dag["airflow_status"] = "unknown"
    except Exception:
        for dag in dags:
            dag["airflow_status"] = "airflow_unavailable"
    
    return dags
