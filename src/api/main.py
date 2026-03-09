"""
FastAPI Application for Blood Cell Classification

Provides REST API endpoints:
- GET /health - Health check
- GET /metrics - Model performance metrics
- GET /model/info - Model architecture info
- POST /predict - Image classification
- POST /pipelines/* - Trigger Airflow DAGs
"""

import base64
import json
import io
from functools import lru_cache
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Depends
from PIL import Image

from src.api.schemas import (
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
    MetricsResponse,
    ErrorResponse,
)
from src.services.yaml_loader import YamlLoader
from src.services.inference_service import InferenceService
from src.services.mlflow_service import MLflowService
from src.services.model_loader_service import ModelLoaderService
from src.models.model_factory import ModelFactory
from src.core.constants import CLASS_NAMES
from src.api.routers.pipelines import router as pipelines_router
from src.api.routers.ml_tasks import router as ml_tasks_router
from src.api.metrics import (
    record_prediction,
    record_error,
    set_model_info,
    set_api_ready,
    initialize_class_metrics,
)


# ============================================================
# Application setup
# ============================================================

app = FastAPI(
    title="Blood Cell Classification API",
    description="REST API for classifying blood cell images using deep learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Include routers
app.include_router(pipelines_router)
app.include_router(ml_tasks_router)

# Setup Prometheus metrics
from prometheus_fastapi_instrumentator import Instrumentator

instrumentator = Instrumentator(
    should_group_status_codes=True,
    should_ignore_untemplated=True,
    should_respect_env_var=True,
    should_instrument_requests_inprogress=True,
    excluded_handlers=["/metrics"],
    inprogress_name="bloodcell_inprogress_requests",
    inprogress_labels=True,
)

instrumentator.instrument(app).expose(app, include_in_schema=True, tags=["Monitoring"])

# Initialize class metrics
initialize_class_metrics()


# ============================================================
# Configuration and dependencies
# ============================================================

@lru_cache()
def get_config() -> dict:
    """Load and cache configuration."""
    loader = YamlLoader()
    return loader.config


@lru_cache()
def get_yaml_loader() -> YamlLoader:
    """Get cached YamlLoader instance."""
    return YamlLoader()


def get_checkpoint_path() -> Path:
    """Get path to model checkpoint."""
    loader = get_yaml_loader()
    checkpoint_dir = Path(loader.get_nested_value(
        "paths.models.checkpoints", 
        "./models/checkpoints"
    ))
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = loader.project_root / checkpoint_dir
    return checkpoint_dir / "best_model.pth"


def get_metrics_path() -> Path:
    """Get path to metrics.json file."""
    loader = get_yaml_loader()
    checkpoint_dir = Path(loader.get_nested_value(
        "paths.models.checkpoints", 
        "./models/checkpoints"
    ))
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = loader.project_root / checkpoint_dir
    return checkpoint_dir / "metrics.json"


@lru_cache()
def get_model_loader_service() -> ModelLoaderService:
    """Get cached ModelLoaderService instance."""
    config = get_config()
    return ModelLoaderService(config)


@lru_cache()
def get_mlflow_service() -> MLflowService:
    """Get cached MLflow service instance."""
    return get_model_loader_service().mlflow_service


def get_model_source() -> str:
    """
    Determine the model source: 'mlflow' or 'checkpoint'.
    
    Prefers MLflow if a model is registered, falls back to checkpoint.
    """
    return get_model_loader_service().get_preferred_source()


# Cache for model load result
_model_load_result = None


def get_inference_service() -> Optional[InferenceService]:
    """
    Load and cache inference service.
    
    Uses ModelLoaderService for unified loading logic.
    Tries MLflow first, falls back to checkpoint.
    Returns None if no model is available.
    """
    global _model_load_result
    
    if _model_load_result is not None:
        return _model_load_result.inference_service
    
    try:
        loader = get_model_loader_service()
        _model_load_result = loader.load(prefer_mlflow=True)
        set_api_ready(ready=True)
        set_model_info(
            source=_model_load_result.source,
            version=_model_load_result.version,
            architecture="resnet18",
        )
        return _model_load_result.inference_service
    except FileNotFoundError:
        return None


def clear_inference_service_cache() -> None:
    """Clear the inference service cache (used after model promotion)."""
    global _model_load_result
    _model_load_result = None
    get_model_loader_service.cache_clear()


# ============================================================
# Endpoints
# ============================================================

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health check",
    description="Check if the API is running and model is loaded"
)
def health_check() -> HealthResponse:
    """Health check endpoint."""
    config = get_config()
    model_name = config.get("model", {}).get("name", "unknown")
    checkpoint_path = get_checkpoint_path()
    device = ModelFactory.get_device()

    # Include model source and version if model is loaded
    model_source = None
    model_version = None
    if _model_load_result is not None:
        model_source = _model_load_result.source
        model_version = _model_load_result.version

    return HealthResponse(
        status="healthy",
        model_loaded=checkpoint_path.exists(),
        device=str(device),
        model_name=model_name,
        model_source=model_source,
        model_version=model_version,
    )


@app.get(
    "/metrics",
    response_model=MetricsResponse,
    responses={404: {"model": ErrorResponse}},
    tags=["Model"],
    summary="Get model metrics",
    description="Retrieve performance metrics from the trained model"
)
def get_metrics() -> MetricsResponse:
    """Get model performance metrics."""
    metrics_path = get_metrics_path()
    
    if not metrics_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Metrics file not found at {metrics_path}. Train a model first."
        )
    
    try:
        with open(metrics_path, "r") as f:
            metrics_data = json.load(f)
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Invalid metrics file format: {e}"
        )
    
    return MetricsResponse(
        accuracy=metrics_data.get("accuracy", 0.0),
        best_val_acc=metrics_data.get("best_val_acc", 0.0),
        final_train_acc=metrics_data.get("final_train_acc", 0.0),
        final_train_loss=metrics_data.get("final_train_loss", 0.0),
        class_names=metrics_data.get("class_names", CLASS_NAMES),
        confusion_matrix=metrics_data.get("confusion_matrix")
    )


@app.get(
    "/model/info",
    tags=["Model"],
    summary="Get model information",
    description="Retrieve model architecture and configuration details"
)
def get_model_info() -> dict:
    """Get model architecture information."""
    config = get_config()
    model_config = config.get("model", {})
    
    # Get MLflow info with timeout protection
    mlflow_info = {"available": False, "error": None}
    try:
        mlflow_service = get_mlflow_service()
        latest_version = mlflow_service.get_latest_model_version()
        if latest_version:
            mlflow_info = {
                "available": True,
                "model_name": mlflow_service.model_name,
                "latest_version": latest_version,
                "tracking_uri": mlflow_service.tracking_uri,
            }
    except Exception as e:
        mlflow_info["error"] = str(e)
    
    # Get model source with fallback
    try:
        model_source = get_model_source()
    except Exception:
        model_source = "checkpoint"
    
    return {
        "model_name": model_config.get("name", "resnet18"),
        "pretrained": model_config.get("pretrained", True),
        "pretrained_weights": model_config.get("pretrained_weights", "IMAGENET1K_V1"),
        "num_classes": len(CLASS_NAMES),
        "class_names": CLASS_NAMES,
        "normalization": model_config.get("normalization", {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        }),
        "checkpoint_available": get_checkpoint_path().exists(),
        "model_source": model_source,
        "mlflow": mlflow_info,
    }


@app.get(
    "/mlflow/models",
    tags=["MLflow"],
    summary="List registered model versions",
    description="Get all versions of the registered model from MLflow"
)
def list_mlflow_models() -> dict:
    """List all model versions in MLflow registry."""
    from src.services.mlflow_service import MLflowTimeoutError
    
    try:
        mlflow_service = get_mlflow_service()
        
        # Try to get client with timeout protection
        try:
            client = mlflow_service.client
        except MLflowTimeoutError:
            return {
                "model_name": mlflow_service.model_name,
                "tracking_uri": mlflow_service.tracking_uri,
                "versions": [],
                "error": "MLflow server not available (timeout)",
            }
        
        # Get all versions
        versions = []
        try:
            for mv in client.search_model_versions(f"name='{mlflow_service.model_name}'"):
                versions.append({
                    "version": mv.version,
                    "stage": mv.current_stage,
                    "status": mv.status,
                    "run_id": mv.run_id,
                    "creation_timestamp": mv.creation_timestamp,
                })
        except Exception:
            pass
        
        return {
            "model_name": mlflow_service.model_name,
            "tracking_uri": mlflow_service.tracking_uri,
            "versions": versions,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MLflow error: {e}")


@app.post(
    "/mlflow/promote/{version}",
    tags=["MLflow"],
    summary="Promote model to a stage",
    description="Transition a model version to the specified stage (default: Production)"
)
def promote_model(version: str, stage: str = "Production") -> dict:
    """Promote a model version to the specified stage."""
    try:
        mlflow_service = get_mlflow_service()
        mlflow_service.transition_model_stage(version=version, stage=stage)

        # Clear the inference service cache to reload the new model
        clear_inference_service_cache()

        return {
            "status": "success",
            "message": f"Model version {version} promoted to {stage}",
            "model_name": mlflow_service.model_name,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Promotion failed: {e}")


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse},
        503: {"model": ErrorResponse}
    },
    tags=["Prediction"],
    summary="Classify blood cell image",
    description="Upload a base64-encoded image and get classification results"
)
def predict(
    request: PredictionRequest,
    inference_service: Optional[InferenceService] = Depends(get_inference_service)
) -> PredictionResponse:
    """Classify a blood cell image."""
    import time
    start_time = time.time()
    
    # Check if model is loaded
    if inference_service is None:
        record_error("model_not_loaded")
        raise HTTPException(
            status_code=503,
            detail="Model not available. Train a model first using 'uv run train-model'"
        )
    
    # Decode base64 image
    try:
        image_bytes = base64.b64decode(request.image_base64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        record_error("invalid_image")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image data: {e}"
        )
    
    # Run prediction
    try:
        result = inference_service.predict_image(image)
    except Exception as e:
        record_error("prediction_failed")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {e}"
        )
    
    # Record metrics
    latency = time.time() - start_time
    record_prediction(
        predicted_class=result["predicted_class"],
        confidence=result["confidence"],
        latency=latency,
    )
    
    all_predictions = sorted(
        [{"class": k, "probability": v} for k, v in result["probabilities"].items()],
        key=lambda x: x["probability"],
        reverse=True,
    )

    return PredictionResponse(
        predicted_class=result["predicted_class"],
        confidence=result["confidence"],
        probabilities=result["probabilities"],
        all_predictions=all_predictions,
    )


# ============================================================
# Optional: File upload endpoint (alternative to base64)
# ============================================================

from fastapi import File, UploadFile


@app.post(
    "/predict/upload",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse},
        503: {"model": ErrorResponse}
    },
    tags=["Prediction"],
    summary="Classify uploaded blood cell image",
    description="Upload an image file directly and get classification results"
)
async def predict_upload(
    file: UploadFile = File(..., description="Image file (JPEG/PNG)"),
    inference_service: Optional[InferenceService] = Depends(get_inference_service)
) -> PredictionResponse:
    """Classify an uploaded blood cell image."""
    import time
    start_time = time.time()

    # Check if model is loaded
    if inference_service is None:
        record_error("model_not_loaded")
        raise HTTPException(
            status_code=503,
            detail="Model not available. Train a model first using 'uv run train-model'"
        )

    # Validate file type
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        record_error("invalid_image")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Use JPEG or PNG."
        )

    # Read and decode image
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        record_error("invalid_image")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image file: {e}"
        )

    # Run prediction
    try:
        result = inference_service.predict_image(image)
    except Exception as e:
        record_error("prediction_failed")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {e}"
        )

    # Record metrics
    latency = time.time() - start_time
    record_prediction(
        predicted_class=result["predicted_class"],
        confidence=result["confidence"],
        latency=latency,
    )

    all_predictions = sorted(
        [{"class": k, "probability": v} for k, v in result["probabilities"].items()],
        key=lambda x: x["probability"],
        reverse=True,
    )

    return PredictionResponse(
        predicted_class=result["predicted_class"],
        confidence=result["confidence"],
        probabilities=result["probabilities"],
        all_predictions=all_predictions,
    )


# ============================================================
# Entry point for uvicorn
# ============================================================

def start_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False) -> None:
    """Start the FastAPI server with uvicorn."""
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        reload=reload
    )


if __name__ == "__main__":
    start_server()
