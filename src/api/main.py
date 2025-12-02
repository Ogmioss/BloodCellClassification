"""
FastAPI Application for Blood Cell Classification

Provides REST API endpoints:
- GET /health - Health check
- GET /metrics - Model performance metrics
- GET /model/info - Model architecture info
- POST /predict - Image classification
"""

import base64
import json
import io
from functools import lru_cache
from pathlib import Path
from typing import Optional

import torch
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
from src.models.model_factory import ModelFactory


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

# Class names for blood cell types
CLASS_NAMES = [
    "basophil", "eosinophil", "erythroblast", "immature_granulocyte",
    "lymphocyte", "monocyte", "neutrophil", "platelet"
]


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


def get_device() -> torch.device:
    """Get compute device."""
    return ModelFactory.get_device()


@lru_cache()
def get_inference_service() -> Optional[InferenceService]:
    """
    Load and cache inference service.
    
    Returns None if model checkpoint doesn't exist.
    """
    checkpoint_path = get_checkpoint_path()
    
    if not checkpoint_path.exists():
        return None
    
    config = get_config()
    device = get_device()
    
    return InferenceService.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        config=config,
        device=device,
        class_names=CLASS_NAMES
    )


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
    device = get_device()
    
    return HealthResponse(
        status="healthy",
        model_loaded=checkpoint_path.exists(),
        device=str(device),
        model_name=model_name
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
        "checkpoint_available": get_checkpoint_path().exists()
    }


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
    # Check if model is loaded
    if inference_service is None:
        raise HTTPException(
            status_code=503,
            detail="Model not available. Train a model first using 'uv run train-model'"
        )
    
    # Decode base64 image
    try:
        image_bytes = base64.b64decode(request.image_base64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image data: {e}"
        )
    
    # Run prediction
    try:
        result = inference_service.predict_image(image)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {e}"
        )
    
    return PredictionResponse(
        predicted_class=result["predicted_class"],
        confidence=result["confidence"],
        probabilities=result["probabilities"]
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
    # Check if model is loaded
    if inference_service is None:
        raise HTTPException(
            status_code=503,
            detail="Model not available. Train a model first using 'uv run train-model'"
        )
    
    # Validate file type
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Use JPEG or PNG."
        )
    
    # Read and decode image
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image file: {e}"
        )
    
    # Run prediction
    try:
        result = inference_service.predict_image(image)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {e}"
        )
    
    return PredictionResponse(
        predicted_class=result["predicted_class"],
        confidence=result["confidence"],
        probabilities=result["probabilities"]
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
