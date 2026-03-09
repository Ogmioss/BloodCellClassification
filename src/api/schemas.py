"""
Pydantic Schemas for API Request/Response Models

Defines data validation and serialization for the API.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status", examples=["healthy"])
    model_loaded: bool = Field(..., description="Whether model checkpoint is available")
    device: str = Field(..., description="Compute device (cpu/cuda)")
    model_name: str = Field(..., description="Model architecture name")
    model_source: Optional[str] = Field(None, description="Model source (mlflow or checkpoint)")
    model_version: Optional[str] = Field(None, description="Model version from MLflow registry")


class PredictionRequest(BaseModel):
    """Request for single image prediction (base64 encoded)."""
    image_base64: str = Field(..., description="Base64 encoded image (JPEG/PNG)")


class PredictionResponse(BaseModel):
    """Response for image prediction."""
    predicted_class: str = Field(..., description="Predicted blood cell type")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence")
    probabilities: Dict[str, float] = Field(..., description="Per-class probabilities")
    all_predictions: List[Dict[str, Any]] = Field(..., description="All classes sorted by probability descending")


class MetricsResponse(BaseModel):
    """Model performance metrics."""
    accuracy: float = Field(..., ge=0.0, le=1.0, description="Test set accuracy")
    best_val_acc: float = Field(..., ge=0.0, le=1.0, description="Best validation accuracy")
    final_train_acc: float = Field(..., ge=0.0, le=1.0, description="Final training accuracy")
    final_train_loss: float = Field(..., ge=0.0, description="Final training loss")
    class_names: List[str] = Field(..., description="List of class names")
    confusion_matrix: Optional[List[List[int]]] = Field(None, description="Confusion matrix")


class ErrorResponse(BaseModel):
    """Error response."""
    detail: str = Field(..., description="Error message")
