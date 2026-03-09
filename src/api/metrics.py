"""
Prometheus Metrics for Blood Cell Classification API

Custom metrics for monitoring:
- Prediction counts per class
- Prediction latency
- Confidence distribution
- Model information
"""

from prometheus_client import Counter, Histogram, Gauge, Info

from src.core.constants import CLASS_NAMES


# ============================================================
# Prediction Metrics
# ============================================================

# Counter for predictions per class
PREDICTIONS_TOTAL = Counter(
    "bloodcell_predictions_total",
    "Total number of predictions",
    ["predicted_class"],
)

# Histogram for prediction latency
PREDICTION_LATENCY = Histogram(
    "bloodcell_prediction_latency_seconds",
    "Time spent processing prediction requests",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

# Histogram for confidence scores
PREDICTION_CONFIDENCE = Histogram(
    "bloodcell_prediction_confidence",
    "Confidence scores of predictions",
    ["predicted_class"],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0],
)


# ============================================================
# Model Metrics
# ============================================================

# Gauge for model version
MODEL_VERSION = Gauge(
    "bloodcell_model_version",
    "Current model version loaded",
    ["source"],  # "mlflow" or "checkpoint"
)

# Info about the model
MODEL_INFO = Info(
    "bloodcell_model",
    "Information about the loaded model",
)


# ============================================================
# API Health Metrics
# ============================================================

# Gauge for API readiness (model loaded)
API_READY = Gauge(
    "bloodcell_api_ready",
    "Whether the API has a model loaded and is ready to serve",
)

# Counter for errors
ERRORS_TOTAL = Counter(
    "bloodcell_prediction_errors_total",
    "Total number of prediction errors",
    ["error_type"],
)


# ============================================================
# Helper Functions
# ============================================================

def record_prediction(predicted_class: str, confidence: float, latency: float) -> None:
    """
    Record a prediction in Prometheus metrics.
    
    Args:
        predicted_class: The predicted class name
        confidence: Confidence score (0-1)
        latency: Time taken for prediction in seconds
    """
    PREDICTIONS_TOTAL.labels(predicted_class=predicted_class).inc()
    PREDICTION_CONFIDENCE.labels(predicted_class=predicted_class).observe(confidence)
    PREDICTION_LATENCY.observe(latency)


def record_error(error_type: str) -> None:
    """
    Record an error in Prometheus metrics.
    
    Args:
        error_type: Type of error (e.g., "model_not_loaded", "invalid_image")
    """
    ERRORS_TOTAL.labels(error_type=error_type).inc()


def set_model_info(source: str, version: str = None, architecture: str = None) -> None:
    """
    Set model information in Prometheus metrics.
    
    Args:
        source: Model source ("mlflow" or "checkpoint")
        version: Model version (if from MLflow)
        architecture: Model architecture name
    """
    MODEL_VERSION.labels(source=source).set(1)
    
    info = {"source": source}
    if version:
        info["version"] = version
    if architecture:
        info["architecture"] = architecture
    
    MODEL_INFO.info(info)


def set_api_ready(ready: bool) -> None:
    """
    Set API readiness status.
    
    Args:
        ready: Whether the API is ready to serve predictions
    """
    API_READY.set(1 if ready else 0)


def initialize_class_metrics() -> None:
    """Initialize metrics for all classes (to ensure they appear in Prometheus)."""
    for class_name in CLASS_NAMES:
        # Initialize counters with 0
        PREDICTIONS_TOTAL.labels(predicted_class=class_name)
        PREDICTION_CONFIDENCE.labels(predicted_class=class_name)
