"""
Tests for FastAPI endpoints.

TDD approach: tests written first, then implementation.
"""

import base64
import json
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from src.api.main import app, get_inference_service


# Test fixtures
@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def sample_image_base64():
    """Create a minimal valid image as base64 using PIL."""
    from PIL import Image
    import io
    
    # Create a small RGB image
    img = Image.new("RGB", (64, 64), color=(255, 0, 0))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


# ============================================================
# Health endpoint tests
# ============================================================

class TestHealthEndpoint:
    """Tests for GET /health endpoint."""

    def test_health_returns_200(self, client):
        """Health endpoint should return 200 OK."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_structure(self, client):
        """Health response should have required fields."""
        response = client.get("/health")
        data = response.json()

        assert "status" in data
        assert "model_loaded" in data
        assert "device" in data
        assert "model_name" in data
        assert "model_source" in data
        assert "model_version" in data

    def test_health_status_healthy(self, client):
        """Health status should be 'healthy'."""
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "healthy"


# ============================================================
# Metrics endpoint tests
# ============================================================

class TestMetricsEndpoint:
    """Tests for GET /metrics endpoint."""

    def test_metrics_returns_200_when_file_exists(self, client, tmp_path, monkeypatch):
        """Metrics endpoint should return 200 when metrics.json exists."""
        # Create mock metrics file
        metrics_data = {
            "accuracy": 0.85,
            "best_val_acc": 0.87,
            "final_train_acc": 0.92,
            "final_train_loss": 0.25,
            "class_names": ["basophil", "eosinophil"],
            "confusion_matrix": [[10, 2], [1, 12]]
        }
        metrics_path = tmp_path / "metrics.json"
        metrics_path.write_text(json.dumps(metrics_data))
        
        # Mock the metrics path
        with patch("src.api.main.get_metrics_path", return_value=metrics_path):
            response = client.get("/metrics")
        
        assert response.status_code == 200
        data = response.json()
        assert data["accuracy"] == 0.85
        assert data["best_val_acc"] == 0.87

    def test_metrics_returns_404_when_file_missing(self, client, tmp_path):
        """Metrics endpoint should return 404 when metrics.json doesn't exist."""
        missing_path = tmp_path / "nonexistent" / "metrics.json"
        
        with patch("src.api.main.get_metrics_path", return_value=missing_path):
            response = client.get("/metrics")
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()


# ============================================================
# Predict endpoint tests
# ============================================================

class TestPredictEndpoint:
    """Tests for POST /predict endpoint."""

    def test_predict_returns_200_with_valid_image(self, client, sample_image_base64):
        """Predict endpoint should return 200 with valid base64 image."""
        # Mock inference service
        mock_service = MagicMock()
        mock_service.predict_image.return_value = {
            "predicted_class": "lymphocyte",
            "confidence": 0.95,
            "probabilities": {"lymphocyte": 0.95, "monocyte": 0.05}
        }
        
        def override_inference_service():
            return mock_service
        
        app.dependency_overrides[get_inference_service] = override_inference_service
        
        try:
            response = client.post(
                "/predict",
                json={"image_base64": sample_image_base64}
            )
            assert response.status_code == 200
            data = response.json()
            assert data["predicted_class"] == "lymphocyte"
            assert data["confidence"] == 0.95
        finally:
            app.dependency_overrides.clear()

    def test_predict_returns_422_with_invalid_base64(self, client):
        """Predict endpoint should return 422 with invalid base64."""
        mock_service = MagicMock()
        
        def override_inference_service():
            return mock_service
        
        app.dependency_overrides[get_inference_service] = override_inference_service
        
        try:
            response = client.post(
                "/predict",
                json={"image_base64": "not-valid-base64!!!"}
            )
            # Should fail during image decoding
            assert response.status_code in [400, 422, 500]
        finally:
            app.dependency_overrides.clear()

    def test_predict_response_structure(self, client, sample_image_base64):
        """Predict response should have required fields."""
        mock_service = MagicMock()
        mock_service.predict_image.return_value = {
            "predicted_class": "neutrophil",
            "confidence": 0.88,
            "probabilities": {"neutrophil": 0.88, "basophil": 0.12}
        }
        
        def override_inference_service():
            return mock_service
        
        app.dependency_overrides[get_inference_service] = override_inference_service
        
        try:
            response = client.post(
                "/predict",
                json={"image_base64": sample_image_base64}
            )
            data = response.json()
            
            assert "predicted_class" in data
            assert "confidence" in data
            assert "probabilities" in data
            assert isinstance(data["probabilities"], dict)
            assert "all_predictions" in data
            assert isinstance(data["all_predictions"], list)
            assert len(data["all_predictions"]) > 0
            assert data["all_predictions"][0]["class"] == data["predicted_class"]
        finally:
            app.dependency_overrides.clear()


# ============================================================
# Model info endpoint tests
# ============================================================

class TestModelInfoEndpoint:
    """Tests for GET /model/info endpoint."""

    def test_model_info_returns_200(self, client):
        """Model info endpoint should return 200."""
        response = client.get("/model/info")
        assert response.status_code == 200

    def test_model_info_contains_architecture(self, client):
        """Model info should contain architecture details."""
        response = client.get("/model/info")
        data = response.json()
        
        assert "model_name" in data
        assert "num_classes" in data
        assert "class_names" in data
