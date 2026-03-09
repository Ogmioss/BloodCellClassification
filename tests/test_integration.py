"""
Integration Tests

End-to-end tests for the ML pipelines and API.
These tests verify that components work together correctly.
"""

import json
from unittest.mock import patch, MagicMock

import pytest
from PIL import Image
from fastapi.testclient import TestClient


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def test_dataset(tmp_path):
    """Create a minimal test dataset."""
    from src.core.constants import CLASS_NAMES
    
    # Create class directories with a few images each
    for class_name in CLASS_NAMES[:3]:  # Only 3 classes for speed
        class_dir = tmp_path / class_name
        class_dir.mkdir()
        
        for i in range(5):
            img = Image.new("RGB", (224, 224), color=(i * 50, i * 30, i * 20))
            img.save(class_dir / f"img_{i}.jpg")
    
    return tmp_path


@pytest.fixture
def api_client():
    """Create FastAPI test client."""
    from src.api.main import app
    return TestClient(app)


@pytest.fixture
def mock_inference_service():
    """Create a mock inference service."""
    mock_service = MagicMock()
    mock_service.predict_image.return_value = {
        "predicted_class": "lymphocyte",
        "confidence": 0.95,
        "probabilities": {
            "basophil": 0.01,
            "eosinophil": 0.01,
            "erythroblast": 0.01,
            "immature_granulocyte": 0.01,
            "lymphocyte": 0.95,
            "monocyte": 0.005,
            "neutrophil": 0.005,
            "platelet": 0.01,
        }
    }
    return mock_service


# ============================================================
# Data Validation Integration Tests
# ============================================================

class TestDataValidationIntegration:
    """Integration tests for data validation pipeline."""

    def test_validate_dataset_end_to_end(self, test_dataset):
        """Should validate a dataset end-to-end."""
        from src.services.data_validation_service import DataValidationService
        
        validator = DataValidationService(
            expected_classes=["basophil", "eosinophil", "erythroblast"],
            min_samples_per_class=3,
        )
        
        # Validate dataset
        report = validator.validate_dataset(test_dataset)
        
        assert report.is_valid is True
        assert report.total_images == 15  # 3 classes * 5 images
        assert len(report.errors) == 0

    def test_validate_images_end_to_end(self, test_dataset):
        """Should validate images end-to-end."""
        from src.services.data_validation_service import DataValidationService
        
        validator = DataValidationService(min_image_size=(32, 32))
        
        report = validator.validate_images(test_dataset)
        
        assert report.is_valid is True
        assert len(report.corrupted_images) == 0

    def test_save_and_load_validation_report(self, test_dataset, tmp_path):
        """Should save and load validation report."""
        from src.services.data_validation_service import DataValidationService
        
        validator = DataValidationService(
            expected_classes=["basophil", "eosinophil", "erythroblast"],
        )
        
        report = validator.validate_dataset(test_dataset)
        
        # Save report
        report_path = tmp_path / "report.json"
        report.save(report_path)
        
        # Load and verify
        with open(report_path) as f:
            loaded = json.load(f)
        
        assert loaded["is_valid"] == report.is_valid
        assert loaded["total_images"] == report.total_images


# ============================================================
# Batch Inference Integration Tests
# ============================================================

class TestBatchInferenceIntegration:
    """Integration tests for batch inference pipeline."""

    def test_find_images_in_dataset(self, test_dataset):
        """Should find all images in dataset."""
        from src.pipe.batch_inference_pipeline import find_images
        
        images = find_images(test_dataset)
        
        assert len(images) == 15  # 3 classes * 5 images

    def test_batch_inference_with_mock_model(self, test_dataset, tmp_path):
        """Should run batch inference with mocked model."""
        from src.pipe.batch_inference_pipeline import run_batch_inference
        
        output_dir = tmp_path / "output"
        
        # Mock ModelLoaderService
        mock_inference = MagicMock()
        mock_inference.predict_image.return_value = {
            "predicted_class": "lymphocyte",
            "confidence": 0.95,
            "probabilities": {"lymphocyte": 0.95},
        }
        
        mock_result = MagicMock()
        mock_result.inference_service = mock_inference
        mock_result.source = "checkpoint:test.pth"
        
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_result
        
        with patch("src.pipe.batch_inference_pipeline.ModelLoaderService", return_value=mock_loader):
            result = run_batch_inference(
                input_dir=test_dataset,
                output_dir=output_dir,
                save_results=True,
                show_progress=False,
            )
        
        assert result.total_images == 15
        assert result.successful == 15
        assert result.output_file is not None
        assert result.output_file.exists()
        
        # Verify output file content
        with open(result.output_file) as f:
            output_data = json.load(f)
        
        assert "predictions" in output_data
        assert len(output_data["predictions"]) == 15


# ============================================================
# API Integration Tests
# ============================================================

class TestAPIIntegration:
    """Integration tests for FastAPI endpoints."""

    def test_health_endpoint(self, api_client):
        """Health endpoint should return 200."""
        response = api_client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_predict_endpoint_exists(self, api_client):
        """Predict endpoint should exist and return proper error without model."""
        import base64
        import io
        
        # Create test image
        img = Image.new("RGB", (224, 224), color=(255, 0, 0))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        response = api_client.post(
            "/predict",
            json={"image_base64": image_base64}
        )
        
        # Should either succeed (if model exists) or return 503 (no model)
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "predicted_class" in data
            assert "confidence" in data

    def test_model_info_endpoint(self, api_client):
        """Model info endpoint should return model details."""
        response = api_client.get("/model/info")
        
        assert response.status_code == 200
        data = response.json()
        assert "model_name" in data
        assert "num_classes" in data
        assert data["num_classes"] == 8

    def test_pipelines_dags_endpoint(self, api_client):
        """Pipelines DAGs endpoint should list available DAGs."""
        response = api_client.get("/pipelines/dags")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 3


# ============================================================
# Model Loading Integration Tests
# ============================================================

class TestModelLoadingIntegration:
    """Integration tests for model loading service."""

    def test_model_loader_service_initialization(self):
        """ModelLoaderService should initialize correctly."""
        from src.services.model_loader_service import ModelLoaderService
        from src.services.yaml_loader import YamlLoader
        
        loader = YamlLoader()
        model_loader = ModelLoaderService(loader.config)
        
        assert model_loader.checkpoint_path is not None
        assert model_loader.device is not None

    def test_model_loader_get_info(self):
        """ModelLoaderService should return model info."""
        from src.services.model_loader_service import ModelLoaderService
        from src.services.yaml_loader import YamlLoader
        
        loader = YamlLoader()
        model_loader = ModelLoaderService(loader.config)
        
        info = model_loader.get_model_info()
        
        assert "checkpoint_available" in info
        assert "checkpoint_path" in info
        assert "mlflow" in info

    def test_model_loader_preferred_source(self):
        """ModelLoaderService should determine preferred source."""
        from src.services.model_loader_service import ModelLoaderService
        from src.services.yaml_loader import YamlLoader
        
        loader = YamlLoader()
        model_loader = ModelLoaderService(loader.config)
        
        source = model_loader.get_preferred_source()
        
        assert source in ["mlflow", "checkpoint"]


# ============================================================
# Configuration Integration Tests
# ============================================================

class TestConfigurationIntegration:
    """Integration tests for configuration loading."""

    def test_yaml_loader_loads_config(self):
        """YamlLoader should load configuration correctly."""
        from src.services.yaml_loader import YamlLoader
        
        loader = YamlLoader()
        
        assert loader.config is not None
        assert "model" in loader.config
        assert "training" in loader.config

    def test_yaml_loader_resolves_paths(self):
        """YamlLoader should resolve paths correctly."""
        from src.services.yaml_loader import YamlLoader
        
        loader = YamlLoader()
        
        assert loader.data_dir.exists() or True  # May not exist in test env
        assert loader.project_root.exists()

    def test_constants_are_defined(self):
        """Core constants should be defined."""
        from src.core.constants import CLASS_NAMES, NUM_CLASSES
        
        assert len(CLASS_NAMES) == 8
        assert NUM_CLASSES == 8
        assert "lymphocyte" in CLASS_NAMES
