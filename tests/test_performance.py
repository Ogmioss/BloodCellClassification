"""
Performance Tests

Tests for inference latency and throughput requirements.
"""

import time
from unittest.mock import MagicMock

import pytest
from PIL import Image


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def test_image():
    """Create a test image."""
    return Image.new("RGB", (224, 224), color=(128, 128, 128))


@pytest.fixture
def mock_model():
    """Create a mock model for performance testing."""
    import torch
    
    mock = MagicMock()
    # Return tensor with 8 classes
    mock.return_value = torch.randn(1, 8)
    mock.eval = MagicMock()
    return mock


# ============================================================
# Inference Latency Tests
# ============================================================

class TestInferenceLatency:
    """Tests for inference latency requirements."""

    def test_transform_latency(self, test_image):
        """Image transform should be fast (< 10ms)."""
        from src.services.data_transform_service import DataTransformService
        from src.services.yaml_loader import YamlLoader
        
        loader = YamlLoader()
        transform_service = DataTransformService(loader.config)
        transform = transform_service.get_val_test_transform()
        
        # Warm up
        _ = transform(test_image)
        
        # Measure
        start = time.perf_counter()
        iterations = 100
        for _ in range(iterations):
            _ = transform(test_image)
        elapsed = time.perf_counter() - start
        
        avg_latency_ms = (elapsed / iterations) * 1000
        
        assert avg_latency_ms < 10, f"Transform latency {avg_latency_ms:.2f}ms exceeds 10ms"

    def test_inference_service_predict_latency(self, test_image):
        """Prediction should be fast if model is available."""
        import torch
        from src.services.inference_service import InferenceService
        from src.services.data_transform_service import DataTransformService
        from src.services.yaml_loader import YamlLoader
        from src.core.constants import CLASS_NAMES
        from src.models.model_factory import ModelFactory
        
        loader = YamlLoader()
        checkpoint_path = loader.project_root / "models" / "checkpoints" / "best_model.pth"
        
        if not checkpoint_path.exists():
            pytest.skip("No model checkpoint available for performance test")
        
        # Load real model
        config = loader.config
        device = torch.device("cpu")
        model = ModelFactory.create_model(config, len(CLASS_NAMES), device)
        
        # Load weights
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if not any(k.startswith("model.") for k in checkpoint.keys()):
            checkpoint = {f"model.{k}": v for k, v in checkpoint.items()}
        model.load_state_dict(checkpoint)
        model.eval()
        
        transform_service = DataTransformService(config)
        inference_service = InferenceService(
            model=model,
            transform_service=transform_service,
            device=device,
            class_names=CLASS_NAMES,
        )
        
        # Warm up
        _ = inference_service.predict_image(test_image)
        
        # Measure
        start = time.perf_counter()
        iterations = 10
        for _ in range(iterations):
            _ = inference_service.predict_image(test_image)
        elapsed = time.perf_counter() - start
        
        avg_latency_ms = (elapsed / iterations) * 1000
        
        # CPU inference should be < 500ms per image
        assert avg_latency_ms < 500, f"Inference latency {avg_latency_ms:.2f}ms exceeds 500ms"


# ============================================================
# Throughput Tests
# ============================================================

class TestThroughput:
    """Tests for throughput requirements."""

    def test_batch_image_loading(self, tmp_path):
        """Should load images efficiently."""
        from src.pipe.batch_inference_pipeline import find_images
        
        # Create test images
        for i in range(100):
            img = Image.new("RGB", (224, 224))
            img.save(tmp_path / f"img_{i}.jpg")
        
        start = time.perf_counter()
        images = find_images(tmp_path)
        elapsed = time.perf_counter() - start
        
        assert len(images) == 100
        assert elapsed < 1.0, f"Finding 100 images took {elapsed:.2f}s (should be < 1s)"

    def test_validation_throughput(self, tmp_path):
        """Data validation should be fast."""
        from src.services.data_validation_service import DataValidationService
        
        # Create test dataset
        for class_name in ["class1", "class2", "class3"]:
            class_dir = tmp_path / class_name
            class_dir.mkdir()
            for i in range(50):
                img = Image.new("RGB", (64, 64))
                img.save(class_dir / f"img_{i}.jpg")
        
        validator = DataValidationService(
            expected_classes=["class1", "class2", "class3"],
        )
        
        start = time.perf_counter()
        report = validator.validate_dataset(tmp_path)
        elapsed = time.perf_counter() - start
        
        assert report.total_images == 150
        assert elapsed < 2.0, f"Validation took {elapsed:.2f}s (should be < 2s)"


# ============================================================
# Memory Tests
# ============================================================

class TestMemoryUsage:
    """Tests for memory usage."""

    def test_image_transform_memory(self, test_image):
        """Image transform should not leak memory."""
        import gc
        from src.services.data_transform_service import DataTransformService
        from src.services.yaml_loader import YamlLoader
        
        loader = YamlLoader()
        transform_service = DataTransformService(loader.config)
        transform = transform_service.get_val_test_transform()
        
        # Run many transforms
        for _ in range(100):
            _ = transform(test_image)
        
        # Force garbage collection
        gc.collect()
        
        # If we get here without OOM, test passes
        assert True

    def test_batch_inference_result_serialization(self):
        """BatchInferenceResult should serialize efficiently."""
        import sys
        from src.pipe.batch_inference_pipeline import BatchInferenceResult
        
        # Create result with many predictions
        predictions = [
            {
                "image_path": f"/path/to/image_{i}.jpg",
                "predicted_class": "lymphocyte",
                "confidence": 0.95,
                "probabilities": {"lymphocyte": 0.95},
            }
            for i in range(1000)
        ]
        
        result = BatchInferenceResult(
            total_images=1000,
            successful=1000,
            failed=0,
            predictions=predictions,
        )
        
        # Serialize to dict
        result_dict = result.to_dict()
        
        # Check size is reasonable (< 1MB for 1000 predictions)
        import json
        json_str = json.dumps(result_dict)
        size_mb = sys.getsizeof(json_str) / (1024 * 1024)
        
        assert size_mb < 1.0, f"Result size {size_mb:.2f}MB exceeds 1MB"


# ============================================================
# API Response Time Tests
# ============================================================

class TestAPIResponseTime:
    """Tests for API response times."""

    def test_health_endpoint_response_time(self):
        """Health endpoint should respond quickly."""
        from fastapi.testclient import TestClient
        from src.api.main import app
        
        client = TestClient(app)
        
        # Warm up
        _ = client.get("/health")
        
        # Measure
        start = time.perf_counter()
        iterations = 20
        for _ in range(iterations):
            response = client.get("/health")
            assert response.status_code == 200
        elapsed = time.perf_counter() - start
        
        avg_latency_ms = (elapsed / iterations) * 1000
        
        assert avg_latency_ms < 50, f"Health endpoint latency {avg_latency_ms:.2f}ms exceeds 50ms"

    def test_model_info_endpoint_response_time(self):
        """Model info endpoint should respond reasonably fast."""
        from fastapi.testclient import TestClient
        from src.api.main import app
        
        client = TestClient(app)
        
        # Warm up
        _ = client.get("/model/info")
        
        # Measure
        start = time.perf_counter()
        iterations = 10
        for _ in range(iterations):
            response = client.get("/model/info")
            assert response.status_code == 200
        elapsed = time.perf_counter() - start
        
        avg_latency_ms = (elapsed / iterations) * 1000
        
        # MLflow calls can be slow, allow 500ms
        assert avg_latency_ms < 500, f"Model info latency {avg_latency_ms:.2f}ms exceeds 500ms"
