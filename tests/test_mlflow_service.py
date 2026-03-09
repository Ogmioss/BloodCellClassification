"""
Tests for MLflow Service.

Tests experiment tracking and model registry functionality.
"""


import pytest
import torch.nn as nn

from src.services.mlflow_service import MLflowService


class SimpleModel(nn.Module):
    """Simple model for testing."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 2)
    
    def forward(self, x):
        return self.linear(x)


@pytest.fixture
def temp_mlruns(tmp_path):
    """Create a temporary MLflow tracking directory."""
    mlruns_dir = tmp_path / "mlruns"
    mlruns_dir.mkdir()
    return str(mlruns_dir)


@pytest.fixture
def mlflow_service(temp_mlruns):
    """Create MLflow service with temporary tracking."""
    return MLflowService(
        tracking_uri=temp_mlruns,
        experiment_name="test-experiment",
        model_name="test-model",
    )


class TestMLflowServiceInit:
    """Tests for MLflow service initialization."""

    def test_init_creates_service(self, temp_mlruns):
        """Service should initialize with given parameters."""
        service = MLflowService(
            tracking_uri=temp_mlruns,
            experiment_name="my-experiment",
            model_name="my-model",
        )
        
        assert service.tracking_uri == temp_mlruns
        assert service.experiment_name == "my-experiment"
        assert service.model_name == "my-model"

    def test_from_config(self, temp_mlruns):
        """Service should be created from config dict."""
        config = {
            "mlflow": {
                "tracking_uri": temp_mlruns,
                "experiment_name": "config-experiment",
                "model_name": "config-model",
            }
        }
        
        service = MLflowService.from_config(config)
        
        assert service.experiment_name == "config-experiment"
        assert service.model_name == "config-model"

    def test_from_config_with_defaults(self):
        """Service should use defaults when config is empty."""
        service = MLflowService.from_config({})
        
        assert service.experiment_name == "bloodcells-classification"
        assert service.model_name == "bloodcells-classifier"


class TestMLflowRuns:
    """Tests for MLflow run management."""

    def test_start_run_returns_run_id(self, mlflow_service):
        """Starting a run should return a run ID."""
        run_id = mlflow_service.start_run(run_name="test-run")
        
        assert run_id is not None
        assert mlflow_service.run_id == run_id
        
        mlflow_service.end_run()

    def test_end_run_clears_run_id(self, mlflow_service):
        """Ending a run should clear the run ID."""
        mlflow_service.start_run()
        mlflow_service.end_run()
        
        assert mlflow_service.run_id is None


class TestMLflowLogging:
    """Tests for MLflow logging functionality."""

    def test_log_params(self, mlflow_service):
        """Should log parameters without error."""
        mlflow_service.start_run()
        
        # Should not raise
        mlflow_service.log_params({
            "learning_rate": 0.001,
            "batch_size": 32,
            "model": {"name": "resnet18", "pretrained": True},
        })
        
        mlflow_service.end_run()

    def test_log_metrics(self, mlflow_service):
        """Should log metrics without error."""
        mlflow_service.start_run()
        
        # Should not raise
        mlflow_service.log_metrics({
            "accuracy": 0.95,
            "loss": 0.05,
        })
        
        mlflow_service.end_run()

    def test_log_metrics_with_step(self, mlflow_service):
        """Should log metrics with step number."""
        mlflow_service.start_run()
        
        for epoch in range(3):
            mlflow_service.log_metrics({"loss": 1.0 / (epoch + 1)}, step=epoch)
        
        mlflow_service.end_run()

    def test_set_tags(self, mlflow_service):
        """Should set tags without error."""
        mlflow_service.start_run()
        
        mlflow_service.set_tags({
            "dataset": "bloodcells",
            "version": "1.0",
        })
        
        mlflow_service.end_run()

    def test_log_artifact(self, mlflow_service, tmp_path):
        """Should log artifact file."""
        mlflow_service.start_run()
        
        # Create a test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")
        
        # Should not raise
        mlflow_service.log_artifact(str(test_file))
        
        mlflow_service.end_run()


class TestMLflowModelRegistry:
    """Tests for MLflow model registry."""

    def test_log_model(self, mlflow_service):
        """Should log a PyTorch model."""
        mlflow_service.start_run()
        
        model = SimpleModel()
        
        # Should not raise
        mlflow_service.log_model(
            model,
            artifact_path="model",
            registered_model_name="test-model",
        )
        
        mlflow_service.end_run()

    def test_get_latest_model_version_when_none(self, mlflow_service):
        """Should return None when no model is registered."""
        version = mlflow_service.get_latest_model_version()
        assert version is None

    def test_get_model_uri_when_none(self, mlflow_service):
        """Should return None when no model is registered."""
        uri = mlflow_service.get_model_uri()
        assert uri is None


class TestFlattenDict:
    """Tests for dictionary flattening utility."""

    def test_flatten_simple_dict(self, mlflow_service):
        """Should flatten simple dict."""
        d = {"a": 1, "b": 2}
        result = mlflow_service._flatten_dict(d)
        assert result == {"a": 1, "b": 2}

    def test_flatten_nested_dict(self, mlflow_service):
        """Should flatten nested dict with dot notation."""
        d = {
            "model": {
                "name": "resnet18",
                "pretrained": True,
            },
            "training": {
                "lr": 0.001,
            },
        }
        result = mlflow_service._flatten_dict(d)
        
        assert result["model.name"] == "resnet18"
        assert result["model.pretrained"] is True
        assert result["training.lr"] == 0.001

    def test_flatten_deeply_nested_dict(self, mlflow_service):
        """Should flatten deeply nested dict."""
        d = {"a": {"b": {"c": 1}}}
        result = mlflow_service._flatten_dict(d)
        assert result == {"a.b.c": 1}
