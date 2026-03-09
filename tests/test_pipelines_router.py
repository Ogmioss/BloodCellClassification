"""
Tests for pipelines router.

Tests the Airflow DAG trigger endpoints.
"""

from unittest.mock import patch, AsyncMock
import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestListDags:
    """Tests for GET /pipelines/dags endpoint."""

    def test_list_dags_returns_200(self, client):
        """Should return list of available DAGs."""
        response = client.get("/pipelines/dags")
        
        assert response.status_code == 200
        data = response.json()
        
        assert isinstance(data, list)
        assert len(data) == 4
        
        dag_ids = [d["dag_id"] for d in data]
        assert "bloodcells_data_validation_api" in dag_ids
        assert "bloodcells_train_model_api" in dag_ids
        assert "bloodcells_evaluate_model_api" in dag_ids
        assert "bloodcells_batch_inference_api" in dag_ids

    def test_list_dags_contains_endpoints(self, client):
        """Each DAG should have a trigger endpoint."""
        response = client.get("/pipelines/dags")
        data = response.json()
        
        for dag in data:
            assert "trigger_endpoint" in dag
            assert dag["trigger_endpoint"].startswith("/pipelines/")


class TestTriggerTraining:
    """Tests for POST /pipelines/train endpoint."""

    def test_trigger_training_success(self, client):
        """Should trigger training DAG successfully."""
        mock_response = {
            "dag_run_id": "manual__2024-01-01T00:00:00+00:00",
            "execution_date": "2024-01-01T00:00:00+00:00",
        }
        
        with patch("src.api.routers.pipelines._trigger_dag", new_callable=AsyncMock) as mock_trigger:
            mock_trigger.return_value = mock_response
            
            response = client.post("/pipelines/train")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "triggered"
        assert data["dag_id"] == "bloodcells_train_model_api"
        assert data["dag_run_id"] == mock_response["dag_run_id"]

    def test_trigger_training_airflow_unavailable(self, client):
        """Should return 503 when Airflow is unavailable."""
        from fastapi import HTTPException
        
        with patch("src.api.routers.pipelines._trigger_dag", new_callable=AsyncMock) as mock_trigger:
            mock_trigger.side_effect = HTTPException(
                status_code=503,
                detail="Cannot connect to Airflow"
            )
            
            response = client.post("/pipelines/train")
        
        assert response.status_code == 503


class TestTriggerEvaluation:
    """Tests for POST /pipelines/evaluate endpoint."""

    def test_trigger_evaluation_success(self, client):
        """Should trigger evaluation DAG successfully."""
        mock_response = {
            "dag_run_id": "manual__2024-01-01T00:00:00+00:00",
            "execution_date": "2024-01-01T00:00:00+00:00",
        }
        
        with patch("src.api.routers.pipelines._trigger_dag", new_callable=AsyncMock) as mock_trigger:
            mock_trigger.return_value = mock_response
            
            response = client.post("/pipelines/evaluate")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "triggered"
        assert data["dag_id"] == "bloodcells_evaluate_model_api"


class TestTriggerBatchInference:
    """Tests for POST /pipelines/batch-inference endpoint."""

    def test_trigger_batch_inference_success(self, client):
        """Should trigger batch inference DAG successfully."""
        mock_response = {
            "dag_run_id": "manual__2024-01-01T00:00:00+00:00",
            "execution_date": "2024-01-01T00:00:00+00:00",
        }
        
        with patch("src.api.routers.pipelines._trigger_dag", new_callable=AsyncMock) as mock_trigger:
            mock_trigger.return_value = mock_response
            
            response = client.post("/pipelines/batch-inference")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "triggered"
        assert data["dag_id"] == "bloodcells_batch_inference_api"

    def test_trigger_batch_inference_with_config(self, client):
        """Should pass configuration to DAG."""
        mock_response = {
            "dag_run_id": "manual__2024-01-01T00:00:00+00:00",
            "execution_date": "2024-01-01T00:00:00+00:00",
        }
        
        with patch("src.api.routers.pipelines._trigger_dag", new_callable=AsyncMock) as mock_trigger:
            mock_trigger.return_value = mock_response
            
            response = client.post(
                "/pipelines/batch-inference",
                json={
                    "input_dir": "/data/images",
                    "output_dir": "/data/results",
                    "max_images": 100,
                }
            )
        
        assert response.status_code == 200
        
        # Verify config was passed
        mock_trigger.assert_called_once()
        call_args = mock_trigger.call_args
        assert call_args[1]["conf"]["input_dir"] == "/data/images"
        assert call_args[1]["conf"]["max_images"] == 100


class TestGetDagRunStatus:
    """Tests for GET /pipelines/status/{dag_id}/{dag_run_id} endpoint."""

    def test_get_status_success(self, client):
        """Should return DAG run status."""
        mock_response = {
            "dag_id": "bloodcells_train_model_api",
            "dag_run_id": "manual__2024-01-01T00:00:00+00:00",
            "state": "success",
            "execution_date": "2024-01-01T00:00:00+00:00",
            "start_date": "2024-01-01T00:00:01+00:00",
            "end_date": "2024-01-01T00:05:00+00:00",
        }

        with patch("src.api.routers.pipelines._get_dag_run_status", new_callable=AsyncMock) as mock_status:
            mock_status.return_value = mock_response

            response = client.get(
                "/pipelines/status/bloodcells_train_model_api/manual__2024-01-01T00:00:00+00:00"
            )

        assert response.status_code == 200
        data = response.json()

        assert data["state"] == "success"
        assert data["dag_id"] == "bloodcells_train_model_api"


class TestListDagRuns:
    """Tests for GET /pipelines/runs/{dag_id} endpoint."""

    def test_list_runs_airflow_unavailable(self, client):
        """Should return error when Airflow is unavailable."""
        # This test verifies the endpoint handles connection errors
        # In real scenario, Airflow would not be running during tests
        response = client.get("/pipelines/runs/bloodcells_train_model")

        # Should return an error since Airflow is not running
        # 503 = connection refused, 403/401 = wrong auth on another service
        assert response.status_code >= 400
