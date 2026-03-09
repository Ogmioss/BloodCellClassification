"""
Tests for Airflow DAGs (API versions).

Tests DAG structure and import correctness (without running Airflow).
Requires apache-airflow to be installed — skipped otherwise.
"""

import sys
from pathlib import Path

import pytest

# Add project root to path for DAG imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

airflow = pytest.importorskip("airflow", reason="apache-airflow not installed")


# ============================================================
# Test DAG imports
# ============================================================

class TestDAGImports:
    """Test that all API DAGs can be imported without errors."""

    def test_train_dag_api_imports(self):
        """Training API DAG should import without errors."""
        from dags.bloodcells_train_dag_api import dag

        assert dag is not None
        assert dag.dag_id == "bloodcells_train_model_api"

    def test_evaluate_dag_api_imports(self):
        """Evaluation API DAG should import without errors."""
        from dags.bloodcells_evaluate_dag_api import dag

        assert dag is not None
        assert dag.dag_id == "bloodcells_evaluate_model_api"

    def test_batch_inference_dag_api_imports(self):
        """Batch inference API DAG should import without errors."""
        from dags.bloodcells_batch_inference_dag_api import dag

        assert dag is not None
        assert dag.dag_id == "bloodcells_batch_inference_api"

    def test_data_validation_dag_api_imports(self):
        """Data validation API DAG should import without errors."""
        from dags.bloodcells_data_validation_dag_api import dag

        assert dag is not None
        assert dag.dag_id == "bloodcells_data_validation_api"


# ============================================================
# Test DAG structure
# ============================================================

class TestTrainDAGStructure:
    """Test training API DAG structure."""

    def test_train_dag_has_required_tasks(self):
        """Training API DAG should have all required tasks."""
        from dags.bloodcells_train_dag_api import dag

        task_ids = [task.task_id for task in dag.tasks]

        assert "check_api_health" in task_ids
        assert "trigger_training" in task_ids
        assert "decide_promotion" in task_ids
        assert "promote_model" in task_ids
        assert "skip_promotion" in task_ids
        assert "training_complete" in task_ids

    def test_train_dag_task_dependencies(self):
        """Training API DAG should have correct task order."""
        from dags.bloodcells_train_dag_api import dag

        check_api = dag.get_task("check_api_health")
        trigger = dag.get_task("trigger_training")
        decide = dag.get_task("decide_promotion")

        assert "trigger_training" in [t.task_id for t in check_api.downstream_list]
        assert "decide_promotion" in [t.task_id for t in trigger.downstream_list]
        downstream_ids = [t.task_id for t in decide.downstream_list]
        assert "promote_model" in downstream_ids
        assert "skip_promotion" in downstream_ids

    def test_train_dag_is_manual_trigger(self):
        """Training API DAG should be manual trigger only."""
        from dags.bloodcells_train_dag_api import dag

        assert dag.schedule_interval is None

    def test_train_dag_has_api_tag(self):
        """Training API DAG should be tagged with 'api'."""
        from dags.bloodcells_train_dag_api import dag

        assert "api" in dag.tags

    def test_train_dag_promotes_to_staging(self):
        """Training DAG should promote to Staging (not Production)."""
        from dags.bloodcells_train_dag_api import promote_model, FASTAPI_URL

        # Verify the promote function sends stage=Staging by inspecting source
        import inspect
        source = inspect.getsource(promote_model)
        assert "Staging" in source
        assert '"stage": "Staging"' in source or "'stage': 'Staging'" in source


class TestEvaluateDAGStructure:
    """Test evaluation API DAG structure."""

    def test_evaluate_dag_has_required_tasks(self):
        """Evaluation API DAG should have all required tasks."""
        from dags.bloodcells_evaluate_dag_api import dag

        task_ids = [task.task_id for task in dag.tasks]

        assert "check_api_health" in task_ids
        assert "trigger_evaluation" in task_ids
        assert "report_results" in task_ids


class TestBatchInferenceDAGStructure:
    """Test batch inference API DAG structure."""

    def test_batch_inference_dag_has_required_tasks(self):
        """Batch inference API DAG should have all required tasks."""
        from dags.bloodcells_batch_inference_dag_api import dag

        task_ids = [task.task_id for task in dag.tasks]

        assert "check_api_health" in task_ids
        assert "trigger_batch_inference" in task_ids
        assert "report_results" in task_ids


class TestDataValidationDAGStructure:
    """Test data validation API DAG structure."""

    def test_data_validation_dag_has_required_tasks(self):
        """Data validation API DAG should have all required tasks."""
        from dags.bloodcells_data_validation_dag_api import dag

        task_ids = [task.task_id for task in dag.tasks]

        assert "check_api_health" in task_ids
        assert "trigger_data_validation" in task_ids
        assert "report_results" in task_ids

    def test_data_validation_dag_task_dependencies(self):
        """Data validation API DAG should have correct task order."""
        from dags.bloodcells_data_validation_dag_api import dag

        check_api = dag.get_task("check_api_health")
        validate = dag.get_task("trigger_data_validation")

        assert "trigger_data_validation" in [t.task_id for t in check_api.downstream_list]
        assert "report_results" in [t.task_id for t in validate.downstream_list]


# ============================================================
# Test API client functions
# ============================================================

class TestAPIClient:
    """Test API client utility functions."""

    def test_api_client_imports(self):
        """API client should import all functions without errors."""
        from dags.utils.api_client import (
            check_api_health,
            wait_for_api,
            start_training,
            start_evaluation,
            start_batch_inference,
            start_data_validation,
            run_training_and_wait,
            run_evaluation_and_wait,
            run_batch_inference_and_wait,
            run_data_validation_and_wait,
            predict_image,
            get_model_info,
            get_task_status,
            wait_for_task,
        )

        # Verify all functions are callable
        assert callable(check_api_health)
        assert callable(wait_for_api)
        assert callable(start_training)
        assert callable(start_evaluation)
        assert callable(start_batch_inference)
        assert callable(start_data_validation)
        assert callable(run_training_and_wait)
        assert callable(run_evaluation_and_wait)
        assert callable(run_batch_inference_and_wait)
        assert callable(run_data_validation_and_wait)
        assert callable(predict_image)
        assert callable(get_model_info)
        assert callable(get_task_status)
        assert callable(wait_for_task)
