"""
MLflow Service

Single Responsibility: Handles MLflow experiment tracking and model registry.
"""

import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient


class MLflowService:
    """
    Service responsible for MLflow experiment tracking and model registry.
    Follows Single Responsibility Principle.
    """

    def __init__(
        self,
        tracking_uri: str = "./mlruns",
        experiment_name: str = "bloodcells-classification",
        model_name: str = "bloodcells-classifier",
    ):
        """
        Initialize MLflow service.

        Args:
            tracking_uri: URI for MLflow tracking server or local directory
            experiment_name: Name of the experiment
            model_name: Name for the model in the registry
        """
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.model_name = model_name
        self._run_id: Optional[str] = None

        # Setup MLflow
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.client = MlflowClient(tracking_uri=tracking_uri)

    @staticmethod
    def from_config(config: Dict[str, Any]) -> "MLflowService":
        """
        Create MLflowService from configuration dictionary.

        Args:
            config: Configuration dictionary with 'mlflow' section

        Returns:
            MLflowService instance
        """
        mlflow_config = config.get("mlflow", {})
        return MLflowService(
            tracking_uri=mlflow_config.get("tracking_uri", "./mlruns"),
            experiment_name=mlflow_config.get("experiment_name", "bloodcells-classification"),
            model_name=mlflow_config.get("model_name", "bloodcells-classifier"),
        )

    def start_run(self, run_name: Optional[str] = None) -> str:
        """
        Start a new MLflow run.

        Args:
            run_name: Optional name for the run

        Returns:
            Run ID
        """
        run = mlflow.start_run(run_name=run_name)
        self._run_id = run.info.run_id
        return self._run_id

    def end_run(self) -> None:
        """End the current MLflow run."""
        mlflow.end_run()
        self._run_id = None

    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log parameters to the current run.

        Args:
            params: Dictionary of parameters to log
        """
        # Flatten nested dicts for MLflow
        flat_params = self._flatten_dict(params)
        mlflow.log_params(flat_params)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        Log metrics to the current run.

        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number (epoch)
        """
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """
        Log an artifact file to the current run.

        Args:
            local_path: Path to the local file
            artifact_path: Optional path within the artifact store
        """
        mlflow.log_artifact(local_path, artifact_path)

    def log_model(
        self,
        model: Any,
        artifact_path: str = "model",
        registered_model_name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Log a PyTorch model to MLflow.

        Args:
            model: PyTorch model to log
            artifact_path: Path within the artifact store
            registered_model_name: If provided, register the model
            **kwargs: Additional arguments for mlflow.pytorch.log_model
        """
        mlflow.pytorch.log_model(
            model,
            artifact_path=artifact_path,
            registered_model_name=registered_model_name or self.model_name,
            **kwargs,
        )

    def log_git_info(self) -> None:
        """Log git commit hash and branch as tags."""
        try:
            commit_hash = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            ).decode().strip()
            branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL
            ).decode().strip()
            mlflow.set_tags({
                "git.commit": commit_hash,
                "git.branch": branch,
            })
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass  # Git not available or not a git repo

    def set_tags(self, tags: Dict[str, str]) -> None:
        """
        Set tags on the current run.

        Args:
            tags: Dictionary of tags
        """
        mlflow.set_tags(tags)

    def get_latest_model_version(self, stage: Optional[str] = None) -> Optional[str]:
        """
        Get the latest version of the registered model.

        Args:
            stage: Optional stage filter ("Production", "Staging", "Archived", "None")

        Returns:
            Model version number or None if not found
        """
        try:
            if stage:
                versions = self.client.get_latest_versions(self.model_name, stages=[stage])
            else:
                versions = self.client.get_latest_versions(self.model_name)
            
            if versions:
                return versions[0].version
            return None
        except mlflow.exceptions.MlflowException:
            return None

    def get_model_uri(self, version: Optional[str] = None, stage: Optional[str] = None) -> Optional[str]:
        """
        Get the URI for loading a model from the registry.

        Args:
            version: Specific version number
            stage: Stage name ("Production", "Staging", etc.)

        Returns:
            Model URI or None if not found
        """
        if version:
            return f"models:/{self.model_name}/{version}"
        elif stage:
            return f"models:/{self.model_name}/{stage}"
        else:
            # Get latest version
            latest_version = self.get_latest_model_version()
            if latest_version:
                return f"models:/{self.model_name}/{latest_version}"
            return None

    def load_model(self, version: Optional[str] = None, stage: Optional[str] = None) -> Any:
        """
        Load a model from the MLflow registry.

        Args:
            version: Specific version number
            stage: Stage name ("Production", "Staging", etc.)

        Returns:
            Loaded PyTorch model
        """
        model_uri = self.get_model_uri(version=version, stage=stage)
        if model_uri is None:
            raise ValueError(f"No model found in registry: {self.model_name}")
        return mlflow.pytorch.load_model(model_uri)

    def transition_model_stage(
        self,
        version: str,
        stage: str,
        archive_existing: bool = True,
    ) -> None:
        """
        Transition a model version to a new stage.

        Args:
            version: Model version number
            stage: Target stage ("Production", "Staging", "Archived", "None")
            archive_existing: Whether to archive existing models in the target stage
        """
        self.client.transition_model_version_stage(
            name=self.model_name,
            version=version,
            stage=stage,
            archive_existing_versions=archive_existing,
        )

    def promote_to_production(self, version: Optional[str] = None) -> str:
        """
        Promote a model version to Production stage.

        Args:
            version: Version to promote (defaults to latest)

        Returns:
            Version number that was promoted
        """
        if version is None:
            version = self.get_latest_model_version()
            if version is None:
                raise ValueError("No model versions found in registry")

        self.transition_model_stage(version, "Production", archive_existing=True)
        return version

    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
        """Flatten a nested dictionary."""
        items: List[tuple] = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    @property
    def run_id(self) -> Optional[str]:
        """Get the current run ID."""
        return self._run_id
