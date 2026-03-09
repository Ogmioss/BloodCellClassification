"""
Model Loader Service

Single Responsibility: Unified model loading from MLflow or checkpoint.
Provides a single entry point for loading models across API and DAGs.
"""

from pathlib import Path
from typing import Optional
from dataclasses import dataclass


from src.core.constants import CLASS_NAMES
from src.services.mlflow_service import MLflowService
from src.services.inference_service import InferenceService
from src.services.data_transform_service import DataTransformService
from src.models.model_factory import ModelFactory


@dataclass
class ModelLoadResult:
    """Result of model loading operation."""
    inference_service: InferenceService
    source: str  # "mlflow:v{version}" or "checkpoint:{filename}"
    version: Optional[str] = None


class ModelLoaderService:
    """
    Service unifié pour charger un modèle depuis MLflow ou checkpoint.
    
    Utilisé par:
    - FastAPI (src/api/main.py)
    - DAG batch inference (dags/bloodcells_batch_inference_dag.py)
    - Tout autre composant nécessitant un modèle d'inférence
    
    Stratégie de chargement:
    1. Si prefer_mlflow=True, essaie MLflow d'abord (stage spécifié ou latest)
    2. Fallback sur checkpoint local si MLflow échoue ou n'a pas de modèle
    3. Lève une exception si aucun modèle n'est disponible
    """

    def __init__(
        self,
        config: dict,
        checkpoint_dir: Optional[Path] = None,
        checkpoint_name: str = "best_model.pth"
    ):
        """
        Initialize ModelLoaderService.

        Args:
            config: Configuration dictionary (from YamlLoader)
            checkpoint_dir: Directory containing checkpoints (default: from config)
            checkpoint_name: Name of the checkpoint file
        """
        self.config = config
        self.checkpoint_name = checkpoint_name
        self.device = ModelFactory.get_device()
        
        # Resolve checkpoint directory
        if checkpoint_dir is not None:
            self.checkpoint_dir = checkpoint_dir
        else:
            from src.services.yaml_loader import YamlLoader
            loader = YamlLoader()
            checkpoint_path = loader.get_nested_value(
                "paths.models.checkpoints",
                "./models/checkpoints"
            )
            self.checkpoint_dir = loader._resolve_dir(checkpoint_path)
        
        # Initialize MLflow service (lazy)
        self._mlflow_service: Optional[MLflowService] = None

    @property
    def mlflow_service(self) -> MLflowService:
        """Lazy initialization of MLflow service."""
        if self._mlflow_service is None:
            self._mlflow_service = MLflowService.from_config(self.config)
        return self._mlflow_service

    @property
    def checkpoint_path(self) -> Path:
        """Full path to checkpoint file."""
        return self.checkpoint_dir / self.checkpoint_name

    def load(
        self,
        prefer_mlflow: bool = True,
        mlflow_stage: Optional[str] = None,
        mlflow_version: Optional[str] = None,
    ) -> ModelLoadResult:
        """
        Load model and return InferenceService with source info.

        Args:
            prefer_mlflow: If True, try MLflow first, fallback to checkpoint
            mlflow_stage: MLflow stage to load from ("Production", "Staging", etc.)
            mlflow_version: Specific MLflow version to load

        Returns:
            ModelLoadResult with inference_service and source info

        Raises:
            FileNotFoundError: If no model is available
        """
        if prefer_mlflow:
            result = self._try_load_from_mlflow(mlflow_stage, mlflow_version)
            if result is not None:
                return result
        
        # Fallback to checkpoint
        return self._load_from_checkpoint()

    def _try_load_from_mlflow(
        self,
        stage: Optional[str] = None,
        version: Optional[str] = None,
    ) -> Optional[ModelLoadResult]:
        """
        Try to load model from MLflow registry.

        Returns:
            ModelLoadResult if successful, None otherwise
        """
        try:
            # Determine version to load
            if version is None:
                version = self.mlflow_service.get_latest_model_version(stage=stage)
            
            if version is None:
                print("ℹ️ No model found in MLflow registry")
                return None
            
            print(f"📦 Loading model from MLflow (version {version})...")
            model = self.mlflow_service.load_model(version=version)
            model = model.to(self.device)
            model.eval()
            
            # Create transform service
            transform_service = DataTransformService(self.config)
            
            # Create inference service
            inference_service = InferenceService(
                model=model,
                transform_service=transform_service,
                device=self.device,
                class_names=CLASS_NAMES
            )
            
            source = f"mlflow:v{version}"
            if stage:
                source = f"mlflow:{stage}:v{version}"
            
            print(f"✅ Model loaded from: {source}")
            
            return ModelLoadResult(
                inference_service=inference_service,
                source=source,
                version=version
            )
            
        except Exception as e:
            print(f"⚠️ Could not load from MLflow: {e}")
            return None

    def _load_from_checkpoint(self) -> ModelLoadResult:
        """
        Load model from local checkpoint.

        Returns:
            ModelLoadResult

        Raises:
            FileNotFoundError: If checkpoint doesn't exist
        """
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"No model available. Checkpoint not found at {self.checkpoint_path}. "
                "Train a model first using 'uv run train-model'"
            )
        
        print(f"📦 Loading model from checkpoint: {self.checkpoint_path}")
        
        inference_service = InferenceService.load_from_checkpoint(
            checkpoint_path=self.checkpoint_path,
            config=self.config,
            device=self.device,
            class_names=CLASS_NAMES
        )
        
        source = f"checkpoint:{self.checkpoint_path.name}"
        print(f"✅ Model loaded from: {source}")
        
        return ModelLoadResult(
            inference_service=inference_service,
            source=source,
            version=None
        )

    def get_model_info(self) -> dict:
        """
        Get information about available models.

        Returns:
            Dictionary with model availability info
        """
        info = {
            "checkpoint_available": self.checkpoint_path.exists(),
            "checkpoint_path": str(self.checkpoint_path),
            "mlflow": {
                "available": False,
                "model_name": None,
                "latest_version": None,
                "tracking_uri": None,
            }
        }
        
        try:
            latest_version = self.mlflow_service.get_latest_model_version()
            if latest_version:
                info["mlflow"] = {
                    "available": True,
                    "model_name": self.mlflow_service.model_name,
                    "latest_version": latest_version,
                    "tracking_uri": self.mlflow_service.tracking_uri,
                }
        except Exception:
            pass
        
        return info

    def get_preferred_source(self) -> str:
        """
        Determine which source would be used for loading.

        Returns:
            "mlflow" or "checkpoint"
        """
        try:
            latest_version = self.mlflow_service.get_latest_model_version()
            if latest_version:
                return "mlflow"
        except Exception:
            pass
        return "checkpoint"
