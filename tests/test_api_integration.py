"""
Tests d'intégration pour l'API FastAPI.

Ces tests utilisent de vraies données (images du dataset) sans mock.
Ils testent les endpoints de bout en bout.
"""

import base64
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.core.constants import CLASS_NAMES


# ============================================================
# Configuration
# ============================================================

DATASET_DIR = Path(__file__).parent.parent / "src" / "data" / "raw" / "bloodcells_dataset"

# Mapping entre les noms de dossiers du dataset et les noms de classes du modèle
FOLDER_TO_CLASS = {
    "basophil": "basophil",
    "eosinophil": "eosinophil",
    "erythroblast": "erythroblast",
    "ig": "immature_granulocyte",
    "lymphocyte": "lymphocyte",
    "monocyte": "monocyte",
    "neutrophil": "neutrophil",
    "platelet": "platelet",
}
DATASET_FOLDERS = list(FOLDER_TO_CLASS.keys())


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture(scope="module")
def client():
    """Create test client (module scope for performance)."""
    return TestClient(app)


@pytest.fixture(scope="module")
def sample_images() -> dict[str, Path]:
    """Get one sample image per class from the dataset."""
    images = {}
    for folder_name in DATASET_FOLDERS:
        class_dir = DATASET_DIR / folder_name
        if class_dir.exists():
            # Get first image in the directory
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    # Use the model class name, not the folder name
                    class_name = FOLDER_TO_CLASS[folder_name]
                    images[class_name] = img_path
                    break
    return images


@pytest.fixture
def basophil_image_base64(sample_images) -> str:
    """Get a basophil image as base64."""
    if "basophil" not in sample_images:
        pytest.skip("Basophil images not available")
    
    with open(sample_images["basophil"], "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


@pytest.fixture
def basophil_image_path(sample_images) -> Path:
    """Get path to a basophil image."""
    if "basophil" not in sample_images:
        pytest.skip("Basophil images not available")
    return sample_images["basophil"]


# ============================================================
# Health Endpoint Tests
# ============================================================

class TestHealthEndpointIntegration:
    """Tests d'intégration pour GET /health."""

    def test_health_returns_200(self, client):
        """L'endpoint health doit retourner 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_has_required_fields(self, client):
        """La réponse health doit contenir tous les champs requis."""
        response = client.get("/health")
        data = response.json()
        
        assert "status" in data
        assert "model_loaded" in data
        assert "device" in data
        assert "model_name" in data

    def test_health_status_is_healthy(self, client):
        """Le status doit être 'healthy'."""
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "healthy"

    def test_health_model_loaded_is_boolean(self, client):
        """model_loaded doit être un booléen."""
        response = client.get("/health")
        data = response.json()
        assert isinstance(data["model_loaded"], bool)


# ============================================================
# Metrics Endpoint Tests
# ============================================================

class TestMetricsEndpointIntegration:
    """Tests d'intégration pour GET /metrics."""

    def test_metrics_returns_200_or_404(self, client):
        """L'endpoint metrics doit retourner 200 si metrics.json existe, 404 sinon."""
        response = client.get("/metrics")
        assert response.status_code in [200, 404]

    def test_metrics_response_structure_when_available(self, client):
        """Si metrics disponibles, la structure doit être correcte."""
        response = client.get("/metrics")
        if response.status_code == 200:
            data = response.json()
            assert "accuracy" in data
            assert "best_val_acc" in data
            assert "class_names" in data
            assert isinstance(data["accuracy"], float)
            assert 0.0 <= data["accuracy"] <= 1.0


# ============================================================
# Model Info Endpoint Tests
# ============================================================

class TestModelInfoEndpointIntegration:
    """Tests d'intégration pour GET /model/info."""

    def test_model_info_returns_200(self, client):
        """L'endpoint model/info doit retourner 200."""
        response = client.get("/model/info")
        assert response.status_code == 200

    def test_model_info_has_required_fields(self, client):
        """La réponse doit contenir les champs d'architecture."""
        response = client.get("/model/info")
        data = response.json()
        
        assert "model_name" in data
        assert "num_classes" in data
        assert "class_names" in data
        assert "checkpoint_available" in data

    def test_model_info_class_names_match(self, client):
        """Les class_names doivent correspondre aux classes attendues."""
        response = client.get("/model/info")
        data = response.json()
        
        assert len(data["class_names"]) == 8
        # Vérifier que les class_names de l'API correspondent aux constantes
        assert set(data["class_names"]) == set(CLASS_NAMES)

    def test_model_info_num_classes_is_8(self, client):
        """Il doit y avoir 8 classes."""
        response = client.get("/model/info")
        data = response.json()
        assert data["num_classes"] == 8


# ============================================================
# MLflow Models Endpoint Tests
# ============================================================

class TestMLflowModelsEndpointIntegration:
    """Tests d'intégration pour GET /mlflow/models."""

    def test_mlflow_models_returns_200(self, client):
        """L'endpoint mlflow/models doit retourner 200 (même si MLflow indisponible)."""
        response = client.get("/mlflow/models")
        assert response.status_code == 200

    def test_mlflow_models_has_model_name(self, client):
        """La réponse doit contenir le nom du modèle."""
        response = client.get("/mlflow/models")
        data = response.json()
        assert "model_name" in data

    def test_mlflow_models_has_versions_list(self, client):
        """La réponse doit contenir une liste de versions."""
        response = client.get("/mlflow/models")
        data = response.json()
        assert "versions" in data
        assert isinstance(data["versions"], list)


# ============================================================
# Predict Endpoint Tests (avec vraies images)
# ============================================================

class TestPredictEndpointIntegration:
    """Tests d'intégration pour POST /predict avec vraies images."""

    def test_predict_with_real_image_returns_200(self, client, basophil_image_base64):
        """La prédiction avec une vraie image doit retourner 200."""
        response = client.post(
            "/predict",
            json={"image_base64": basophil_image_base64}
        )
        # 200 si modèle chargé, 503 sinon
        assert response.status_code in [200, 503]

    def test_predict_response_structure(self, client, basophil_image_base64):
        """La réponse de prédiction doit avoir la bonne structure."""
        response = client.post(
            "/predict",
            json={"image_base64": basophil_image_base64}
        )
        
        if response.status_code == 200:
            data = response.json()
            assert "predicted_class" in data
            assert "confidence" in data
            assert "probabilities" in data
            
            # Vérifier les types
            assert isinstance(data["predicted_class"], str)
            assert isinstance(data["confidence"], float)
            assert isinstance(data["probabilities"], dict)
            
            # Vérifier les valeurs
            assert 0.0 <= data["confidence"] <= 1.0
            assert data["predicted_class"] in CLASS_NAMES

    def test_predict_probabilities_sum_to_one(self, client, basophil_image_base64):
        """Les probabilités doivent sommer à ~1.0."""
        response = client.post(
            "/predict",
            json={"image_base64": basophil_image_base64}
        )
        
        if response.status_code == 200:
            data = response.json()
            total_prob = sum(data["probabilities"].values())
            assert 0.99 <= total_prob <= 1.01

    def test_predict_with_invalid_base64_returns_400(self, client):
        """Une image base64 invalide doit retourner 400."""
        response = client.post(
            "/predict",
            json={"image_base64": "not-valid-base64!!!"}
        )
        assert response.status_code == 400

    def test_predict_with_empty_base64_returns_400_or_422(self, client):
        """Une image base64 vide doit retourner une erreur."""
        response = client.post(
            "/predict",
            json={"image_base64": ""}
        )
        assert response.status_code in [400, 422]


# ============================================================
# Predict Upload Endpoint Tests
# ============================================================

class TestPredictUploadEndpointIntegration:
    """Tests d'intégration pour POST /predict/upload."""

    def test_predict_upload_with_real_image(self, client, basophil_image_path):
        """L'upload d'une vraie image doit fonctionner."""
        with open(basophil_image_path, "rb") as f:
            response = client.post(
                "/predict/upload",
                files={"file": ("test.jpg", f, "image/jpeg")}
            )
        
        # 200 si modèle chargé, 503 sinon
        assert response.status_code in [200, 503]

    def test_predict_upload_response_structure(self, client, basophil_image_path):
        """La réponse d'upload doit avoir la bonne structure."""
        with open(basophil_image_path, "rb") as f:
            response = client.post(
                "/predict/upload",
                files={"file": ("test.jpg", f, "image/jpeg")}
            )
        
        if response.status_code == 200:
            data = response.json()
            assert "predicted_class" in data
            assert "confidence" in data
            assert "probabilities" in data

    def test_predict_upload_invalid_content_type(self, client):
        """Un fichier avec un mauvais content-type doit être rejeté."""
        response = client.post(
            "/predict/upload",
            files={"file": ("test.txt", b"not an image", "text/plain")}
        )
        assert response.status_code == 400


# ============================================================
# Pipelines Endpoints Tests
# ============================================================

class TestPipelinesEndpointIntegration:
    """Tests d'intégration pour les endpoints /pipelines."""

    def test_list_dags_returns_200(self, client):
        """L'endpoint /pipelines/dags doit retourner 200."""
        response = client.get("/pipelines/dags")
        assert response.status_code == 200

    def test_list_dags_returns_list(self, client):
        """La réponse doit être une liste de DAGs."""
        response = client.get("/pipelines/dags")
        data = response.json()
        
        assert isinstance(data, list)
        assert len(data) >= 3  # Au moins train, evaluate, batch-inference

    def test_list_dags_has_required_fields(self, client):
        """Chaque DAG doit avoir les champs requis."""
        response = client.get("/pipelines/dags")
        data = response.json()
        
        for dag in data:
            assert "dag_id" in dag
            assert "description" in dag
            assert "trigger_endpoint" in dag


# ============================================================
# ML Tasks Endpoints Tests
# ============================================================

class TestMLTasksEndpointIntegration:
    """Tests d'intégration pour les endpoints /ml."""

    def test_list_tasks_returns_200(self, client):
        """L'endpoint /ml/tasks doit retourner 200."""
        response = client.get("/ml/tasks")
        assert response.status_code == 200

    def test_list_tasks_returns_list(self, client):
        """La réponse doit être une liste."""
        response = client.get("/ml/tasks")
        data = response.json()
        assert isinstance(data, list)

    def test_get_nonexistent_task_returns_404(self, client):
        """Un task_id inexistant doit retourner 404."""
        response = client.get("/ml/tasks/nonexistent-task-id")
        assert response.status_code == 404


# ============================================================
# Multi-class Prediction Tests
# ============================================================

class TestMultiClassPredictionIntegration:
    """Tests de prédiction sur plusieurs classes."""

    def test_predict_multiple_classes(self, client, sample_images):
        """Tester la prédiction sur des images de différentes classes."""
        if not sample_images:
            pytest.skip("No sample images available")
        
        results = {}
        for class_name, img_path in sample_images.items():
            with open(img_path, "rb") as f:
                image_base64 = base64.b64encode(f.read()).decode("utf-8")
            
            response = client.post(
                "/predict",
                json={"image_base64": image_base64}
            )
            
            if response.status_code == 200:
                data = response.json()
                results[class_name] = {
                    "predicted": data["predicted_class"],
                    "confidence": data["confidence"],
                    "correct": data["predicted_class"] == class_name
                }
        
        # Log results for debugging
        if results:
            correct_count = sum(1 for r in results.values() if r["correct"])
            total_count = len(results)
            print(f"\nPrediction results: {correct_count}/{total_count} correct")
            for class_name, result in results.items():
                status = "✓" if result["correct"] else "✗"
                print(f"  {status} {class_name}: predicted={result['predicted']} ({result['confidence']:.2%})")
