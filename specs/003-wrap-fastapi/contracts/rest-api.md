# REST API Contract

**Base URL**: `http://localhost:8000`

## Health & System

### GET /health

Retourne le statut de santé de l'API.

**Response 200**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cpu",
  "model_name": "resnet18",
  "model_source": "mlflow",
  "model_version": "3"
}
```

### GET /metrics (Model Performance)

Retourne les métriques de performance du modèle chargé.

**Response 200**:
```json
{
  "accuracy": 0.92,
  "best_val_acc": 0.93,
  "final_train_acc": 0.95,
  "final_train_loss": 0.12,
  "class_names": ["basophil", "eosinophil", ...],
  "confusion_matrix": [[...]]
}
```

**Response 404**: Pas de métriques disponibles (modèle non entraîné).

### GET /model/info

Retourne les informations d'architecture du modèle.

**Response 200**:
```json
{
  "model_name": "resnet18",
  "pretrained": true,
  "num_classes": 8,
  "checkpoint_available": true,
  "model_source": "mlflow",
  "mlflow": {
    "tracking_uri": "http://mlflow:5000",
    "model_name": "bloodcells-classifier",
    "latest_version": "3"
  }
}
```

---

## Prediction

### POST /predict

Prédiction depuis image encodée en base64.

**Request**:
```json
{
  "image_base64": "<base64-encoded-image>"
}
```

**Response 200**:
```json
{
  "predicted_class": "neutrophil",
  "confidence": 0.95,
  "probabilities": {
    "basophil": 0.01,
    "eosinophil": 0.02,
    "erythroblast": 0.00,
    "immature_granulocyte": 0.01,
    "lymphocyte": 0.00,
    "monocyte": 0.01,
    "neutrophil": 0.95,
    "platelet": 0.00
  },
  "all_predictions": [
    {"class": "neutrophil", "probability": 0.95},
    {"class": "eosinophil", "probability": 0.02},
    ...
  ]
}
```

**Response 400**: Image invalide ou corrompue.
**Response 503**: Modèle non chargé.

### POST /predict/upload

Prédiction depuis upload de fichier.

**Request**: `multipart/form-data` avec champ `file` (JPEG, PNG).

**Response 200**: Identique à POST /predict.
**Response 400**: Fichier non-image ou corrompu.

---

## MLflow Model Registry

### GET /mlflow/models

Liste les versions du modèle enregistrées dans MLflow.

**Response 200**:
```json
{
  "model_name": "bloodcells-classifier",
  "tracking_uri": "http://mlflow:5000",
  "versions": [
    {
      "version": "3",
      "stage": "Staging",
      "status": "READY",
      "run_id": "abc123",
      "creation_timestamp": 1709913000000
    }
  ]
}
```

**Response 503**: MLflow indisponible.

### POST /mlflow/promote/{version}

Promeut une version de modèle (default: Production).

**Path params**: `version` (int) — numéro de version MLflow.
**Query params**: `stage` (string, default: "Production") — Staging ou Production.

**Response 200**:
```json
{
  "status": "success",
  "message": "Model version 3 promoted to Production"
}
```

**Response 404**: Version inexistante.
**Response 503**: MLflow indisponible.

---

## ML Tasks (Background)

### POST /ml/train

Lance un entraînement en arrière-plan.

**Request** (tous optionnels):
```json
{
  "epochs": 20,
  "learning_rate": 0.001,
  "batch_size": 32
}
```

**Response 202**:
```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending",
  "task_type": "train"
}
```

**Response 409**: Une tâche GPU est déjà en cours.

### POST /ml/evaluate

Lance une évaluation en arrière-plan.

**Request** (optionnel):
```json
{
  "checkpoint_path": "./models/checkpoints/best_model.pth"
}
```

**Response 202**: Identique structure (task_type: "evaluate").
**Response 409**: Une tâche GPU est déjà en cours.

### POST /ml/batch-inference

Lance une inférence en batch.

**Request** (optionnel):
```json
{
  "input_dir": "./data/raw",
  "output_dir": "./data/results",
  "max_images": 1000
}
```

**Response 202**: Identique structure (task_type: "batch_inference").

### POST /ml/validate-data

Lance une validation du dataset.

**Request** (optionnel):
```json
{
  "dataset_path": "./data/processed",
  "check_images": true
}
```

**Response 202**: Identique structure (task_type: "validate_data").

### GET /ml/tasks/{task_id}

Consulte le statut d'une tâche.

**Response 200**:
```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "task_type": "train",
  "status": "completed",
  "created_at": "2026-03-08T14:30:00Z",
  "started_at": "2026-03-08T14:30:01Z",
  "completed_at": "2026-03-08T14:45:00Z",
  "result": {
    "training_metrics": {"final_train_loss": 0.12, "final_train_acc": 0.95},
    "test_results": {"accuracy": 0.92},
    "mlflow_run_id": "abc123",
    "model_version": "3"
  },
  "error": null
}
```

**Response 404**: Tâche inexistante.

### GET /ml/tasks

Liste toutes les tâches.

**Query params**:
- `status` (string, optionnel) — filtre par statut (pending, running, completed, failed)
- `task_type` (string, optionnel) — filtre par type

**Response 200**: Liste de TaskResult.

### DELETE /ml/tasks/{task_id}

Supprime une tâche terminée du store.

**Response 200**: `{"message": "Task deleted"}`.
**Response 404**: Tâche inexistante.

---

## Airflow Pipelines

### POST /pipelines/train

Déclenche le DAG d'entraînement dans Airflow.

**Response 200**:
```json
{
  "status": "success",
  "dag_id": "bloodcells_train_model_api",
  "dag_run_id": "manual__2026-03-08T14:30:00",
  "execution_date": "2026-03-08T14:30:00Z"
}
```

**Response 503**: Airflow indisponible.

### POST /pipelines/evaluate, /pipelines/batch-inference, /pipelines/validate-data

Même structure que /pipelines/train avec le dag_id correspondant.

### GET /pipelines/status/{dag_id}/{dag_run_id}

Consulte le statut d'un DAG run.

**Response 200**:
```json
{
  "dag_id": "bloodcells_train_model_api",
  "dag_run_id": "manual__2026-03-08T14:30:00",
  "state": "success",
  "execution_date": "2026-03-08T14:30:00Z",
  "start_date": "2026-03-08T14:30:01Z",
  "end_date": "2026-03-08T14:45:00Z"
}
```

### GET /pipelines/runs/{dag_id}

Liste les runs récents d'un DAG.

**Query params**: `limit` (int, default: 10).

**Response 200**: Liste de DAG run status.

### GET /pipelines/dags

Liste tous les DAGs disponibles.

**Response 200**:
```json
[
  {
    "dag_id": "bloodcells_train_model_api",
    "description": "Training pipeline",
    "trigger_endpoint": "/pipelines/train",
    "is_paused": false,
    "airflow_status": "available"
  }
]
```

---

## Prometheus Metrics

### GET /metrics (Prometheus format)

Endpoint instrumenté par `prometheus-fastapi-instrumentator`.

**Métriques exposées**:

| Metric | Type | Description |
|--------|------|-------------|
| `bloodcell_predictions_total` | Counter | Compteur de prédictions par classe (label: `predicted_class`) |
| `bloodcell_prediction_errors_total` | Counter | Compteur d'erreurs de prédiction |
| `bloodcell_prediction_latency_seconds` | Histogram | Latence des prédictions |
| `bloodcell_prediction_confidence` | Histogram | Distribution des scores de confiance |
| `bloodcell_model_info` | Gauge | Informations modèle (labels: source, version, architecture) |
| `bloodcell_api_ready` | Gauge | API prête (1) ou non (0) |
| `http_requests_total` | Counter | Compteur HTTP standard (instrumentator) |
| `http_request_duration_seconds` | Histogram | Latence HTTP standard |

---

## Error Responses

Toutes les erreurs suivent le format :

```json
{
  "detail": "Description détaillée de l'erreur"
}
```

| Code | Usage |
|------|-------|
| 400 | Image invalide, paramètres invalides |
| 404 | Tâche/version inexistante, métriques non disponibles |
| 409 | Tâche GPU déjà en cours (conflit de concurrence) |
| 503 | Service externe indisponible (MLflow, Airflow, modèle non chargé) |
