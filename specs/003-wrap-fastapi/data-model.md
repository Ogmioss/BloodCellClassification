# Data Model: Wrap FastAPI - MLOps Platform

**Branch**: `003-wrap-fastapi` | **Date**: 2026-03-08

## Entities

### Prediction

Résultat d'une classification d'image de cellule sanguine.

| Field | Type | Description |
|-------|------|-------------|
| predicted_class | string | Classe prédite parmi CLASS_NAMES (8 classes) |
| confidence | float | Score de confiance [0.0, 1.0] |
| probabilities | dict[string, float] | Probabilités par classe |
| all_predictions | list[dict] | Classes triées par probabilité décroissante |

**Validation rules**:
- `predicted_class` ∈ {basophil, eosinophil, erythroblast, immature_granulocyte, lymphocyte, monocyte, neutrophil, platelet}
- `confidence` ∈ [0.0, 1.0]
- Somme des probabilités ≈ 1.0

**State transitions**: N/A (entité immutable)

### MLTask (TaskResult)

Tâche ML en arrière-plan, persistée dans le task store JSON.

| Field | Type | Description |
|-------|------|-------------|
| task_id | string (UUID) | Identifiant unique |
| task_type | string | Type de tâche : train, evaluate, batch_inference, validate_data |
| status | TaskStatus enum | PENDING → RUNNING → COMPLETED \| FAILED |
| created_at | string (ISO 8601) | Timestamp de création |
| started_at | string (ISO 8601) \| null | Timestamp de début d'exécution |
| completed_at | string (ISO 8601) \| null | Timestamp de fin |
| result | dict \| null | Résultat de la tâche (structure variable selon task_type) |
| error | string \| null | Message d'erreur si FAILED |

**Validation rules**:
- `task_id` : UUID v4 unique
- `task_type` ∈ {train, evaluate, batch_inference, validate_data}
- `status` transition : PENDING → RUNNING → COMPLETED | FAILED (pas de retour en arrière)

**State transitions**:

```
PENDING ──→ RUNNING ──→ COMPLETED
                    └──→ FAILED
```

**GPU exclusivity**: Les tâches GPU (train, evaluate) vérifient qu'aucune autre tâche GPU n'est PENDING ou RUNNING. Retour 409 sinon.

### TaskResult.result — Structures par type

#### Train result
```json
{
  "training_metrics": {"final_train_loss": 0.12, "final_train_acc": 0.95},
  "test_results": {"accuracy": 0.92, "per_class": {...}},
  "mlflow_run_id": "abc123",
  "model_version": "3"
}
```

#### Evaluate result
```json
{
  "accuracy": 0.92,
  "loss": 0.15,
  "per_class_metrics": {...},
  "confusion_matrix": [[...]]
}
```

#### Batch Inference result
```json
{
  "total_images": 500,
  "successful": 498,
  "failed": 2,
  "output_file": "/path/to/results.json",
  "distribution": {"basophil": 62, "eosinophil": 65, ...}
}
```

#### Validate Data result
```json
{
  "total_images": 12000,
  "classes": {"basophil": 1500, ...},
  "corrupted_images": ["/path/to/bad.jpg"],
  "warnings": ["Class imbalance: basophil has 30% fewer images"],
  "is_valid": true
}
```

### Experiment (MLflow)

Run MLflow — géré par le service MLflow, pas par l'application directement.

| Field | Type | Description |
|-------|------|-------------|
| run_id | string | ID du run MLflow |
| experiment_name | string | "bloodcells-classification" |
| parameters | dict | Hyperparamètres (epochs, lr, batch_size, model_name) |
| metrics | dict | Métriques finales (accuracy, loss, val_accuracy, val_loss) |
| artifacts | list[string] | Chemins S3 des artefacts (modèle, confusion matrix, metrics.json) |
| model_version | string \| null | Version dans le registre MLflow |
| model_stage | string \| null | None → Staging → Production |

**Storage**: MLflow tracking server (SQLite/PostgreSQL) + MinIO S3 (artefacts)

**State transitions (model stage)**:

```
None ──→ Staging ──→ Production
          (auto if accuracy >= 90%)  (manual via POST /mlflow/promote)
```

### HealthResponse

Réponse du endpoint de santé.

| Field | Type | Description |
|-------|------|-------------|
| status | string | "healthy" |
| model_loaded | bool | Modèle chargé en mémoire |
| device | string | "cpu" ou "cuda" |
| model_name | string | Nom du modèle (ex: "resnet18") |
| model_source | string \| null | "mlflow" ou "checkpoint" |
| model_version | string \| null | Version MLflow si applicable |

### ValidationReport

Rapport de validation du dataset.

| Field | Type | Description |
|-------|------|-------------|
| total_images | int | Nombre total d'images |
| classes | dict[string, int] | Distribution par classe |
| corrupted_images | list[string] | Chemins des images corrompues |
| warnings | list[string] | Avertissements (déséquilibre, etc.) |
| is_valid | bool | Dataset valide pour entraînement |

## Relationships

```
Prediction ←── InferenceService ←── Model (ResNet18/CNN)
                                      ↑
                               ModelLoaderService
                              /                 \
                      MLflowService          Checkpoint (local)
                           ↓
                    Experiment (MLflow)
                           ↓
                      MinIO S3 (artefacts)

MLTask ←── TaskStore (JSON file)
  ├── train → TrainingService → Experiment
  ├── evaluate → EvaluationService
  ├── batch_inference → BatchInferencePipeline → Predictions[]
  └── validate_data → DataValidationService → ValidationReport
```

## Data Volume Assumptions

- ~12 000 images dans le dataset (8 classes × ~1 500 images)
- ~10-50 tâches ML en parallèle max dans le task store
- ~10-100 expériences MLflow
- Modèle ResNet18 : ~44 Mo de poids
- Prédiction unique : < 1 Mo d'image en entrée, < 1 Ko de JSON en sortie
