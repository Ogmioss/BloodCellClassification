# Quickstart: Wrap FastAPI - MLOps Platform

**Branch**: `003-wrap-fastapi` | **Date**: 2026-03-08

## Prérequis

- Docker & Docker Compose v2
- Python 3.11+ avec `uv` (pour développement local)
- ~6 Go d'espace disque (images Docker)

## 1. Déploiement rapide (Docker - profil light)

```bash
# Cloner et checkout
git clone <repo-url>
cd BloodCellClassification
git checkout 003-wrap-fastapi

# Lancer le profil light (API + Streamlit + MinIO + MLflow)
docker compose -f docker/docker-compose.light.yml up -d

# Vérifier la santé
curl http://localhost:8001/health
```

**Services accessibles** :
- API : http://localhost:8001
- Streamlit : http://localhost:8502
- MinIO Console : http://localhost:9001 (minio/minio123)
- MLflow : http://localhost:5001

## 2. Première prédiction

```bash
# Upload d'une image
curl -X POST http://localhost:8001/predict/upload \
  -F "file=@path/to/cell_image.jpg"

# Réponse attendue :
# {"predicted_class":"neutrophil","confidence":0.95,"probabilities":{...}}
```

## 3. Lancer un entraînement

```bash
# Déclencher un entraînement (background task)
curl -X POST http://localhost:8001/ml/train \
  -H "Content-Type: application/json" \
  -d '{"epochs": 5, "learning_rate": 0.001}'

# Réponse : {"task_id": "xxx", "status": "pending"}

# Suivre la progression
curl http://localhost:8001/ml/tasks/<task_id>
```

## 4. Déploiement complet (tous les services)

```bash
# Full stack (API, Streamlit, MinIO, MLflow, Airflow, PostgreSQL, Prometheus, Grafana)
docker compose -f docker/docker-compose.yml up -d

# Services additionnels :
# Airflow : http://localhost:8081 (airflow/airflow)
# Prometheus : http://localhost:9090
# Grafana : http://localhost:3000
```

## 5. Développement local

```bash
# Installer les dépendances
uv sync

# Lancer MinIO + MLflow (prérequis)
docker compose -f docker/docker-compose.light.yml up -d minio mlflow

# Lancer l'API en local
uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Lancer Streamlit en local
uv run streamlit run src/app.py

# Lancer les tests
uv run pytest tests/ -v --ignore=tests/test_integration.py --ignore=tests/test_performance.py
```

## 6. Vérification end-to-end

```bash
# 1. Health check
curl http://localhost:8001/health
# → {"status":"healthy","model_loaded":true,...}

# 2. Prédiction
curl -X POST http://localhost:8001/predict/upload -F "file=@image.jpg"
# → {"predicted_class":"neutrophil","confidence":0.95,...}

# 3. Métriques Prometheus
curl http://localhost:8001/metrics | grep bloodcell
# → bloodcell_predictions_total{predicted_class="neutrophil"} 1

# 4. MLflow models
curl http://localhost:8001/mlflow/models
# → {"model_name":"bloodcells-classifier","versions":[...]}

# 5. Validation données
curl -X POST http://localhost:8001/ml/validate-data
# → {"task_id":"xxx","status":"pending"}
```

## Troubleshooting

| Problème | Solution |
|----------|----------|
| `model_loaded: false` | Premier appel déclenche le chargement (~10s). Réessayer. |
| `503 MLflow unavailable` | Vérifier `docker compose ps mlflow`. MinIO doit être up aussi. |
| `409 GPU task conflict` | Un entraînement/évaluation est déjà en cours. Attendre ou vérifier `GET /ml/tasks`. |
| Port déjà utilisé | Modifier les ports dans le docker-compose correspondant. |
