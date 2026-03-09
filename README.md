# BloodCellClassification

Application de classification de cellules sanguines par deep learning (ResNet18), avec API REST, orchestration Airflow et monitoring.

## Architecture

```
src/
  api/          # FastAPI REST API (prédiction, MLflow, pipelines)
  core/         # Constantes et configuration centralisées
  models/       # CNN classifier (ResNet18)
  services/     # Inference, évaluation, MLflow, validation de données
  pipe/         # Pipelines (entraînement, batch inference)
  pages/        # Interface Streamlit (6 pages)
  utils/        # Utilitaires (charts, GradCAM, RGB, stats)
dags/           # DAGs Airflow (train, evaluate, batch inference, validation)
docker/         # Dockerfiles et Docker Compose
monitoring/     # Prometheus + Grafana
```

## Démarrage rapide

### Prérequis

- Python 3.11+, [uv](https://docs.astral.sh/uv/)
- Docker et Docker Compose (pour les stacks conteneurisées)
- Credentials Kaggle (`kaggle.json`) pour le dataset

### Installation locale

```bash
uv pip install -e .
```

### Chargement du dataset

```bash
./scripts/load_dataset.sh
```

Le dataset contient 17 092 images de cellules sanguines normales (360x363 px) réparties en 8 classes : neutrophiles, éosinophiles, basophiles, lymphocytes, monocytes, granulocytes immatures, érythroblastes et plaquettes.

### Configuration Kaggle

1. Aller sur https://www.kaggle.com/ → Profil → Account → API → "Create New API Token"
2. Placer `kaggle.json` dans `~/.kaggle/` ou à la racine du projet

## Utilisation

### Entraînement

```bash
uv run train-model
```

### API FastAPI

```bash
uv run start-api
```

L'API est accessible sur **http://localhost:8000/docs** avec les endpoints :
- `GET /health` — Health check
- `POST /predict` — Prédiction sur image (base64)
- `POST /predict/upload` — Prédiction sur fichier uploadé
- `GET /model/info` — Informations du modèle
- `GET /metrics` — Métriques de performance
- `POST /pipelines/*` — Déclenchement des DAGs Airflow

### Interface Streamlit

```bash
uv run streamlit run src/app.py
```

Accessible sur **http://localhost:8501**

### Batch inference

```bash
uv run batch-inference
```

## Docker

### Stack light (développement)

```bash
docker compose -f docker/docker-compose.light.yml up -d
```

| Service    | URL                          |
|------------|------------------------------|
| API        | http://localhost:8000/docs    |
| Streamlit  | http://localhost:8501         |
| MLflow     | http://localhost:5000         |
| MinIO      | http://localhost:9001         |

### Stack complète (MLOps)

```bash
docker compose -f docker/docker-compose.yml up -d
```

| Service    | URL                          | Credentials     |
|------------|------------------------------|-----------------|
| Streamlit  | http://localhost:8502         |                 |
| API        | http://localhost:8001/docs    |                 |
| Airflow    | http://localhost:8081         | admin / admin   |
| MLflow     | http://localhost:5002         |                 |
| MinIO      | http://localhost:9003         | minio / minio123|

### Monitoring (overlay sur stack complète)

```bash
docker compose -f docker/docker-compose.yml -f docker/docker-compose.monitoring.yml up -d
```

| Service    | URL                          | Credentials     |
|------------|------------------------------|-----------------|
| Prometheus | http://localhost:9090         |                 |
| Grafana    | http://localhost:3000         | admin / admin   |

### Airflow standalone

```bash
docker compose -f docker/docker-compose.airflow.yml up -d
```

## Tests

```bash
# Tests unitaires
uv run pytest tests/ --ignore=tests/test_dags.py --ignore=tests/test_performance.py --ignore=tests/test_integration.py

# Tests des DAGs Airflow
uv run pytest tests/test_dags.py

# Tests d'intégration
uv run pytest tests/test_integration.py

# Tests de performance
uv run pytest tests/test_performance.py
```

## CI/CD

Les workflows GitHub Actions exécutent automatiquement :
- **CI** : lint (ruff), tests unitaires, tests DAGs, tests d'intégration, build Docker, audit de sécurité (pip-audit)
- **CD** : build et push des images Docker, déploiement

## Technologies

- **ML** : PyTorch (ResNet18), torchvision, Captum (GradCAM)
- **API** : FastAPI, Uvicorn, Pydantic v2
- **Tracking** : MLflow, MinIO (S3)
- **Orchestration** : Apache Airflow
- **Monitoring** : Prometheus, Grafana
- **UI** : Streamlit
- **CI/CD** : GitHub Actions, Docker

## Dataset

> A. Acevedo et al., "A dataset of microscopic peripheral blood cell images for development of automatic recognition systems", Data in Brief, 2020.

Le dataset est disponible sur [Kaggle](https://www.kaggle.com/datasets/unclesamulus/blood-cells-image-dataset).
