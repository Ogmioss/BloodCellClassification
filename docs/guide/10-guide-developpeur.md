# 10 — Guide développeur

Ce document est destiné à un développeur qui rejoint le projet. Il couvre l'installation, les commandes essentielles et les conventions.

## Prérequis

| Outil | Version | Installation |
|-------|---------|-------------|
| Python | 3.11+ | — |
| uv | dernière | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| Docker | 24+ | docs.docker.com |
| Docker Compose | v2 | Inclus avec Docker Desktop |
| Git | 2.x | — |

## Installation locale

```bash
# 1. Cloner le repo
git clone <repo-url>
cd BloodCellClassification

# 2. Installer les dépendances Python
uv sync

# 3. Vérifier l'installation
uv run pytest tests/test_model_factory.py -v
```

## Lancer le projet

### Option A : tout en local (sans Docker)

```bash
# Entraîner un modèle
uv run train-model

# Lancer l'API
uv run start-api

# Lancer le dashboard Streamlit
uv run streamlit run src/app.py
```

### Option B : avec Docker (recommandé)

```bash
# Stack légère (API + Streamlit + MLflow + MinIO)
docker compose -f docker/docker-compose.light.yml up -d

# Stack complète (+ Airflow + PostgreSQL + Pushgateway)
docker compose -f docker/docker-compose.yml up -d

# Avec monitoring (+ Prometheus + Grafana)
docker compose -f docker/docker-compose.yml -f docker/docker-compose.monitoring.yml up -d
```

## URLs des services

| Service | Stack légère | Stack complète | Avec monitoring |
|---------|-------------|----------------|-----------------|
| API (docs) | localhost:8000/docs | localhost:8000/docs | idem |
| Streamlit | localhost:8501 | localhost:8501 | idem |
| MLflow | localhost:5000 | localhost:5002 | idem |
| MinIO console | localhost:9001 | localhost:9001 | idem |
| Airflow | — | localhost:8081 | idem |
| Prometheus | — | — | localhost:9090 |
| Grafana | — | — | localhost:3000 |

## Commandes de développement

### Tests

```bash
# Tests unitaires (rapide)
uv run pytest tests/ --ignore=tests/test_dags.py --ignore=tests/test_performance.py --ignore=tests/test_integration.py

# Un test spécifique
uv run pytest tests/test_inference_service.py -v

# Avec couverture
uv run pytest tests/ --cov=src --cov-report=html
```

### Lint

```bash
# Vérifier
uv run ruff check .

# Corriger automatiquement
uv run ruff check . --fix

# Formater
uv run ruff format .
```

### Entraînement

```bash
# Entraînement avec config par défaut (conf.yaml)
uv run train-model

# Via l'API (en Docker)
curl -X POST http://localhost:8000/ml/train \
  -H "Content-Type: application/json" \
  -d '{"epochs": 10, "learning_rate": 0.001}'
```

### Prédiction

```bash
# Via l'API
curl -X POST http://localhost:8000/predict/upload \
  -F "file=@path/to/image.jpg"
```

## Configuration (`conf.yaml`)

Le fichier `conf.yaml` à la racine est la source de vérité pour tous les paramètres :

```yaml
model:
  name: "resnet18"          # Architecture (resnet18, resnet34, cnn)
  pretrained: true           # Utiliser les poids ImageNet

training:
  epochs: 20                 # Nombre d'epochs
  learning_rate: 0.001       # Taux d'apprentissage
  batch_size: 32             # Taille des batches
  subset_size: 1400          # 0 = dataset complet

mlflow:
  tracking_uri: "./mlruns"   # Surchargé par $MLFLOW_TRACKING_URI en Docker
  experiment_name: "bloodcells-classification"
```

Modifier ce fichier affecte tous les entraînements lancés sans surcharge explicite.

## Ajouter un nouveau modèle

1. Créer une classe dans `src/models/` héritant de `BaseClassifier`
2. Implémenter `get_num_features()` et `set_classifier_head()`
3. Enregistrer le modèle dans `ModelFactory` (`src/models/model_factory.py`)
4. Tester avec `uv run pytest tests/test_model_factory.py`

## Ajouter un nouveau DAG Airflow

1. Créer un fichier dans `dags/`
2. Utiliser le client HTTP de `dags/utils/api_client.py`
3. Suivre le pattern : health check → trigger → poll → exploiter le résultat
4. Tester avec `uv run pytest tests/test_dags.py`

## Ajouter une métrique Prometheus

1. Définir le compteur/histogramme dans `src/api/metrics.py`
2. L'incrémenter dans le code de l'endpoint
3. Ajouter un panneau dans le dashboard Grafana correspondant

## Conventions du projet

| Convention | Détail |
|------------|--------|
| Langue du code | Anglais (noms de variables, fonctions, classes) |
| Langue de l'UI | Français (pages Streamlit) |
| Gestionnaire de paquets | uv (jamais pip directement) |
| Formatage | Ruff |
| Architecture | Services injectés, Factory pour les modèles |
| Configs | `conf.yaml` pour les défauts, env vars pour les surcharges |
| Branches | Feature branches (ex: `004-training-config-propagation`) |

## Dépannage rapide

| Problème | Solution |
|----------|----------|
| L'API dit `model_loaded: false` | Vérifier que `models/checkpoints/best_model.pth` existe ou que MLflow est accessible |
| Airflow ne voit pas les DAGs | Vérifier le montage du volume `./dags` dans le compose |
| MLflow timeout | Vérifier que le service mlflow est démarré et que `MLFLOW_TRACKING_URI` est correct |
| MinIO inaccessible | Vérifier les credentials et que le service est healthy |
| Import torch échoue en local | `uv sync` pour installer les dépendances |

Pour plus de détails sur le dépannage, voir `docs/troubleshooting.md`.
