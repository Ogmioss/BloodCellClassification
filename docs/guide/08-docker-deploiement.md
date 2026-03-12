# 08 — Docker & déploiement

Le projet utilise Docker pour conteneuriser chaque service et Docker Compose pour les orchestrer.

## Les trois images Docker

### 1. `Dockerfile.api` — API + PyTorch (~3 Go)

C'est la seule image lourde. Elle contient PyTorch et toutes les dépendances ML.

**Build multi-stage** :
```
Stage 1 (builder) :
  - Image Python slim
  - Installe les dépendances avec uv
  - Compile les packages natifs

Stage 2 (runtime) :
  - Image Python slim propre
  - Copie uniquement les packages installés
  - Pas de compilateur, pas de cache
  - Résultat : image plus petite et plus sécurisée
```

**Points notables** :
- `PYTORCH_INDEX_URL` configurable : CPU, CUDA 11.8, CUDA 12.1, ROCm
- Healthcheck intégré : `GET /health`
- 2 workers Uvicorn
- Port 8000

### 2. `Dockerfile.airflow` — Orchestration (~500 Mo)

Image légère basée sur `apache/airflow:slim-2.9.3`.

**Ne contient PAS** de PyTorch. Uniquement :
- httpx (appels HTTP vers l'API)
- pydantic (validation)
- psycopg2-binary (connexion PostgreSQL)
- apache-airflow-providers-http

### 3. `Dockerfile.streamlit` — Dashboard (léger)

Image légère pour l'interface Streamlit. Appelle l'API pour les prédictions.

## Profils Docker Compose

Le projet propose trois niveaux de déploiement, du plus simple au plus complet :

### Stack légère — développement rapide

```bash
docker compose -f docker/docker-compose.light.yml up -d
```

| Service | Port | Description |
|---------|------|-------------|
| API | 8000 | FastAPI + PyTorch |
| Streamlit | 8501 | Dashboard |
| MLflow | 5000 | Tracking UI |
| MinIO | 9000 / 9001 | Stockage S3 + console |

**Quand l'utiliser** : développement local, tests rapides, démo.

### Stack complète — MLOps

```bash
docker compose -f docker/docker-compose.yml up -d
```

Ajoute à la stack légère :

| Service | Port | Description |
|---------|------|-------------|
| Airflow webserver | 8081 | UI d'orchestration |
| Airflow scheduler | — | Exécute les DAGs |
| PostgreSQL | 5432 | Métadonnées Airflow |
| Pushgateway | 9091 | Métriques d'entraînement |

**Quand l'utiliser** : quand vous avez besoin d'orchestration automatique.

### Stack monitoring — observabilité

```bash
docker compose -f docker/docker-compose.yml -f docker/docker-compose.monitoring.yml up -d
```

Ajoute en overlay :

| Service | Port | Description |
|---------|------|-------------|
| Prometheus | 9090 | Collecte de métriques |
| Grafana | 3000 | Dashboards (admin/admin) |

**Quand l'utiliser** : suivi en production, debugging de performance.

## Variables d'environnement clés

| Variable | Valeur (Docker) | Description |
|----------|-----------------|-------------|
| `MLFLOW_TRACKING_URI` | `http://mlflow:5000` | URI du serveur MLflow |
| `MLFLOW_S3_ENDPOINT_URL` | `http://minio:9000` | Endpoint MinIO pour les artefacts |
| `AWS_ACCESS_KEY_ID` | `minio` | Identifiant MinIO |
| `AWS_SECRET_ACCESS_KEY` | `minio123` | Secret MinIO |
| `FASTAPI_URL` | `http://api:8000` | URL de l'API (pour Airflow/Streamlit) |
| `DOCKER` | `true` | Indique l'exécution dans Docker |

> **Note** : les identifiants MinIO sont en dur pour le développement. En production, utilisez des secrets Docker ou un gestionnaire de secrets.

## Réseau Docker

Tous les services partagent un réseau Docker `bloodcell-network`. Les noms de services (api, mlflow, minio, etc.) sont résolus par le DNS interne Docker.

```
Streamlit ──(http://api:8000)──▶ FastAPI
Airflow   ──(http://api:8000)──▶ FastAPI
FastAPI   ──(http://mlflow:5000)──▶ MLflow
MLflow    ──(http://minio:9000)──▶ MinIO
```

## Volumes persistants

| Volume | Contenu | Service |
|--------|---------|---------|
| `./data` | Dataset d'images | API |
| `./models` | Checkpoints de modèles | API |
| `./mlruns` | Données MLflow locales | MLflow |
| `./dags` | DAGs Airflow | Airflow |
| `postgres-data` | Métadonnées Airflow | PostgreSQL |
| `minio-data` | Artefacts S3 | MinIO |

## Commandes utiles

```bash
# Construire toutes les images
docker compose -f docker/docker-compose.yml build

# Démarrer en arrière-plan
docker compose -f docker/docker-compose.yml up -d

# Voir les logs d'un service
docker compose -f docker/docker-compose.yml logs -f api

# Redémarrer un service
docker compose -f docker/docker-compose.yml restart api

# Tout arrêter et nettoyer
docker compose -f docker/docker-compose.yml down

# Tout arrêter ET supprimer les volumes (reset complet)
docker compose -f docker/docker-compose.yml down -v
```

## Build GPU

Pour utiliser un GPU NVIDIA, configurez l'index PyTorch au build :

```bash
# CUDA 12.1
docker compose -f docker/docker-compose.yml build --build-arg PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu121

# CPU uniquement (défaut)
docker compose -f docker/docker-compose.yml build --build-arg PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
```

Et ajoutez le runtime GPU dans le compose :

```yaml
services:
  api:
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
```
