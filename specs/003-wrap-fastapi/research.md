# Research: Wrap FastAPI - MLOps Platform

**Branch**: `003-wrap-fastapi` | **Date**: 2026-03-08

## R1: GPU Task Concurrency Strategy

**Decision**: Mutex — une seule tâche GPU (entraînement, évaluation) à la fois. Erreur 409 si tentative concurrente. Tâches CPU-only (batch inference, validation données) restent parallèles.

**Rationale**: Le projet tourne sur une seule machine avec un GPU (ou CPU-only en Docker). Deux entraînements concurrents satureraient la VRAM et provoqueraient des OOM. Un mutex simple est plus fiable qu'un système de queue.

**Alternatives considered**:
- Queue avec priorité : Plus complexe, pas nécessaire pour un usage mono-utilisateur
- Limitation par sémaphore : Over-engineering pour le cas d'usage actuel
- Aucune limitation : Risque d'OOM et de crash silencieux

**Implementation**: Vérifier `TaskStore` pour une tâche GPU en status PENDING ou RUNNING avant d'accepter une nouvelle. Les tâches CPU-only ne sont pas soumises à cette contrainte.

## R2: Artifact Storage Strategy

**Decision**: MinIO S3 unique — tous les artefacts ML passent par MinIO. Prérequis obligatoire, pas de fallback local.

**Rationale**: Une stratégie de stockage unique simplifie l'architecture et garantit la cohérence. MLflow artifact store pointe vers MinIO, les modèles enregistrés y sont stockés, et les DAGs Airflow peuvent y accéder sans monter de volumes partagés.

**Alternatives considered**:
- Stockage local avec fallback : Deux chemins de code à maintenir, risque de divergence
- S3 cloud (AWS) : Plus cher, latence réseau, le projet est on-premise
- NFS partagé : Plus complexe à configurer, pas S3-compatible

**Implementation**:
- MLflow `--default-artifact-root s3://mlflow-artifacts/`
- Variables d'environnement : `MLFLOW_S3_ENDPOINT_URL`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`
- Exception : le fallback checkpoint local est conservé **uniquement** pour le chargement initial du modèle en mémoire (pas pour le stockage d'artefacts)

## R3: Model Promotion Strategy

**Decision**: Semi-automatique — le DAG promeut en Staging si accuracy >= 90%. Staging → Production nécessite un appel API manuel (`POST /mlflow/promote/{version}`).

**Rationale**: L'automatisation complète vers Production est risquée sans validation humaine. Le semi-automatique offre un bon compromis : les modèles performants sont pré-qualifiés automatiquement, mais un humain décide de la mise en production.

**Alternatives considered**:
- Full automatique : Risque de mettre en production un modèle avec des biais non détectés
- Full manuel : Frein à l'itération, oubli de promotion
- Multi-stage (Staging → Canary → Production) : Over-engineering pour le MVP

**Implementation**:
- DAG training : étape `decide_promotion` vérifie accuracy >= 90%, appelle `POST /mlflow/promote/{version}` avec `stage=Staging`
- Endpoint dédié : `POST /mlflow/promote/{version}` accepte un paramètre `stage` (default: Production)

## R4: Docker Compose Profiles

**Decision**: 4 profils — full, light (API + Streamlit + MinIO + MLflow), airflow-only, monitoring.

**Rationale**: Différents besoins selon le contexte : développement léger (light), orchestration (airflow), observabilité (monitoring), démo complète (full).

**Alternatives considered**:
- Profil unique : Trop lourd pour le développement quotidien
- 2 profils (dev/prod) : Pas assez granulaire pour les différents cas d'usage
- Kubernetes : Over-engineering pour un projet mono-machine

**Implementation**: 4 fichiers docker-compose séparés plutôt que des profiles Docker Compose natifs, pour une meilleure lisibilité et compatibilité.

| Profil | Services | Commande |
|--------|----------|----------|
| full | Tous (API, Streamlit, MinIO, MLflow, Airflow, PostgreSQL, Prometheus, Grafana) | `docker compose -f docker/docker-compose.yml up` |
| light | API, Streamlit, MinIO, MLflow | `docker compose -f docker/docker-compose.light.yml up` |
| airflow-only | Airflow (webserver, scheduler), PostgreSQL | `docker compose -f docker/docker-compose.airflow.yml up` |
| monitoring | Prometheus, Grafana | `docker compose -f docker/docker-compose.monitoring.yml up` |

## R5: Deferred ML Loading

**Decision**: Chargement différé (lazy loading) de PyTorch et du modèle au premier appel de prédiction, pas au démarrage de l'API.

**Rationale**: PyTorch prend ~2-3 secondes à importer et le modèle ~5-10 secondes à charger. Le health check et les endpoints non-ML doivent répondre immédiatement. Les routers ML utilisent des imports différés (`from src.pipe.train_model import main`).

**Alternatives considered**:
- Chargement au startup : Bloque le health check, le startup probe Docker timeout
- Worker dédié ML : Architecture plus complexe (process pool, IPC)
- Pre-fork avec modèle en mémoire partagée : Complexité mémoire, pas nécessaire avec 2 workers

**Implementation**:
- `main.py` : modèle chargé via `ModelLoaderService` au premier appel prédiction (cached)
- `routers/ml_tasks.py` : imports différés des pipelines dans les fonctions de background task
- Health check retourne `model_loaded: false` tant que le modèle n'est pas chargé

## R6: Airflow DAG Architecture

**Decision**: DAGs légers — Airflow appelle l'API FastAPI via HTTP, pas d'exécution ML directe. Image Airflow sans PyTorch.

**Rationale**:
- Image Airflow reste < 500 Mo (vs ~4 Go avec PyTorch)
- Séparation des responsabilités : Airflow orchestre, FastAPI exécute
- Pas de duplication de code ML entre Airflow et l'API
- Les DAGs sont testables sans environnement ML

**Alternatives considered**:
- PyTorch dans Airflow (KubernetesPodOperator) : Image trop grosse, duplication de code
- Celery workers avec PyTorch : Architecture distribuée over-engineered
- Scripts bash dans Airflow : Pas d'API, pas de monitoring, pas de task tracking

**Implementation**:
- `dags/utils/api_client.py` : client HTTP partagé (health check, trigger task, poll status)
- Chaque DAG : health check → trigger API → poll → extract results → decide next step
- Variables d'environnement : `API_URL`, `AIRFLOW_DAG_SUFFIX`

## R7: Task Persistence

**Decision**: JSON file (`./data/tasks.json`) avec mutex thread-safe et écriture atomique (temp + rename).

**Rationale**: Simple, sans dépendance externe (pas de Redis, pas de DB). Suffisant pour un usage mono-utilisateur avec ~10-50 tâches max. Le fichier survit aux redémarrages de l'API.

**Alternatives considered**:
- SQLite : Plus robuste mais over-engineering pour ~50 entrées
- Redis : Dépendance externe supplémentaire
- En mémoire seul : Perte des tâches au redémarrage

**Implementation**: `TaskStore` avec `threading.Lock`, rechargement depuis disque avant chaque lecture (support multi-worker uvicorn).
