# Docker Compose Profiles Contract

**Branch**: `003-wrap-fastapi` | **Date**: 2026-03-08

## Profiles Overview

| Profile | Compose File | Services | Use Case |
|---------|-------------|----------|----------|
| full | `docker/docker-compose.yml` | Tous (10 services) | Démo complète, production |
| light | `docker/docker-compose.light.yml` | API, Streamlit, MinIO, MLflow | Développement, démos rapides |
| airflow-only | `docker/docker-compose.airflow.yml` | Airflow (webserver, scheduler), PostgreSQL | Orchestration uniquement |
| monitoring | `docker/docker-compose.monitoring.yml` | Prometheus, Grafana | Observabilité uniquement |

## Service Matrix

| Service | Port (interne) | Port (externe) | full | light | airflow | monitoring |
|---------|---------------|----------------|------|-------|---------|------------|
| api | 8000 | 8001 | X | X | | |
| streamlit | 8501 | 8502 | X | X | | |
| minio | 9000/9001 | 9000/9001 | X | X | | |
| mlflow | 5000 | 5001 | X | X | | |
| mlflow-serve | 5002 | 5002 | X | | | |
| postgres | 5432 | — | X | | X | |
| airflow-webserver | 8080 | 8081 | X | | X | |
| airflow-scheduler | — | — | X | | X | |
| prometheus | 9090 | 9090 | X | | | X |
| grafana | 3000 | 3000 | X | | | X |

## Network

Tous les profils utilisent un réseau Docker custom : `bloodcell-network` (bridge).

## Environment Variables

### Partagées (tous les profils)

| Variable | Default | Description |
|----------|---------|-------------|
| `MLFLOW_TRACKING_URI` | `http://mlflow:5000` | URI du serveur MLflow |
| `MLFLOW_S3_ENDPOINT_URL` | `http://minio:9000` | Endpoint MinIO pour MLflow |
| `AWS_ACCESS_KEY_ID` | `minio` | Credentials MinIO |
| `AWS_SECRET_ACCESS_KEY` | `minio123` | Credentials MinIO |

### API spécifiques

| Variable | Default | Description |
|----------|---------|-------------|
| `API_URL` | `http://api:8000` | URL interne de l'API |
| `API_EXTERNAL_URL` | `http://localhost:8001` | URL externe de l'API |

### Airflow spécifiques

| Variable | Default | Description |
|----------|---------|-------------|
| `AIRFLOW_API_URL` | `http://airflow-webserver:8080` | URL interne Airflow |
| `AIRFLOW_USERNAME` | `airflow` | Admin username |
| `AIRFLOW_PASSWORD` | `airflow` | Admin password |
| `AIRFLOW_DAG_SUFFIX` | `_api` | Suffixe des DAG IDs |

## Volumes persistants

| Volume | Service | Contenu |
|--------|---------|---------|
| `postgres-db` | postgres | Métadonnées Airflow |
| `minio-data` | minio | Artefacts ML (modèles, datasets) |
| `airflow-logs` | airflow-* | Logs des DAG runs |
| `grafana-data` | grafana | Dashboards et settings |

## Health Checks

| Service | Endpoint / Command | Interval | Retries |
|---------|-------------------|----------|---------|
| api | `curl -f http://localhost:8000/health` | 30s | 3 |
| mlflow | `curl -f http://localhost:5000/health` | 30s | 3 |
| minio | `mc ready local` | 30s | 3 |
| postgres | `pg_isready` | 10s | 5 |
| airflow-webserver | `curl -f http://localhost:8080/health` | 30s | 3 |
| prometheus | `curl -f http://localhost:9090/-/healthy` | 30s | 3 |
| grafana | `curl -f http://localhost:3000/api/health` | 30s | 3 |

## Startup Commands

```bash
# Full stack
docker compose -f docker/docker-compose.yml up -d

# Light (dev/demo)
docker compose -f docker/docker-compose.light.yml up -d

# Airflow only (requires API running separately or via light/full)
docker compose -f docker/docker-compose.airflow.yml up -d

# Monitoring only (requires API running separately)
docker compose -f docker/docker-compose.monitoring.yml up -d

# Combinaison (light + monitoring)
docker compose -f docker/docker-compose.light.yml -f docker/docker-compose.monitoring.yml up -d
```
