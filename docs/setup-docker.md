# Installation et configuration Docker

## Prerequis

- Docker et Docker Compose installes
- Le dataset dans `data/raw/bloodcells_dataset/` (voir section Dataset)
- Optionnel : `kaggle.json` pour telecharger le dataset automatiquement

## Demarrage rapide

### Stack complete (recommandee)

Inclut : FastAPI, MLflow, Airflow, MinIO, PostgreSQL, Streamlit.

```bash
docker compose -f docker/docker-compose.yml up -d
```

### Stack legere (sans Airflow)

Inclut : FastAPI, MLflow, MinIO, Streamlit. Suffisant pour du dev local.

```bash
docker compose -f docker/docker-compose.light.yml up -d
```

### Avec monitoring (Prometheus + Grafana)

```bash
docker compose -f docker/docker-compose.yml -f docker/docker-compose.monitoring.yml up -d
```

## Services et ports

| Service           | Container                    | Port  | URL                          |
|-------------------|------------------------------|-------|------------------------------|
| FastAPI           | bloodcell-api                | 8001  | http://localhost:8001/docs   |
| Streamlit         | bloodcell-streamlit          | 8502  | http://localhost:8502        |
| MLflow            | bloodcell-mlflow             | 5002  | http://localhost:5002        |
| MinIO S3 API      | bloodcell-minio              | 9002  | -                            |
| MinIO Console     | bloodcell-minio              | 9003  | http://localhost:9003        |
| Airflow           | bloodcell-airflow-webserver  | 8081  | http://localhost:8081        |
| PostgreSQL        | bloodcell-postgres           | 5432  | -                            |
| Prometheus        | bloodcell-prometheus         | 9090  | http://localhost:9090        |
| Grafana           | bloodcell-grafana            | 3000  | http://localhost:3000        |

### Identifiants par defaut

| Service  | Utilisateur | Mot de passe |
|----------|-------------|--------------|
| Airflow  | admin       | admin        |
| MinIO    | minio       | minio123     |
| Grafana  | admin       | admin        |

## Configuration du dataset

### Option 1 : Placement manuel

Telecharger le dataset depuis Kaggle et le placer dans :

```
data/raw/bloodcells_dataset/
  basophil/
  eosinophil/
  erythroblast/
  granulocyte/
  lymphocyte/
  monocyte/
  neutrophil/
  platelet/
```

### Option 2 : Telechargement automatique avec Kaggle API

1. Obtenir vos credentials Kaggle :
   - https://www.kaggle.com/ → Profil → Account → API → Create New API Token
   - Telecharger `kaggle.json`

2. Placer `kaggle.json` a la racine du projet

3. Le script `scripts/load_dataset.sh` sera execute au build Docker

> **Securite** : ne jamais commiter `kaggle.json` dans Git. Il est dans `.gitignore`.

## Architecture des containers

```
                ┌─────────────────────────────────────────┐
                │           bloodcell-network              │
                │                                         │
                │  ┌─────────┐  ┌─────────┐  ┌────────┐ │
                │  │   api    │  │ mlflow  │  │ minio  │ │
                │  │ PyTorch  │  │ tracking│  │   S3   │ │
                │  │ FastAPI  │  │ server  │  │ storage│ │
                │  └────┬────┘  └────┬────┘  └────┬───┘ │
                │       │            │             │      │
                │  ┌────┴────┐  ┌────┴────┐            │
                │  │ airflow │  │ airflow │              │
                │  │webserver│  │scheduler│              │
                │  └────┬────┘  └─────────┘              │
                │       │                                 │
                │  ┌────┴────┐  ┌─────────┐              │
                │  │postgres │  │streamlit│              │
                │  │metadata │  │   UI    │              │
                │  └─────────┘  └─────────┘              │
                └─────────────────────────────────────────┘
```

**Point important** : seul `api` contient PyTorch (~2 GB image).
Airflow est construit sur `apache/airflow:slim-2.9.3` (~500 MB) sans dependances ML.

## Buckets MinIO

Crees automatiquement au demarrage par `minio-init` :

| Bucket             | Usage                              |
|--------------------|------------------------------------|
| `mlflow-artifacts` | Artefacts MLflow (modeles, images) |
| `datasets`         | Datasets                           |
| `checkpoints`      | Checkpoints modeles                |

## Variables d'environnement cles

Ces variables sont configurees dans `docker-compose.yml` :

| Variable                  | Valeur Docker            | Description                    |
|---------------------------|--------------------------|--------------------------------|
| `MLFLOW_TRACKING_URI`     | `http://mlflow:5000`     | URL interne du serveur MLflow  |
| `FASTAPI_URL`             | `http://api:8000`        | URL interne de l'API           |
| `AWS_ACCESS_KEY_ID`       | `minio`                  | Credentials MinIO (S3)         |
| `AWS_SECRET_ACCESS_KEY`   | `minio123`               | Credentials MinIO (S3)         |
| `MLFLOW_S3_ENDPOINT_URL`  | `http://minio:9000`      | Endpoint S3 pour MLflow        |

## Commandes utiles

```bash
# Voir les logs d'un service
docker logs bloodcell-api --tail 50 -f

# Entrer dans un container
docker exec -it bloodcell-api bash

# Redemarrer un service
docker compose -f docker/docker-compose.yml restart api

# Rebuild un service apres modification
docker compose -f docker/docker-compose.yml up -d --build api

# Tout arreter
docker compose -f docker/docker-compose.yml down

# Tout arreter + supprimer les volumes (reset complet)
docker compose -f docker/docker-compose.yml down -v

# Verifier la sante
docker compose -f docker/docker-compose.yml ps
```

## Volumes persistes

| Volume Docker    | Contenu                         |
|------------------|---------------------------------|
| `postgres-db`    | Base de donnees Airflow         |
| `airflow-logs`   | Logs d'execution Airflow        |
| `minio-data`     | Donnees MinIO (artefacts, etc.) |

Les dossiers locaux suivants sont montes en bind mount :
- `src/`, `dags/`, `models/`, `data/`, `conf.yaml`, `mlruns/`
- Les modifications locales sont refletees sans rebuild.
