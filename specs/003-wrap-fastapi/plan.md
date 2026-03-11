# Implementation Plan: Wrap FastAPI - MLOps Platform

**Branch**: `003-wrap-fastapi` | **Date**: 2026-03-08 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/003-wrap-fastapi/spec.md`

## Summary

Refactoring du projet BloodCellClassification pour wrapper l'application existante avec une couche API REST FastAPI complète, incluant : restructuration de l'API avec routers modulaires, métriques Prometheus, service de validation de données, batch inference pipeline, intégration Airflow via DAGs légers, monitoring Prometheus/Grafana, et Docker Compose multi-profils (full, light, airflow-only, monitoring).

Le projet est **déjà substantiellement implémenté** sur la branche `wrap_fastapi`. Ce plan documente l'architecture cible et identifie les écarts à combler pour atteindre la conformité complète avec la spec.

## Technical Context

**Language/Version**: Python 3.11+ (managed with `uv`)
**Primary Dependencies**: FastAPI, Uvicorn, PyTorch (torchvision ResNet18), MLflow, Prometheus (prometheus-fastapi-instrumentator), Pydantic v2, httpx, boto3
**Storage**: MinIO S3 (artefacts ML, modèles, datasets) — prérequis obligatoire. PostgreSQL (métadonnées Airflow). JSON fichier (task store)
**Testing**: pytest (unit, integration, performance, DAG structure). FastAPI TestClient. ~86k lignes de tests existantes
**Target Platform**: Linux server (Docker). GPU optionnel en local, CPU-only en Docker
**Project Type**: Web service (ML serving API) + orchestration platform
**Performance Goals**: Prédiction < 2s, premier appel < 30s, plateforme complète < 5 min au démarrage
**Constraints**: 1 tâche GPU à la fois (409 si concurrent). Image Airflow < 500 Mo (pas de PyTorch). MinIO obligatoire (pas de fallback local)
**Scale/Scope**: Mono-utilisateur, pas d'authentification pour le MVP. 8 classes de cellules sanguines. ~15 endpoints API

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Evidence |
|-----------|--------|----------|
| I. Simplicity First | PASS | Architecture existante suit le pattern minimal : services SRP, factory pattern, pas d'abstraction superflue |
| II. Verification Before Done | PASS | Suite de tests complète (~86k lignes), CI/CD avec 7 jobs (lint, unit, DAG, integration, performance, Docker, security) |
| III. MLOps Reproducibility | PASS | MLflow tracking obligatoire, conf.yaml centralisé, promotion avec gate accuracy >= 90% |
| IV. Observability | PASS | Prometheus metrics (latency, prediction counts, confidence histograms), health endpoint, structured logging requis |
| V. Service-Oriented Architecture | PASS | Services dans `src/services/` (SRP), routers thin dans `src/api/routers/`, DAGs thin dans `dags/`, pipelines dans `src/pipe/` |
| VI. Test Discipline | PASS | Unit tests services, integration tests API (TestClient), performance tests latence, DAG structure tests, CI obligatoire |
| VII. Minimal Impact | PASS | Wrapping autour du code existant, pas de réécriture des services métier |

**Gate result**: PASS — Aucune violation. L'architecture existante est conforme à la constitution.

## Project Structure

### Documentation (this feature)

```text
specs/003-wrap-fastapi/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output
│   ├── rest-api.md      # REST API contracts
│   └── docker-profiles.md # Docker Compose profiles
└── tasks.md             # Phase 2 output (/speckit.tasks)
```

### Source Code (repository root)

```text
src/
├── api/                          # FastAPI application
│   ├── main.py                   # App entry point, health/predict/mlflow endpoints
│   ├── metrics.py                # Prometheus metrics definitions
│   ├── schemas.py                # Pydantic request/response models
│   ├── task_store.py             # Persistent JSON task storage
│   └── routers/
│       ├── ml_tasks.py           # /ml/* endpoints (train, evaluate, batch, validate)
│       └── pipelines.py          # /pipelines/* endpoints (Airflow DAG triggers)
│
├── core/                         # Shared constants & config
│   ├── config.py                 # Service URL management
│   └── constants.py              # CLASS_NAMES, NUM_CLASSES
│
├── models/                       # PyTorch model definitions
│   ├── base_classifier.py        # Abstract base class
│   ├── resnet_classifier.py      # ResNet18 pretrained
│   ├── cnn_classifier.py         # Custom CNN
│   └── model_factory.py          # Factory pattern
│
├── services/                     # Business logic (SRP)
│   ├── inference_service.py      # Prediction (single + batch)
│   ├── mlflow_service.py         # MLflow tracking & registry
│   ├── model_loader_service.py   # MLflow → checkpoint fallback
│   ├── training_service.py       # Training loop
│   ├── evaluation_service.py     # Model evaluation
│   ├── data_validation_service.py # Dataset quality checks
│   ├── data_transform_service.py  # Augmentation & preprocessing
│   ├── dataset_service.py        # DataLoader creation
│   └── yaml_loader.py            # conf.yaml management
│
├── pipe/                         # Pipeline orchestrators
│   ├── train_model.py            # Training pipeline
│   ├── evaluate_model.py         # Evaluation pipeline
│   └── batch_inference_pipeline.py # Batch inference
│
├── pages/                        # Streamlit UI (6 pages)
├── utils/                        # Visualization & analysis utilities
└── app.py                        # Streamlit entry point

dags/                             # Airflow DAGs (thin orchestrators)
├── utils/api_client.py           # FastAPI client helpers
├── bloodcells_train_dag_api.py
├── bloodcells_evaluate_dag_api.py
├── bloodcells_batch_inference_dag_api.py
└── bloodcells_data_validation_dag_api.py

docker/                           # Containerization
├── Dockerfile.api                # FastAPI + PyTorch (CPU-only)
├── Dockerfile.airflow            # Airflow (no PyTorch)
├── Dockerfile.streamlit          # Streamlit UI
├── docker-compose.yml            # Full stack
├── docker-compose.light.yml      # API + Streamlit + MinIO + MLflow
├── docker-compose.airflow.yml    # Airflow-only
└── docker-compose.monitoring.yml # Prometheus + Grafana

monitoring/                       # Monitoring configs
├── prometheus.yml
└── grafana/provisioning/

tests/                            # Test suite (~86k lines)
├── test_api.py                   # API endpoint tests
├── test_api_integration.py       # E2E integration
├── test_batch_inference_pipeline.py
├── test_dags.py                  # DAG structure validation
├── test_data_validation_service.py
├── test_inference_service.py
├── test_mlflow_service.py
├── test_performance.py           # Latency benchmarks
├── test_pipelines_router.py
├── test_task_store.py
├── test_integration.py
└── ...
```

**Structure Decision**: Single project (Option 1 variant). All source code under `src/` with clear module separation (api, core, models, services, pipe, pages, utils). DAGs externalized in `dags/` for Airflow volume mounting. Docker configs in `docker/`. Tests flat in `tests/`.

## Complexity Tracking

> No violations detected — no justifications needed.
