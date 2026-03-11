# Tasks: Wrap FastAPI - MLOps Platform

**Input**: Design documents from `/specs/003-wrap-fastapi/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/

**Tests**: pytest (unit, integration, performance, DAG). Suite existante ~86k lignes. Les tâches de test valident la conformité avec la spec.

**Organization**: Tasks grouped by user story. Le projet est déjà substantiellement implémenté — les tâches ciblent les écarts identifiés entre le code existant et la spec.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Paths relative to repository root

---

## Phase 1: Setup (Verify Infrastructure)

**Purpose**: Vérifier que l'infrastructure existante est conforme

- [X] T001 Vérifier et mettre à jour .gitignore pour couvrir les patterns Python/Docker/ML (data/, mlruns/, __pycache__/, *.pyc, .env*, .venv/) dans .gitignore
- [X] T002 [P] Vérifier et mettre à jour .dockerignore pour exclure .git/, data/, mlruns/, tests/, notebooks/, specs/, .claude/ dans .dockerignore

---

## Phase 2: Foundational (Shared Schema Fixes)

**Purpose**: Corrections de schémas et modèles partagés entre plusieurs user stories

**⚠️ CRITICAL**: Ces corrections impactent plusieurs endpoints et doivent être faites avant les phases user story.

- [X] T003 Ajouter les champs `model_source: Optional[str]` et `model_version: Optional[str]` à `HealthResponse` dans src/api/schemas.py
- [X] T004 [P] Ajouter le champ `all_predictions: List[Dict[str, Any]]` à `PredictionResponse` (liste des classes triées par probabilité décroissante) dans src/api/schemas.py
- [X] T005 [P] Renommer la métrique `bloodcell_errors_total` en `bloodcell_prediction_errors_total` pour cohérence avec le contrat dans src/api/metrics.py

**Checkpoint**: Schémas mis à jour, prêts pour l'intégration dans les endpoints.

---

## Phase 3: User Story 1 - Prédire la classe d'une cellule sanguine via l'API (Priority: P1) 🎯 MVP

**Goal**: L'API retourne une prédiction complète (classe, confiance, probabilités, all_predictions) et le health check inclut la source et version du modèle.

**Independent Test**: POST /predict/upload avec une image → réponse contenant `predicted_class`, `confidence`, `probabilities`, `all_predictions`. GET /health → contient `model_source` et `model_version`.

### Implementation for User Story 1

- [X] T006 [US1] Mettre à jour le endpoint `/health` pour retourner `model_source` et `model_version` depuis l'état du modèle chargé (mlflow ou checkpoint) dans src/api/main.py
- [X] T007 [US1] Mettre à jour le endpoint `/predict` pour inclure le champ `all_predictions` (classes triées par probabilité décroissante) dans la réponse dans src/api/main.py
- [X] T008 [US1] Mettre à jour le endpoint `/predict/upload` pour inclure `all_predictions` dans la réponse dans src/api/main.py
- [X] T009 [US1] Mettre à jour les tests de prédiction pour vérifier la présence de `all_predictions` et des champs `model_source`/`model_version` dans le health check dans tests/test_api.py

**Checkpoint**: GET /health retourne model_source et model_version. POST /predict et /predict/upload retournent all_predictions. Tests passent.

---

## Phase 4: User Story 2 - Lancer et suivre un entraînement via l'API (Priority: P1)

**Goal**: L'API refuse les tâches GPU concurrentes avec 409 et permet de suivre l'avancement des tâches ML.

**Independent Test**: Lancer POST /ml/train, puis immédiatement relancer POST /ml/train → 409. Lancer POST /ml/train puis POST /ml/batch-inference → succès (batch = CPU-only).

### Implementation for User Story 2

- [X] T010 [US2] Implémenter le GPU task mutex : avant d'accepter un train ou evaluate, vérifier dans le TaskStore qu'aucune tâche GPU (train, evaluate) n'est en status PENDING ou RUNNING. Retourner 409 avec message explicite si conflit dans src/api/routers/ml_tasks.py
- [X] T011 [US2] Ajouter une méthode `has_running_gpu_task() -> bool` au TaskStore pour vérifier les tâches GPU actives dans src/api/task_store.py
- [X] T012 [US2] Écrire des tests pour le GPU mutex : tâche GPU concurrent → 409, tâche CPU pendant GPU → succès, tâche GPU après complétion → succès dans tests/test_task_store.py

**Checkpoint**: POST /ml/train pendant un train actif → 409. POST /ml/batch-inference pendant un train actif → 202. Tests passent.

---

## Phase 5: User Story 3 - Orchestrer les pipelines ML via Airflow (Priority: P2)

**Goal**: Le DAG d'entraînement promeut en Staging (pas Production) quand accuracy >= 90%. Production nécessite POST /mlflow/promote manuel.

**Independent Test**: Simuler un DAG run avec accuracy >= 90% → modèle promu en Staging. Vérifier que la promotion Production nécessite un appel API séparé.

### Implementation for User Story 3

- [X] T013 [US3] Modifier `decide_promotion()` dans le DAG d'entraînement pour promouvoir en Staging (pas Production) quand accuracy >= 90% dans dags/bloodcells_train_dag_api.py
- [X] T014 [US3] Modifier `promote_model()` dans le DAG pour appeler POST /mlflow/promote/{version} avec `stage=Staging` dans dags/bloodcells_train_dag_api.py
- [X] T015 [US3] Mettre à jour le endpoint POST /mlflow/promote/{version} pour accepter un paramètre query `stage` (default: "Production") dans src/api/main.py
- [X] T016 [US3] Mettre à jour les tests DAG pour vérifier la promotion en Staging (pas Production) dans tests/test_dags.py

**Checkpoint**: DAG promeut en Staging. POST /mlflow/promote/{version}?stage=Production pour la mise en production manuelle. Tests passent.

---

## Phase 6: User Story 4 - Monitorer la santé et les performances (Priority: P2)

**Goal**: Les métriques Prometheus sont correctement nommées et le dashboard Grafana reflète les métriques réelles.

**Independent Test**: POST /predict → GET /metrics (Prometheus format) → vérifier que `bloodcell_prediction_errors_total` apparaît (pas `bloodcell_errors_total`).

### Implementation for User Story 4

- [X] T017 [US4] Mettre à jour toutes les références à `bloodcell_errors_total` vers `bloodcell_prediction_errors_total` dans src/api/metrics.py et src/api/main.py
- [X] T018 [P] [US4] Mettre à jour le dashboard Grafana pour utiliser `bloodcell_prediction_errors_total` dans monitoring/grafana/dashboards/bloodcell-api.json
- [X] T019 [P] [US4] Vérifier que le scrape Prometheus est configuré correctement (interval 10s, target api:8000) dans monitoring/prometheus.yml

**Checkpoint**: Métriques Prometheus cohérentes avec le contrat. Dashboard Grafana fonctionnel.

---

## Phase 7: User Story 5 - Valider les données avant entraînement (Priority: P3)

**Goal**: L'endpoint de validation de données fonctionne conformément à la spec.

**Independent Test**: POST /ml/validate-data → task_id → poll → rapport avec distribution classes, images corrompues, warnings.

### Implementation for User Story 5

- [X] T020 [US5] Vérifier que POST /ml/validate-data retourne un rapport conforme au data-model (total_images, classes, corrupted_images, warnings, is_valid) dans src/api/routers/ml_tasks.py
- [X] T021 [US5] Vérifier que les tests de validation de données couvrent les cas edge (dataset vide, images corrompues, déséquilibre de classes) dans tests/test_data_validation_service.py

**Checkpoint**: Validation de données conforme à la spec. Tests existants passent.

---

## Phase 8: User Story 6 - Exécuter une inférence en batch (Priority: P3)

**Goal**: Le batch inference produit un JSON de résultats avec statistiques de distribution.

**Independent Test**: POST /ml/batch-inference avec un répertoire → task_id → poll → résultat avec total_images, distribution, output_file.

### Implementation for User Story 6

- [X] T022 [US6] Vérifier que le batch inference retourne un résultat conforme au data-model (total_images, successful, failed, output_file, distribution) dans src/pipe/batch_inference_pipeline.py
- [X] T023 [US6] Vérifier que les tests de batch inference couvrent les cas edge (répertoire vide, images corrompues) dans tests/test_batch_inference_pipeline.py

**Checkpoint**: Batch inference conforme à la spec. Tests existants passent.

---

## Phase 9: User Story 7 - Déployer via Docker Compose (Priority: P2)

**Goal**: Le profil "light" contient uniquement API + Streamlit + MinIO + MLflow (sans Airflow, sans PostgreSQL).

**Independent Test**: `docker compose -f docker/docker-compose.light.yml config` → vérifier que seuls api, streamlit, minio, minio-init, mlflow sont listés.

### Implementation for User Story 7

- [X] T024 [US7] Refactorer docker-compose.light.yml pour ne contenir que api, streamlit, minio, minio-init et mlflow — supprimer tous les services Airflow (airflow-common, airflow-init, airflow-webserver, airflow-scheduler) et les volumes airflow-* dans docker/docker-compose.light.yml
- [X] T025 [P] [US7] Vérifier que docker-compose.airflow.yml contient bien Airflow + PostgreSQL comme profil séparé dans docker/docker-compose.airflow.yml
- [X] T026 [P] [US7] Vérifier que docker-compose.monitoring.yml contient Prometheus + Grafana dans docker/docker-compose.monitoring.yml
- [X] T027 [US7] Vérifier que le full docker-compose.yml contient tous les services (API, Streamlit, MinIO, MLflow, Airflow, PostgreSQL, Prometheus, Grafana) dans docker/docker-compose.yml

**Checkpoint**: `docker compose -f docker/docker-compose.light.yml config --services` → api, streamlit, minio, minio-init, mlflow uniquement.

---

## Phase 10: Polish & Cross-Cutting Concerns

**Purpose**: Validation finale, cohérence, tests d'intégration

- [X] T028 Lancer la suite de tests unitaires : `uv run pytest tests/ -v --ignore=tests/test_integration.py --ignore=tests/test_performance.py` et corriger les échecs
- [X] T029 [P] Lancer les tests DAG : `uv run pytest tests/test_dags.py -v` et corriger les échecs
- [X] T030 [P] Vérifier la cohérence des contrats REST avec l'implémentation : comparer contracts/rest-api.md avec les endpoints réels (response shapes, status codes)
- [X] T031 Valider le quickstart.md : exécuter les commandes listées et vérifier que chaque étape fonctionne
- [X] T032 Vérifier que le CI (.github/workflows/ci.yml) couvre tous les types de tests (lint, unit, DAG, integration, performance, Docker, security)

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — vérification immédiate
- **Foundational (Phase 2)**: Depends on Setup — BLOCKS US1 (schémas partagés)
- **US1 Prédiction (Phase 3)**: Depends on Foundational (T003, T004) — peut démarrer en premier (P1, MVP)
- **US2 Entraînement (Phase 4)**: Depends on Setup — peut démarrer en parallèle avec US1 (P1)
- **US3 Airflow (Phase 5)**: Depends on Setup — peut démarrer en parallèle (P2)
- **US4 Monitoring (Phase 6)**: Depends on Foundational (T005) — peut démarrer après T005 (P2)
- **US5 Validation (Phase 7)**: Depends on Setup — vérification, peut démarrer en parallèle (P3)
- **US6 Batch (Phase 8)**: Depends on Setup — vérification, peut démarrer en parallèle (P3)
- **US7 Docker (Phase 9)**: Depends on Setup — peut démarrer en parallèle (P2)
- **Polish (Phase 10)**: Depends on toutes les US complétées

### User Story Dependencies

- **US1 (Prédiction)**: Dépend de T003, T004 (schémas). Indépendante des autres US
- **US2 (Entraînement)**: Indépendante. Modifie ml_tasks.py et task_store.py
- **US3 (Airflow)**: Indépendante. Modifie DAGs et main.py (promote endpoint)
- **US4 (Monitoring)**: Dépend de T005. Modifie metrics.py et Grafana
- **US5 (Validation)**: Indépendante. Vérification uniquement
- **US6 (Batch)**: Indépendante. Vérification uniquement
- **US7 (Docker)**: Indépendante. Modifie docker-compose.light.yml

### Within Each User Story

- Schema fixes before endpoint updates
- Endpoint updates before test updates
- Tests run after implementation

### Parallel Opportunities

- T001, T002 (setup) can run in parallel
- T003, T004, T005 (foundational) can run in parallel
- US2, US3, US5, US6, US7 can all start in parallel (no shared files)
- US1 and US4 share dependency on Foundational but modify different files
- T018, T019 (monitoring) can run in parallel
- T025, T026 (Docker verification) can run in parallel

---

## Implementation Strategy

### MVP First (US1 + US2)

1. Complete Phase 1: Setup verification
2. Complete Phase 2: Foundational schema fixes
3. Complete Phase 3: US1 — prédiction complète (health + predict endpoints)
4. Complete Phase 4: US2 — GPU mutex (409)
5. **STOP and VALIDATE**: Tests passent, API refuse les GPU concurrents, réponses conformes au contrat

### Incremental Delivery

1. Setup + Foundational → Schémas corrigés
2. US1 (Prédiction) → Health + predict conformes → **MVP**
3. US2 (Entraînement) → GPU mutex actif → **Core API complete**
4. US3 (Airflow) → Promotion Staging → **Orchestration conforme**
5. US7 (Docker) → Profil light correct → **Deployment conforme**
6. US4 (Monitoring) → Métriques cohérentes → **Observabilité conforme**
7. US5 + US6 (Validation + Batch) → Vérifications → **Feature complete**
8. Polish → Tests d'intégration, quickstart → **Release ready**

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story
- Le projet est déjà ~90% implémenté — les tâches ciblent les écarts avec la spec
- Les US5 et US6 sont principalement des vérifications (code déjà existant)
- Les tâches GPU mutex (US2) et profil light Docker (US7) sont les changements les plus significatifs
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
