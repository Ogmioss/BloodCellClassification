# Architecture du projet

## Vue d'ensemble

Blood Cell Classification est une plateforme MLOps pour la classification de cellules sanguines
(8 classes) basee sur ResNet18 (PyTorch), avec orchestration Airflow, tracking MLflow,
stockage MinIO S3, API FastAPI et interface Streamlit.

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│  Streamlit   │────>│   FastAPI     │────>│   MLflow      │
│  (UI)        │     │  (ML runtime) │     │  (tracking)   │
│  port 8502   │     │  port 8001    │     │  port 5002    │
└─────────────┘     └──────┬───────┘     └──────┬───────┘
                           │                     │
                    ┌──────┴───────┐     ┌──────┴───────┐
                    │   Airflow     │     │   MinIO S3    │
                    │ (orchestration│     │  (artefacts)  │
                    │  port 8081)   │     │  port 9002    │
                    └──────────────┘     └──────────────┘
```

**Principe cle** : seul le container FastAPI (`api`) contient PyTorch.
Airflow reste leger (~500 MB) et declenche l'entrainement via HTTP.

## Organisation du code

```
src/
  api/                    # API REST FastAPI
    main.py               # App FastAPI, endpoints /health, /predict, /metrics, /model/info
    routers/
      ml_tasks.py         # Endpoints ML background : /ml/train, /ml/evaluate, etc.
      pipelines.py        # Trigger Airflow DAGs : /pipelines/train, etc.
    schemas.py            # Modeles Pydantic (request/response)
    task_store.py         # Stockage persistant des taches (JSON)
    metrics.py            # Metriques Prometheus custom

  services/               # Services metier (Single Responsibility)
    mlflow_service.py     # Tracking MLflow + Model Registry
    training_service.py   # Boucle d'entrainement (Adam, CrossEntropy)
    evaluation_service.py # Evaluation + confusion matrix + metriques par classe
    inference_service.py  # Prediction sur images (preprocess + forward)
    model_loader_service.py  # Chargement unifie (MLflow > checkpoint local)
    dataset_service.py    # Chargement dataset, splits, class weights
    data_transform_service.py  # Augmentations + normalisation ImageNet
    data_validation_service.py # Validation qualite du dataset
    yaml_loader.py        # Chargement de conf.yaml

  models/                 # Modeles PyTorch
    base_classifier.py    # Classe abstraite
    resnet_classifier.py  # ResNet18/34/50 (torchvision, pretrained ImageNet)
    cnn_classifier.py     # CNN custom (optionnel)
    model_factory.py      # Factory : cree le modele selon conf.yaml

  pipe/                   # Pipelines d'execution
    train_model.py        # Orchestration complete : data → train → eval → MLflow
    evaluate_model.py     # Evaluation d'un modele existant
    batch_inference_pipeline.py  # Predictions en batch sur un dossier

  core/
    config.py             # URLs des services (API, MLflow, Airflow, Grafana)

  pages/                  # Interface Streamlit (6 pages)
  utils/                  # Utilitaires (charts, GradCAM, RGB, stats)

dags/                     # DAGs Airflow
  bloodcells_train_dag_api.py       # Train → decide → promote/skip
  bloodcells_evaluate_dag_api.py    # Evaluate → report
  bloodcells_batch_inference_dag_api.py  # Batch inference → report
  bloodcells_data_validation_dag_api.py  # Validate data → report
  utils/
    api_client.py         # Client HTTP leger pour appeler FastAPI

docker/                   # Configuration Docker
  Dockerfile.api          # FastAPI + PyTorch (seul container ML)
  Dockerfile.airflow      # Airflow leger (pas de PyTorch)
  Dockerfile.streamlit    # Interface Streamlit
  docker-compose.yml      # Stack complete
  docker-compose.light.yml     # Stack legere (sans Airflow)
  docker-compose.monitoring.yml  # Overlay Prometheus + Grafana

monitoring/               # Observabilite
  prometheus.yml          # Config scraping
  grafana/                # Dashboards + datasources

conf.yaml                 # Configuration centrale
```

## Services et responsabilites

### MLflowService (`src/services/mlflow_service.py`)

- Initialisation avec timeout (5s) pour ne pas bloquer si MLflow est down
- Log des parametres, metriques, artefacts, modeles
- Model Registry : versionning + transitions de stage (None → Staging → Production)
- Priorite tracking URI : `MLFLOW_TRACKING_URI` env > `conf.yaml` > `./mlruns`

### TrainingService (`src/services/training_service.py`)

- Boucle d'entrainement : forward, loss, backward, optimizer step
- Optimiseur Adam, loss CrossEntropyLoss (avec class weights optionnels)
- Sauvegarde du meilleur checkpoint (best val accuracy)
- Metriques : train_loss, train_acc, val_loss, val_acc par epoque

### ModelFactory (`src/models/model_factory.py`)

- Cree le modele selon `conf.yaml` (resnet18/34/50, cnn custom)
- Detection auto du device : CUDA > CPU
- ResNet18 avec poids pretrained ImageNet-1K par defaut

### InferenceService (`src/services/inference_service.py`)

- Preprocessing : resize 224x224, normalisation ImageNet
- Prediction : classe, confiance, probabilites par classe
- Chargement : MLflow registry (prioritaire) > checkpoint local

### DatasetService (`src/services/dataset_service.py`)

- Source : `data/raw/bloodcells_dataset/` (ImageFolder)
- 8 classes : basophil, eosinophil, erythroblast, granulocyte, lymphocyte, monocyte, neutrophil, platelet
- Splits : 70% train / 15% val / 15% test
- Sous-echantillonnage configurable (`subset_size`)
- Calcul des class weights pour donnees desequilibrees

## Configuration (`conf.yaml`)

```yaml
training:
  batch_size: 32
  img_size: 224
  epochs: 20
  learning_rate: 0.001
  train_split: 0.7
  val_split: 0.15
  subset_size: 1400       # null pour dataset complet

model:
  name: "resnet18"
  pretrained: true
  pretrained_weights: "IMAGENET1K_V1"

mlflow:
  tracking_uri: "./mlruns"
  experiment_name: "bloodcells-classification"
  model_name: "bloodcells-classifier"

minio:
  endpoint_url: "http://localhost:9000"
  access_key: "minio"
  secret_key: "minio123"
  buckets:
    artifacts: "mlflow-artifacts"
    datasets: "datasets"
    checkpoints: "checkpoints"
```

## Endpoints API FastAPI

### Prediction
- `POST /predict` — Prediction depuis image base64
- `POST /predict/upload` — Prediction depuis upload fichier

### Taches ML (background, asynchrone)
- `POST /ml/train` — Lance un entrainement (params optionnels : epochs, lr, batch_size)
- `POST /ml/evaluate` — Evalue le modele charge
- `POST /ml/batch-inference` — Predictions en batch
- `POST /ml/validate-data` — Validation du dataset
- `GET /ml/tasks/{task_id}` — Statut d'une tache

### MLflow
- `GET /mlflow/models` — Liste les versions enregistrees
- `POST /mlflow/promote/{version}?stage=Production` — Promotion de modele

### Airflow Pipelines
- `POST /pipelines/train` — Trigger le DAG d'entrainement
- `POST /pipelines/evaluate` — Trigger le DAG d'evaluation
- `GET /pipelines/status/{dag_id}/{dag_run_id}` — Statut d'un DAG run

### Monitoring
- `GET /health` — Sante de l'API
- `GET /metrics` — Metriques Prometheus
- `GET /model/info` — Info sur le modele charge

## Contraintes de concurrence

- **Taches GPU** (train, evaluate) : 1 seule a la fois (rejet 409 si deja en cours)
- **Taches CPU** (batch inference, validation) : paralleles autorisees
- **Persistance** : les taches survivent au redemarrage de l'API (JSON store)

## Strategie de chargement du modele

1. MLflow Model Registry (si `MLFLOW_TRACKING_URI` configure)
2. Checkpoint local (`models/checkpoints/best_model.pth`)
3. Si rien disponible : `/predict` retourne 503
