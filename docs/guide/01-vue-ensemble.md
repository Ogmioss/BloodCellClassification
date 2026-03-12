# 01 — Vue d'ensemble

## Le projet en une phrase

BloodCellClassification est une plateforme MLOps complète qui entraîne, sert et monitore un modèle de classification d'images de cellules sanguines (8 types) via une architecture microservices conteneurisée.

## Les 8 classes de cellules

| Classe | Description |
|--------|-------------|
| Basophil | Granulocyte basophile |
| Eosinophil | Granulocyte éosinophile |
| Erythroblast | Précurseur des globules rouges |
| Immature granulocyte | Granulocyte immature |
| Lymphocyte | Lymphocyte |
| Monocyte | Monocyte |
| Neutrophil | Granulocyte neutrophile |
| Platelet | Plaquette sanguine |

Le dataset contient ~17 000 images de 360x363 pixels, organisées en sous-dossiers par classe.

## Architecture globale

Le projet s'articule autour de **6 composants principaux** qui communiquent entre eux :

```
                        ┌──────────────────────────────────────────────────┐
                        │              Couche présentation                 │
                        │  ┌───────────────────────────────────────────┐   │
                        │  │  Streamlit (6 pages)                     │   │
                        │  │  - Exploration du dataset                │   │
                        │  │  - Démo de prédiction                    │   │
                        │  │  - Interprétabilité (Grad-CAM)           │   │
                        │  └───────────────┬───────────────────────────┘   │
                        └──────────────────┼───────────────────────────────┘
                                           │ HTTP
                        ┌──────────────────▼───────────────────────────────┐
                        │              Couche API                          │
                        │  ┌───────────────────────────────────────────┐   │
                        │  │  FastAPI                                  │   │
                        │  │  - /predict, /predict/upload              │   │
                        │  │  - /ml/train, /ml/evaluate                │   │
                        │  │  - /health, /metrics                      │   │
                        │  └───────────────┬───────────────────────────┘   │
                        └──────────────────┼───────────────────────────────┘
                                           │
                 ┌─────────────────────────┼──────────────────────────┐
                 ▼                         ▼                          ▼
   ┌─────────────────────┐  ┌─────────────────────┐  ┌──────────────────────┐
   │   Couche ML         │  │   Couche stockage   │  │  Couche monitoring   │
   │                     │  │                     │  │                      │
   │  PyTorch (ResNet18) │  │  MinIO (artefacts)  │  │  Prometheus          │
   │  Services :         │  │  MLflow (tracking)  │  │  Grafana             │
   │  - Training         │  │  PostgreSQL         │  │  Pushgateway         │
   │  - Inference        │  │    (metadata Airflow)│  │  Alertes             │
   │  - Evaluation       │  │                     │  │                      │
   └─────────────────────┘  └─────────────────────┘  └──────────────────────┘

   ┌──────────────────────────────────────────────────────────────────────────┐
   │                     Couche orchestration                                │
   │  Airflow — déclenche les pipelines ML via des appels HTTP à FastAPI     │
   └──────────────────────────────────────────────────────────────────────────┘
```

## Principe clé : séparation PyTorch / orchestration

Un choix architectural central du projet est que **seul le conteneur API embarque PyTorch**. Airflow, Streamlit et les autres services n'ont aucune dépendance ML lourde. Ils communiquent avec l'API via HTTP.

Pourquoi ?

- **Images Docker légères** : le conteneur Airflow fait ~500 Mo au lieu de 3-4 Go
- **Scalabilité** : on peut scaler l'API indépendamment
- **Simplicité** : un seul point d'entrée pour toutes les opérations ML

## Flux de données principaux

### 1. Prédiction (temps réel)

```
Utilisateur → Streamlit → POST /predict → InferenceService → ResNet18 → réponse JSON
```

### 2. Entraînement (batch)

```
Airflow DAG → POST /ml/train → BackgroundTask → TrainingService
    → DatasetService (charge les images)
    → train_model.main() (boucle d'entraînement)
    → MLflow (log métriques + modèle)
    → MinIO (stocke artefacts)
```

### 3. Monitoring (continu)

```
FastAPI /metrics → Prometheus (scrape toutes les 10s) → Grafana (dashboards)
Pushgateway ← TrainingService (métriques d'entraînement par epoch)
```

## Organisation du code source

```
src/
├── api/                    # FastAPI — endpoints REST
│   ├── main.py             # Application principale, routes de base
│   ├── schemas.py          # Modèles Pydantic (request/response)
│   ├── metrics.py          # Compteurs Prometheus custom
│   ├── task_store.py       # Stockage persistant des tâches async
│   └── routers/
│       ├── ml_tasks.py     # Routes /ml/* (train, evaluate, batch)
│       └── pipelines.py    # Routes pour déclencher des DAGs Airflow
│
├── models/                 # Définitions de modèles PyTorch
│   ├── base_classifier.py  # Classe abstraite
│   ├── resnet_classifier.py# ResNet (18/34/50)
│   ├── cnn_classifier.py   # CNN custom léger
│   └── model_factory.py    # Factory pattern
│
├── services/               # Logique métier (couche services)
│   ├── training_service.py       # Boucle d'entraînement
│   ├── inference_service.py      # Prédiction
│   ├── evaluation_service.py     # Métriques (accuracy, F1, etc.)
│   ├── dataset_service.py        # Chargement et split du dataset
│   ├── data_transform_service.py # Augmentations et normalisation
│   ├── mlflow_service.py         # Intégration MLflow
│   ├── model_loader_service.py   # Chargement modèle (MLflow ou checkpoint)
│   ├── data_validation_service.py# Validation qualité des données
│   ├── training_metrics.py       # Push métriques vers Pushgateway
│   └── yaml_loader.py            # Chargement de conf.yaml
│
├── pipe/                   # Pipelines ML exécutables
│   ├── train_model.py      # Pipeline d'entraînement complet
│   ├── evaluate_model.py   # Pipeline d'évaluation
│   └── batch_inference_pipeline.py  # Inférence en batch
│
├── core/                   # Constantes et configuration
│   ├── constants.py        # Noms des classes, NUM_CLASSES
│   └── config.py           # Résolution d'URLs de services
│
├── pages/                  # Pages Streamlit
│   ├── 1_Presentation_du_projet.py
│   ├── 2_Exploration_du_dataset.py
│   ├── 3_Modele.py
│   ├── 4_Demo.py
│   ├── 5_Interpretabilite.py
│   └── 6_Conclusion.py
│
├── utils/                  # Utilitaires Streamlit
│   ├── gradcam.py          # Grad-CAM (interprétabilité)
│   ├── charts.py           # Graphiques Plotly
│   └── ...
│
└── app.py                  # Point d'entrée Streamlit
```

## Fichiers de configuration clés

| Fichier | Rôle |
|---------|------|
| `conf.yaml` | Configuration centrale (modèle, entraînement, MLflow, MinIO) |
| `docker/docker-compose.yml` | Stack complète (API + Airflow + MLflow + MinIO + PostgreSQL) |
| `docker/docker-compose.light.yml` | Stack légère (API + Streamlit + MLflow + MinIO) |
| `docker/docker-compose.monitoring.yml` | Overlay monitoring (Prometheus + Grafana) |
| `monitoring/prometheus.yml` | Cibles de scraping Prometheus |
| `pyproject.toml` | Dépendances Python et métadonnées du projet |
| `.github/workflows/ci.yml` | Pipeline CI (lint, tests, build Docker) |

## Pour aller plus loin

Chaque composant est détaillé dans les documents suivants de ce guide. Commencez par celui qui vous intéresse ou lisez-les dans l'ordre pour une compréhension progressive.
