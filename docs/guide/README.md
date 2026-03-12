# Guide technique — BloodCellClassification

Ce dossier contient une documentation pédagogique complète du projet. L'objectif est de permettre à tout développeur ou utilisateur de comprendre rapidement la stack technique, l'architecture et les workflows sans avoir à lire l'intégralité du code source.

## Sommaire

| # | Document | Description |
|---|----------|-------------|
| 01 | [Vue d'ensemble](01-vue-ensemble.md) | Architecture générale, flux de données, composants principaux |
| 02 | [Stack technique](02-stack-technique.md) | Technologies utilisées, rôle de chaque outil, versions |
| 03 | [API FastAPI](03-api-fastapi.md) | Endpoints, injection de dépendances, couche services |
| 04 | [Pipeline ML](04-pipeline-ml.md) | Entraînement, inférence, évaluation — le cycle de vie du modèle |
| 05 | [Orchestration Airflow](05-orchestration-airflow.md) | DAGs, interaction avec l'API, patterns utilisés |
| 06 | [MLflow & gestion de modèles](06-mlflow-tracking.md) | Experiment tracking, registry, promotion de modèles |
| 07 | [Monitoring](07-monitoring.md) | Prometheus, Grafana, alertes, métriques métier |
| 08 | [Docker & déploiement](08-docker-deploiement.md) | Images, Compose profiles, stratégies de build |
| 09 | [CI/CD](09-ci-cd.md) | GitHub Actions, lint, tests, publication d'images |
| 10 | [Guide développeur](10-guide-developpeur.md) | Onboarding, commandes utiles, conventions du projet |

## Comment lire cette documentation

- **Vous débarquez sur le projet ?** Commencez par le document **01** puis **10**.
- **Vous voulez comprendre un composant précis ?** Allez directement au document concerné.
- **Vous cherchez à dépanner ?** Consultez aussi `docs/troubleshooting.md` à la racine de `docs/`.

## Schéma d'architecture rapide

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Streamlit   │────▶│   FastAPI     │────▶│   PyTorch    │
│  (dashboard) │     │   (API REST)  │     │  (ResNet18)  │
└─────────────┘     └──────┬───────┘     └─────────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │  MLflow   │ │Prometheus│ │  MinIO   │
        │(tracking) │ │(métriques│ │  (S3)    │
        └──────────┘ └────┬─────┘ └──────────┘
                          ▼
                    ┌──────────┐
                    │ Grafana  │
                    │(tableaux)│
                    └──────────┘

        ┌──────────────────────────────┐
        │  Airflow (orchestration)     │
        │  déclenche l'API pour les    │
        │  tâches ML lourdes           │
        └──────────────────────────────┘
```
