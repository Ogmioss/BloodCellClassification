# 09 — CI/CD

L'intégration et le déploiement continus sont gérés par **GitHub Actions**.

## Pipeline CI (`ci.yml`)

Déclenché sur chaque push et pull request :

```
Push / PR
  │
  ├─▶ Lint (Ruff)
  │     Vérifie le formatage et les règles de style
  │
  ├─▶ Tests unitaires (pytest)
  │     Exclut: DAG, intégration, performance
  │
  ├─▶ Tests DAG (pytest)
  │     Valide la structure des DAGs Airflow
  │
  ├─▶ Tests d'intégration (pytest)
  │     Scénarios end-to-end
  │
  ├─▶ Tests de performance (pytest)
  │     Benchmarks de latence et throughput
  │
  ├─▶ Build Docker
  │     Construit les images API, Airflow, Streamlit
  │
  └─▶ Audit sécurité (pip-audit)
        Vérifie les vulnérabilités des dépendances
```

### Exécution locale des tests

```bash
# Lint
uv run ruff check .
uv run ruff format --check .

# Tests unitaires
uv run pytest tests/ --ignore=tests/test_dags.py --ignore=tests/test_performance.py --ignore=tests/test_integration.py

# Tests DAG
uv run pytest tests/test_dags.py

# Tests d'intégration
uv run pytest tests/test_integration.py

# Tests de performance
uv run pytest tests/test_performance.py

# Tous les tests
uv run pytest tests/
```

## Pipeline CD (`cd.yml`)

Déclenché sur release GitHub ou manuellement :

```
Release / Manual
  │
  ├─▶ Build images Docker
  │     ├── bloodcell-api
  │     ├── bloodcell-airflow
  │     └── bloodcell-streamlit
  │
  ├─▶ Push vers GHCR
  │     Tags: semver, SHA, latest
  │     Registry: ghcr.io/{owner}/{repo}/
  │
  ├─▶ Deploy Staging (manuel)
  │
  └─▶ Deploy Production (sur release)
```

### Images publiées

| Image | Registry | Contenu |
|-------|----------|---------|
| `bloodcell-api` | `ghcr.io/{repo}/bloodcell-api` | FastAPI + PyTorch |
| `bloodcell-airflow` | `ghcr.io/{repo}/bloodcell-airflow` | Airflow léger |
| `bloodcell-streamlit` | `ghcr.io/{repo}/bloodcell-streamlit` | Dashboard Streamlit |

### Tags des images

| Tag | Exemple | Description |
|-----|---------|-------------|
| Semver | `v1.2.3` | Version de la release |
| SHA | `sha-abc1234` | Commit exact |
| `latest` | — | Dernière version publiée |

## Structure des tests

```
tests/
├── test_inference_service.py       # Prédictions unitaires
├── test_training_pipeline.py       # Pipeline d'entraînement
├── test_evaluation_service.py      # Calcul de métriques
├── test_model_factory.py           # Création de modèles
├── test_mlflow_service.py          # Intégration MLflow
├── test_data_validation_service.py # Validation de données
├── test_task_store.py              # Stockage de tâches
├── test_pipelines_router.py        # Routes API
├── test_batch_inference_pipeline.py# Inférence batch
├── test_resnet_classifier.py       # Tests ResNet
├── test_device.py                  # Détection GPU/CPU
├── test_dags.py                    # Structure des DAGs Airflow
├── test_integration.py             # Tests end-to-end
└── test_performance.py             # Benchmarks
```

## Conventions de qualité

- **Ruff** pour le formatage et le linting (configuration dans `pyproject.toml`)
- **pytest** avec des fixtures pour l'isolation des tests
- **pip-audit** pour détecter les vulnérabilités connues dans les dépendances
