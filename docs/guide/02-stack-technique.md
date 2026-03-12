# 02 — Stack technique

Ce document présente chaque technologie utilisée dans le projet, son rôle et pourquoi elle a été choisie.

## Vue synthétique

| Couche | Technologie | Rôle |
|--------|-------------|------|
| API REST | FastAPI + Uvicorn | Servir les prédictions et piloter les tâches ML |
| Validation | Pydantic v2 | Schémas de requêtes/réponses typés |
| Deep Learning | PyTorch + torchvision | Modèle ResNet18, entraînement, inférence |
| Experiment tracking | MLflow | Versionner les expériences, modèles et artefacts |
| Stockage objets | MinIO (compatible S3) | Stocker modèles, datasets, artefacts |
| Orchestration | Apache Airflow 2.9 | Planifier et chaîner les pipelines ML |
| Métriques | Prometheus | Collecter les métriques de l'API et de l'entraînement |
| Dashboards | Grafana | Visualiser les métriques en temps réel |
| Interface utilisateur | Streamlit | Dashboard interactif (démo, exploration, Grad-CAM) |
| Conteneurisation | Docker + Compose | Isoler et déployer chaque service |
| CI/CD | GitHub Actions | Automatiser lint, tests, build et publication |
| Gestion de paquets | uv | Installation rapide des dépendances Python |
| Linting | Ruff | Formatage et analyse statique du code |
| Tests | pytest | Tests unitaires, intégration, performance |
| Client HTTP | httpx | Appels inter-services (Airflow → API) |

## Détail par technologie

### FastAPI + Uvicorn

**Rôle** : Point d'entrée unique pour toutes les opérations ML (prédiction, entraînement, évaluation).

FastAPI a été choisi pour :
- La **validation automatique** des requêtes via Pydantic
- Le support natif de l'**asynchrone** (important pour les tâches longues)
- La **documentation OpenAPI** générée automatiquement (`/docs`)
- L'intégration native avec `BackgroundTasks` pour lancer des entraînements sans bloquer l'API

Uvicorn sert l'application avec 2 workers en production.

### PyTorch + torchvision

**Rôle** : Entraînement et inférence du modèle de classification.

Le modèle principal est un **ResNet18 pré-entraîné sur ImageNet** dont la dernière couche fully-connected est remplacée pour 8 classes. Ce choix offre :
- Un bon compromis **performance/taille** pour la classification d'images médicales
- Le **transfer learning** : les poids ImageNet accélèrent la convergence
- La possibilité de changer facilement de backbone (ResNet34, 50, etc.) via la factory

Un CNN custom léger est aussi disponible comme alternative.

### MLflow

**Rôle** : Tracker les expériences, versionner les modèles, gérer leur cycle de vie.

MLflow intervient à trois niveaux :
1. **Tracking** : log des hyperparamètres, métriques par epoch, artefacts
2. **Registry** : chaque modèle entraîné est enregistré avec un numéro de version
3. **Lifecycle** : promotion `None → Staging → Production`

L'API charge automatiquement le modèle marqué "Production" dans le registry. Si MLflow est indisponible, elle tombe sur le checkpoint local.

### MinIO

**Rôle** : Stockage S3-compatible pour les artefacts ML.

Trois buckets :
- `mlflow-artifacts` — modèles et artefacts MLflow
- `datasets` — datasets uploadés
- `checkpoints` — sauvegardes de modèles

MinIO remplace AWS S3 en développement local, avec la même interface. En production, on peut pointer vers un vrai S3 en changeant les variables d'environnement.

### Apache Airflow

**Rôle** : Orchestrer les pipelines ML (entraînement, évaluation, batch inference, validation de données).

Point important : Airflow **ne fait pas de ML directement**. Il appelle l'API FastAPI via HTTP. Cela signifie :
- Le conteneur Airflow est léger (~500 Mo, pas de PyTorch)
- Les DAGs sont de simples séquences d'appels HTTP
- L'API gère l'exécution ML en tâche de fond

### Prometheus + Grafana

**Rôle** : Observabilité de la plateforme.

Prometheus scrape trois sources :
1. L'API FastAPI (`/metrics`) — latence, requêtes, prédictions par classe
2. Le Pushgateway — métriques d'entraînement poussées par epoch
3. MinIO — métriques de stockage

Grafana affiche deux dashboards :
- **API** : taux de requêtes, latence, distribution des prédictions, confiance
- **Training** : loss et accuracy par epoch, durée d'entraînement

Des alertes sont configurées (API down, taux d'erreur élevé, confiance basse, entraînement bloqué).

### Streamlit

**Rôle** : Interface utilisateur pour explorer le dataset, tester le modèle et visualiser les résultats.

6 pages couvrent :
1. Présentation du projet
2. Exploration statistique du dataset
3. Architecture et métriques du modèle
4. Démo de prédiction en temps réel
5. Interprétabilité via Grad-CAM
6. Conclusion

Streamlit appelle l'API pour les prédictions — il n'embarque pas PyTorch.

### Docker

**Rôle** : Conteneuriser et orchestrer tous les services.

Trois Dockerfiles optimisés :
- `Dockerfile.api` — Multi-stage build, seul conteneur avec PyTorch (~3 Go)
- `Dockerfile.airflow` — Image légère, uniquement httpx et config (~500 Mo)
- `Dockerfile.streamlit` — Image légère, Streamlit + httpx

Trois profils Compose pour adapter la stack au besoin (voir [08-docker-deploiement.md](08-docker-deploiement.md)).

### uv

**Rôle** : Gestionnaire de paquets Python ultra-rapide (remplacement de pip/pip-tools).

`uv` est utilisé partout :
- Dans les Dockerfiles pour installer les dépendances
- En local pour le développement
- Dans la CI pour les tests

### GitHub Actions

**Rôle** : CI/CD automatisé.

Deux workflows :
- **CI** : lint (Ruff), tests unitaires/intégration/performance, build Docker, audit sécurité
- **CD** : build, push vers GHCR, déploiement staging/production

## Diagramme des interactions

```
┌─────────┐  HTTP   ┌──────────┐  HTTP   ┌─────────┐
│Streamlit │───────▶│  FastAPI  │◀────────│ Airflow │
└─────────┘        └────┬─────┘         └─────────┘
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
    ┌──────────┐  ┌──────────┐  ┌──────────┐
    │ PyTorch  │  │  MLflow   │  │Prometheus│
    │(interne) │  │          │  │          │
    └──────────┘  └────┬─────┘  └────┬─────┘
                       ▼             ▼
                 ┌──────────┐  ┌──────────┐
                 │  MinIO   │  │ Grafana  │
                 │  (S3)    │  │          │
                 └──────────┘  └──────────┘
```

> **Note** : PyTorch est embarqué *dans* le processus FastAPI, pas en tant que service séparé. C'est le seul composant non-HTTP du schéma.
