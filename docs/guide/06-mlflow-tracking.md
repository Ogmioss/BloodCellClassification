# 06 — MLflow & gestion de modèles

MLflow assure le suivi des expériences, le versionnage des modèles et leur promotion vers la production.

## Rôle dans le projet

```
Entraînement                    MLflow                         API
┌───────────┐    log_params()   ┌─────────────────┐           ┌─────────┐
│ train()   │───────────────────▶│ Experiment      │           │ FastAPI │
│           │    log_metrics()  │ ├── Run 1        │           │         │
│           │───────────────────▶│ │   params, metrics│        │         │
│           │    log_model()    │ ├── Run 2        │           │         │
│           │───────────────────▶│ │   ...          │           │         │
└───────────┘                   │ └── Run N        │           │         │
                                │                   │           │         │
                                │ Model Registry    │  load()   │         │
                                │ ├── v1 (None)     │──────────▶│ sert le │
                                │ ├── v2 (Staging)  │           │ modèle  │
                                │ └── v3 (Production)│          │ Prod    │
                                └─────────────────┘           └─────────┘
```

## Ce qui est logué

### Paramètres (une fois par run)

- Hyperparamètres : epochs, learning_rate, batch_size, img_size
- Modèle : nom (resnet18), pretrained, weights
- Données : train_split, val_split, subset_size, seed
- Environnement : device (cpu/cuda), git commit, git branch

### Métriques (par epoch et finales)

| Métrique | Quand | Description |
|----------|-------|-------------|
| `train_loss` | Chaque epoch | Loss d'entraînement |
| `train_accuracy` | Chaque epoch | Accuracy sur le train set |
| `val_loss` | Chaque epoch | Loss de validation |
| `val_accuracy` | Chaque epoch | Accuracy de validation |
| `test_accuracy` | Fin | Accuracy sur le test set |
| `test_precision` | Fin | Precision (weighted) |
| `test_recall` | Fin | Recall (weighted) |
| `test_f1` | Fin | F1-score (weighted) |

### Artefacts

- Le modèle PyTorch complet (sérialisé)
- `metrics.json` — métriques détaillées
- Matrice de confusion (si générée)

## Cycle de vie d'un modèle

```
                log_model()              promote()              promote()
Entraînement ─────────────▶ None ────────────────▶ Staging ────────────────▶ Production
                            (enregistré)          (test)                    (sert le trafic)
```

### Stages

| Stage | Signification | Qui l'utilise |
|-------|---------------|---------------|
| `None` | Juste enregistré, pas encore validé | Personne |
| `Staging` | En cours de validation | Tests d'intégration, évaluation |
| `Production` | Validé, sert le trafic réel | L'API FastAPI charge ce modèle |

### Promotion via l'API

```bash
# Promouvoir la version 3 en Production
curl -X POST http://localhost:8000/mlflow/promote/3?stage=Production

# Lister les versions disponibles
curl http://localhost:8000/mlflow/models
```

### Promotion automatique (via Airflow)

Le DAG d'entraînement peut automatiquement promouvoir un modèle si son accuracy dépasse un seuil :

```
Si test_accuracy > 0.90 :
  → Transition vers Staging
  → (optionnel) Transition vers Production
```

## MLflowService (`src/services/mlflow_service.py`)

Le service encapsule toutes les interactions avec MLflow et ajoute de la résilience :

### Timeout protection

Chaque appel MLflow a un timeout de 5 secondes par défaut. Si le serveur MLflow est lent ou indisponible, l'entraînement continue sans tracking plutôt que de planter.

### Lazy initialization

La connexion à MLflow n'est établie que lors du premier appel effectif. Cela évite de ralentir le démarrage de l'API si MLflow n'est pas encore prêt.

### Résolution de l'URI

La priorité pour déterminer l'URI du serveur MLflow :

```
1. Variable d'environnement MLFLOW_TRACKING_URI  (priorité haute)
2. Valeur dans conf.yaml → mlflow.tracking_uri
3. Défaut: "./mlruns" (stockage local)               (priorité basse)
```

## Stockage des artefacts

MLflow stocke les artefacts (modèles, fichiers) dans MinIO via le protocole S3 :

```
MLflow ──s3://──▶ MinIO
                  ├── mlflow-artifacts/
                  │   ├── 1/  (experiment)
                  │   │   ├── abc123/  (run)
                  │   │   │   ├── artifacts/
                  │   │   │   │   ├── model/
                  │   │   │   │   └── metrics.json
```

**Variables d'environnement nécessaires** :
- `MLFLOW_S3_ENDPOINT_URL=http://minio:9000`
- `AWS_ACCESS_KEY_ID=minio`
- `AWS_SECRET_ACCESS_KEY=minio123`

## Interface MLflow

L'UI MLflow est accessible à `http://localhost:5000` (stack light) ou `http://localhost:5002` (stack full).

Elle permet de :
- Comparer les runs (graphiques de métriques)
- Voir les paramètres de chaque expérience
- Télécharger les artefacts
- Gérer le registry (promouvoir/archiver des versions)

## Fallback sans MLflow

Si MLflow est totalement indisponible, le projet fonctionne quand même :
- L'entraînement sauvegarde un checkpoint local (`models/checkpoints/best_model.pth`)
- L'API charge ce checkpoint au démarrage
- Les métriques sont tout de même affichées via Prometheus/Grafana
