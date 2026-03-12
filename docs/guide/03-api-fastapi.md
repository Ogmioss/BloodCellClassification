# 03 — API FastAPI

L'API est le coeur du système. C'est le seul composant qui embarque PyTorch et c'est à travers elle que transitent toutes les opérations ML.

## Points d'entrée

### Endpoints principaux (`src/api/main.py`)

| Méthode | Route | Description |
|---------|-------|-------------|
| `GET` | `/health` | État de santé (modèle chargé, device, source) |
| `GET` | `/metrics` | Métriques Prometheus (format texte) |
| `GET` | `/model/info` | Informations sur le modèle (architecture, version MLflow) |
| `POST` | `/predict` | Prédiction depuis une image base64 |
| `POST` | `/predict/upload` | Prédiction depuis un fichier uploadé |
| `GET` | `/mlflow/models` | Liste des versions du modèle dans le registry |
| `POST` | `/mlflow/promote/{version}` | Promouvoir une version (Staging/Production) |

### Endpoints ML (`src/api/routers/ml_tasks.py`)

| Méthode | Route | Description |
|---------|-------|-------------|
| `GET` | `/ml/datasets` | Lister les datasets disponibles |
| `POST` | `/ml/train` | Lancer un entraînement (tâche asynchrone) |
| `POST` | `/ml/evaluate` | Lancer une évaluation |
| `POST` | `/ml/batch-inference` | Lancer une inférence batch |
| `POST` | `/ml/validate-data` | Valider l'intégrité d'un dataset |
| `GET` | `/ml/tasks/{task_id}` | Consulter le statut d'une tâche |

### Endpoints pipelines (`src/api/routers/pipelines.py`)

| Méthode | Route | Description |
|---------|-------|-------------|
| `POST` | `/pipelines/trigger/{dag_id}` | Déclencher un DAG Airflow |

## Architecture interne

### Injection de dépendances

L'API utilise le pattern d'injection de dépendances de FastAPI avec `@lru_cache()` pour créer des singletons :

```python
@lru_cache()
def get_yaml_loader():
    return YamlLoader()

@lru_cache()
def get_mlflow_service():
    config = get_yaml_loader()
    return MLflowService(config)

@lru_cache()
def get_inference_service():
    model = get_model_loader_service().load_model()
    transforms = DataTransformService(get_yaml_loader())
    return InferenceService(model, transforms)
```

Ce mécanisme garantit qu'une seule instance de chaque service existe en mémoire, et que le modèle n'est chargé qu'une seule fois.

### Graphe de dépendances

```
FastAPI App
├── YamlLoader (conf.yaml)
│   └── utilisé par presque tous les services
├── ModelLoaderService
│   ├── MLflowService (tente de charger le modèle "Production")
│   ├── ModelFactory (crée l'architecture ResNet/CNN)
│   └── fallback: checkpoint local (best_model.pth)
├── InferenceService
│   ├── DataTransformService (resize + normalisation)
│   └── modèle chargé
├── TaskStore (stockage JSON des tâches async)
└── Prometheus Instrumentator (métriques HTTP auto)
```

### Chargement du modèle

Le chargement suit une stratégie de fallback :

```
1. Tenter MLflow Registry → chercher le modèle au stage "Production"
   ├── Succès → utiliser ce modèle (source: "mlflow", version: X)
   └── Échec (timeout, indisponible, pas de modèle Production)
       ↓
2. Charger le checkpoint local → models/checkpoints/best_model.pth
   ├── Succès → utiliser ce modèle (source: "checkpoint")
   └── Échec → API démarre sans modèle (health: model_loaded=false)
```

Cela rend l'API **résiliente** : elle fonctionne même sans MLflow.

## Tâches asynchrones

Les opérations longues (entraînement, évaluation, batch inference) sont exécutées en arrière-plan via `FastAPI.BackgroundTasks` :

```
POST /ml/train
  ↓
1. Créer un TaskResult (status=PENDING) dans le TaskStore
2. Retourner immédiatement le task_id au client
3. Lancer _run_training() en BackgroundTask
  ↓
BackgroundTask:
4. Importer les modules lourds (torch, etc.) — import différé
5. Appeler train_model.main() avec la config
6. Mettre à jour le TaskResult (status=COMPLETED ou FAILED)
  ↓
GET /ml/tasks/{task_id}
7. Le client poll régulièrement pour suivre l'avancement
```

### Import différé

Les imports PyTorch ne sont faits que lorsqu'une tâche ML est effectivement lancée. Cela optimise le temps de démarrage de l'API et réduit la consommation mémoire quand seules les routes non-ML sont utilisées.

## Schémas Pydantic (`src/api/schemas.py`)

```python
class PredictionRequest(BaseModel):
    image: str  # base64-encoded

class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    probabilities: dict[str, float]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_source: str | None
    device: str
```

## TaskStore (`src/api/task_store.py`)

Le TaskStore est un stockage persistant basé sur un fichier JSON (`data/tasks.json`). Il est thread-safe et supporte le mode multi-workers de Uvicorn :

- **Écriture atomique** : écrit dans un fichier temporaire puis renomme (pas de corruption)
- **Rechargement** : relit le fichier à chaque lecture (cohérence éventuelle entre workers)
- **Cycle de vie** : PENDING → RUNNING → COMPLETED / FAILED

## Métriques Prometheus (`src/api/metrics.py`)

L'API expose des métriques custom en plus des métriques HTTP automatiques :

| Métrique | Type | Description |
|----------|------|-------------|
| `bloodcell_predictions_total` | Counter | Nombre de prédictions par classe |
| `bloodcell_prediction_errors_total` | Counter | Nombre d'erreurs de prédiction |
| `bloodcell_prediction_confidence` | Histogram | Distribution de la confiance |
| `bloodcell_prediction_latency_ms` | Histogram | Latence des prédictions |

Ces métriques sont scrapées par Prometheus et visualisées dans Grafana.

## Flux complet d'une prédiction

```
Client envoie POST /predict avec {"image": "<base64>"}
  ↓
1. Pydantic valide la requête (PredictionRequest)
2. Décodage base64 → bytes → PIL.Image
3. InferenceService.predict_image(image)
   a. DataTransformService applique les transformations (resize 224x224, normalisation ImageNet)
   b. Tensor envoyé sur le device (CPU ou CUDA)
   c. model.eval() + torch.no_grad() → forward pass
   d. Softmax → probabilités par classe
   e. argmax → classe prédite
4. Enregistrement des métriques Prometheus (classe, confiance, latence)
5. Retour PredictionResponse JSON
```

## Documentation interactive

L'API génère automatiquement une documentation Swagger accessible à `http://localhost:8000/docs` quand elle est lancée.
