# 05 — Orchestration Airflow

Airflow planifie et chaîne les pipelines ML. Son rôle est de **coordonner**, pas d'exécuter du ML directement.

## Principe architectural

```
┌──────────────────────┐          ┌──────────────────────┐
│  Airflow (léger)     │  HTTP    │  FastAPI (lourd)     │
│  ~500 Mo             │─────────▶│  ~3 Go (PyTorch)     │
│  Pas de PyTorch      │          │  Exécute le ML       │
│  Planifie les tâches │◀─────────│  Retourne les résultats│
└──────────────────────┘  JSON    └──────────────────────┘
```

Airflow ne fait que des appels HTTP vers l'API FastAPI. Tout le calcul ML se fait côté API. Cela permet de garder le conteneur Airflow léger et de scaler l'API indépendamment.

## DAGs disponibles

### 1. `bloodcells_train_dag_api` — Entraînement

Le DAG le plus important. Il orchestre un cycle complet d'entraînement.

```
check_api_health → trigger_training → wait_for_completion → decide_promotion
```

**Étapes** :

| Tâche | Action | Endpoint appelé |
|-------|--------|-----------------|
| `check_api_health` | Vérifie que l'API répond | `GET /health` |
| `trigger_training` | Lance l'entraînement avec les paramètres | `POST /ml/train` |
| `wait_for_completion` | Poll toutes les 5s jusqu'à COMPLETED/FAILED | `GET /ml/tasks/{id}` |
| `decide_promotion` | Si accuracy > seuil, promeut le modèle | `POST /mlflow/promote/{v}` |

**Paramètres configurables** (via l'UI Airflow) :
- `epochs`, `learning_rate`, `batch_size`
- `model_name` (resnet18, resnet34, cnn...)
- `dataset_path` (chemin du dataset à utiliser)

### 2. `bloodcells_evaluate_dag_api` — Évaluation

```
check_api_health → trigger_evaluation → wait_for_completion → check_metrics
```

Évalue le modèle actuel et compare les métriques à un seuil.

### 3. `bloodcells_batch_inference_dag_api` — Inférence batch

```
check_api_health → trigger_batch_inference → wait_for_completion → collect_results
```

Traite un dossier d'images et produit un fichier de prédictions JSON.

### 4. `bloodcells_data_validation_dag_api` — Validation de données

```
check_api_health → trigger_validation → wait_for_completion → report
```

Vérifie l'intégrité du dataset (structure, images corrompues, statistiques).

## Client HTTP (`dags/utils/api_client.py`)

Un client HTTP réutilisable encapsule toutes les interactions avec l'API :

```python
# Fonctions principales
check_api_health(api_url)          # GET /health
wait_for_api(api_url, timeout=60)  # Poll jusqu'à disponibilité
start_training(api_url, config)    # POST /ml/train → task_id
wait_for_task(api_url, task_id)    # Poll GET /ml/tasks/{id}
```

**Timeouts** :
- Requête HTTP : 30 secondes
- Poll entre vérifications : 5 secondes
- Attente maximale d'une tâche : 1 heure

**Gestion d'erreurs** : exceptions custom `APIError` et `TaskTimeoutError` pour des messages clairs dans les logs Airflow.

## Pattern d'interaction

Le pattern est le même pour tous les DAGs :

```
1. Vérifier que l'API est prête (health check)
2. Déclencher la tâche via POST → recevoir un task_id
3. Poller le statut via GET /ml/tasks/{task_id}
   - PENDING → continuer à poller
   - RUNNING → continuer à poller
   - COMPLETED → récupérer le résultat
   - FAILED → lever une exception (le DAG échoue)
4. Exploiter le résultat (promotion, rapport, etc.)
```

Ce pattern asynchrone permet à Airflow de ne pas bloquer de worker pendant l'exécution ML.

## Airflow Datasets

Les DAGs utilisent les **Airflow Datasets** pour exprimer des dépendances data-driven :

```python
# Le DAG d'entraînement produit un dataset
Dataset("bloodcells_trained_model")

# Le DAG d'évaluation se déclenche quand le modèle est entraîné
schedule=[Dataset("bloodcells_trained_model")]
```

Cela permet de chaîner automatiquement : entraînement → évaluation → promotion.

## Configuration Airflow

| Paramètre | Valeur |
|-----------|--------|
| Executor | LocalExecutor |
| Base de données | PostgreSQL |
| Interface web | `http://localhost:8081` |
| Login | airflow / airflow |
| Dossier DAGs | `/app/dags` (monté depuis `./dags/`) |
| Variable d'environnement | `FASTAPI_URL` pointe vers l'API |

## Comment déclencher un DAG manuellement

1. Ouvrir l'UI Airflow (`http://localhost:8081`)
2. Activer le DAG souhaité (toggle ON)
3. Cliquer sur "Trigger DAG" (bouton play)
4. Optionnel : modifier les paramètres dans le JSON de configuration
5. Suivre l'exécution dans la vue Graph ou Grid
