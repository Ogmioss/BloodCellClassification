# Guide d'entrainement des modeles

Ce guide explique comment entrainer un modele de classification de cellules sanguines,
du lancement local jusqu'a l'orchestration automatisee avec Airflow et MLflow.

## Architecture globale

```
                    +------------------+
                    |   Airflow UI     |  Declenchement manuel ou programme
                    |  (port 8081)     |
                    +--------+---------+
                             |
                    Trigger DAG via API
                             |
                    +--------v---------+
                    |   FastAPI        |  Execution du training PyTorch
                    |  (port 8001)     |  (seul container avec GPU/PyTorch)
                    +--------+---------+
                             |
              +--------------+--------------+
              |                             |
     +--------v---------+         +--------v---------+
     |   MLflow          |         |   MinIO S3       |
     |  (port 5002)      |         |  (port 9002)     |
     |  Tracking server  |         |  Stockage objets |
     +-------------------+         +------------------+
```

**Principe cle** : Airflow ne fait PAS de ML. Il appelle l'API FastAPI via HTTP.
Seul le container `api` a PyTorch installe (~2 GB). Airflow reste leger (~500 MB).

---

## 1. Prerequis

### Lancer l'infrastructure Docker

```bash
# Stack complete (Airflow + MLflow + MinIO + API + Streamlit)
docker compose -f docker/docker-compose.yml up -d

# Verifier que tout est lance
docker compose -f docker/docker-compose.yml ps
```

### Verifier la sante des services

| Service        | URL                          | Identifiants    |
|----------------|------------------------------|-----------------|
| FastAPI (docs) | http://localhost:8001/docs   | -               |
| Airflow        | http://localhost:8081        | admin / admin   |
| MLflow         | http://localhost:5002        | -               |
| MinIO Console  | http://localhost:9003        | minio / minio123|
| Streamlit      | http://localhost:8502        | -               |

### Verifier la sante de l'API

```bash
curl http://localhost:8001/health
```

Reponse attendue :
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cpu"
}
```

---

## 2. Le dataset

Le dataset doit etre place dans `data/raw/bloodcells_dataset/` avec cette structure :

```
data/raw/bloodcells_dataset/
  basophil/
    img001.jpg
    img002.jpg
    ...
  eosinophil/
  erythroblast/
  granulocyte/
  lymphocyte/
  monocyte/
  neutrophil/
  platelet/
```

8 classes de cellules sanguines. Le nom du dossier = le label de la classe.

### Configuration du dataset (conf.yaml)

```yaml
training:
  batch_size: 32        # Taille des batchs
  img_size: 224         # Taille des images (224x224)
  epochs: 20            # Nombre d'epoques
  learning_rate: 0.001  # Taux d'apprentissage
  train_split: 0.7      # 70% train
  val_split: 0.15       # 15% validation, 15% test
  subset_size: 1400     # Sous-echantillon (dev rapide)
```

`subset_size: 1400` limite le nombre d'images pour accelerer le developpement.
Passer a `null` ou une valeur elevee pour un entrainement complet.

---

## 3. Entrainement : 3 methodes

### Methode A — Script local (sans Docker)

```bash
# Depuis la racine du projet
uv run python -m src.pipe.train_model
```

Le script :
1. Charge `conf.yaml`
2. Initialise MLflow (local `./mlruns/`)
3. Charge le dataset, cree les data loaders
4. Entraine ResNet18 avec les augmentations configurees
5. Evalue sur le jeu de test
6. Sauvegarde le checkpoint dans `models/checkpoints/best_model.pth`
7. Enregistre le modele dans le MLflow Model Registry

### Methode B — Via l'API FastAPI (Docker)

```bash
# Lancer un entrainement avec les parametres par defaut (conf.yaml)
curl -X POST http://localhost:8001/ml/train \
  -H "Content-Type: application/json" \
  -d '{}'

# Ou avec des parametres personnalises
curl -X POST http://localhost:8001/ml/train \
  -H "Content-Type: application/json" \
  -d '{"epochs": 10, "learning_rate": 0.0005, "batch_size": 16}'
```

Reponse :
```json
{
  "task_id": "abc123-...",
  "status": "pending",
  "message": "Training task started"
}
```

L'entrainement s'execute en arriere-plan. Suivre l'avancement :

```bash
curl http://localhost:8001/ml/tasks/{task_id}
```

Reponse en cours :
```json
{"task_id": "abc123-...", "status": "running"}
```

Reponse terminee :
```json
{
  "task_id": "abc123-...",
  "status": "completed",
  "result": {
    "test_accuracy": 0.9234,
    "best_val_acc": 0.9156,
    "model_version": "3",
    "mlflow_run_id": "a1b2c3d4..."
  }
}
```

### Methode C — Via Airflow (orchestration automatisee)

C'est la methode recommandee pour la production. Voir la section 5.

---

## 4. Comprendre MLflow dans ce projet

### Qu'est-ce que MLflow fait ici ?

MLflow remplit 3 roles :

1. **Tracking** : enregistre les parametres, metriques et artefacts de chaque run
2. **Model Registry** : versionne les modeles et gere leur cycle de vie (Staging / Production)
3. **Artifact Store** : stocke les fichiers (checkpoints, images) dans MinIO S3

### Ce qui est enregistre a chaque entrainement

| Type       | Donnees                                                        |
|------------|----------------------------------------------------------------|
| Parametres | model.name, training.epochs, training.learning_rate, etc.      |
| Metriques  | test_accuracy, best_val_acc, final_train_loss, macro_f1, etc.  |
| Artefacts  | best_model.pth, metrics.json, confusion_matrix.png             |
| Tags       | git.commit, git.branch, training_env, triggered_by             |
| Modele     | Enregistre dans le registry comme `bloodcells-classifier` vN   |

### Consulter les resultats dans MLflow UI

1. Ouvrir http://localhost:5002
2. Cliquer sur l'experience `bloodcells-classification`
3. Chaque ligne = un run d'entrainement
4. Cliquer sur un run pour voir :
   - Les parametres (onglet Parameters)
   - Les metriques (onglet Metrics) avec courbes d'evolution
   - Les artefacts (onglet Artifacts) : checkpoint, confusion matrix, etc.

### Comparer des runs

1. Selectionner plusieurs runs (cases a cocher)
2. Cliquer "Compare"
3. Comparer les metriques cote a cote

### Cycle de vie des modeles (Model Registry)

```
   Entrainement
       |
       v
   [Version N] -----> "None" (par defaut)
       |
       | (accuracy >= 90%)
       v
   [Version N] -----> "Staging"
       |
       | (promotion manuelle)
       v
   [Version N] -----> "Production"
       |
       | (nouveau modele promu)
       v
   [Version N] -----> "Archived"
```

#### Voir les versions enregistrees

```bash
curl http://localhost:8001/mlflow/models
```

#### Promouvoir un modele manuellement

```bash
# Promouvoir la version 3 en Production
curl -X POST "http://localhost:8001/mlflow/promote/3?stage=Production"

# Promouvoir en Staging
curl -X POST "http://localhost:8001/mlflow/promote/3?stage=Staging"
```

### Configuration MLflow (conf.yaml)

```yaml
mlflow:
  tracking_uri: "./mlruns"                    # Local (Docker override: http://mlflow:5000)
  experiment_name: "bloodcells-classification" # Nom de l'experience
  model_name: "bloodcells-classifier"          # Nom dans le registry
  auto_log: false                              # Logging manuel pour plus de controle
```

En Docker, la variable `MLFLOW_TRACKING_URI=http://mlflow:5000` surcharge `conf.yaml`.

### Stockage des artefacts (MinIO S3)

MLflow stocke les artefacts dans MinIO, un stockage S3-compatible :

- **Bucket** : `mlflow-artifacts`
- **Console** : http://localhost:9003 (minio / minio123)
- Les artefacts sont organises par `run_id`

---

## 5. Comprendre Airflow dans ce projet

### Qu'est-ce qu'Airflow fait ici ?

Airflow orchestre les pipelines ML **sans executer de code ML**.
Il agit comme un chef d'orchestre qui appelle l'API FastAPI.

### Les 4 DAGs disponibles

| DAG                                    | Declenchement | Description                          |
|----------------------------------------|---------------|--------------------------------------|
| `bloodcells_train_model_api`           | Manuel        | Entraine un modele + promotion auto  |
| `bloodcells_evaluate_model_api`        | Manuel        | Evalue le modele charge              |
| `bloodcells_batch_inference_api`       | Manuel        | Predictions sur un dossier d'images  |
| `bloodcells_data_validation_api`       | Manuel        | Valide la qualite du dataset         |

### DAG d'entrainement en detail

```
check_api_health
       |
       v
trigger_training         <-- POST /ml/train + polling /ml/tasks/{id}
       |
       v
decide_promotion         <-- Si accuracy >= 90% : promouvoir
      / \
     /   \
    v     v
promote  skip            <-- POST /mlflow/promote/{version}?stage=Staging
    \   /
     \ /
      v
training_complete
```

**Etapes :**

1. **check_api_health** : Verifie que FastAPI repond (retry 30x, 2s interval)
2. **trigger_training** : Envoie `POST /ml/train`, attend la fin (polling toutes les 5s, max 1h)
3. **decide_promotion** : Si `test_accuracy >= 0.90`, branche vers `promote_model`, sinon `skip_promotion`
4. **promote_model** : Envoie `POST /mlflow/promote/{version}?stage=Staging`
5. **training_complete** : Log final

### Lancer un DAG depuis l'UI Airflow

1. Ouvrir http://localhost:8081 (admin / admin)
2. Dans la liste des DAGs, trouver `bloodcells_train_model_api`
3. Activer le DAG (toggle ON)
4. Cliquer sur le bouton "Play" (Trigger DAG)
5. Optionnel : passer des parametres JSON dans "Trigger DAG w/ config"
6. Suivre l'execution dans la vue "Graph" ou "Grid"

### Lancer un DAG via l'API FastAPI

```bash
# Trigger le DAG d'entrainement
curl -X POST http://localhost:8001/pipelines/train \
  -H "Content-Type: application/json" \
  -d '{}'

# Verifier le statut
curl "http://localhost:8001/pipelines/status/bloodcells_train_model_api/{dag_run_id}"
```

### Lancer un DAG via la CLI Airflow

```bash
docker exec bloodcell-airflow-webserver \
  airflow dags trigger bloodcells_train_model_api
```

### Consulter les logs d'un DAG

1. Airflow UI > cliquer sur le DAG > cliquer sur un run > cliquer sur une tache
2. Onglet "Log" : logs complets de la tache
3. Les logs incluent la progression de l'entrainement (accuracy par epoque)

---

## 6. Workflow complet recommande

### Premier entrainement

```bash
# 1. Lancer l'infrastructure
docker compose -f docker/docker-compose.yml up -d

# 2. Verifier que tout est up
curl http://localhost:8001/health

# 3. Lancer un entrainement via l'API
curl -X POST http://localhost:8001/ml/train -H "Content-Type: application/json" -d '{}'

# 4. Suivre l'avancement
curl http://localhost:8001/ml/tasks/{task_id}

# 5. Consulter les resultats dans MLflow
# -> http://localhost:5002

# 6. Si satisfait, promouvoir en Production
curl -X POST "http://localhost:8001/mlflow/promote/{version}?stage=Production"
```

### Entrainement iteratif avec Airflow

```bash
# 1. Activer le DAG dans Airflow UI (http://localhost:8081)
# 2. Trigger le DAG
# 3. Le DAG entraine, evalue, et promeut automatiquement si accuracy >= 90%
# 4. Comparer les runs dans MLflow UI (http://localhost:5002)
# 5. Promouvoir manuellement le meilleur modele en Production si necessaire
```

### Experimenter avec des hyperparametres

```bash
# Essayer differentes configurations
curl -X POST http://localhost:8001/ml/train \
  -H "Content-Type: application/json" \
  -d '{"epochs": 30, "learning_rate": 0.0001, "batch_size": 64}'

# Comparer dans MLflow UI
```

---

## 7. Depannage

### L'entrainement ne demarre pas

```bash
# Verifier l'API
curl http://localhost:8001/health

# Verifier les logs
docker logs bloodcell-api --tail 50

# Verifier qu'aucune tache n'est deja en cours (1 seule GPU task a la fois)
curl http://localhost:8001/ml/tasks/{task_id}
```

### MLflow n'enregistre rien

```bash
# Verifier que MLflow est accessible
curl http://localhost:5002/api/2.0/mlflow/experiments/search

# Verifier la connexion MinIO
docker logs bloodcell-mlflow --tail 20
```

### Airflow ne trouve pas les DAGs

```bash
# Verifier que les DAGs sont montes
docker exec bloodcell-airflow-webserver ls /app/dags/

# Forcer la re-detection
docker exec bloodcell-airflow-scheduler airflow dags reserialize
```

### Le modele n'est pas charge au demarrage de l'API

L'API charge le modele dans cet ordre :
1. MLflow Model Registry (si `MLFLOW_TRACKING_URI` est configure)
2. Checkpoint local (`models/checkpoints/best_model.pth`)
3. Si rien n'est disponible : l'API repond 503 sur `/predict`

Entrainer un modele ou placer un checkpoint pour resoudre le probleme.
