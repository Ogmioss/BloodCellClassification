# Guide de promotion des modeles

Ce guide explique comment promouvoir un modele entraine vers Staging ou Production
en utilisant les differentes interfaces disponibles : Swagger UI, MLflow UI, Airflow UI et curl.

## Rappel du cycle de vie

```
   Entrainement
       |
       v
   [Version N] -----> "None" (par defaut)
       |
       | (auto si accuracy >= 90%, ou manuel)
       v
   [Version N] -----> "Staging"
       |
       | (promotion manuelle uniquement)
       v
   [Version N] -----> "Production"  <-- modele servi par l'API
       |
       | (remplace par un nouveau modele)
       v
   [Version N] -----> "Archived"
```

**Regle** : la promotion vers Staging peut etre automatique (via Airflow).
La promotion vers Production est toujours manuelle.

---

## 1. Connaitre les versions disponibles

Avant de promouvoir, il faut identifier la version cible.

### Via Swagger UI (FastAPI)

1. Ouvrir http://localhost:8001/docs
2. Dans la section **MLflow**, deployer `GET /mlflow/models`
3. Cliquer **Try it out** puis **Execute**
4. La reponse liste toutes les versions avec leur stage actuel :

```json
{
  "model_name": "bloodcells-classifier",
  "versions": [
    {"version": "1", "stage": "Archived", "status": "READY"},
    {"version": "2", "stage": "Production", "status": "READY"},
    {"version": "3", "stage": "None", "status": "READY"}
  ]
}
```

### Via curl

```bash
curl http://localhost:8001/mlflow/models
```

### Via MLflow UI

1. Ouvrir http://localhost:5002
2. Dans le menu lateral, cliquer **Models**
3. Cliquer sur `bloodcells-classifier`
4. La page affiche toutes les versions, leur stage et leur run d'origine

---

## 2. Promouvoir via Swagger UI (FastAPI)

C'est la methode la plus intuitive pour une promotion manuelle.

### Etapes

1. Ouvrir http://localhost:8001/docs
2. Descendre a la section **MLflow**
3. Deployer `POST /mlflow/promote/{version}`
4. Cliquer **Try it out**
5. Remplir les champs :
   - **version** : le numero de version (ex: `3`)
   - **stage** : le stage cible (ex: `Production`)
6. Cliquer **Execute**

### Reponse attendue (200 OK)

```json
{
  "status": "success",
  "message": "Model version 3 promoted to Production",
  "model_name": "bloodcells-classifier"
}
```

### Ce qui se passe en coulisse

- Le modele version N est transitionne vers le stage demande
- L'ancien modele dans ce stage est automatiquement archive
- Le cache du service d'inference est vide
- La prochaine prediction utilisera le nouveau modele Production

### Stages disponibles

| Stage        | Signification                                   |
|--------------|------------------------------------------------|
| `None`       | Modele enregistre mais pas encore evalue        |
| `Staging`    | Modele valide, pret pour revue avant production |
| `Production` | Modele actif, servi par l'API `/predict`        |
| `Archived`   | Modele retire, conserve pour historique          |

---

## 3. Promouvoir via curl (ligne de commande)

### Promouvoir en Staging

```bash
curl -X POST "http://localhost:8001/mlflow/promote/3?stage=Staging"
```

### Promouvoir en Production

```bash
curl -X POST "http://localhost:8001/mlflow/promote/3?stage=Production"
```

### Archiver un modele

```bash
curl -X POST "http://localhost:8001/mlflow/promote/2?stage=Archived"
```

### Verifier le resultat

```bash
# Lister les versions et leur stage
curl http://localhost:8001/mlflow/models

# Verifier quel modele est charge par l'API
curl http://localhost:8001/model/info
```

La reponse de `/model/info` indique la source du modele actif :

```json
{
  "model_source": "mlflow",
  "mlflow": {
    "latest_version": "3",
    "tracking_uri": "http://mlflow:5000"
  }
}
```

---

## 4. Promouvoir via MLflow UI

MLflow offre une interface graphique pour gerer le cycle de vie des modeles.

### Etapes

1. Ouvrir http://localhost:5002
2. Cliquer **Models** dans le menu lateral
3. Cliquer sur `bloodcells-classifier`
4. Reperer la version a promouvoir dans la liste
5. Cliquer sur le numero de version (ex: `Version 3`)
6. Dans la page de la version, reperer le champ **Stage**
7. Cliquer sur le menu deroulant du stage
8. Selectionner le stage cible : `Staging` ou `Production`
9. Confirmer la transition dans la boite de dialogue

> **Note** : la promotion via MLflow UI ne vide pas le cache de l'API FastAPI.
> Apres une promotion via MLflow UI, relancer l'API ou appeler l'endpoint
> de promotion FastAPI pour forcer le rechargement :
>
> ```bash
> curl -X POST "http://localhost:8001/mlflow/promote/3?stage=Production"
> ```

### Comparer avant de promouvoir

1. Aller dans l'experience `bloodcells-classification`
2. Cocher les runs a comparer
3. Cliquer **Compare**
4. Verifier `test_accuracy`, `macro_f1`, la matrice de confusion
5. Promouvoir le meilleur candidat

---

## 5. Promotion automatique via Airflow

Le DAG `bloodcells_train_model_api` inclut une logique de promotion automatique
vers Staging apres l'entrainement.

### Comment ca marche

```
trigger_training
       |
       v
decide_promotion  <-- test_accuracy >= 0.90 ?
      / \
     /   \
    v     v
promote  skip
```

- Si `test_accuracy >= 90%` : le modele est automatiquement promu en **Staging**
- Sinon : le modele reste en stage `None`

### Lancer le pipeline complet depuis Airflow UI

1. Ouvrir http://localhost:8081 (admin / admin)
2. Trouver le DAG `bloodcells_train_model_api`
3. Activer le DAG (toggle ON a gauche)
4. Cliquer le bouton **Play** > **Trigger DAG**
5. Suivre l'execution dans la vue **Graph** :
   - Vert = tache reussie
   - Rouge = tache en erreur
   - Jaune = tache en cours
6. Cliquer sur la tache `decide_promotion` pour voir le choix effectue dans les logs
7. Cliquer sur `promote_model` ou `skip_promotion` pour voir le resultat

### Lancer le pipeline via l'API FastAPI

```bash
# Declencher le DAG
curl -X POST http://localhost:8001/pipelines/train \
  -H "Content-Type: application/json" \
  -d '{}'

# Reponse
{
  "status": "triggered",
  "dag_id": "bloodcells_train_model_api",
  "dag_run_id": "2026-03-12T10:30:45.123456+00:00"
}
```

### Apres la promotion automatique en Staging

La promotion en Production reste manuelle. Une fois le modele en Staging :

1. Verifier ses metriques dans MLflow UI (http://localhost:5002)
2. Tester le modele en Staging si necessaire
3. Promouvoir en Production via Swagger ou curl :

```bash
curl -X POST "http://localhost:8001/mlflow/promote/3?stage=Production"
```

---

## 6. Verification post-promotion

Apres toute promotion en Production, verifier que le modele est bien charge.

### Verifier le modele actif

```bash
curl http://localhost:8001/model/info
```

Verifier que `model_source` est `"mlflow"` et que la version correspond.

### Verifier la sante de l'API

```bash
curl http://localhost:8001/health
```

Verifier que `model_loaded` est `true`.

### Tester une prediction

```bash
curl -X POST http://localhost:8001/predict \
  -F "file=@chemin/vers/image_cellule.jpg"
```

Verifier que la prediction renvoie un resultat coherent avec les classes attendues.

---

## 7. Resume des interfaces

| Action                    | Swagger UI                | curl / API                          | MLflow UI            | Airflow UI             |
|---------------------------|---------------------------|-------------------------------------|----------------------|------------------------|
| Lister les versions       | GET /mlflow/models        | `curl .../mlflow/models`            | Models > bloodcells  | -                      |
| Promouvoir (manuel)       | POST /mlflow/promote/{v}  | `curl -X POST .../promote/3?...`    | Version > Stage menu | -                      |
| Promotion auto (Staging)  | -                         | -                                   | -                    | DAG train > auto       |
| Voir metriques            | GET /model/info           | `curl .../model/info`               | Experiment > Run     | Task logs              |
| Verifier modele charge    | GET /health               | `curl .../health`                   | -                    | -                      |
