# Problemes connus et solutions

## PyTorch : "could not create a primitive"

**Contexte** : PyTorch 2.9.0+cpu, erreur lors des operations de convolution.

**Cause** : Incompatibilite du backend oneDNN (MKL-DNN) sur certains processeurs.

**Solution** : Desactiver oneDNN au debut de chaque script :

```python
import torch
torch.backends.mkldnn.enabled = False
```

Deja applique dans : `src/api/main.py`, `src/pipe/train_model.py`, `src/pages/3_Modele.py`,
`src/pages/4_Demo.py`, `src/models/model_factory.py`.

**Impact** : legere baisse de performance CPU. Negligeable pour l'inference.

---

## GradCAM : heatmaps similaires entre images

**Contexte** : Les heatmaps GradCAM semblaient identiques d'une image a l'autre.

**Cause** : Accumulation de gradients residuels entre les appels successifs.

**Solution** (appliquee dans `src/utils/gradcam_analyzer.py`) :

1. `model.zero_grad()` avant et apres chaque analyse
2. `torch.no_grad()` pour l'inference (economies memoire)
3. `.detach().cpu().numpy()` pour liberer la VRAM
4. Protection contre la division par zero sur la normalisation de la heatmap

```python
model.eval()
model.zero_grad()

with torch.no_grad():
    output = model(input_tensor)
    probs = F.softmax(output, dim=1)

model.zero_grad()
input_tensor.requires_grad = True
attribution = gradcam.attribute(input_tensor, target=pred_class)
model.zero_grad()

heatmap = upsampled_attr.squeeze().detach().cpu().numpy()
heatmap = np.maximum(heatmap, 0)
if heatmap.max() > 0:
    heatmap /= heatmap.max()
```

---

## MLflow : connexion timeout au demarrage

**Contexte** : L'API FastAPI bloque au demarrage si MLflow n'est pas encore pret.

**Solution** : Le `MLflowService` a un timeout de 5 secondes. Si MLflow est inaccessible,
l'API demarre quand meme et charge le modele depuis le checkpoint local.

**Verification** :
```bash
curl http://localhost:8001/health
# model_source indique "checkpoint" si MLflow est down, "mlflow" sinon
```

---

## Airflow : DAGs non visibles

**Cause possible** : Les DAGs ne sont pas montes correctement.

**Verification** :
```bash
docker exec bloodcell-airflow-webserver ls /app/dags/
```

**Solution** :
```bash
# Forcer la re-detection
docker exec bloodcell-airflow-scheduler airflow dags reserialize

# Ou redemarrer
docker compose -f docker/docker-compose.yml restart airflow-webserver airflow-scheduler
```

---

## Entrainement : "409 Conflict"

**Cause** : Une tache GPU (train ou evaluate) est deja en cours. Une seule a la fois.

**Verification** :
```bash
curl http://localhost:8001/ml/tasks/{task_id}
```

**Solution** : Attendre la fin de la tache en cours, ou redemarrer l'API.

---

## MinIO : buckets non crees

**Cause** : Le container `minio-init` a echoue.

**Verification** :
```bash
docker logs bloodcell-minio-init
```

**Solution** :
```bash
docker compose -f docker/docker-compose.yml restart minio-init
```

---

## Modele : 503 sur /predict

**Cause** : Aucun modele disponible (ni dans MLflow, ni en checkpoint local).

**Solution** : Entrainer un modele :
```bash
# Via l'API
curl -X POST http://localhost:8001/ml/train -H "Content-Type: application/json" -d '{}'

# Ou en local
uv run python -m src.pipe.train_model
```

---

## mlflow-serve : container qui crash en boucle

**Cause** : Le service `mlflow-serve` necessite un modele en stage "Production" dans le registry.
Si aucun modele n'est promu, il echoue au demarrage.

**Solution** : Entrainer un modele, puis le promouvoir :
```bash
curl -X POST "http://localhost:8001/mlflow/promote/{version}?stage=Production"
```

Le `restart: on-failure` dans docker-compose le relancera automatiquement apres la promotion.
