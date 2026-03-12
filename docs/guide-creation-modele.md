# Guide de creation d'un nouveau modele

Ce guide explique comment creer et entrainer un nouveau modele de classification
de cellules sanguines, de la configuration jusqu'au deploiement.

---

## 1. Architecture des modeles

Le projet utilise un **pattern Factory** pour creer les modeles.
Tous les modeles heritent de `BaseClassifier` et implementent deux methodes :

- `get_num_features()` : nombre de features avant la couche de classification
- `set_classifier_head(num_classes)` : remplacement de la tete de classification

### Modeles disponibles

| Modele     | Cle `conf.yaml`     | Description                              |
|------------|----------------------|------------------------------------------|
| ResNet18   | `resnet18` (defaut)  | Transfer learning ImageNet, 11M params   |
| ResNet34   | `resnet34`           | ResNet plus profond, 21M params          |
| ResNet50   | `resnet50`           | ResNet avec bottleneck, 25M params       |
| CNN simple | `cnn`                | 3 couches conv (32→64→128), sans pretrain|

### Organisation des fichiers

```
src/models/
├── base_classifier.py       # Interface abstraite (BaseClassifier)
├── resnet_classifier.py     # Implementation ResNet (torchvision)
├── cnn_classifier.py        # Implementation CNN simple
└── model_factory.py         # Factory : cree le modele selon conf.yaml
```

---

## 2. Configurer le modele

Toute la configuration se fait dans `conf.yaml` a la racine du projet.

### Section `model`

```yaml
model:
    name: "resnet18"                    # Architecture (resnet18, resnet34, resnet50, cnn)
    pretrained: true                    # Utiliser les poids pretrained (ResNet uniquement)
    pretrained_weights: "IMAGENET1K_V1" # Version des poids ImageNet

    normalization:
        mean: [0.485, 0.456, 0.406]    # Normalisation ImageNet
        std: [0.229, 0.224, 0.225]
```

### Section `training`

```yaml
training:
    batch_size: 32          # Taille des batchs (augmenter si GPU le permet)
    img_size: 224           # Taille des images (224x224)
    epochs: 20              # Nombre d'epoques
    learning_rate: 0.001    # Taux d'apprentissage (Adam)
    num_workers: 0          # Workers DataLoader (0 = thread principal)
    seed: 42                # Graine aleatoire (reproductibilite)

    train_split: 0.7        # 70% entrainement
    val_split: 0.15         # 15% validation (15% test = reste)

    subset_size: 1400       # Sous-echantillon rapide (null = tout le dataset)
```

### Section `augmentation`

```yaml
augmentation:
    train:
        horizontal_flip: true
        vertical_flip: true
        rotation_degrees: 30
        color_jitter:
            brightness: 0.2
            contrast: 0.2
            saturation: 0.2

    val_test:
        enabled: false      # Pas d'augmentation en validation/test
```

### Exemples de configurations

**Entrainement rapide (dev/test) :**
```yaml
training:
    epochs: 5
    subset_size: 500
    batch_size: 16
```

**Entrainement complet (production) :**
```yaml
training:
    epochs: 50
    subset_size: null       # Tout le dataset
    batch_size: 64
    learning_rate: 0.0005
```

**CNN simple (sans pretrained) :**
```yaml
model:
    name: "cnn"
    pretrained: false
```

---

## 3. Preparer le dataset

Le dataset doit suivre la structure ImageFolder de PyTorch :
un dossier par classe, contenant les images correspondantes.

```
data/raw/bloodcells_dataset/
├── basophil/
│   ├── img001.jpg
│   └── ...
├── eosinophil/
├── erythroblast/
├── immature_granulocyte/
├── lymphocyte/
├── monocyte/
├── neutrophil/
└── platelet/
```

8 classes de cellules sanguines. Le nom du dossier = le label.

### Valider le dataset avant l'entrainement

```bash
# Via l'API
curl -X POST http://localhost:8001/ml/validate-data \
  -H "Content-Type: application/json" \
  -d '{"check_images": true}'
```

La validation verifie :
- La structure des dossiers
- Le nombre d'images par classe
- L'integrite des images (si `check_images: true`)

### Lister les datasets disponibles

```bash
curl http://localhost:8001/ml/datasets
```

---

## 4. Lancer l'entrainement

3 methodes disponibles, du plus simple au plus automatise.

### Methode A — Via Swagger UI

1. Ouvrir http://localhost:8001/docs
2. Section **ML Tasks**, deployer `POST /ml/train`
3. Cliquer **Try it out**
4. Remplir le body (tous les champs sont optionnels) :

```json
{
  "model_name": "resnet50",
  "epochs": 30,
  "learning_rate": 0.0005,
  "batch_size": 64,
  "dataset_path": "raw/bloodcells_dataset"
}
```

5. Cliquer **Execute**
6. Noter le `task_id` dans la reponse
7. Suivre l'avancement dans `GET /ml/tasks/{task_id}`

### Methode B — Via curl

```bash
# Lancer avec les parametres par defaut (conf.yaml)
curl -X POST http://localhost:8001/ml/train \
  -H "Content-Type: application/json" \
  -d '{}'

# Ou avec des parametres personnalises (modele + hyperparametres)
curl -X POST http://localhost:8001/ml/train \
  -H "Content-Type: application/json" \
  -d '{"model_name": "resnet50", "epochs": 30, "learning_rate": 0.0005, "batch_size": 64}'
```

Reponse :
```json
{
  "task_id": "abc123-...",
  "task_type": "train",
  "status": "pending",
  "message": "Training task started"
}
```

Suivre l'avancement :
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

> **Note** : une seule tache GPU (train/evaluate) peut tourner a la fois.
> L'API renvoie HTTP 409 si une tache est deja en cours.

### Methode C — Via script local (sans Docker)

```bash
uv run python -m src.pipe.train_model
```

Le script utilise directement `conf.yaml` sans possibilite d'override par parametres.

### Methode D — Via Airflow (orchestration complete)

1. Ouvrir http://localhost:8081 (admin / admin)
2. Activer le DAG `bloodcells_train_model_api`
3. Cliquer **Play** > **Trigger DAG w/ config**
4. Passer le JSON de configuration (tous les champs sont optionnels) :

```json
{
  "model_name": "resnet50",
  "epochs": 30,
  "learning_rate": 0.0005,
  "batch_size": 64,
  "dataset_path": "raw/bloodcells_dataset"
}
```

5. Le pipeline entraine, evalue, et promeut automatiquement en Staging si accuracy >= 90%

Voir [guide-promotion-modele.md](guide-promotion-modele.md) pour la suite du cycle de vie.

### Methode E — Via Streamlit

1. Ouvrir http://localhost:8502
2. Naviguer vers la page **Modele**
3. Cliquer **Lancer l'entrainement**
4. Suivre la progression en temps reel
5. Consulter les metriques et la matrice de confusion une fois termine

---

## 5. Ce qui se passe pendant l'entrainement

```
1. Chargement conf.yaml
2. Initialisation MLflow (experience + run)
3. Creation des transforms (augmentation train / normalisation val-test)
4. Chargement du dataset (ImageFolder + split stratifie)
5. Calcul des poids de classe (equilibrage des classes)
6. Creation du modele (ModelFactory)
7. Entrainement (Adam + CrossEntropyLoss)
   └── Pour chaque epoque :
       ├── Train sur le train set
       ├── Evaluation sur le val set
       └── Sauvegarde du meilleur checkpoint
8. Evaluation finale sur le test set
9. Enregistrement MLflow (metriques, artefacts, modele)
10. Enregistrement dans le Model Registry (version N, stage "None")
```

### Artefacts generes

| Artefact                  | Emplacement                             |
|---------------------------|-----------------------------------------|
| Checkpoint PyTorch        | `models/checkpoints/best_model.pth`     |
| Metriques JSON            | `models/checkpoints/metrics.json`       |
| Modele MLflow             | MLflow Registry (`bloodcells-classifier`)|
| Matrice de confusion      | MLflow Artifacts (image PNG)            |
| Parametres et metriques   | MLflow Tracking                         |

### Metriques enregistrees

| Metrique           | Description                          |
|--------------------|--------------------------------------|
| `test_accuracy`    | Accuracy sur le jeu de test          |
| `best_val_acc`     | Meilleure accuracy en validation     |
| `final_train_loss` | Loss du dernier epoch                |
| `final_train_acc`  | Accuracy du dernier epoch            |
| `macro_f1`         | F1-score macro-average               |
| `macro_precision`  | Precision macro-average              |
| `macro_recall`     | Recall macro-average                 |

Plus les metriques par classe : `{classe}_precision`, `{classe}_recall`, `{classe}_f1`.

---

## 6. Consulter les resultats

### Via MLflow UI

1. Ouvrir http://localhost:5002
2. Cliquer sur l'experience `bloodcells-classification`
3. Chaque ligne = un run d'entrainement
4. Cliquer sur un run pour voir :
   - **Parameters** : hyperparametres utilises
   - **Metrics** : courbes d'evolution (loss, accuracy)
   - **Artifacts** : checkpoint, matrice de confusion

### Comparer des runs

1. Cocher plusieurs runs dans la liste
2. Cliquer **Compare**
3. Comparer les metriques cote a cote

### Via l'API

```bash
# Informations sur le modele charge
curl http://localhost:8001/model/info

# Metriques du dernier entrainement
curl http://localhost:8001/metrics

# Versions dans le registry
curl http://localhost:8001/mlflow/models
```

### Via Streamlit

La page **Modele** affiche :
- Les metriques principales (accuracy, val accuracy, train accuracy)
- La matrice de confusion en heatmap
- L'accuracy par classe en bar chart
- Les metriques detaillees en JSON

---

## 7. Apres l'entrainement

Le modele entraine est enregistre dans le MLflow Model Registry avec le stage `None`.
Pour le rendre actif, il doit etre promu.

### Workflow recommande

```bash
# 1. Verifier les metriques dans MLflow UI
#    -> http://localhost:5002

# 2. Comparer avec les modeles precedents

# 3. Promouvoir en Staging pour validation
curl -X POST "http://localhost:8001/mlflow/promote/{version}?stage=Staging"

# 4. Tester le modele en Staging

# 5. Promouvoir en Production
curl -X POST "http://localhost:8001/mlflow/promote/{version}?stage=Production"

# 6. Verifier que le modele est charge
curl http://localhost:8001/health
```

Pour le detail complet de la promotion, voir [guide-promotion-modele.md](guide-promotion-modele.md).

---

## 8. Ajouter une nouvelle architecture

Pour integrer un nouveau type de modele (ex: EfficientNet) :

### Etape 1 — Creer le classifier

Creer `src/models/efficientnet_classifier.py` :

```python
import torch
import torch.nn as nn
from torchvision import models
from src.models.base_classifier import BaseClassifier


class EfficientNetClassifier(BaseClassifier):

    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        self.num_classes = num_classes

        if pretrained:
            self.model = models.efficientnet_b0(weights="IMAGENET1K_V1")
        else:
            self.model = models.efficientnet_b0(weights=None)

        self.set_classifier_head(num_classes)

    def get_num_features(self) -> int:
        return self.model.classifier[1].in_features

    def set_classifier_head(self, num_classes: int) -> None:
        self.model.classifier[1] = nn.Linear(self.get_num_features(), num_classes)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
```

### Etape 2 — Enregistrer dans la factory

Modifier `src/models/model_factory.py` :

```python
from src.models.efficientnet_classifier import EfficientNetClassifier

# Dans create_model(), ajouter :
elif model_name.startswith('efficientnet'):
    model = EfficientNetClassifier(
        num_classes=num_classes,
        pretrained=pretrained,
    )
```

### Etape 3 — Configurer

Modifier `conf.yaml` :

```yaml
model:
    name: "efficientnet_b0"
    pretrained: true
    pretrained_weights: "IMAGENET1K_V1"
```

### Etape 4 — Entrainer et verifier

```bash
curl -X POST http://localhost:8001/ml/train \
  -H "Content-Type: application/json" \
  -d '{}'
```

Le modele sera automatiquement cree par la factory et enregistre dans MLflow.

---

## 9. Conseils pratiques

### Choix du modele

| Besoin                         | Modele recommande  |
|--------------------------------|--------------------|
| Bon compromis vitesse/accuracy | `resnet18` (defaut)|
| Plus de capacite               | `resnet50`         |
| Entrainement tres rapide       | `cnn`              |
| Prototype / debug              | `cnn` + subset_size|

### Hyperparametres a ajuster en priorite

1. **`learning_rate`** : commencer a 0.001, reduire a 0.0005 ou 0.0001 si le training est instable
2. **`epochs`** : 20 pour un test rapide, 50+ pour un entrainement complet
3. **`batch_size`** : 32 par defaut, augmenter si la memoire GPU le permet
4. **`subset_size`** : `null` pour un entrainement sur tout le dataset

### Erreurs courantes

| Probleme                    | Cause probable                        | Solution                              |
|-----------------------------|---------------------------------------|---------------------------------------|
| HTTP 409 au lancement       | Une tache GPU est deja en cours       | Attendre la fin ou redemarrer l'API   |
| Accuracy tres basse (<50%)  | Learning rate trop haut               | Reduire a 0.0005 ou 0.0001           |
| Overfitting (val << train)  | Trop d'epoques, pas assez de donnees  | Augmenter augmentation, reduire epochs|
| Out of memory               | Batch size trop grand                 | Reduire batch_size (16 ou 8)          |
| MLflow non accessible       | Service MLflow non demarre            | `docker compose up -d mlflow`         |
