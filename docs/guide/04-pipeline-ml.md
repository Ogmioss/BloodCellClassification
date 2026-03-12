# 04 — Pipeline ML

Ce document détaille le cycle de vie complet du modèle : de la préparation des données à l'inférence en production.

## Vue d'ensemble du cycle

```
  Données brutes          Entraînement           Registry            Production
 ┌─────────────┐    ┌──────────────────┐    ┌─────────────┐    ┌─────────────┐
 │ data/       │───▶│ train_model.py   │───▶│ MLflow      │───▶│ FastAPI     │
 │ 8 classes   │    │ 20 epochs        │    │ v1, v2...   │    │ /predict    │
 │ ~17k images │    │ ResNet18         │    │ Staging →   │    │             │
 └─────────────┘    └──────────────────┘    │ Production  │    └─────────────┘
                                            └─────────────┘
```

## 1. Préparation des données

### DatasetService (`src/services/dataset_service.py`)

Le service charge les images avec `torchvision.ImageFolder` qui s'attend à la structure :

```
data/
├── basophil/
│   ├── img001.jpg
│   └── ...
├── eosinophil/
│   └── ...
└── ... (8 dossiers = 8 classes)
```

**Split des données** :

| Split | Proportion | Usage |
|-------|-----------|-------|
| Train | 70% | Entraînement du modèle |
| Validation | 15% | Suivi pendant l'entraînement (early stopping implicite) |
| Test | 15% | Évaluation finale |

**Gestion du déséquilibre de classes** : un `WeightedRandomSampler` est utilisé pour suréchantillonner les classes minoritaires pendant l'entraînement.

**Mode subset** : en développement, on peut limiter à N images (par défaut 1400 dans `conf.yaml`) pour des itérations rapides.

### DataTransformService (`src/services/data_transform_service.py`)

Deux pipelines de transformations :

**Entraînement** (augmentation) :
```
Resize(224) → RandomHorizontalFlip → RandomVerticalFlip
→ RandomRotation(30°) → ColorJitter(0.2, 0.2, 0.2)
→ ToTensor → Normalize(ImageNet)
```

**Validation/Test** (pas d'augmentation) :
```
Resize(224) → ToTensor → Normalize(ImageNet)
```

La normalisation utilise les statistiques d'ImageNet (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`) car le modèle est pré-entraîné sur ImageNet.

### DataValidationService (`src/services/data_validation_service.py`)

Avant d'entraîner, on peut valider le dataset :
- Vérification de la structure des dossiers
- Intégrité des images (décodage, dimensions)
- Statistiques et signalement des anomalies

## 2. Modèle

### Architecture

Le modèle principal est un **ResNet18 pré-entraîné sur ImageNet**. Le transfer learning fonctionne ainsi :

```
ResNet18 original (ImageNet, 1000 classes)
  ↓
Remplacement de la dernière couche FC :
  nn.Linear(512, 1000) → nn.Linear(512, 8)
  ↓
Fine-tuning sur le dataset BloodCells (8 classes)
```

### Factory Pattern (`src/models/model_factory.py`)

```python
model = ModelFactory.create_model(config, num_classes=8, device="cuda")
# config["model"]["name"] = "resnet18" → crée un ResNetClassifier
# config["model"]["name"] = "cnn"      → crée un CNNClassifier
```

Le pattern Factory permet de changer de backbone sans toucher au code d'entraînement.

### Hiérarchie des classes

```
nn.Module
└── BaseClassifier (ABC)
    ├── ResNetClassifier  ← modèle principal
    └── CNNClassifier     ← alternative légère
```

## 3. Entraînement

### Pipeline complète (`src/pipe/train_model.py`)

Le script `train_model.main()` orchestre l'entraînement de bout en bout :

```python
def main(dataset_path=None, config_overrides=None):
    # 1. Charger la configuration
    config = YamlLoader().get_config()
    if config_overrides:
        config.update(config_overrides)  # permet de surcharger epochs, lr, etc.

    # 2. Initialiser MLflow
    mlflow_service.start_run()
    mlflow_service.log_params(config)
    mlflow_service.log_git_info()

    # 3. Préparer les données
    transforms = DataTransformService(config)
    dataset = DatasetService(config).load_dataset(dataset_path)
    train_loader, val_loader, test_loader = dataset.get_loaders()

    # 4. Créer le modèle
    model = ModelFactory.create_model(config, num_classes=8)

    # 5. Entraîner
    training_service = TrainingService(config)
    training_service.train(model, train_loader, val_loader)

    # 6. Évaluer
    metrics = EvaluationService().evaluate(model, test_loader)

    # 7. Enregistrer dans MLflow
    mlflow_service.log_metrics(metrics)
    mlflow_service.log_model(model)
```

### Boucle d'entraînement (`src/services/training_service.py`)

```
Pour chaque epoch (1 à 20) :
  ├── _train_epoch()
  │   ├── model.train()
  │   ├── Pour chaque batch :
  │   │   ├── forward pass
  │   │   ├── CrossEntropyLoss (avec poids de classes)
  │   │   ├── backward pass
  │   │   └── Adam optimizer step
  │   └── Retourne loss et accuracy moyennes
  │
  ├── _validate_epoch()
  │   ├── model.eval() + torch.no_grad()
  │   ├── Pour chaque batch : forward pass
  │   └── Retourne val_loss et val_accuracy
  │
  ├── Si val_accuracy > best_accuracy :
  │   └── _save_checkpoint() → models/checkpoints/best_model.pth
  │
  └── (optionnel) Push métriques vers Pushgateway
```

**Hyperparamètres par défaut** (`conf.yaml`) :

| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| `epochs` | 20 | Nombre d'epochs |
| `learning_rate` | 0.001 | Taux d'apprentissage |
| `batch_size` | 32 | Taille des batches |
| `img_size` | 224 | Taille des images (pixels) |
| `seed` | 42 | Graine aléatoire (reproductibilité) |
| `train_split` | 0.7 | Proportion d'entraînement |
| `val_split` | 0.15 | Proportion de validation |
| `subset_size` | 1400 | Nombre d'images en mode rapide |

### Surcharge de configuration

Quand l'API ou Airflow lance un entraînement, ils peuvent surcharger n'importe quel paramètre :

```json
POST /ml/train
{
  "epochs": 10,
  "learning_rate": 0.0005,
  "batch_size": 64
}
```

Ces valeurs écrasent celles de `conf.yaml` pour cette exécution.

## 4. Évaluation

### EvaluationService (`src/services/evaluation_service.py`)

Après l'entraînement, le modèle est évalué sur le split de test :

| Métrique | Description |
|----------|-------------|
| Accuracy | Taux de bonnes classifications global |
| Precision | Précision par classe (moyenne pondérée) |
| Recall | Rappel par classe (moyenne pondérée) |
| F1-score | Moyenne harmonique precision/recall |
| Confusion matrix | Matrice 8x8 des prédictions vs réalité |

Les résultats sont logués dans MLflow et sauvegardés en `metrics.json`.

## 5. Inférence

### InferenceService (`src/services/inference_service.py`)

**Prédiction unitaire** :

```python
result = inference_service.predict_image(pil_image)
# → {"predicted_class": "neutrophil", "confidence": 0.94, "probabilities": {...}}
```

**Prédiction batch** (`src/pipe/batch_inference_pipeline.py`) :

```python
# Traite un dossier d'images
results = batch_inference(input_dir="./new_images/", output_file="predictions.json")
```

### Flux d'inférence détaillé

```
Image PIL (taille quelconque)
  ↓
Resize(224, 224)
  ↓
ToTensor → [0, 1] float
  ↓
Normalize(mean_imagenet, std_imagenet)
  ↓
Unsqueeze → batch de 1 : (1, 3, 224, 224)
  ↓
model.eval() + torch.no_grad()
  ↓
Forward pass → logits (1, 8)
  ↓
Softmax → probabilités (1, 8)
  ↓
argmax → index de la classe prédite
  ↓
CLASS_NAMES[index] → nom de la classe
```

## 6. Interprétabilité

Le module Grad-CAM (`src/utils/gradcam.py`) permet de visualiser quelles régions de l'image ont le plus contribué à la prédiction. C'est accessible depuis la page Streamlit "Interprétabilité".

Grad-CAM fonctionne en extrayant les gradients de la dernière couche convolutionnelle du ResNet et en les projetant sur l'image originale sous forme de heatmap.
