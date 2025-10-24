# Implémentation de la Matrice de Confusion

## Vue d'ensemble

La matrice de confusion a été ajoutée au projet BloodCellClassification pour fournir une analyse détaillée des performances du modèle par classe. Cette fonctionnalité permet d'identifier les classes difficiles à classifier et les confusions fréquentes entre classes.

## Architecture

### 1. Service d'Évaluation (`src/services/evaluation_service.py`)

Nouvelle méthode ajoutée:

```python
def compute_confusion_matrix(
    self, 
    predictions: List[int], 
    labels: List[int],
    num_classes: int
) -> np.ndarray:
    """
    Compute confusion matrix from predictions and labels.
    
    Returns:
        Confusion matrix as numpy array of shape (num_classes, num_classes)
    """
    return confusion_matrix(labels, predictions, labels=list(range(num_classes)))
```

**Responsabilité**: Calcul de la matrice de confusion à partir des prédictions et des vraies étiquettes.

### 2. Pipeline d'Entraînement (`src/pipe/train_model.py`)

- Calcule la matrice de confusion sur le test set après l'entraînement
- Sauvegarde la matrice dans `metrics.json` au format liste imbriquée

### 3. Script d'Évaluation (`src/pipe/evaluate_model.py`)

- Charge un modèle existant
- Évalue sur train/val/test sets
- Calcule et sauvegarde la matrice de confusion

**Usage**: `uv run evaluate-model`

### 4. Interface Streamlit (`src/pages/3_Modele.py`)

#### Visualisations ajoutées:

1. **Heatmap interactive Plotly**
   - Colorscale bleue
   - Annotations avec les valeurs exactes
   - Axes: Prédictions (x) vs Vraies étiquettes (y)
   - Taille: 800x600 pixels

2. **Tableau de précision par classe**
   - Colonnes: Classe, Correct, Total, Précision
   - Trié par ordre des classes
   - Format: DataFrame pandas

## Format des Données

### Structure de `metrics.json`

```json
{
  "best_val_acc": 0.6810,
  "final_train_loss": 0.0,
  "final_train_acc": 0.7211,
  "accuracy": 0.7251,
  "class_names": [
    "basophil",
    "eosinophil",
    "erythroblast",
    "ig",
    "lymphocyte",
    "monocyte",
    "neutrophil",
    "platelet"
  ],
  "confusion_matrix": [
    [13, 0, 0, 1, 0, 0, 0, 0],
    [1, 37, 0, 0, 0, 0, 0, 0],
    ...
  ]
}
```

### Interprétation de la Matrice

- **Lignes**: Vraies classes
- **Colonnes**: Classes prédites
- **Diagonale**: Prédictions correctes
- **Hors diagonale**: Confusions

Exemple: `confusion_matrix[i][j]` = nombre d'échantillons de la classe `i` prédits comme classe `j`

## Résultats Actuels

### Performance Globale
- **Test Accuracy**: 72.5%
- **Validation Accuracy**: 68.1%
- **Training Accuracy**: 72.1%

### Précision par Classe

| Classe | Précision | Observations |
|--------|-----------|--------------|
| Eosinophil | 97.4% | ✅ Excellente |
| Basophil | 92.9% | ✅ Très bonne |
| Neutrophil | 85.0% | ✅ Bonne |
| Lymphocyte | 75.0% | ⚠️ Moyenne |
| Erythroblast | 68.4% | ⚠️ Moyenne |
| IG | 59.5% | ❌ Difficile |
| Monocyte | 55.6% | ❌ Difficile |
| Platelet | 41.4% | ❌ Très difficile |

### Confusions Principales

1. **IG → Basophil**: 14 cas (37.8% des IG)
2. **Platelet → Basophil**: 8 cas (27.6% des Platelet)
3. **Monocyte → IG**: 7 cas (38.9% des Monocyte)
4. **Neutrophil → Basophil**: 6 cas (15% des Neutrophil)
5. **Platelet → Erythroblast**: 4 cas (13.8% des Platelet)

## Améliorations Possibles

### Court terme
1. **Augmentation de données ciblée** pour les classes difficiles (Platelet, Monocyte, IG)
2. **Pondération des classes** ajustée pour réduire les confusions
3. **Analyse des images mal classées** pour identifier les patterns

### Moyen terme
1. **Métriques supplémentaires**: Precision, Recall, F1-score par classe
2. **Courbes ROC** multi-classes
3. **Analyse des erreurs** avec visualisation des images

### Long terme
1. **Ensemble de modèles** pour améliorer les classes difficiles
2. **Attention mechanisms** pour se concentrer sur les caractéristiques distinctives
3. **Transfer learning** avec des modèles pré-entraînés sur des données médicales

## Utilisation

### Générer les métriques pour un modèle existant

```bash
uv run evaluate-model
```

### Visualiser dans Streamlit

1. Lancer l'application: `./start.sh`
2. Naviguer vers la page "Modèle"
3. La matrice de confusion s'affiche automatiquement si disponible

### Accéder aux données programmatiquement

```python
import json
import numpy as np

# Charger les métriques
with open('models/checkpoints/metrics.json') as f:
    metrics = json.load(f)

# Récupérer la matrice
confusion_mat = np.array(metrics['confusion_matrix'])
class_names = metrics['class_names']

# Analyser
for i, class_name in enumerate(class_names):
    total = confusion_mat[i].sum()
    correct = confusion_mat[i, i]
    accuracy = correct / total
    print(f"{class_name}: {accuracy:.1%}")
```

## Dépendances

- `scikit-learn`: Pour `confusion_matrix()`
- `numpy`: Pour les opérations matricielles
- `plotly`: Pour la visualisation heatmap
- `pandas`: Pour le tableau de précision
- `streamlit`: Pour l'interface web

Toutes ces dépendances sont déjà incluses dans le projet.

## Tests

Un test de validation a été créé pour vérifier:
- ✅ Chargement de la matrice depuis JSON
- ✅ Création du DataFrame pandas
- ✅ Génération de la heatmap Plotly
- ✅ Calcul de la précision par classe

## Références

- [Scikit-learn Confusion Matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)
- [Plotly Annotated Heatmap](https://plotly.com/python/annotated-heatmap/)
- [Understanding Confusion Matrix](https://en.wikipedia.org/wiki/Confusion_matrix)
