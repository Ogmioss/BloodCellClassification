# Fix PyTorch "could not create a primitive" Error

## Problème

Lors de l'exécution de l'application Streamlit avec PyTorch 2.9.0+cpu, l'erreur suivante se produit:

```
RuntimeError: could not create a primitive
```

Cette erreur se produit lors de l'exécution d'opérations de convolution, même sur CPU.

## Cause racine

Le problème est causé par le backend **oneDNN (anciennement MKL-DNN)** de PyTorch 2.9.0+cpu qui a des incompatibilités sur certains processeurs ou configurations.

## Solution appliquée

Désactiver le backend oneDNN en ajoutant au début de chaque script:

```python
import torch
torch.backends.mkldnn.enabled = False
```

## Fichiers modifiés

- `src/app.py` - Application principale (fix global)
- `src/pages/3_Modele.py` - Page modèle
- `src/pages/4_Demo.py` - Page démo
- `src/models/model_factory.py` - Désactivation de MPS (incompatible aussi)

## Impact sur les performances

La désactivation de oneDNN peut légèrement réduire les performances sur CPU, mais garantit la compatibilité et le bon fonctionnement de l'application.

## Alternative

Si les performances CPU sont critiques, considérer:
1. Downgrade vers PyTorch 2.8.x ou 2.7.x
2. Utiliser une version avec support CUDA si GPU disponible
3. Compiler PyTorch depuis les sources avec des optimisations spécifiques

## Références

- PyTorch Issue Tracker: oneDNN/MKL-DNN compatibility issues
- Date du fix: 21 octobre 2025
