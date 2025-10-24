# Fix GradCAM - Résolution du problème de variabilité des heatmaps

**Date:** 24 octobre 2024  
**Problème:** Les heatmaps GradCAM semblaient similaires d'une image à l'autre, suggérant une pollution des gradients ou un problème de cache.

## 🔴 Problèmes identifiés

### 1. Pollution des gradients entre les analyses successives

**Localisation:** `src/utils/gradcam_analyzer.py`, fonction `gradcam_analysis()`

**Problème:**
```python
# Code original - PROBLÉMATIQUE
def gradcam_analysis(...):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)
    
    # Pas de nettoyage des gradients
    output = model(input_tensor)
    probs = F.softmax(output, dim=1)
    
    # GradCAM sans réinitialisation
    gradcam = LayerGradCam(model, target_layer)
    attribution = gradcam.attribute(input_tensor, target=pred_class)
```

**Conséquences:**
- Les gradients s'accumulent entre les appels successifs
- Les hooks de Captum peuvent laisser des états résiduels
- Les heatmaps deviennent de plus en plus similaires

### 2. Cache Streamlit trop agressif

**Localisation:** `src/pages/5_Ameliorations_potentielles.py`, fonction `load_model()`

**Problème:**
```python
# Code original - PROBLÉMATIQUE
@st.cache_resource
def load_model(checkpoint_file: str) -> object:
    return load_resnet_model(checkpoint_path, num_classes=8)

model = load_model(checkpoint_filename)
```

**Conséquences:**
- Le modèle est chargé une seule fois et réutilisé pour toutes les images
- L'état interne du modèle n'est jamais réinitialisé
- Les gradients résiduels persistent entre les analyses

### 3. Absence de gestion du contexte torch

**Problème:**
- Pas de `torch.no_grad()` pour l'inférence (gaspillage de mémoire)
- Pas de `model.zero_grad()` entre les analyses
- Pas de `.cpu()` pour libérer la mémoire GPU

### 4. Division par zéro potentielle

**Problème:**
```python
# Code original - RISQUE
heatmap /= heatmap.max()
```

Si `heatmap.max() == 0`, cela provoque une division par zéro.

## ✅ Solutions implémentées

### 1. Nettoyage explicite des gradients

**Fichier:** `src/utils/gradcam_analyzer.py`

```python
def gradcam_analysis(...):
    # Ensure model is in eval mode and gradients are cleared
    model.eval()
    model.zero_grad()
    
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)
    
    # Prediction with no_grad context for efficiency
    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)
        pred_class = probs.argmax(dim=1).item()
        pred_conf = probs[0, pred_class].item() * 100
    
    # Clear any residual gradients before GradCAM
    model.zero_grad()
    
    # Grad-CAM computation (requires gradients)
    target_layer = model.layer4[-1]
    gradcam = LayerGradCam(model, target_layer)
    
    # Enable gradients only for attribution
    input_tensor.requires_grad = True
    attribution = gradcam.attribute(input_tensor, target=pred_class)
    
    # Clean up gradients after attribution
    model.zero_grad()
    
    upsampled_attr = LayerAttribution.interpolate(attribution, input_tensor.shape[2:])
    heatmap = upsampled_attr.squeeze().detach().cpu().numpy()
    heatmap = np.maximum(heatmap, 0)
    
    # Avoid division by zero
    if heatmap.max() > 0:
        heatmap /= heatmap.max()
    else:
        heatmap = np.zeros_like(heatmap)
```

**Améliorations:**
- ✅ `model.zero_grad()` avant et après chaque analyse
- ✅ `torch.no_grad()` pour l'inférence (économie de mémoire)
- ✅ `.cpu()` pour libérer la mémoire GPU
- ✅ Protection contre la division par zéro
- ✅ `input_tensor.requires_grad = True` explicite pour GradCAM

### 2. Amélioration du cache Streamlit

**Fichier:** `src/pages/5_Ameliorations_potentielles.py`

```python
# Load model with proper caching
@st.cache_resource
def load_model(checkpoint_file: str) -> object:
    yaml_loader = YamlLoader()
    checkpoint_dir = yaml_loader.get_nested_value(
        'paths.models.checkpoints', 
        './models/checkpoints'
    )
    checkpoint_path = Path(checkpoint_dir) / checkpoint_file
    model = load_resnet_model(checkpoint_path, num_classes=8)
    # Ensure clean state
    model.eval()
    model.zero_grad()
    return model

model = load_model(checkpoint_filename)
transform = get_image_transform()

# Add option to clear cache if needed
if st.sidebar.button(f"🔄 Recharger le modèle ({tab_title[:20]}...)", key=f"reload_{checkpoint_filename}"):
    st.cache_resource.clear()
    st.rerun()
```

**Améliorations:**
- ✅ Initialisation propre du modèle dans le cache
- ✅ Bouton pour forcer le rechargement du modèle
- ✅ Nettoyage du cache Streamlit

### 3. Gestion du sampling aléatoire

```python
if selected_class and class_images.get(selected_class):
    # Use a seed for reproducibility but allow refresh
    if st.button("🎲 Nouvelles images aléatoires", key=f"refresh_{checkpoint_filename}"):
        st.session_state[f"seed_{checkpoint_filename}"] = random.randint(0, 10000)
    
    seed = st.session_state.get(f"seed_{checkpoint_filename}", 42)
    random.seed(seed)
    
    sample_images = random.sample(
        class_images[selected_class],
        min(3, len(class_images[selected_class]))
    )
    st.info(
        f"Analyse Grad-CAM sur **3 images aléatoires** "
        f"de la classe **{selected_class}** (seed: {seed})."
    )
```

**Améliorations:**
- ✅ Seed reproductible pour le debugging
- ✅ Bouton pour rafraîchir les images
- ✅ Affichage du seed pour la traçabilité

## 📊 Impact attendu

### Avant les corrections:
- Heatmaps très similaires entre images différentes
- Accumulation de gradients résiduels
- Comportement non déterministe

### Après les corrections:
- ✅ Heatmaps distinctes pour chaque image
- ✅ Pas d'accumulation de gradients
- ✅ Comportement reproductible (avec seed)
- ✅ Meilleure utilisation de la mémoire
- ✅ Robustesse accrue (protection division par zéro)

## 🧪 Tests recommandés

Pour vérifier que les corrections fonctionnent:

1. **Test de variabilité:**
   ```bash
   uv run python test_gradcam_issue.py
   ```
   Vérifier que:
   - Les heatmaps ont des distributions différentes
   - La corrélation entre heatmaps est faible (<0.8)
   - Les prédictions sont cohérentes

2. **Test dans Streamlit:**
   - Sélectionner une classe
   - Observer 3 images différentes
   - Cliquer sur "🎲 Nouvelles images aléatoires"
   - Vérifier que les heatmaps changent significativement

3. **Test de reproductibilité:**
   - Noter le seed affiché
   - Recharger la page
   - Vérifier que les mêmes images produisent les mêmes heatmaps

## 🔍 Diagnostic technique

### Pourquoi les gradients s'accumulaient?

1. **PyTorch conserve les gradients par défaut:**
   - Chaque forward pass calcule des gradients
   - Sans `zero_grad()`, ils s'additionnent

2. **Captum crée des hooks:**
   - LayerGradCam attache des hooks au modèle
   - Ces hooks peuvent laisser des états résiduels

3. **Cache Streamlit:**
   - Le modèle est partagé entre toutes les analyses
   - L'état interne n'est jamais réinitialisé

### Pourquoi `.cpu()` est important?

```python
heatmap = upsampled_attr.squeeze().detach().cpu().numpy()
```

- `.detach()`: Détache le tensor du graphe de calcul
- `.cpu()`: Transfère le tensor vers la RAM (libère la VRAM)
- `.numpy()`: Convertit en numpy array

Sans `.cpu()`, les tensors restent en VRAM et peuvent causer des fuites mémoire.

## 📚 Références

- [Captum LayerGradCam Documentation](https://captum.ai/api/layer.html#layergradcam)
- [PyTorch Autograd Documentation](https://pytorch.org/docs/stable/autograd.html)
- [Streamlit Caching Best Practices](https://docs.streamlit.io/library/advanced-features/caching)

## 🎯 Prochaines étapes

- [ ] Ajouter des tests unitaires pour `gradcam_analysis()`
- [ ] Implémenter d'autres méthodes d'interprétabilité (Integrated Gradients, LIME)
- [ ] Comparer les heatmaps entre modèles avec/sans masque
- [ ] Analyser les classes les plus confondues avec GradCAM
