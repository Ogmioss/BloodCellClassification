import streamlit as st


st.title("🚀 Pistes d'amélioration du modèle")
# gradcam_page.py

import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from captum.attr import LayerGradCam, LayerAttribution
from PIL import Image, UnidentifiedImageError
import matplotlib.pyplot as plt
import numpy as np
import cv2
import random
from pathlib import Path
from src.services.yaml_loader import YamlLoader


# ---- PAGE CONFIG ----
st.set_page_config(page_title="Pistes d'amélioration du modèle", layout="wide")

st.title("🔍 Analyse d’interprétabilité – Grad-CAM")
st.markdown("""
Cette page te permet de visualiser les zones d’attention de ton modèle **ResNet18** sur les images de cellules sanguines.
Sélectionne une classe pour générer les cartes Grad-CAM correspondantes.
""")

st.divider()

# =======================================================
# 📂 CHARGEMENT DU DATASET (logique existante)
# =======================================================
loader = YamlLoader()
DATA_DIR = loader.data_raw_dir / "bloodcells_dataset"

if not DATA_DIR.exists():
    st.warning("⚠️ Dossier introuvable. Vérifie le chemin du dataset.")
    st.stop()
else:
    st.success("Dataset détecté ✅")

    counts = {}
    widths, heights = [], []
    class_images = {}

    for d in DATA_DIR.iterdir():
        if d.is_dir():
            image_files = [f for f in list(d.glob("*.jpeg")) + list(d.glob("*.jpg")) if not f.name.startswith(".")]
            valid_images = []
            for f in image_files:
                try:
                    img = Image.open(f)
                    widths.append(img.width)
                    heights.append(img.height)
                    valid_images.append(f)
                except UnidentifiedImageError:
                    st.warning(f"Fichier ignoré (non-image ou corrompu) : {f.name}")
            counts[d.name] = len(valid_images)
            class_images[d.name] = valid_images

# =======================================================
# ⚙️ CHARGEMENT DU MODÈLE RESNET18
# =======================================================
@st.cache_resource
def load_model():
    yaml_loader = YamlLoader()
    checkpoint_dir = yaml_loader.get_nested_value('paths.models.checkpoints', './models/checkpoints')
    checkpoint_path = Path(checkpoint_dir) / 'best_model.pth'
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 8)  # 8 classes du dataset
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()
    print("XXXXXXXXXXXXXXXXXXXXXXXXX", dir(model))
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def gradcam_analysis(model, image_path, class_names):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    # Prédiction
    output = model(input_tensor)
    probs = F.softmax(output, dim=1)
    pred_class = probs.argmax(dim=1).item()
    pred_conf = probs[0, pred_class].item() * 100

    target_layer = model.layer4[-1]
    gradcam = LayerGradCam(model, target_layer)
    attribution = gradcam.attribute(input_tensor, target=pred_class)
    upsampled_attr = LayerAttribution.interpolate(attribution, input_tensor.shape[2:])
    heatmap = upsampled_attr.squeeze().detach().numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= heatmap.max()

    img = np.array(image)
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    overlay = np.uint8(0.6 * heatmap_colored + 0.4 * img)

    pred_label = class_names[pred_class] if pred_class < len(class_names) else f"Classe {pred_class}"
    print("Pred label:", pred_label)
    print("Pred conf:", pred_conf)
    print("Class names:", class_names)
    print("pred_class index:", pred_class)

    return img, heatmap_colored, overlay, pred_label, pred_conf

st.subheader("🎯 Sélection par classe")

class_names = [
    'basophil', 'eosinophil', 'erythroblast', 'ig',
    'lymphocyte', 'monocyte', 'neutrophil', 'platelet'
] 
selected_class = st.selectbox("Choisis une classe pour effectuer l’analyse Grad-CAM :", class_names)

if selected_class and class_images[selected_class]:
    sample_images = random.sample(class_images[selected_class], min(3, len(class_images[selected_class])))
    st.info(f"Analyse Grad-CAM sur **3 images aléatoires** de la classe **{selected_class}**.")
else:
    st.warning("Aucune image disponible pour cette classe.")
    st.stop()

# =======================================================
# 🧠 AFFICHAGE DES ANALYSES
# =======================================================
for img_path in sample_images:
    try:
        img, heatmap_colored, overlay, pred_label, pred_conf = gradcam_analysis(model, img_path, class_names)

        cols = st.columns(3)
        cols[0].image(img, caption="Image originale", use_container_width=True)
        cols[1].image(heatmap_colored, caption="Carte Grad-CAM", use_container_width=True)
        cols[2].image(overlay, caption="Overlay", use_container_width=True)

        st.markdown(
            f"<div style='text-align:center; font-size:16px; color:#333;'>"
            f"<b> Prédiction du modèle : </b> {pred_label} ({pred_conf:.1f}%)"
            f"</div>",
            unsafe_allow_html=True
        )

        st.markdown("---")

    except Exception as e:
        st.error(f"Erreur lors de l’analyse de {img_path.name} : {e}")











# # ---- CHARGEMENT DU MODÈLE ----
# @st.cache_resource
# def load_model():
#     yaml_loader = YamlLoader()
#     checkpoint_dir = yaml_loader.get_nested_value('paths.models.checkpoints', './models/checkpoints')
#     checkpoint_path = Path(checkpoint_dir) / 'best_model.pth'
#     model = models.resnet18(pretrained=False)
#     model.fc = torch.nn.Linear(model.fc.in_features, 8)  # 8 classes
#     model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
#     model.eval()
#     return model

# model = load_model()

# # ---- PRÉPROCESSING ----
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225])
# ])

# # ---- GRAD-CAM FUNCTION ----
# def gradcam_analysis(model, image_path, class_names):
#     image = Image.open(image_path).convert("RGB")
#     input_tensor = transform(image).unsqueeze(0)

#     # Forward + prédiction
#     output = model(input_tensor)
#     probs = F.softmax(output, dim=1)
#     pred_class = probs.argmax(dim=1).item()
#     pred_conf = probs[0, pred_class].item() * 100

#     # Grad-CAM
#     target_layer = model.layer4[-1]
#     gradcam = LayerGradCam(model, target_layer)
#     attribution = gradcam.attribute(input_tensor, target=pred_class)
#     upsampled_attr = LayerAttribution.interpolate(attribution, input_tensor.shape[2:])
#     heatmap = upsampled_attr.squeeze().detach().numpy()
#     heatmap = np.maximum(heatmap, 0)
#     heatmap /= heatmap.max()

#     # Resize heatmap à la taille originale
#     img = np.array(image)
#     heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
#     heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
#     heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
#     overlay = np.uint8(0.6 * heatmap_colored + 0.4 * img)

#     pred_label = class_names[pred_class] if pred_class < len(class_names) else f"Classe {pred_class}"

#     return img, heatmap_colored, overlay, pred_label, pred_conf

# # ---- CHARGEMENT DES DONNÉES ----
# # Suppose que tu as un dictionnaire {classe: [liste de chemins d’images]}
# # Par exemple construit à partir de ton dossier de dataset :
# loader = YamlLoader()
# DATA_DIR = loader.data_raw_dir / "bloodcells_dataset"

# if not DATA_DIR.exists():
#     st.warning("⚠️ Dossier introuvable. Vérifie le chemin du dataset.")
#     st.stop()
# else:
#     st.success("Dataset détecté ✅")

#     counts = {}
#     widths, heights = [], []
#     class_images = {}

#     for d in DATA_DIR.iterdir():
#         if d.is_dir():
#             image_files = [f for f in list(d.glob("*.jpeg")) + list(d.glob("*.jpg")) if not f.name.startswith(".")]
#             valid_images = []
#             for f in image_files:
#                 try:
#                     img = Image.open(f)
#                     widths.append(img.width)
#                     heights.append(img.height)
#                     valid_images.append(f)
#                 except UnidentifiedImageError:
#                     st.warning(f"Fichier ignoré (non-image ou corrompu) : {f.name}")
#             counts[d.name] = len(valid_images)
#             class_images[d.name] = valid_images



st.markdown("""
### Idées d’amélioration :
- Augmentation de données plus riche (rotation, zoom, contraste)
- Utiliser des **techniques d’interprétabilité** (Grad-CAM, LIME)
- Équilibrage de classes (oversampling ou focal loss)
- Entraînement sur GPU (Google Colab, Kaggle, etc.)
- Optimisation des hyperparamètres (Keras Tuner, Optuna)
""")