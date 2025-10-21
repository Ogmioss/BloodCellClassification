import streamlit as st
from PIL import Image
import numpy as np
import torch
from torchvision import transforms

st.title("🧪 Démonstration interactive avec PyTorch")
st.markdown("Upload une image de frottis sanguin pour prédire le type de cellule.")

# Charger ton modèle PyTorch
model_path = "models/best_model.pt"
try:
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval()
    st.success("Modèle chargé ✅")
except Exception as e:
    model = None
    st.error(f"Modèle introuvable ou erreur : {e}")

uploaded = st.file_uploader("Choisir une image (.jpg / .png)", type=["jpg", "jpeg", "png"])

if uploaded and model:
    img = Image.open(uploaded).convert("RGB").resize((224, 224))
    st.image(img, caption="Image uploadée", use_column_width=False)

    # Transformation pour le modèle
    transform = transforms.Compose([
        transforms.ToTensor(),          # Convertir en tensor
        transforms.Normalize([0.5]*3, [0.5]*3)  # Normalisation simple
    ])
    x = transform(img).unsqueeze(0)  # Ajouter dimension batch

    with torch.no_grad():
        preds = model(x)
        class_names = ['basophil','eosinophil','erythroblast','immature_granulocyte',
                       'lymphocyte','monocyte','neutrophil','platelet']
        pred_class = class_names[torch.argmax(preds).item()]
        st.success(f"**Prédiction :** {pred_class}")
