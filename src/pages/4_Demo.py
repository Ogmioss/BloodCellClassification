import streamlit as st
from PIL import Image
import torch
from pathlib import Path
import sys
import os

# Fix for "could not create a primitive" error in PyTorch 2.9.0+cpu
torch.backends.mkldnn.enabled = False

# Add project root to path for imports
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.services.yaml_loader import YamlLoader
from src.services.inference_service import InferenceService
from src.models.model_factory import ModelFactory

st.title("🧪 Démonstration interactive avec PyTorch")
st.markdown("Upload une image de frottis sanguin pour prédire le type de cellule.")

# Class names pour la prédiction
CLASS_NAMES = [
    'basophil', 'eosinophil', 'erythroblast', 'immature_granulocyte',
    'lymphocyte', 'monocyte', 'neutrophil', 'platelet'
]

# Charger la configuration
@st.cache_resource
def load_config():
    try:
        yaml_loader = YamlLoader()
        return yaml_loader.config
    except Exception as e:
        st.error(f"Erreur lors du chargement de la configuration: {e}")
        return None

config = load_config()

if config is None:
    st.stop()

# Charger le service d'inférence
@st.cache_resource
def load_inference_service(_config, device_str: str):
    try:
        # Convert device string to torch.device
        device = torch.device(device_str)
        st.info(f"Device utilisé: {device}")
        
        # Load checkpoint
        yaml_loader = YamlLoader()
        checkpoint_dir = yaml_loader.get_nested_value('paths.models.checkpoints', './models/checkpoints')
        checkpoint_path = Path(checkpoint_dir) / 'best_model.pth'
        
        if not checkpoint_path.exists():
            return None, f"Checkpoint introuvable: {checkpoint_path}"
        
        # Create inference service
        inference_service = InferenceService.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            config=_config,
            device=device,
            class_names=CLASS_NAMES
        )
        
        return inference_service, None
    except Exception as e:
        return None, str(e)

# Get device and pass as string to enable cache invalidation
device = ModelFactory.get_device()
inference_service, error = load_inference_service(config, str(device))

if error:
    st.error(f"Erreur lors du chargement du modèle: {error}")
else:
    st.success("Modèle chargé ✅")

uploaded = st.file_uploader("Choisir une image (.jpg / .png)", type=["jpg", "jpeg", "png"])

if uploaded and inference_service:
    # Load and display image
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Image uploadée", width=300)

    # Make prediction using InferenceService
    prediction = inference_service.predict_image(img)
    
    # Display results
    st.success(f"**Prédiction:** {prediction['predicted_class']}")
    st.info(f"**Confiance:** {prediction['confidence']:.2%}")
    
    # Display all class probabilities
    with st.expander("Voir toutes les probabilités"):
        for class_name, prob in prediction['probabilities'].items():
            st.write(f"{class_name}: {prob:.2%}")
