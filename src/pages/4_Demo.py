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
from src.services.keras_inference_service import KerasInferenceService
from src.models.model_factory import ModelFactory

st.title("🧪 Démonstration interactive - Comparaison de modèles")
st.markdown("Upload une image de frottis sanguin pour comparer les prédictions des deux modèles.")

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

# Charger le service d'inférence PyTorch
@st.cache_resource
def load_pytorch_inference_service(_config, device_str: str):
    try:
        # Convert device string to torch.device
        device = torch.device(device_str)
        
        # Load checkpoint
        yaml_loader = YamlLoader()
        checkpoint_dir = yaml_loader.get_nested_value('paths.models.checkpoints', './models/checkpoints')
        checkpoint_path = Path(checkpoint_dir) / 'best_model.pth'
        
        if not checkpoint_path.exists():
            return None, f"Checkpoint PyTorch introuvable: {checkpoint_path}"
        
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

# Charger le service d'inférence Keras
@st.cache_resource
def load_keras_inference_service():
    try:
        # Load checkpoint
        yaml_loader = YamlLoader()
        checkpoint_dir = yaml_loader.get_nested_value('paths.models.checkpoints', './models/checkpoints')
        
        # Try .keras format first, then .h5
        keras_path = Path(checkpoint_dir) / 'baseline_pbc_model.keras'
        h5_path = Path(checkpoint_dir) / 'baseline_pbc_model.h5'
        
        checkpoint_path = keras_path if keras_path.exists() else h5_path
        
        if not checkpoint_path.exists():
            return None, f"Checkpoint Keras introuvable: {checkpoint_path}"
        
        # Create inference service
        inference_service = KerasInferenceService.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            class_names=CLASS_NAMES,
            input_size=(200, 200)  # Baseline PBC model expects 200x200 images
        )
        
        return inference_service, None
    except Exception as e:
        return None, str(e)

# Load both models
device = ModelFactory.get_device()
pytorch_service, pytorch_error = load_pytorch_inference_service(config, str(device))
keras_service, keras_error = load_keras_inference_service()

# Display loading status
col1, col2 = st.columns(2)

with col1:
    st.subheader("🔥 Modèle PyTorch (ResNet)")
    if pytorch_error:
        st.error(f"Erreur: {pytorch_error}")
    else:
        st.success("Modèle chargé ✅")
        st.info(f"Device: {device}")

with col2:
    st.subheader("🧠 Modèle Keras (Baseline PBC)")
    if keras_error:
        st.error(f"Erreur: {keras_error}")
    else:
        st.success("Modèle chargé ✅")

st.divider()

uploaded = st.file_uploader("Choisir une image (.jpg / .png)", type=["jpg", "jpeg", "png"])

if uploaded:
    # Load and display image
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Image uploadée", width=300)
    
    st.divider()
    
    # Create two columns for predictions
    col1, col2 = st.columns(2)
    
    # PyTorch model prediction
    with col1:
        st.subheader("🔥 Prédiction PyTorch (ResNet)")
        if pytorch_service:
            try:
                prediction = pytorch_service.predict_image(img)
                
                st.success(f"**Classe prédite:** {prediction['predicted_class']}")
                st.metric("Confiance", f"{prediction['confidence']:.2%}")
                
                with st.expander("Voir toutes les probabilités"):
                    for class_name, prob in sorted(
                        prediction['probabilities'].items(), 
                        key=lambda x: x[1], 
                        reverse=True
                    ):
                        st.write(f"**{class_name}:** {prob:.2%}")
            except Exception as e:
                st.error(f"Erreur lors de la prédiction: {e}")
        else:
            st.warning("Modèle PyTorch non disponible")
    
    # Keras model prediction
    with col2:
        st.subheader("🧠 Prédiction Keras (Baseline PBC)")
        if keras_service:
            try:
                prediction = keras_service.predict_image(img)
                
                st.success(f"**Classe prédite:** {prediction['predicted_class']}")
                st.metric("Confiance", f"{prediction['confidence']:.2%}")
                
                with st.expander("Voir toutes les probabilités"):
                    for class_name, prob in sorted(
                        prediction['probabilities'].items(), 
                        key=lambda x: x[1], 
                        reverse=True
                    ):
                        st.write(f"**{class_name}:** {prob:.2%}")
            except Exception as e:
                st.error(f"Erreur lors de la prédiction: {e}")
        else:
            st.warning("Modèle Keras non disponible")
