import streamlit as st
import torch
from pathlib import Path
import sys
import json

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.yaml_loader import YamlLoader
from models.model_factory import ModelFactory

st.title("🧠 Modèle de classification")

# Class names
CLASS_NAMES = [
    'basophil', 'eosinophil', 'erythroblast', 'immature_granulocyte',
    'lymphocyte', 'monocyte', 'neutrophil', 'platelet'
]

# Load configuration
@st.cache_resource
def load_config():
    try:
        yaml_loader = YamlLoader()
        return yaml_loader.config, yaml_loader
    except Exception as e:
        st.error(f"Erreur lors du chargement de la configuration: {e}")
        return None, None

config, yaml_loader = load_config()

if config is None:
    st.stop()

# Section 1: Pré-traitement des images
st.header("📊 1. Pré-traitement des images")

with st.expander("⚡ Détails du pré-traitement", expanded=True):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Configuration d'entraînement")
        training_config = config.get('training', {})
        st.write(f"- **Taille d'image:** {training_config.get('img_size', 224)}x{training_config.get('img_size', 224)}")
        st.write(f"- **Batch size:** {training_config.get('batch_size', 32)}")
        st.write(f"- **Epochs:** {training_config.get('epochs', 20)}")
        st.write(f"- **Learning rate:** {training_config.get('learning_rate', 0.001)}")
    
    with col2:
        st.subheader("🔄 Augmentations")
        aug_config = config.get('augmentation', {}).get('train', {})
        if aug_config.get('horizontal_flip'):
            st.write("✅ Flip horizontal")
        if aug_config.get('vertical_flip'):
            st.write("✅ Flip vertical")
        rotation = aug_config.get('rotation_degrees', 0)
        if rotation > 0:
            st.write(f"✅ Rotation: ±{rotation}°")
        if aug_config.get('color_jitter'):
            st.write("✅ Color jitter")
    
    st.subheader("🎯 Normalisation")
    norm_config = config.get('model', {}).get('normalization', {})
    st.write(f"- **Mean:** {norm_config.get('mean', [0.485, 0.456, 0.406])}")
    st.write(f"- **Std:** {norm_config.get('std', [0.229, 0.224, 0.225])}")

# Section 2: Architecture du modèle
st.header("🏛️ 2. Architecture du modèle")

model_config = config.get('model', {})
model_name = model_config.get('name', 'resnet18')
pretrained = model_config.get('pretrained', True)

st.markdown(f"""
### Modèle utilisé: **{model_name.upper()}**

- **Type:** Transfer Learning (ResNet)
- **Pré-entraîné:** {'Oui' if pretrained else 'Non'}
- **Poids:** {model_config.get('pretrained_weights', 'IMAGENET1K_V1')}
- **Nombre de classes:** {len(CLASS_NAMES)}
""")

with st.expander("🔍 Voir les classes"):
    cols = st.columns(4)
    for i, class_name in enumerate(CLASS_NAMES):
        with cols[i % 4]:
            st.write(f"{i+1}. {class_name}")

# Load model info
@st.cache_resource
def get_model_info(_config):
    try:
        device = ModelFactory.get_device()
        model = ModelFactory.create_model(_config, len(CLASS_NAMES), device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'device': str(device)
        }
    except Exception as e:
        return {'error': str(e)}

model_info = get_model_info(config)

if 'error' not in model_info:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Paramètres totaux", f"{model_info['total_params']:,}")
    with col2:
        st.metric("Paramètres entraînables", f"{model_info['trainable_params']:,}")
    with col3:
        st.metric("Device", model_info['device'])

# Section 3: Entraînement
st.header("🚀 3. Entraînement")

st.markdown("""
- **Dataset:** Mendeley Blood Cell Images
- **Optimiseur:** Adam
- **Loss function:** CrossEntropyLoss avec pondération des classes
- **Data split:** 70% train / 15% validation / 15% test
""")

# Section 4: Évaluation
st.header("🎯 4. Évaluation")

# Check if metrics file exists
metrics_path = Path(yaml_loader.project_root) / "models" / "checkpoints" / "metrics.json"

if metrics_path.exists():
    try:
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        st.success("✅ Métriques chargées depuis le fichier")
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Accuracy", f"{metrics.get('accuracy', 0):.2%}")
        with col2:
            st.metric("Best Val Acc", f"{metrics.get('best_val_acc', 0):.2%}")
        with col3:
            st.metric("Final Train Acc", f"{metrics.get('final_train_acc', 0):.2%}")
        
        st.json(metrics)
    except Exception as e:
        st.error(f"Erreur lors du chargement des métriques: {e}")
else:
    st.warning("⚠️ Aucun fichier de métriques trouvé. Entraînez d'abord le modèle avec `train_model.py`.")
    st.info("💡 Après entraînement, cette section affichera: accuracy, précision, rappel, F1-score, et matrice de confusion.")

st.markdown("---")
st.caption("🛠️ Cette page utilise les services backend: ModelFactory, DataTransformService, YamlLoader")