import streamlit as st
import torch
from pathlib import Path
import sys
import json
import subprocess
import time

# Fix for "could not create a primitive" error in PyTorch 2.9.0+cpu
torch.backends.mkldnn.enabled = False

# Add project root to path for imports
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.services.yaml_loader import YamlLoader
from src.models.model_factory import ModelFactory

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
def get_model_info(_config, device_str: str):
    try:
        device = torch.device(device_str)
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

device = ModelFactory.get_device()
model_info = get_model_info(config, str(device))

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

# Section 4: Ré-entraînement du modèle
st.header("🔄 4. Ré-entraînement du modèle")

st.markdown("""
Cliquez sur le bouton ci-dessous pour lancer l'entraînement d'un nouveau modèle.
Le modèle sera automatiquement sauvegardé dans le répertoire `checkpoints`.
""")

# Initialize session state for training status
if 'training_in_progress' not in st.session_state:
    st.session_state.training_in_progress = False

col1, col2 = st.columns([1, 3])

with col1:
    if st.button("🚀 Lancer l'entraînement", type="primary", disabled=st.session_state.training_in_progress):
        st.session_state.training_in_progress = True
        st.rerun()

with col2:
    if st.session_state.training_in_progress:
        st.info("⏳ Entraînement en cours... Veuillez patienter.")

# Training execution
if st.session_state.training_in_progress:
    with st.expander("📊 Détails de l'entraînement", expanded=True):
        status_placeholder = st.empty()
        log_placeholder = st.empty()
        
        try:
            status_placeholder.info("🔄 Démarrage de l'entraînement...")
            
            # Run training script using uv
            train_script = Path(yaml_loader.project_root) / "src" / "pipe" / "train_model.py"
            
            # Use subprocess to run the training
            process = subprocess.Popen(
                ["uv", "run", "python", "-m", "src.pipe.train_model"],
                cwd=yaml_loader.project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            # Capture output in real-time
            output_lines = []
            for line in process.stdout:
                output_lines.append(line)
                log_placeholder.text_area("📝 Logs d'entraînement", 
                                         value="".join(output_lines[-50:]),  # Show last 50 lines
                                         height=300)
            
            # Wait for process to complete
            process.wait()
            
            if process.returncode == 0:
                status_placeholder.success("✅ Entraînement terminé avec succès!")
                st.success(f"🎉 Le nouveau modèle a été sauvegardé dans: `{config['paths']['models']['checkpoints']}`")
                st.info("💡 Rafraîchissez la page pour voir les nouvelles métriques.")
                
                # Reset training state
                time.sleep(2)
                st.session_state.training_in_progress = False
                st.rerun()
            else:
                status_placeholder.error(f"❌ Erreur lors de l'entraînement (code: {process.returncode})")
                st.session_state.training_in_progress = False
                
        except Exception as e:
            status_placeholder.error(f"❌ Erreur: {str(e)}")
            st.session_state.training_in_progress = False

st.markdown("---")

# Section 5: Évaluation
st.header("🎯 5. Évaluation")

# Load metrics by default from checkpoints directory
@st.cache_data
def load_evaluation_metrics():
    """Load evaluation metrics from the checkpoints directory."""
    metrics_path = Path(yaml_loader.project_root) / "models" / "checkpoints" / "metrics.json"
    
    if metrics_path.exists():
        try:
            with open(metrics_path, 'r') as f:
                return json.load(f), None
        except Exception as e:
            return None, f"Erreur lors du chargement des métriques: {e}"
    else:
        return None, "Aucun fichier de métriques trouvé"

# Check for saved model
checkpoint_path = Path(yaml_loader.project_root) / "models" / "checkpoints" / "best_model.pth"
model_exists = checkpoint_path.exists()

if model_exists:
    import os
    from datetime import datetime
    
    # Get model file info
    model_size = os.path.getsize(checkpoint_path) / (1024 * 1024)  # Size in MB
    model_mtime = datetime.fromtimestamp(os.path.getmtime(checkpoint_path))
    
    st.info(f"📦 Modèle trouvé: `best_model.pth` ({model_size:.1f} MB) - Dernière modification: {model_mtime.strftime('%Y-%m-%d %H:%M:%S')}")

# Load metrics automatically
metrics, error = load_evaluation_metrics()

if metrics is not None:
    st.success("✅ Métriques d'évaluation chargées depuis le modèle entraîné")
    
    # Display main metrics in prominent cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🎯 Test Accuracy", f"{metrics.get('accuracy', 0):.2%}")
    with col2:
        st.metric("📈 Best Val Accuracy", f"{metrics.get('best_val_acc', 0):.2%}")
    with col3:
        st.metric("🏋️ Final Train Accuracy", f"{metrics.get('final_train_acc', 0):.2%}")
    
    # Additional metrics details
    with st.expander("📊 Détails complets des métriques", expanded=False):
        st.json(metrics)
        
        # Display training loss if available
        if 'final_train_loss' in metrics:
            st.write(f"**Final Training Loss:** {metrics['final_train_loss']:.4f}")
        
        # Display class names
        if 'class_names' in metrics:
            st.write("**Classes détectées:**")
            cols = st.columns(4)
            for i, class_name in enumerate(metrics['class_names']):
                with cols[i % 4]:
                    st.write(f"• {class_name}")
elif model_exists:
    st.warning(f"⚠️ {error}")
    st.info("💡 Le modèle existe mais les métriques n'ont pas été sauvegardées. Ré-entraînez le modèle pour générer les métriques.")
else:
    st.warning(f"⚠️ {error}")
    st.info("💡 Entraînez d'abord le modèle en cliquant sur le bouton '🚀 Lancer l'entraînement' ci-dessus.")
    st.markdown("""
    **Après entraînement, cette section affichera:**
    - Accuracy sur le test set
    - Meilleure accuracy de validation
    - Accuracy finale d'entraînement
    - Loss finale d'entraînement
    """)

st.markdown("---")
st.caption("🛠️ Cette page utilise les services backend: ModelFactory, DataTransformService, YamlLoader")