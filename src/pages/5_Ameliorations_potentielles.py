"""Page d'analyse d'interprétabilité avec Grad-CAM pour les modèles ResNet."""

import random
from pathlib import Path

import streamlit as st

from src.services.yaml_loader import YamlLoader
from src.utils.gradcam_analyzer import (
    display_gradcam_results,
    get_image_transform,
    gradcam_analysis,
    load_dataset_images,
    load_resnet_model,
)

# ---- PAGE CONFIG ----
st.set_page_config(page_title="Pistes d'amélioration du modèle", layout="wide")

st.title("🔍 Analyse d'interprétabilité – Grad-CAM")
st.markdown("""
Cette page te permet de visualiser les zones d'attention de ton modèle **ResNet18** sur les images de cellules sanguines.
Sélectionne une classe pour générer les cartes Grad-CAM correspondantes.
""")

st.divider()

# =======================================================
# 📊 ONGLETS DE COMPARAISON
# =======================================================
tab1, tab2 = st.tabs(["📷 ResNet sans masque", "🎭 ResNet avec masque"])

# =======================================================
# CONSTANTES
# =======================================================
CLASS_NAMES = [
    'basophil', 'eosinophil', 'erythroblast', 'ig',
    'lymphocyte', 'monocyte', 'neutrophil', 'platelet'
]


def run_gradcam_analysis(
    data_dir: Path,
    checkpoint_filename: str,
    tab_title: str
) -> None:
    """
    Execute Grad-CAM analysis for a specific model configuration.
    
    Args:
        data_dir: Path to the dataset directory
        checkpoint_filename: Name of the checkpoint file
        tab_title: Title for the analysis section
    """
    st.subheader(f"🎯 {tab_title}")
    
    # Load dataset
    if not data_dir.exists():
        st.warning("⚠️ Dossier introuvable. Vérifie le chemin du dataset.")
        st.stop()
    
    st.success(f"Dataset détecté : `{data_dir}` ✅")
    counts, class_images = load_dataset_images(data_dir)
    
    # Load model
    @st.cache_resource
    def load_model(checkpoint_file: str) -> object:
        yaml_loader = YamlLoader()
        checkpoint_dir = yaml_loader.get_nested_value(
            'paths.models.checkpoints', 
            './models/checkpoints'
        )
        checkpoint_path = Path(checkpoint_dir) / checkpoint_file
        return load_resnet_model(checkpoint_path, num_classes=8)
    
    model = load_model(checkpoint_filename)
    transform = get_image_transform()
    
    # Class selection
    st.subheader("🎯 Sélection par classe")
    selected_class = st.selectbox(
        "Choisis une classe pour effectuer l'analyse Grad-CAM :",
        CLASS_NAMES,
        key=f"class_select_{checkpoint_filename}"
    )
    
    if selected_class and class_images.get(selected_class):
        sample_images = random.sample(
            class_images[selected_class],
            min(3, len(class_images[selected_class]))
        )
        st.info(
            f"Analyse Grad-CAM sur **3 images aléatoires** "
            f"de la classe **{selected_class}**."
        )
    else:
        st.warning("Aucune image disponible pour cette classe.")
        st.stop()
    
    # Display analyses
    st.subheader("🧠 Résultats de l'analyse")
    for img_path in sample_images:
        try:
            img, heatmap_colored, overlay, pred_label, pred_conf = gradcam_analysis(
                model, img_path, CLASS_NAMES, transform
            )
            display_gradcam_results(
                img, heatmap_colored, overlay, pred_label, pred_conf
            )
        except Exception as e:
            st.error(f"Erreur lors de l'analyse de {img_path.name} : {e}")


# =======================================================
# TAB 1: ResNet sans masque
# =======================================================
with tab1:
    loader = YamlLoader()
    data_dir_raw = loader.data_raw_dir / "bloodcells_dataset"
    run_gradcam_analysis(
        data_dir=data_dir_raw,
        checkpoint_filename='best_model.pth',
        tab_title="Analyse sur données brutes (sans masque)"
    )

# =======================================================
# TAB 2: ResNet avec masque
# =======================================================
with tab2:
    loader = YamlLoader()
    data_dir_processed = loader.data_processed_dir / "bloodcells_dataset"
    run_gradcam_analysis(
        data_dir=data_dir_processed,
        checkpoint_filename='best_model_masked.pth',
        tab_title="Analyse sur données masquées"
    )

# =======================================================
# 💡 SUGGESTIONS D'AMÉLIORATION
# =======================================================
st.divider()
st.markdown("""
### 💡 Idées d'amélioration :
- **Augmentation de données** plus riche (rotation, zoom, contraste)
- Utiliser des **techniques d'interprétabilité** (Grad-CAM, LIME)
- **Équilibrage de classes** (oversampling ou focal loss)
- **Entraînement sur GPU** (Google Colab, Kaggle, etc.)
- **Optimisation des hyperparamètres** (Keras Tuner, Optuna)
- **Comparaison des performances** entre modèles avec/sans masque
""")