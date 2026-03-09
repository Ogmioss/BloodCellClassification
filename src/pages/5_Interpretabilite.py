"""Page d'analyse d'interpretabilite avec Grad-CAM pour les modeles ResNet."""

import random
from pathlib import Path

import streamlit as st

from src.core.constants import CLASS_NAMES
from src.services.yaml_loader import YamlLoader
from src.utils.gradcam_analyzer import (
    LayerGradCam,
    display_gradcam_results,
    get_image_transform,
    gradcam_analysis,
    load_dataset_images,
    load_resnet_model,
)
from src.utils.streamlit_style import apply_global_style
from src.utils.streamlit_sidebar import render_mlops_sidebar

# ---- PAGE CONFIG ----
st.set_page_config(page_title="Interpretabilite", layout="wide")
apply_global_style()
render_mlops_sidebar()

st.title("Interpretabilite du modele")
st.markdown("""
Cette page permet de visualiser les zones d'attention du modele **ResNet18** sur les images
des cellules sanguines via l'analyse **Grad-CAM**.
""")

if LayerGradCam is None:
    st.error(
        "Le module `captum` n'est pas installe. "
        "Installez-le avec `uv add captum` pour activer l'analyse Grad-CAM."
    )
    st.stop()

st.divider()

# =======================================================
# ONGLETS DE COMPARAISON
# =======================================================
tab1, tab2 = st.tabs(["ResNet sans masque", "ResNet avec masque"])


def run_gradcam_analysis(
    data_dir: Path,
    checkpoint_filename: str,
    tab_title: str,
) -> None:
    """Execute Grad-CAM analysis for a specific model configuration."""
    st.subheader(tab_title)

    # Load dataset
    if not data_dir.exists():
        st.warning("Dossier introuvable. Verifiez le chemin du dataset.")
        st.stop()

    st.success(f"Dataset detecte : `{data_dir}`")
    with st.spinner("Chargement des images du dataset..."):
        counts, class_images = load_dataset_images(data_dir)

    # Load model with proper caching
    @st.cache_resource
    def load_model(checkpoint_file: str) -> object:
        yaml_loader = YamlLoader()
        checkpoint_dir = yaml_loader.get_nested_value(
            "paths.models.checkpoints",
            "./models/checkpoints",
        )
        checkpoint_path = Path(checkpoint_dir) / checkpoint_file
        model = load_resnet_model(checkpoint_path, num_classes=len(CLASS_NAMES))
        # Ensure clean state
        model.eval()
        model.zero_grad()
        return model

    model = load_model(checkpoint_filename)
    transform = get_image_transform()

    # Add option to clear cache if needed
    if st.button("Recharger le modele", key=f"reload_{checkpoint_filename}"):
        st.cache_resource.clear()
        st.rerun()

    # Class selection
    st.subheader("Selection par classe")
    selected_class = st.selectbox(
        "Choisis une classe pour effectuer l'analyse Grad-CAM :",
        CLASS_NAMES,
        key=f"class_select_{checkpoint_filename}",
    )

    if selected_class and class_images.get(selected_class):
        # Use a seed for reproducibility but allow refresh
        if st.button(
            "Nouvelles images aleatoires", key=f"refresh_{checkpoint_filename}"
        ):
            st.session_state[f"seed_{checkpoint_filename}"] = random.randint(0, 10000)

        seed = st.session_state.get(f"seed_{checkpoint_filename}", 42)
        random.seed(seed)

        sample_images = random.sample(
            class_images[selected_class],
            min(3, len(class_images[selected_class])),
        )
        st.info(
            f"Analyse Grad-CAM sur **3 images aleatoires** "
            f"de la classe **{selected_class}**."
        )
    else:
        st.warning("Aucune image disponible pour cette classe.")
        st.stop()

    # Display analyses
    st.subheader("Resultats de l'analyse")
    for idx, img_path in enumerate(sample_images):
        with st.spinner(
            f"Analyse Grad-CAM de l'image {idx + 1}/{len(sample_images)}..."
        ):
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
        checkpoint_filename="best_model.pth",
        tab_title="Analyse sur donnees brutes (sans masque)",
    )

# =======================================================
# TAB 2: ResNet avec masque
# =======================================================
with tab2:
    loader = YamlLoader()
    data_dir_processed = loader.data_processed_dir / "bloodcells_dataset"
    run_gradcam_analysis(
        data_dir=data_dir_processed,
        checkpoint_filename="best_model_masked.pth",
        tab_title="Analyse sur donnees masquees",
    )

st.divider()
st.page_link("app.py", label="Retour a l'accueil")
