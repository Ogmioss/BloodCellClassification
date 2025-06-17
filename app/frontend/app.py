import os
import sys
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path

# Ajouter le répertoire racine du projet au début du path pour assurer que les imports fonctionnent
project_root = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, project_root)

# Import direct du module
from app.backend.services.display_img import ImageVisualizer

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Visualisation de Cellules Sanguines",
    page_icon="🔬",
    layout="wide"
)

# Titre de l'application
st.title("🔬 Visualisation de Cellules Sanguines")
st.markdown("Cette application permet de visualiser des échantillons d'images de cellules sanguines.")

# Chemin vers le dataset
data_path = os.path.join(Path(__file__).parent.parent, "data", "raw", "bloodcells_dataset")

# Initialisation du visualiseur
@st.cache_resource
def get_visualizer(path):
    """
    Initialise et met en cache le visualiseur d'images.
    
    Args:
        path: Chemin vers le dataset.
        
    Returns:
        ImageVisualizer: Instance du visualiseur.
    """
    try:
        return ImageVisualizer(path)
    except Exception as e:
        st.error(f"Erreur lors de l'initialisation du visualiseur: {e}")
        return None

# Récupération du visualiseur
visualizer = get_visualizer(data_path)

if visualizer:
    # Affichage des types de cellules disponibles
    cell_types = visualizer.get_available_cell_types()
    st.write(f"**Types de cellules disponibles:** {', '.join(cell_types)}")
    
    # Paramètres de visualisation
    st.sidebar.header("Paramètres")
    
    # Nombre d'échantillons par type de cellule
    num_samples = st.sidebar.slider(
        "Nombre d'échantillons par type de cellule",
        min_value=1,
        max_value=5,
        value=3
    )
    
    # Affichage en couleur ou en niveaux de gris
    display_color = st.sidebar.checkbox("Afficher en couleur", value=True)
    
    # Randomisation des images
    randomize = st.sidebar.checkbox("Sélection aléatoire des images", value=True)
    
    # Bouton pour actualiser les images
    if st.sidebar.button("🔄 Actualiser les images"):
        st.session_state.refresh_counter = st.session_state.get('refresh_counter', 0) + 1
    
    # Affichage des images
    st.subheader("Échantillons d'images par type de cellule")
    
    # Utilisation d'un compteur de rafraîchissement pour forcer la mise à jour
    refresh_counter = st.session_state.get('refresh_counter', 0)
    
    try:
        # Génération de la figure avec les échantillons d'images
        with st.spinner("Chargement des images..."):
            fig = visualizer.display_sample_images_per_subdir(
                num_samples_per_subdir=num_samples,
                display_color=display_color,
                randomize=randomize
            )
            
            # Affichage de la figure
            st.pyplot(fig)
            
    except Exception as e:
        st.error(f"Erreur lors de l'affichage des images: {e}")
        
    # Affichage d'une image individuelle
    st.subheader("Visualiser une image spécifique")
    
    # Sélection du type de cellule
    selected_cell_type = st.selectbox("Type de cellule", cell_types)
    
    if selected_cell_type:
        # Nombre d'images disponibles pour ce type
        num_images = visualizer.get_image_count(selected_cell_type)
        st.write(f"{num_images} images disponibles pour {selected_cell_type}")
        
        # Sélection de l'index de l'image
        selected_index = st.slider(
            "Index de l'image",
            min_value=0,
            max_value=num_images - 1 if num_images > 0 else 0,
            value=0
        )
        
        # Affichage de l'image sélectionnée
        if st.button("Afficher l'image"):
            try:
                # Création d'une figure pour l'image sélectionnée
                plt.figure(figsize=(8, 6))
                visualizer.show_image(
                    subdir_name=selected_cell_type,
                    index_img=selected_index,
                    color=display_color
                )
                
                # Affichage de la figure
                st.pyplot(plt.gcf())
                
            except Exception as e:
                st.error(f"Erreur lors de l'affichage de l'image: {e}")
else:
    st.error("Le visualiseur n'a pas pu être initialisé. Vérifiez le chemin vers le dataset.")

# Informations sur l'application
st.sidebar.markdown("---")
st.sidebar.info(
    """
    **À propos de cette application**
    
    Cette application utilise la classe `ImageVisualizer` pour afficher des images de cellules sanguines.
    
    Développée dans le cadre du projet BloodCellClassification.
    """
)
