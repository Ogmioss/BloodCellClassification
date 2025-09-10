import os
import sys
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path

# Ajouter le répertoire racine du projet au début du path pour assurer que les imports fonctionnent
project_root = str(Path(__file__).parent.parent.parent.parent)
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

# Définition des chemins vers les datasets
raw_data_path = os.path.join(Path(__file__).parent.parent.parent, "data", "raw", "bloodcells_dataset")
masked_data_path = os.path.join(Path(__file__).parent.parent.parent, "data", "processed", "masqued_bloodcells_dataset")

# Paramètres de visualisation dans la sidebar
st.sidebar.header("Paramètres")

# Ajout du bouton pour appliquer le masque
apply_mask = st.sidebar.checkbox("Appliquer masque", value=False)

# Initialisation des deux visualiseurs (original et masqué)
@st.cache_resource
def get_visualizers():
    """Initialise et met en cache les deux visualiseurs d'images."""
    try:
        raw_visualizer = ImageVisualizer(raw_data_path)
        masked_visualizer = ImageVisualizer(masked_data_path)
        return raw_visualizer, masked_visualizer
    except Exception as e:
        st.error(f"Erreur lors de l'initialisation des visualiseurs: {e}")
        return None, None

# Récupération des visualiseurs
raw_visualizer, masked_visualizer = get_visualizers()

# Sélection du visualiseur en fonction de l'état du bouton
visualizer = masked_visualizer if apply_mask else raw_visualizer

if visualizer:
    # Affichage des types de cellules disponibles
    cell_types = visualizer.get_available_cell_types()
    st.write(f"**Types de cellules disponibles:** {', '.join(cell_types)}")
    
    # Paramètres supplémentaires
    
    # Nombre d'échantillons par type de cellule
    num_samples = st.sidebar.slider(
        "Nombre d'échantillons par type de cellule",
        min_value=1,
        max_value=5,
        value=3
    )
    
    # Mode d'affichage (niveaux de gris, couleur, canaux RGB)
    display_mode = st.sidebar.selectbox(
        "Mode d'affichage",
        options=["gray", "color", "red", "green", "blue"],
        format_func=lambda x: {
            "gray": "Niveaux de gris",
            "color": "Couleur",
            "red": "Canal Rouge",
            "green": "Canal Vert",
            "blue": "Canal Bleu"
        }.get(x, x),
        index=1  # Par défaut: couleur
    )
    
    # Randomisation des images
    randomize = st.sidebar.checkbox("Sélection aléatoire des images", value=True)
    
    # Bouton pour actualiser les images
    if st.sidebar.button("🔄 Actualiser les images"):
        st.session_state.refresh_counter = st.session_state.get('refresh_counter', 0) + 1
        if 'image_indices' in st.session_state:
            del st.session_state['image_indices']
    
    # Affichage des images
    st.subheader("Échantillons d'images par type de cellule")
    
    # Utilisation d'un compteur de rafraîchissement pour forcer la mise à jour
    refresh_counter = st.session_state.get('refresh_counter', 0)
    
    try:
        # Stockage des indices aléatoires dans session_state pour conserver les mêmes images
        if 'image_indices' not in st.session_state or st.session_state.get('refresh_counter', 0) != refresh_counter:
            st.session_state.image_indices = {}
            for cell_type in cell_types:
                # Génère des indices aléatoires si randomize est activé, sinon utilise les premiers indices
                if randomize:
                    try:
                        indices = raw_visualizer.get_random_indices(cell_type, num_samples)
                    except ValueError:
                        indices = list(range(min(num_samples, raw_visualizer.get_image_count(cell_type))))
                else:
                    indices = list(range(min(num_samples, raw_visualizer.get_image_count(cell_type))))
                st.session_state.image_indices[cell_type] = indices
        
        # Génération de la figure avec les échantillons d'images en utilisant les mêmes indices
        with st.spinner("Chargement des images..."):
            # Création d'une figure avec des subplots
            fig, axes = plt.subplots(
                nrows=len(cell_types),
                ncols=num_samples,
                figsize=(5 * num_samples, 4.5 * len(cell_types)),
                squeeze=False
            )
            
            # Affiche chaque image avec les mêmes indices pour les deux visualiseurs
            for i, cell_type in enumerate(cell_types):
                indices = st.session_state.image_indices.get(cell_type, [])
                
                # Affiche chaque image
                for j, idx in enumerate(indices):
                    if j < num_samples:  # Sécurité supplémentaire
                        success = visualizer.show_image(
                            subdir_name=cell_type,
                            index_img=idx,
                            display_mode=display_mode,
                            ax=axes[i, j]
                        )
                        
                        if not success:
                            axes[i, j].set_title(f"{cell_type}\nNon disponible")
                            axes[i, j].axis('off')
                
                # Désactive les axes des subplots restants si pas assez d'images
                for j_empty in range(len(indices), num_samples):
                    axes[i, j_empty].set_title(f"{cell_type}\nPas assez d'images")
                    axes[i, j_empty].axis('off')
            
            # Détermine le titre en fonction du mode d'affichage
            mode_title = {
                'gray': "en niveaux de gris",
                'color': "en couleur",
                'red': "canal rouge",
                'green': "canal vert",
                'blue': "canal bleu"
            }.get(display_mode, "")
            
            fig.suptitle(f"Échantillons d'images par type de cellule ({mode_title})", fontsize=16, y=1.01)
            plt.tight_layout(rect=[0, 0, 1, 0.99])
            
            # Affichage de la figure
            st.pyplot(fig)
            
    except Exception as e:
        st.error(f"Erreur lors de l'affichage des images: {e}")
        
#     # Affichage d'une image individuelle
#     st.subheader("Visualiser une image spécifique")
    
#     # Sélection du type de cellule
#     selected_cell_type = st.selectbox("Type de cellule", cell_types)
    
#     if selected_cell_type:
#         # Nombre d'images disponibles pour ce type
#         num_images = visualizer.get_image_count(selected_cell_type)
#         st.write(f"{num_images} images disponibles pour {selected_cell_type}")
        
#         # Sélection de l'index de l'image
#         selected_index = st.slider(
#             "Index de l'image",
#             min_value=0,
#             max_value=num_images - 1 if num_images > 0 else 0,
#             value=0
#         )
        
#         # Affichage de l'image sélectionnée
#         if st.button("Afficher l'image"):
#             try:
#                 # Création d'une figure pour l'image sélectionnée
#                 plt.figure(figsize=(8, 6))
#                 visualizer.show_image(
#                     subdir_name=selected_cell_type,
#                     index_img=selected_index,
#                     display_mode=display_mode
#                 )
                
#                 # Affichage de la figure
#                 st.pyplot(plt.gcf())
                
#             except Exception as e:
#                 st.error(f"Erreur lors de l'affichage de l'image: {e}")
# else:
#     st.error("Le visualiseur n'a pas pu être initialisé. Vérifiez le chemin vers le dataset.")

# Informations sur l'application
st.sidebar.markdown("---")
st.sidebar.info(
    """
    **À propos de cette application**
    
    Cette application utilise la classe `ImageVisualizer` pour afficher des images de cellules sanguines.
    
    Fonctionnalités:
    - Visualisation en niveaux de gris
    - Visualisation en couleur
    - Visualisation des canaux RGB séparés
    
    Développée dans le cadre du projet BloodCellClassification.
    """
)
