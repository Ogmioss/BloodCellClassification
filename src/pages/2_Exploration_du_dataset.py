# ============================
# pages/2_Exploration_du_dataset.py (version finale)
# ============================

import streamlit as st
import pandas as pd
from PIL import Image, UnidentifiedImageError
from pathlib import Path
import random
import plotly.express as px
from src.services.yaml_loader import YamlLoader
from src.utils.spectral_visualization import visualize_cell_types_distribution

loader = YamlLoader()

st.title("🔍 Exploration du dataset")

# Chemin du dataset
DATA_DIR = loader.data_raw_dir / "bloodcells_dataset"

if not DATA_DIR.exists():
    st.warning("⚠️ Dossier introuvable. Vérifie le chemin du dataset.")
else:
    st.success("Dataset détecté ✅")

    counts = {}
    widths, heights = [], []
    class_images = {}

    # Parcours des sous-dossiers
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

    if counts:
        df = pd.DataFrame.from_dict(counts, orient='index', columns=['Nombre d\'images'])
        df = df.sort_values(by='Nombre d\'images', ascending=False)

        # Onglets Streamlit
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Statistiques", "📈 Distribution", "🖼️ Exemples d'images", "🌈 Visualisation spectrale"])

        # ---------------- Statistiques ----------------
        with tab1:
            st.subheader("📋 Statistiques globales")
            total_images = df["Nombre d'images"].sum()
            st.write(f"Nombre total d'images : {total_images}")
            st.write(f"Nombre de classes : {len(df)}")
            st.write(f"Classe la plus représentée : {df.index[0]} ({df.iloc[0,0]} images)")
            st.write(f"Classe la moins représentée : {df.index[-1]} ({df.iloc[-1,0]} images)")
            if widths and heights:
                st.write(f"Taille moyenne des images : {sum(widths)/len(widths):.1f} x {sum(heights)/len(heights):.1f}")
                st.write(f"Taille min : {min(widths)} x {min(heights)}, Taille max : {max(widths)} x {max(heights)}")

            # Histogrammes dimensions
            st.subheader("Histogrammes des dimensions")
            hist_w = px.histogram(widths, nbins=30, labels={'value':'Largeur (px)'}, title="Distribution des largeurs")
            hist_h = px.histogram(heights, nbins=30, labels={'value':'Hauteur (px)'}, title="Distribution des hauteurs")
            st.plotly_chart(hist_w)
            st.plotly_chart(hist_h)

            # Ratio largeur/hauteur
            ratios = [w/h for w,h in zip(widths, heights) if h != 0]
            hist_ratio = px.histogram(ratios, nbins=30, labels={'value':'Ratio L/H'}, title="Distribution du ratio L/H")
            st.plotly_chart(hist_ratio)

        # ---------------- Distribution ----------------
        with tab2:
            st.subheader("Distribution des classes")
            st.bar_chart(df)

            # Pie chart interactif
            fig = px.pie(df, names=df.index, values='Nombre d\'images', title="Répartition en pourcentage des classes")
            st.plotly_chart(fig)

            # Tableau
            st.subheader("Table des classes et nombre d'images")
            st.dataframe(df)

        # ---------------- Exemples d'images ----------------
        with tab3:
            st.subheader("Sélection par classe")
            selected_class = st.selectbox("Choisis une classe pour voir des images", df.index)
            if selected_class in class_images and class_images[selected_class]:
                sample_images = random.sample(class_images[selected_class], min(8, len(class_images[selected_class])))
                cols = st.columns(4)
                for i, img_path in enumerate(sample_images):
                    try:
                        img = Image.open(img_path)
                        cols[i % 4].image(img, caption=selected_class, use_container_width=True)
                    except UnidentifiedImageError:
                        st.warning(f"Impossible d'afficher l'image {img_path.name}")
            else:
                st.info("Aucune image disponible pour cette classe.")

            st.subheader("Exemples aléatoires de plusieurs classes")
            valid_classes = [c for c in class_images if class_images[c]]
            if valid_classes:
                selected_classes = random.sample(valid_classes, min(4, len(valid_classes)))
                cols = st.columns(4)
                for i, c in enumerate(selected_classes):
                    try:
                        sample_img = random.choice(class_images[c])
                        img = Image.open(sample_img)
                        cols[i].image(img, caption=c, use_container_width=True)
                    except UnidentifiedImageError:
                        st.warning(f"Impossible d'afficher l'image {sample_img.name}")
            else:
                st.info("Aucune image trouvée dans les sous-dossiers.")

        # ---------------- Visualisation spectrale ----------------
        with tab4:
            st.subheader("Distribution RGB par type de cellule")
            st.markdown("""
            Cette visualisation montre les histogrammes des intensités des couleurs (rouge, vert, bleu) 
            pour chaque classe de cellules. Les différents pics correspondent généralement à deux parties :
            - La cellule elle-même
            - Le fond de l'image
            
            Les statistiques μ(R), μ(G), μ(B) représentent les moyennes des intensités pour chaque canal de couleur.
            """)
            
            # Bouton pour générer la visualisation
            if st.button("Générer la visualisation spectrale", key="spectral_viz"):
                with st.spinner("Génération de la visualisation en cours... Cela peut prendre quelques instants."):
                    try:
                        # Obtenir la liste des types de cellules
                        cell_types = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir()])
                        
                        # Générer la figure
                        fig = visualize_cell_types_distribution(DATA_DIR, cell_types)
                        
                        # Afficher la figure
                        st.pyplot(fig)
                        
                        st.success("✅ Visualisation générée avec succès !")
                        
                        # Informations supplémentaires
                        st.info("""
                        **Interprétation :**
                        - Les courbes montrent la distribution des intensités de pixels pour chaque canal RGB
                        - Les pics similaires entre classes indiquent des caractéristiques colorimétriques communes
                        - Les différences de distribution peuvent aider à distinguer certains types de cellules
                        """)
                    except Exception as e:
                        st.error(f"Erreur lors de la génération de la visualisation : {str(e)}")
            else:
                st.info("👆 Cliquez sur le bouton ci-dessus pour générer la visualisation spectrale.")

    else:
        st.info("Aucun sous-dossier contenant des images trouvé.")
