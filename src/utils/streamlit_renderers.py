"""Module pour le rendu des composants Streamlit."""

from pathlib import Path
from typing import List, Dict
import random
import streamlit as st
import pandas as pd
from PIL import Image, UnidentifiedImageError
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from src.utils.statistics_calculator import StatisticsCalculator
from src.utils.chart_generator import ChartGenerator
from src.utils.spectral_visualization import visualize_cell_types_distribution
from src.utils.rgb_analyzer import RGBAnalyzer
from src.utils.rgb_chart_generator import RGBChartGenerator
from src.utils.image_similarity import ImageSimilarityCalculator


class StreamlitRenderer:
    """Responsable du rendu des composants Streamlit."""
    
    def __init__(
        self, 
        stats_calculator: StatisticsCalculator, 
        chart_generator: ChartGenerator
    ) -> None:
        """
        Initialise le renderer Streamlit.
        
        Args:
            stats_calculator: Calculateur de statistiques
            chart_generator: Générateur de graphiques
        """
        self.stats_calculator = stats_calculator
        self.chart_generator = chart_generator
        self.rgb_analyzer = RGBAnalyzer()
        self.rgb_chart_generator = RGBChartGenerator()
        self.similarity_calculator = ImageSimilarityCalculator()
    
    def render_statistics_tab(
        self, 
        df: pd.DataFrame, 
        widths: List[int], 
        heights: List[int]
    ) -> None:
        """
        Affiche l'onglet des statistiques.
        
        Args:
            df: DataFrame des classes
            widths: Liste des largeurs
            heights: Liste des hauteurs
        """
        st.subheader("📋 Statistiques globales")
        
        # Statistiques générales
        total_images = df["Nombre d'images"].sum()
        st.write(f"Nombre total d'images : {total_images}")
        st.write(f"Nombre de classes : {len(df)}")
        
        most_class, most_count = self.stats_calculator.get_most_represented_class(df)
        least_class, least_count = self.stats_calculator.get_least_represented_class(df)
        
        st.write(f"Classe la plus représentée : {most_class} ({most_count} images)")
        st.write(f"Classe la moins représentée : {least_class} ({least_count} images)")
        
        # Statistiques sur les dimensions
        if widths and heights:
            dim_stats = self.stats_calculator.calculate_dimension_stats(widths, heights)
            st.write(
                f"Taille moyenne des images : "
                f"{dim_stats['avg_width']:.1f} x {dim_stats['avg_height']:.1f}"
            )
            st.write(
                f"Taille min : {dim_stats['min_width']} x {dim_stats['min_height']}, "
                f"Taille max : {dim_stats['max_width']} x {dim_stats['max_height']}"
            )
            
            # Histogrammes
            st.subheader("Histogrammes des dimensions")
            st.plotly_chart(self.chart_generator.create_width_histogram(widths))
            st.plotly_chart(self.chart_generator.create_height_histogram(heights))
            
            # Ratio
            ratios = self.stats_calculator.calculate_aspect_ratios(widths, heights)
            st.plotly_chart(self.chart_generator.create_ratio_histogram(ratios))
    
    def render_distribution_tab(self, df: pd.DataFrame) -> None:
        """
        Affiche l'onglet de distribution.
        
        Args:
            df: DataFrame des classes
        """
        st.subheader("Distribution des classes")
        st.bar_chart(df)
        
        # Pie chart
        fig = self.chart_generator.create_class_distribution_pie(df)
        st.plotly_chart(fig)
        
        # Tableau
        st.subheader("Table des classes et nombre d'images")
        st.dataframe(df)
    
    def render_image_samples_tab(
        self, 
        df: pd.DataFrame, 
        class_images: Dict[str, List[Path]]
    ) -> None:
        """
        Affiche l'onglet des exemples d'images.
        
        Args:
            df: DataFrame des classes
            class_images: Dictionnaire {classe: liste_chemins}
        """
        # Sélection par classe
        st.subheader("Sélection par classe")
        selected_class = st.selectbox("Choisis une classe pour voir des images", df.index)
        
        if selected_class in class_images and class_images[selected_class]:
            self._display_class_images(selected_class, class_images[selected_class])
        else:
            st.info("Aucune image disponible pour cette classe.")
        
        # Exemples aléatoires
        st.subheader("Exemples aléatoires de plusieurs classes")
        self._display_random_samples(class_images)
    
    def _display_class_images(
        self, 
        class_name: str, 
        image_paths: List[Path], 
        max_images: int = 8
    ) -> None:
        """
        Affiche des images d'une classe.
        
        Args:
            class_name: Nom de la classe
            image_paths: Liste des chemins d'images
            max_images: Nombre maximum d'images à afficher
        """
        sample_images = random.sample(
            image_paths, 
            min(max_images, len(image_paths))
        )
        cols = st.columns(4)
        
        for i, img_path in enumerate(sample_images):
            try:
                img = Image.open(img_path)
                cols[i % 4].image(img, caption=class_name, use_container_width=True)
            except UnidentifiedImageError:
                st.warning(f"Impossible d'afficher l'image {img_path.name}")
    
    def _display_random_samples(
        self, 
        class_images: Dict[str, List[Path]], 
        num_classes: int = 4
    ) -> None:
        """
        Affiche des échantillons aléatoires de plusieurs classes.
        
        Args:
            class_images: Dictionnaire {classe: liste_chemins}
            num_classes: Nombre de classes à afficher
        """
        valid_classes = [c for c in class_images if class_images[c]]
        
        if not valid_classes:
            st.info("Aucune image trouvée dans les sous-dossiers.")
            return
        
        selected_classes = random.sample(
            valid_classes, 
            min(num_classes, len(valid_classes))
        )
        cols = st.columns(4)
        
        for i, class_name in enumerate(selected_classes):
            try:
                sample_img = random.choice(class_images[class_name])
                img = Image.open(sample_img)
                cols[i].image(img, caption=class_name, use_container_width=True)
            except UnidentifiedImageError:
                st.warning(f"Impossible d'afficher l'image {sample_img.name}")
    
    def render_spectral_visualization_tab(self, data_dir: Path) -> None:
        """
        Affiche l'onglet de visualisation spectrale.
        
        Args:
            data_dir: Répertoire du dataset
        """
        st.subheader("Distribution RGB par type de cellule")
        st.markdown("""
        Cette visualisation montre les histogrammes des intensités des couleurs (rouge, vert, bleu) 
        pour chaque classe de cellules. Les différents pics correspondent généralement à deux parties :
        - La cellule elle-même
        - Le fond de l'image
        
        Les statistiques μ(R), μ(G), μ(B) représentent les moyennes des intensités pour chaque canal de couleur.
        """)
        
        if st.button("Générer la visualisation spectrale", key="spectral_viz"):
            with st.spinner("Génération de la visualisation en cours... Cela peut prendre quelques instants."):
                try:
                    cell_types = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
                    fig = visualize_cell_types_distribution(data_dir, cell_types)
                    st.pyplot(fig)
                    st.success("✅ Visualisation générée avec succès !")
                    
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
    
    def render_rgb_distribution_tab(
        self, 
        class_images: Dict[str, List[Path]]
    ) -> None:
        """
        Affiche l'onglet des distributions RGB par classe.
        
        Args:
            class_images: Dictionnaire {classe: liste_chemins}
        """
        st.subheader("🎨 Distribution des canaux RGB par type de cellule")
        
        cell_types = sorted(class_images.keys())
        if not cell_types:
            st.info("Aucune classe trouvée pour l'analyse RGB.")
            return
        
        # Calcul des distributions
        rgb_distributions, y_max = self.rgb_analyzer.compute_rgb_distributions(class_images)
        
        if not rgb_distributions:
            st.info("Aucune donnée RGB disponible.")
            return
        
        # Légende commune
        legend_fig = self.rgb_chart_generator.create_legend_figure()
        st.plotly_chart(legend_fig, use_container_width=True)
        
        # Organisation en deux colonnes
        col1, col2 = st.columns(2)
        cols = [col1, col2]
        
        for i, cell_type in enumerate(cell_types):
            if cell_type not in rgb_distributions:
                continue
            
            x_vals, densities = rgb_distributions[cell_type]
            fig = self.rgb_chart_generator.create_rgb_distribution_figure(
                cell_type, x_vals, densities, y_max
            )
            
            cols[i % 2].plotly_chart(fig, use_container_width=True)
        
        st.caption(
            "🔎 Ces courbes permettent de comparer la distribution des valeurs RGB entre classes. "
            "Les différences de densité peuvent indiquer des contrastes d'exposition ou des variations "
            "de coloration propres à chaque type cellulaire."
        )
    
    def render_mean_images_section(
        self, 
        class_images: Dict[str, List[Path]]
    ) -> None:
        """
        Affiche la section des images moyennes par classe.
        
        Args:
            class_images: Dictionnaire {classe: liste_chemins}
        """
        st.markdown("---")
        st.subheader("Image moyenne par classe")
        
        # Calcul des images moyennes
        mean_images = self.similarity_calculator.compute_mean_images(class_images)
        
        if not mean_images:
            st.info("Aucune image moyenne n'a pu être calculée.")
            return
        
        # Affichage en grille 4 colonnes
        n_cols = 4
        cols = st.columns(n_cols)
        
        for i, (class_name, mean_img) in enumerate(mean_images.items()):
            with cols[i % n_cols]:
                st.image(mean_img, caption=class_name, use_container_width=True)
    
    def render_similarity_matrix_section(
        self, 
        class_images: Dict[str, List[Path]]
    ) -> None:
        """
        Affiche la section de la matrice de similarité cosinus.
        
        Args:
            class_images: Dictionnaire {classe: liste_chemins}
        """
        st.markdown("---")
        st.subheader("Similarité cosinus moyenne entre classes (images réelles)")
        
        # Calcul de la matrice de similarité
        cosine_df = self.similarity_calculator.compute_cosine_similarity_matrix(class_images)
        
        if cosine_df.empty:
            st.info("Impossible de calculer la matrice de similarité.")
            return
        
        # Heatmap stylisée
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cosine_df,
            annot=True,
            cmap="YlGnBu",
            fmt=".3f",
            linewidths=0.5,
            cbar_kws={"label": "Similarité cosinus moyenne"},
        )
        st.pyplot(fig)
        
        # Interprétation
        st.markdown("""
        **Interprétation :**  
        Ce graphique montre la similarité cosinus moyenne entre les images réelles de chaque classe.  
        Une valeur élevée (proche de 1) indique que les images de deux classes présentent des 
        caractéristiques visuelles très proches (couleurs, textures ou structures globales similaires).  
        Des valeurs plus faibles traduisent des différences plus nettes entre les types de cellules.
        """)
