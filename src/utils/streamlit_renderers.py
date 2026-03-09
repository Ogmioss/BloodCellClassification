"""Module pour le rendu des composants Streamlit."""

from pathlib import Path
from typing import Dict, List
import random

import streamlit as st
import pandas as pd
from PIL import Image, UnidentifiedImageError

from src.utils.statistics_calculator import StatisticsCalculator
from src.utils.chart_generator import ChartGenerator
from src.utils.rgb_analyzer import RGBAnalyzer
from src.utils.rgb_chart_generator import RGBChartGenerator
from src.utils.image_similarity import ImageSimilarityCalculator


class StreamlitRenderer:
    """Responsable du rendu des composants Streamlit."""

    def __init__(
        self,
        stats_calculator: StatisticsCalculator,
        chart_generator: ChartGenerator,
    ) -> None:
        self.stats_calculator = stats_calculator
        self.chart_generator = chart_generator
        self.rgb_analyzer = RGBAnalyzer()
        self.rgb_chart_generator = RGBChartGenerator()
        self.similarity_calculator = ImageSimilarityCalculator()

    # ===========================================================================
    # Tab 1 — Vue d'ensemble
    # ===========================================================================
    def render_overview_tab(
        self,
        df: pd.DataFrame,
        class_counts: Dict[str, int],
    ) -> None:
        """KPI cards + bar chart horizontal + donut + tableau."""
        total = int(df[df.columns[0]].sum())
        n_classes = len(df)
        imbalance = self.stats_calculator.calculate_imbalance_ratio(class_counts)
        health = self.stats_calculator.calculate_health_score(class_counts)

        # --- KPI row ---
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total images", f"{total:,}",
                  help="Nombre total d'images dans le dataset")
        c2.metric("Classes", n_classes,
                  help="Nombre de types de cellules distincts")
        c3.metric(
            "Imbalance Ratio", f"{imbalance:.2f}",
            delta=f"{imbalance - 1:.2f} vs ideal (1.0)",
            delta_color="inverse",
            help="Ratio max/min des effectifs. 1.0 = parfaitement equilibre",
        )
        c4.metric(
            "Health Score", f"{health}/100",
            delta=f"{health - 100} vs parfait",
            delta_color="normal",
            help="Score composite (entropie 40%, CV 30%, ratio min/max 30%)",
        )

        # --- Métriques avancées ---
        with st.expander("Métriques avancées"):
            entropy = self.stats_calculator.calculate_shannon_entropy(class_counts)
            cv = self.stats_calculator.calculate_cv(class_counts)
            median = self.stats_calculator.calculate_median_count(class_counts)

            m1, m2, m3 = st.columns(3)
            m1.metric(
                "Entropie de Shannon", f"{entropy:.3f}",
                delta=f"{entropy - 1:.3f} vs uniforme (1.0)",
                delta_color="normal",
                help="Normalisee 0-1. 1 = parfaitement uniforme",
            )
            m2.metric(
                "Coeff. de Variation", f"{cv:.3f}",
                delta_color="inverse",
                help="sigma / mu des comptages. Plus bas = plus homogene",
            )
            m3.metric("Mediane", f"{median:.0f} images")

        # --- Chart ---
        st.plotly_chart(
            self.chart_generator.create_class_bar_chart(df),
            use_container_width=True,
        )

        # --- Tableau ---
        st.subheader("Détail par classe")
        st.dataframe(df, use_container_width=True)

    # ===========================================================================
    # Tab 2 — Propriétés des images
    # ===========================================================================
    def render_image_properties_tab(
        self,
        widths: List[int],
        heights: List[int],
        class_widths: Dict[str, List[int]],
        class_heights: Dict[str, List[int]],
    ) -> None:
        """Scatter, box plots, ratio histogram, outliers."""
        if not widths:
            st.info("Aucune donnée dimensionnelle disponible.")
            return

        dim_stats = self.stats_calculator.calculate_dimension_stats(widths, heights)
        import numpy as np

        w_arr, h_arr = np.array(widths), np.array(heights)
        unique_sizes = len(set(zip(widths, heights)))
        pct_square = float(np.mean(w_arr == h_arr) * 100)

        # --- KPI row ---
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "Taille moyenne",
            f"{dim_stats['avg_width']:.0f} × {dim_stats['avg_height']:.0f}",
        )
        c2.metric(
            "Taille médiane",
            f"{int(np.median(w_arr))} × {int(np.median(h_arr))}",
        )
        c3.metric("Tailles uniques", unique_sizes)
        c4.metric("Images carrées", f"{pct_square:.1f}%")

        # --- Charts ---
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(
                self.chart_generator.create_dimension_scatter(class_widths, class_heights),
                use_container_width=True,
            )
        with col2:
            st.plotly_chart(
                self.chart_generator.create_dimension_boxplots(widths, heights),
                use_container_width=True,
            )

        # Ratio histogram
        ratios = self.stats_calculator.calculate_aspect_ratios(widths, heights)
        st.plotly_chart(
            self.chart_generator.create_ratio_histogram_enhanced(ratios),
            use_container_width=True,
        )

        # Outliers
        outliers_df = self.stats_calculator.calculate_dimension_outliers(widths, heights)
        if not outliers_df.empty:
            st.subheader(f"Outliers dimensionnels ({len(outliers_df)} images > 2σ)")
            st.dataframe(outliers_df, use_container_width=True)
        else:
            st.success("Aucun outlier dimensionnel détecté (seuil 2σ).")

    # ===========================================================================
    # Tab 3 — Analyse colorimétrique
    # ===========================================================================
    def render_colorimetric_tab(
        self,
        class_images: Dict[str, List[Path]],
    ) -> None:
        """RGB distributions + tableau stats + radar + heatmap corrélation."""
        cell_types = sorted(class_images.keys())
        if not cell_types:
            st.info("Aucune classe trouvée pour l'analyse RGB.")
            return

        # --- Section 1 : Distributions RGB existantes ---
        st.header("Distribution des canaux RGB par classe")

        rgb_distributions, y_max = self.rgb_analyzer.compute_rgb_distributions(class_images)

        if rgb_distributions:
            legend_fig = self.rgb_chart_generator.create_legend_figure()
            st.plotly_chart(legend_fig, use_container_width=True)

            col1, col2 = st.columns(2)
            cols = [col1, col2]
            for i, cell_type in enumerate(cell_types):
                if cell_type not in rgb_distributions:
                    continue
                x_vals, densities = rgb_distributions[cell_type]
                fig = self.rgb_chart_generator.create_rgb_distribution_figure(
                    cell_type, x_vals, densities, y_max,
                )
                cols[i % 2].plotly_chart(fig, use_container_width=True)

        st.divider()

        # --- Section 2 : Tableau récapitulatif RGB ---
        st.subheader("Statistiques RGB par classe")
        rgb_stats_df = self.rgb_analyzer.compute_class_rgb_stats(class_images)
        if not rgb_stats_df.empty:
            st.dataframe(rgb_stats_df, use_container_width=True)

            st.divider()

            # --- Section 3 : Radar chart comparatif ---
            with st.expander("Comparaison inter-classes (Radar)", expanded=False):
                r_col1, r_col2 = st.columns(2)
                with r_col1:
                    class_a = st.selectbox("Classe A", cell_types, index=0, key="radar_a")
                with r_col2:
                    default_b = min(1, len(cell_types) - 1)
                    class_b = st.selectbox("Classe B", cell_types, index=default_b, key="radar_b")

                if class_a in rgb_stats_df.index and class_b in rgb_stats_df.index:
                    stats_a = rgb_stats_df.loc[class_a].to_dict()
                    stats_b = rgb_stats_df.loc[class_b].to_dict()
                    fig = self.chart_generator.create_radar_chart(stats_a, stats_b, class_a, class_b)
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Impossible de calculer les statistiques RGB.")

    # ===========================================================================
    # Tab 4 — Exploration visuelle
    # ===========================================================================
    def render_visual_exploration_tab(
        self,
        df: pd.DataFrame,
        class_images: Dict[str, List[Path]],
    ) -> None:
        """Galerie + images moyennes + heatmap Plotly + séparabilité."""
        # --- Sélection par classe ---
        st.subheader("Sélection par classe")
        selected_class = st.selectbox("Choisis une classe pour voir des images", df.index)

        if selected_class in class_images and class_images[selected_class]:
            self._display_class_images(selected_class, class_images[selected_class])
        else:
            st.info("Aucune image disponible pour cette classe.")

        # --- Exemples aléatoires ---
        st.subheader("Exemples aléatoires de plusieurs classes")
        self._display_random_samples(class_images)

        # --- Images moyennes ---
        st.divider()
        st.subheader("Image moyenne par classe")
        mean_images = self.similarity_calculator.compute_mean_images(class_images)
        if mean_images:
            n_cols = 4
            cols = st.columns(n_cols)
            for i, (class_name, mean_img) in enumerate(mean_images.items()):
                with cols[i % n_cols]:
                    st.image(mean_img, caption=class_name, use_container_width=True)
        else:
            st.info("Aucune image moyenne calculée.")

        # --- Matrice de similarité (Plotly heatmap) ---
        st.divider()
        st.subheader("Similarité cosinus moyenne entre classes")
        cosine_df = self.similarity_calculator.compute_cosine_similarity_matrix(class_images)
        if not cosine_df.empty:
            fig = self.chart_generator.create_plotly_heatmap(
                cosine_df, "Matrice de similarité cosinus",
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Une valeur proche de 1 indique des caractéristiques visuelles très proches. "
                "Des valeurs plus faibles traduisent des différences nettes entre les types de cellules."
            )

        # --- Séparabilité ---
        st.divider()
        st.subheader("Indice de séparabilité inter-classes")
        sep_scores = self.similarity_calculator.compute_separability_index(class_images)
        if sep_scores:
            fig = self.chart_generator.create_separability_bar_chart(sep_scores)
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Un score élevé signifie que la classe est visuellement distincte des autres "
                "(variance inter-classes élevée vs variance intra-classe faible)."
            )

    # ===========================================================================
    # Helpers privés
    # ===========================================================================
    def _display_class_images(
        self,
        class_name: str,
        image_paths: List[Path],
        max_images: int = 8,
    ) -> None:
        sample_images = random.sample(
            image_paths, min(max_images, len(image_paths))
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
        num_classes: int = 4,
    ) -> None:
        valid_classes = [c for c in class_images if class_images[c]]
        if not valid_classes:
            st.info("Aucune image trouvée dans les sous-dossiers.")
            return

        selected_classes = random.sample(
            valid_classes, min(num_classes, len(valid_classes))
        )
        cols = st.columns(4)
        for i, class_name in enumerate(selected_classes):
            try:
                sample_img = random.choice(class_images[class_name])
                img = Image.open(sample_img)
                cols[i].image(img, caption=class_name, use_container_width=True)
            except UnidentifiedImageError:
                st.warning(f"Impossible d'afficher l'image {sample_img.name}")
