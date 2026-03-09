"""Page Streamlit pour l'exploration du dataset de cellules sanguines."""

import os
from pathlib import Path

import streamlit as st

from src.utils.image_loader import ImageLoader
from src.utils.dataset_analyzer import DatasetAnalyzer
from src.utils.statistics_calculator import StatisticsCalculator
from src.utils.chart_generator import ChartGenerator
from src.utils.streamlit_renderers import StreamlitRenderer
from src.utils.streamlit_style import apply_global_style
from src.utils.streamlit_sidebar import render_mlops_sidebar


def _get_data_dir() -> Path:
    """Resolve dataset directory from env or default convention."""
    env = os.getenv("BLOODCELL_DATA_DIR")
    if env:
        return Path(env)
    # Default: project_root/data/raw/bloodcells_dataset
    project_root = Path(__file__).resolve().parents[2]
    return project_root / "data" / "raw" / "bloodcells_dataset"


def main() -> None:
    """Point d'entree principal de la page d'exploration."""
    st.set_page_config(page_title="Exploration du dataset", layout="wide")
    apply_global_style()
    render_mlops_sidebar()

    image_loader = ImageLoader(ignore_hidden=True)
    dataset_analyzer = DatasetAnalyzer(image_loader)
    stats_calculator = StatisticsCalculator()
    chart_generator = ChartGenerator()
    renderer = StreamlitRenderer(stats_calculator, chart_generator)

    st.title("Exploration du dataset")

    data_dir = _get_data_dir()

    if not data_dir.exists():
        st.warning(f"Dossier introuvable: {data_dir}")
        return

    st.success("Dataset detecte")

    with st.spinner("Analyse du dataset en cours..."):
        dataset_stats = dataset_analyzer.analyze_dataset(data_dir)

    for invalid_img in dataset_stats.invalid_images:
        st.warning(f"Fichier ignore (non-image ou corrompu) : {invalid_img.name}")

    if not dataset_stats.class_counts:
        st.info("Aucun sous-dossier contenant des images trouve.")
        return

    df = stats_calculator.create_class_dataframe(dataset_stats.class_counts)

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Vue d'ensemble",
            "Proprietes des images",
            "Analyse colorimetrique",
            "Exploration visuelle",
        ]
    )

    with tab1:
        renderer.render_overview_tab(df, dataset_stats.class_counts)

    with tab2:
        renderer.render_image_properties_tab(
            dataset_stats.widths,
            dataset_stats.heights,
            dataset_stats.class_widths,
            dataset_stats.class_heights,
        )

    with tab3:
        renderer.render_colorimetric_tab(dataset_stats.class_images)

    with tab4:
        renderer.render_visual_exploration_tab(df, dataset_stats.class_images)

    st.divider()
    st.page_link("app.py", label="Retour a l'accueil")


if __name__ == "__main__":
    main()
