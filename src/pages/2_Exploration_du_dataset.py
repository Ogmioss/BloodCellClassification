"""Page Streamlit pour l'exploration du dataset de cellules sanguines."""

import streamlit as st
from src.services.yaml_loader import YamlLoader
from src.utils.image_loader import ImageLoader
from src.utils.dataset_analyzer import DatasetAnalyzer
from src.utils.statistics_calculator import StatisticsCalculator
from src.utils.chart_generator import ChartGenerator
from src.utils.streamlit_renderers import StreamlitRenderer


def main() -> None:
    """Point d'entrée principal de la page d'exploration."""
    # Initialisation des composants
    loader = YamlLoader()
    image_loader = ImageLoader(ignore_hidden=True)
    dataset_analyzer = DatasetAnalyzer(image_loader)
    stats_calculator = StatisticsCalculator()
    chart_generator = ChartGenerator()
    renderer = StreamlitRenderer(stats_calculator, chart_generator)
    
    st.title("🔍 Exploration du dataset")
    
    # Chemin du dataset
    data_dir = loader.data_raw_dir / "bloodcells_dataset"
    
    if not data_dir.exists():
        st.warning("⚠️ Dossier introuvable. Vérifie le chemin du dataset.")
        return
    
    st.success("Dataset détecté ✅")
    
    # Analyse du dataset
    dataset_stats = dataset_analyzer.analyze_dataset(data_dir)
    
    # Affichage des avertissements pour les images invalides
    for invalid_img in dataset_stats.invalid_images:
        st.warning(f"Fichier ignoré (non-image ou corrompu) : {invalid_img.name}")
    
    if not dataset_stats.class_counts:
        st.info("Aucun sous-dossier contenant des images trouvé.")
        return
    
    # Création du DataFrame
    df = stats_calculator.create_class_dataframe(dataset_stats.class_counts)
    
    # Onglets Streamlit
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Statistiques", 
        "📈 Distribution", 
        "🎨 Couleurs RGB",
        "🖼️ Exemples d'images", 
        "🌈 Visualisation spectrale"
    ])
    
    with tab1:
        renderer.render_statistics_tab(df, dataset_stats.widths, dataset_stats.heights)
    
    with tab2:
        renderer.render_distribution_tab(df)
    
    with tab3:
        renderer.render_rgb_distribution_tab(dataset_stats.class_images)
    
    with tab4:
        renderer.render_image_samples_tab(df, dataset_stats.class_images)
        renderer.render_mean_images_section(dataset_stats.class_images)
        renderer.render_similarity_matrix_section(dataset_stats.class_images)
    
    with tab5:
        renderer.render_spectral_visualization_tab(data_dir)


if __name__ == "__main__":
    main()
