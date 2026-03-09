"""Module pour la génération de graphiques."""

from typing import Dict, List

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


class ChartGenerator:
    """Responsable de la génération des graphiques."""

    @staticmethod
    def _apply_theme(fig: go.Figure) -> go.Figure:
        """Applique un thème visuel cohérent à toute figure Plotly."""
        fig.update_layout(
            font=dict(family="Inter, sans-serif", size=13, color="#2c3e50"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            title=dict(x=0.5, xanchor="center", font=dict(size=16, color="#2c3e50")),
            margin=dict(l=50, r=30, t=55, b=50),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=12)),
        )
        fig.update_xaxes(
            gridcolor="#e8e8e8", gridwidth=1, griddash="dot",
            linecolor="#bdc3c7", linewidth=1,
        )
        fig.update_yaxes(
            gridcolor="#e8e8e8", gridwidth=1, griddash="dot",
            linecolor="#bdc3c7", linewidth=1,
        )
        return fig

    @staticmethod
    def create_histogram(
        data: List[float], 
        x_label: str, 
        title: str, 
        nbins: int = 30
    ) -> go.Figure:
        """
        Crée un histogramme.
        
        Args:
            data: Données à afficher
            x_label: Label de l'axe X
            title: Titre du graphique
            nbins: Nombre de bins
            
        Returns:
            Figure Plotly
        """
        fig = px.histogram(
            data,
            nbins=nbins,
            labels={'value': x_label},
            title=title,
        )
        return ChartGenerator._apply_theme(fig)
    
    @staticmethod
    def create_pie_chart(
        df: pd.DataFrame, 
        values_column: str, 
        title: str
    ) -> go.Figure:
        """
        Crée un diagramme circulaire.
        
        Args:
            df: DataFrame contenant les données
            values_column: Nom de la colonne des valeurs
            title: Titre du graphique
            
        Returns:
            Figure Plotly
        """
        fig = px.pie(
            df,
            names=df.index,
            values=values_column,
            title=title,
        )
        return ChartGenerator._apply_theme(fig)
    
    @staticmethod
    def create_width_histogram(widths: List[int]) -> go.Figure:
        """
        Crée un histogramme des largeurs.
        
        Args:
            widths: Liste des largeurs
            
        Returns:
            Figure Plotly
        """
        return ChartGenerator.create_histogram(
            widths, 
            'Largeur (px)', 
            'Distribution des largeurs'
        )
    
    @staticmethod
    def create_height_histogram(heights: List[int]) -> go.Figure:
        """
        Crée un histogramme des hauteurs.
        
        Args:
            heights: Liste des hauteurs
            
        Returns:
            Figure Plotly
        """
        return ChartGenerator.create_histogram(
            heights, 
            'Hauteur (px)', 
            'Distribution des hauteurs'
        )
    
    @staticmethod
    def create_ratio_histogram(ratios: List[float]) -> go.Figure:
        """
        Crée un histogramme des ratios largeur/hauteur.
        
        Args:
            ratios: Liste des ratios
            
        Returns:
            Figure Plotly
        """
        return ChartGenerator.create_histogram(
            ratios, 
            'Ratio L/H', 
            'Distribution du ratio L/H'
        )
    
    @staticmethod
    def create_class_distribution_pie(df: pd.DataFrame) -> go.Figure:
        """
        Crée un diagramme circulaire de la distribution des classes.
        
        Args:
            df: DataFrame des classes
            
        Returns:
            Figure Plotly
        """
        return ChartGenerator.create_pie_chart(
            df,
            'Nombre d\'images',
            'Répartition en pourcentage des classes'
        )

    @staticmethod
    def create_class_bar_chart(df: pd.DataFrame) -> go.Figure:
        """Histogramme vertical trié décroissant avec ligne moyenne et couleurs conditionnelles."""
        col = df.columns[0]
        sorted_df = df.sort_values(by=col, ascending=False)
        mean_val = sorted_df[col].mean()

        colors = [
            "#2ecc71" if v >= mean_val else ("#f39c12" if v >= mean_val * 0.7 else "#e74c3c")
            for v in sorted_df[col]
        ]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=sorted_df.index,
            y=sorted_df[col],
            marker_color=colors,
            text=sorted_df[col],
            textposition="outside",
        ))
        fig.add_hline(
            y=mean_val, line_dash="dash", line_color="#555",
            annotation_text=f"Moyenne: {mean_val:.0f}",
            annotation_position="top left",
        )
        fig.update_layout(
            title="Distribution des classes",
            xaxis_title="Classe",
            yaxis_title="Nombre d'images",
            height=450,
        )
        return ChartGenerator._apply_theme(fig)

    @staticmethod
    def create_dimension_scatter(
        class_widths: Dict[str, List[int]],
        class_heights: Dict[str, List[int]],
    ) -> go.Figure:
        """Scatter plot largeur vs hauteur, coloré par classe."""
        rows = []
        for cls in sorted(class_widths.keys()):
            for w, h in zip(class_widths[cls], class_heights[cls]):
                rows.append({"Classe": cls, "Largeur": w, "Hauteur": h})

        scatter_df = pd.DataFrame(rows)
        fig = px.scatter(
            scatter_df, x="Largeur", y="Hauteur", color="Classe",
            title="Dimensions des images par classe",
            opacity=0.6,
        )
        fig.update_layout(height=450)
        return ChartGenerator._apply_theme(fig)

    @staticmethod
    def create_dimension_boxplots(widths: List[int], heights: List[int]) -> go.Figure:
        """Box plots côte à côte pour largeur et hauteur."""
        fig = go.Figure()
        fig.add_trace(go.Box(y=widths, name="Largeur", marker_color="#3498db"))
        fig.add_trace(go.Box(y=heights, name="Hauteur", marker_color="#e74c3c"))
        fig.update_layout(
            title="Distribution des dimensions",
            yaxis_title="Pixels",
            height=400,
        )
        return ChartGenerator._apply_theme(fig)

    @staticmethod
    def create_ratio_histogram_enhanced(ratios: List[float]) -> go.Figure:
        """Histogramme du ratio L/H avec ligne verticale à 1.0."""
        fig = px.histogram(
            ratios, nbins=40,
            labels={"value": "Ratio L/H"},
            title="Distribution du ratio largeur / hauteur",
        )
        fig.add_vline(
            x=1.0, line_dash="dash", line_color="red",
            annotation_text="Carré (1:1)",
        )
        fig.update_layout(height=400)
        return ChartGenerator._apply_theme(fig)

    @staticmethod
    def create_radar_chart(
        stats_a: Dict[str, float],
        stats_b: Dict[str, float],
        label_a: str,
        label_b: str,
    ) -> go.Figure:
        """Radar chart comparant les stats RGB de 2 classes."""
        categories = list(stats_a.keys())
        vals_a = [stats_a[c] for c in categories]
        vals_b = [stats_b[c] for c in categories]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=vals_a + [vals_a[0]],
            theta=categories + [categories[0]],
            fill="toself", name=label_a, opacity=0.6,
        ))
        fig.add_trace(go.Scatterpolar(
            r=vals_b + [vals_b[0]],
            theta=categories + [categories[0]],
            fill="toself", name=label_b, opacity=0.6,
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True)),
            title=f"Comparaison RGB : {label_a} vs {label_b}",
            height=450,
        )
        return ChartGenerator._apply_theme(fig)

    @staticmethod
    def create_plotly_heatmap(df: pd.DataFrame, title: str) -> go.Figure:
        """Heatmap interactive Plotly (remplace seaborn)."""
        fig = go.Figure(go.Heatmap(
            z=df.values,
            x=df.columns.tolist(),
            y=df.index.tolist(),
            colorscale="YlGnBu",
            text=np.round(df.values, 3),
            texttemplate="%{text}",
            hovertemplate="Classe X: %{x}<br>Classe Y: %{y}<br>Similarité: %{z:.3f}<extra></extra>",
        ))
        fig.update_layout(
            title=title,
            height=500,
            xaxis_title="Classe",
            yaxis_title="Classe",
        )
        return ChartGenerator._apply_theme(fig)

    @staticmethod
    def create_separability_bar_chart(scores: Dict[str, float]) -> go.Figure:
        """Bar chart des indices de séparabilité par classe."""
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        classes = [item[0] for item in sorted_items]
        values = [item[1] for item in sorted_items]

        fig = go.Figure(go.Bar(
            x=classes, y=values,
            marker_color=px.colors.qualitative.Set2[:len(classes)],
            text=[f"{v:.2f}" for v in values],
            textposition="outside",
        ))
        fig.update_layout(
            title="Indice de séparabilité par classe",
            xaxis_title="Classe",
            yaxis_title="Séparabilité (variance inter / intra)",
            height=400,
        )
        return ChartGenerator._apply_theme(fig)
