"""Module pour générer les graphiques de distribution RGB."""

from typing import Dict, Tuple
import numpy as np
import plotly.graph_objects as go


class RGBChartGenerator:
    """Génère des graphiques de distribution RGB avec Plotly."""
    
    # Configuration des couleurs et labels
    COLORS = {"R": "#FF6B6B", "G": "#4ECB71", "B": "#4A90E2"}
    LABELS = {"R": "Rouge", "G": "Vert", "B": "Bleu"}
    
    def create_legend_figure(self) -> go.Figure:
        """
        Crée une figure contenant uniquement la légende commune.
        
        Returns:
            Figure Plotly avec légende
        """
        fig = go.Figure()
        
        for channel in ["R", "G", "B"]:
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="lines",
                    line=dict(color=self.COLORS[channel], width=3),
                    name=self.LABELS[channel],
                )
            )
        
        fig.update_layout(
            showlegend=True,
            height=50,
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.0,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(255,255,255,0.7)",
                bordercolor="rgba(180,180,180,0.4)",
                borderwidth=1,
                font=dict(size=12)
            ),
        )
        
        return fig
    
    def create_rgb_distribution_figure(
        self, 
        cell_type: str,
        x_vals: np.ndarray,
        densities: Dict[str, np.ndarray],
        y_max: float
    ) -> go.Figure:
        """
        Crée un graphique de distribution RGB pour une classe.
        
        Args:
            cell_type: Nom du type de cellule
            x_vals: Valeurs de l'axe X (0-255)
            densities: Dictionnaire {canal: densités}
            y_max: Valeur maximale pour l'axe Y
            
        Returns:
            Figure Plotly
        """
        fig = go.Figure()
        
        for channel in ["R", "G", "B"]:
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=densities[channel],
                    mode="lines",
                    line=dict(color=self.COLORS[channel], width=3),
                    fill="tozeroy",
                    opacity=0.3,
                    showlegend=False
                )
            )
        
        fig.update_layout(
            title=dict(
                text=f"<b>{cell_type}</b>",
                font=dict(size=20, color="#222", family="Arial Black"),
                x=0.5,
                y=0.93,
            ),
            xaxis=dict(
                title="Valeur de pixel (0–255)",
                showgrid=True,
                gridcolor="rgba(220,220,220,0.3)",
            ),
            yaxis=dict(
                title="Densité",
                showgrid=True,
                gridcolor="rgba(220,220,220,0.3)",
                range=[0, y_max * 1.1],
            ),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=40, r=20, t=60, b=40),
            hovermode="x unified",
        )
        
        return fig
