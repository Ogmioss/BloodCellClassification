"""Module pour la génération de graphiques."""

from typing import List
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


class ChartGenerator:
    """Responsable de la génération des graphiques."""
    
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
        return px.histogram(
            data, 
            nbins=nbins, 
            labels={'value': x_label}, 
            title=title
        )
    
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
        return px.pie(
            df, 
            names=df.index, 
            values=values_column, 
            title=title
        )
    
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
