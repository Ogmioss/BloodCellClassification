"""Module pour le calcul des statistiques."""

from typing import List, Tuple
import pandas as pd


class StatisticsCalculator:
    """Responsable du calcul des statistiques sur les données."""
    
    @staticmethod
    def create_class_dataframe(class_counts: dict) -> pd.DataFrame:
        """
        Crée un DataFrame à partir des comptages de classes.
        
        Args:
            class_counts: Dictionnaire {classe: nombre_images}
            
        Returns:
            DataFrame trié par nombre d'images décroissant
        """
        df = pd.DataFrame.from_dict(
            class_counts, 
            orient='index', 
            columns=['Nombre d\'images']
        )
        return df.sort_values(by='Nombre d\'images', ascending=False)
    
    @staticmethod
    def calculate_dimension_stats(
        widths: List[int], 
        heights: List[int]
    ) -> dict:
        """
        Calcule les statistiques sur les dimensions des images.
        
        Args:
            widths: Liste des largeurs
            heights: Liste des hauteurs
            
        Returns:
            Dictionnaire contenant les statistiques
        """
        if not widths or not heights:
            return {}
        
        return {
            'avg_width': sum(widths) / len(widths),
            'avg_height': sum(heights) / len(heights),
            'min_width': min(widths),
            'max_width': max(widths),
            'min_height': min(heights),
            'max_height': max(heights)
        }
    
    @staticmethod
    def calculate_aspect_ratios(
        widths: List[int], 
        heights: List[int]
    ) -> List[float]:
        """
        Calcule les ratios largeur/hauteur.
        
        Args:
            widths: Liste des largeurs
            heights: Liste des hauteurs
            
        Returns:
            Liste des ratios
        """
        return [w / h for w, h in zip(widths, heights) if h != 0]
    
    @staticmethod
    def get_most_represented_class(df: pd.DataFrame) -> Tuple[str, int]:
        """
        Récupère la classe la plus représentée.
        
        Args:
            df: DataFrame des classes
            
        Returns:
            Tuple (nom_classe, nombre_images)
        """
        return df.index[0], int(df.iloc[0, 0])
    
    @staticmethod
    def get_least_represented_class(df: pd.DataFrame) -> Tuple[str, int]:
        """
        Récupère la classe la moins représentée.
        
        Args:
            df: DataFrame des classes
            
        Returns:
            Tuple (nom_classe, nombre_images)
        """
        return df.index[-1], int(df.iloc[-1, 0])
