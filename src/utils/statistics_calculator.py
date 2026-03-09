"""Module pour le calcul des statistiques."""

import math
from typing import Dict, List, Tuple

import numpy as np
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

    @staticmethod
    def calculate_shannon_entropy(class_counts: Dict[str, int]) -> float:
        """Entropie de Shannon normalisée (0 = 1 classe, 1 = uniforme)."""
        counts = np.array(list(class_counts.values()), dtype=float)
        n_classes = len(counts)
        if n_classes <= 1:
            return 0.0
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log2(probs))
        return float(entropy / math.log2(n_classes))

    @staticmethod
    def calculate_imbalance_ratio(class_counts: Dict[str, int]) -> float:
        """Ratio max/min des comptages de classes."""
        counts = list(class_counts.values())
        if not counts or min(counts) == 0:
            return float("inf")
        return max(counts) / min(counts)

    @staticmethod
    def calculate_cv(class_counts: Dict[str, int]) -> float:
        """Coefficient de variation (std/mean) des comptages."""
        counts = np.array(list(class_counts.values()), dtype=float)
        mean = counts.mean()
        if mean == 0:
            return 0.0
        return float(counts.std() / mean)

    @staticmethod
    def calculate_median_count(class_counts: Dict[str, int]) -> float:
        """Médiane des comptages de classes."""
        return float(np.median(list(class_counts.values())))

    @staticmethod
    def calculate_health_score(class_counts: Dict[str, int]) -> int:
        """
        Score de santé composite du dataset (0-100).

        Pondération : entropy 40%, CV inversé 30%, min_count 30%.
        """
        counts = np.array(list(class_counts.values()), dtype=float)
        n_classes = len(counts)
        if n_classes <= 1:
            return 0

        # Entropy score (0-1)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log2(probs)) / math.log2(n_classes)

        # CV score : CV=0 → 100, CV>=1 → 0
        cv = float(counts.std() / counts.mean()) if counts.mean() > 0 else 1.0
        cv_score = max(0.0, 1.0 - cv)

        # Min count score : ratio min/max (1 = parfait, 0 = très déséquilibré)
        min_max_ratio = float(counts.min() / counts.max()) if counts.max() > 0 else 0.0

        score = entropy * 40 + cv_score * 30 + min_max_ratio * 30
        return int(round(score))

    @staticmethod
    def calculate_dimension_outliers(
        widths: List[int],
        heights: List[int],
        threshold: float = 2.0,
    ) -> pd.DataFrame:
        """
        Détecte les images dont les dimensions s'écartent de > threshold × σ.

        Returns:
            DataFrame avec colonnes [width, height, ecart_w, ecart_h].
        """
        w = np.array(widths, dtype=float)
        h = np.array(heights, dtype=float)
        w_mean, w_std = w.mean(), w.std()
        h_mean, h_std = h.mean(), h.std()

        if w_std == 0 and h_std == 0:
            return pd.DataFrame(columns=["width", "height", "ecart_w", "ecart_h"])

        w_z = np.abs(w - w_mean) / (w_std if w_std > 0 else 1.0)
        h_z = np.abs(h - h_mean) / (h_std if h_std > 0 else 1.0)
        mask = (w_z > threshold) | (h_z > threshold)

        return pd.DataFrame({
            "width": w[mask].astype(int),
            "height": h[mask].astype(int),
            "ecart_w": np.round(w_z[mask], 2),
            "ecart_h": np.round(h_z[mask], 2),
        })
