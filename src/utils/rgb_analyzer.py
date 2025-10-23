"""Module pour l'analyse des distributions RGB des images."""

from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image, UnidentifiedImageError
from scipy.stats import gaussian_kde


class RGBAnalyzer:
    """Analyse les distributions RGB des images par classe."""
    
    def __init__(self, sample_size: int = 10000, max_images: int = 30) -> None:
        """
        Initialise l'analyseur RGB.
        
        Args:
            sample_size: Nombre de pixels à échantillonner par classe
            max_images: Nombre maximum d'images à analyser par classe
        """
        self.sample_size = sample_size
        self.max_images = max_images
    
    def compute_rgb_distributions(
        self, 
        class_images: Dict[str, List[Path]]
    ) -> Tuple[Dict[str, Tuple[np.ndarray, Dict[str, np.ndarray]]], float]:
        """
        Calcule les distributions RGB pour chaque classe.
        
        Args:
            class_images: Dictionnaire {classe: liste_chemins}
            
        Returns:
            Tuple contenant:
            - Dictionnaire {classe: (x_vals, {canal: densités})}
            - Valeur maximale de densité (pour normaliser l'axe Y)
        """
        rgb_distributions = {}
        all_densities = []
        
        for cell_type, images in sorted(class_images.items()):
            if not images:
                continue
            
            # Collecte des valeurs RGB
            all_red, all_green, all_blue = [], [], []
            
            for img_path in images[:self.max_images]:
                try:
                    img = Image.open(img_path).convert("RGB")
                    arr = np.array(img)
                    all_red.extend(arr[:, :, 0].flatten())
                    all_green.extend(arr[:, :, 1].flatten())
                    all_blue.extend(arr[:, :, 2].flatten())
                except UnidentifiedImageError:
                    continue
            
            # Échantillonnage pour performance
            sample_size = min(self.sample_size, len(all_red))
            if sample_size == 0:
                continue
            
            indices = np.random.choice(len(all_red), sample_size, replace=False)
            rgb_data = {
                "R": np.array(all_red)[indices],
                "G": np.array(all_green)[indices],
                "B": np.array(all_blue)[indices],
            }
            
            # Calcul des densités KDE
            x_vals = np.linspace(0, 255, 256)
            densities = {}
            
            for channel in ["R", "G", "B"]:
                kde = gaussian_kde(rgb_data[channel])
                y_vals = kde(x_vals)
                densities[channel] = y_vals
                all_densities.extend(y_vals)
            
            rgb_distributions[cell_type] = (x_vals, densities)
        
        y_max = max(all_densities) if all_densities else 0.01
        
        return rgb_distributions, y_max
