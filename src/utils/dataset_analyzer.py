"""Module pour l'analyse du dataset."""

from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass

from src.utils.image_loader import ImageLoader


@dataclass
class DatasetStats:
    """Statistiques du dataset."""

    class_counts: Dict[str, int]
    class_images: Dict[str, List[Path]]
    widths: List[int]
    heights: List[int]
    invalid_images: List[Path]
    class_widths: Dict[str, List[int]]
    class_heights: Dict[str, List[int]]


class DatasetAnalyzer:
    """Responsable de l'analyse du dataset d'images."""
    
    def __init__(self, image_loader: ImageLoader) -> None:
        """
        Initialise l'analyseur de dataset.
        
        Args:
            image_loader: Instance de ImageLoader pour charger les images
        """
        self.image_loader = image_loader
    
    def analyze_dataset(self, data_dir: Path) -> DatasetStats:
        """
        Analyse un dataset d'images organisé par classes.
        
        Args:
            data_dir: Répertoire racine du dataset
            
        Returns:
            DatasetStats contenant les statistiques du dataset
        """
        class_counts: Dict[str, int] = {}
        class_images: Dict[str, List[Path]] = {}
        all_widths: List[int] = []
        all_heights: List[int] = []
        all_invalid_images: List[Path] = []
        class_widths: Dict[str, List[int]] = {}
        class_heights: Dict[str, List[int]] = {}

        # Parcours des sous-dossiers (classes)
        for class_dir in data_dir.iterdir():
            if not class_dir.is_dir():
                continue

            class_name = class_dir.name
            valid_images, invalid_images = self.image_loader.load_and_validate_images(
                class_dir,
                class_name
            )

            # Stockage des résultats
            class_counts[class_name] = len(valid_images)
            class_images[class_name] = [img.path for img in valid_images]
            widths = [img.width for img in valid_images]
            heights = [img.height for img in valid_images]
            all_widths.extend(widths)
            all_heights.extend(heights)
            class_widths[class_name] = widths
            class_heights[class_name] = heights
            all_invalid_images.extend(invalid_images)

        return DatasetStats(
            class_counts=class_counts,
            class_images=class_images,
            widths=all_widths,
            heights=all_heights,
            invalid_images=all_invalid_images,
            class_widths=class_widths,
            class_heights=class_heights,
        )
    
    def get_class_directories(self, data_dir: Path) -> List[str]:
        """
        Récupère la liste des noms de classes (sous-dossiers).
        
        Args:
            data_dir: Répertoire racine du dataset
            
        Returns:
            Liste triée des noms de classes
        """
        return sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
