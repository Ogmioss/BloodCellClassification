"""Module pour le chargement et la validation des images."""

from pathlib import Path
from typing import List, Tuple, Optional
from PIL import Image, UnidentifiedImageError
from dataclasses import dataclass


@dataclass
class ImageInfo:
    """Information sur une image."""
    
    path: Path
    width: int
    height: int
    class_name: str


class ImageLoader:
    """Responsable du chargement et de la validation des images."""
    
    SUPPORTED_EXTENSIONS = [".jpeg", ".jpg", ".png"]
    
    def __init__(self, ignore_hidden: bool = True) -> None:
        """
        Initialise le loader d'images.
        
        Args:
            ignore_hidden: Si True, ignore les fichiers cachés (commençant par .)
        """
        self.ignore_hidden = ignore_hidden
    
    def load_image(self, image_path: Path) -> Optional[Image.Image]:
        """
        Charge une image depuis un chemin.
        
        Args:
            image_path: Chemin vers l'image
            
        Returns:
            Image PIL ou None si l'image est invalide
        """
        try:
            return Image.open(image_path)
        except UnidentifiedImageError:
            return None
    
    def get_image_files(self, directory: Path) -> List[Path]:
        """
        Récupère tous les fichiers images d'un répertoire.
        
        Args:
            directory: Répertoire à scanner
            
        Returns:
            Liste des chemins vers les images
        """
        image_files = []
        for ext in self.SUPPORTED_EXTENSIONS:
            image_files.extend(directory.glob(f"*{ext}"))
        
        if self.ignore_hidden:
            image_files = [f for f in image_files if not f.name.startswith(".")]
        
        return image_files
    
    def load_and_validate_images(
        self, 
        directory: Path, 
        class_name: str
    ) -> Tuple[List[ImageInfo], List[Path]]:
        """
        Charge et valide toutes les images d'un répertoire.
        
        Args:
            directory: Répertoire contenant les images
            class_name: Nom de la classe associée
            
        Returns:
            Tuple (liste des ImageInfo valides, liste des chemins invalides)
        """
        image_files = self.get_image_files(directory)
        valid_images: List[ImageInfo] = []
        invalid_images: List[Path] = []
        
        for image_path in image_files:
            img = self.load_image(image_path)
            if img is not None:
                valid_images.append(
                    ImageInfo(
                        path=image_path,
                        width=img.width,
                        height=img.height,
                        class_name=class_name
                    )
                )
            else:
                invalid_images.append(image_path)
        
        return valid_images, invalid_images
