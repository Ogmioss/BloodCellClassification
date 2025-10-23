"""Module pour le calcul de similarité entre images."""

from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from sklearn.metrics.pairwise import cosine_similarity
import random


class ImageSimilarityCalculator:
    """Calcule la similarité cosinus entre images de différentes classes."""
    
    def __init__(self, n_samples: int = 20, image_size: Tuple[int, int] = (64, 64)) -> None:
        """
        Initialise le calculateur de similarité.
        
        Args:
            n_samples: Nombre d'images à échantillonner par classe
            image_size: Taille de redimensionnement des images
        """
        self.n_samples = n_samples
        self.image_size = image_size
    
    def compute_mean_images(
        self, 
        class_images: Dict[str, List[Path]], 
        img_size: Tuple[int, int] = (128, 128),
        max_images: int = 100
    ) -> Dict[str, Image.Image]:
        """
        Calcule les images moyennes pour chaque classe.
        
        Args:
            class_images: Dictionnaire {classe: liste_chemins}
            img_size: Taille de standardisation des images
            max_images: Nombre maximum d'images à moyenner
            
        Returns:
            Dictionnaire {classe: image_moyenne}
        """
        mean_images = {}
        
        for class_name, images in class_images.items():
            if not images:
                continue
            
            imgs = []
            for img_path in images[:max_images]:
                try:
                    img = Image.open(img_path).convert("RGB").resize(img_size)
                    imgs.append(np.array(img, dtype=np.float32))
                except UnidentifiedImageError:
                    continue
            
            if imgs:
                mean_img = np.mean(imgs, axis=0).astype(np.uint8)
                mean_images[class_name] = Image.fromarray(mean_img)
        
        return mean_images
    
    def compute_cosine_similarity_matrix(
        self, 
        class_images: Dict[str, List[Path]]
    ) -> pd.DataFrame:
        """
        Calcule la matrice de similarité cosinus entre classes.
        
        Args:
            class_images: Dictionnaire {classe: liste_chemins}
            
        Returns:
            DataFrame contenant la matrice de similarité
        """
        # Vectorisation des images par classe
        class_vectors = {}
        
        for class_name, image_paths in class_images.items():
            if not image_paths:
                continue
            
            sample_paths = random.sample(
                image_paths, 
                min(self.n_samples, len(image_paths))
            )
            vectors = []
            
            for path in sample_paths:
                try:
                    img = Image.open(path).convert("RGB").resize(self.image_size)
                    arr = np.array(img, dtype=np.float32).flatten()
                    # Normalisation L2
                    arr /= np.linalg.norm(arr) + 1e-8
                    vectors.append(arr)
                except Exception:
                    continue
            
            if vectors:
                class_vectors[class_name] = np.stack(vectors)
        
        # Calcul de la similarité moyenne entre classes
        classes = list(class_vectors.keys())
        similarity_matrix = np.zeros((len(classes), len(classes)))
        
        for i, cls_i in enumerate(classes):
            for j, cls_j in enumerate(classes):
                sims = cosine_similarity(class_vectors[cls_i], class_vectors[cls_j])
                similarity_matrix[i, j] = sims.mean()
        
        # Conversion en DataFrame
        return pd.DataFrame(similarity_matrix, index=classes, columns=classes)
