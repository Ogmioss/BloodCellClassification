"""Module pour le calcul de similarité entre images."""

from pathlib import Path
from typing import Dict, List, Tuple
import random

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError
from sklearn.metrics.pairwise import cosine_similarity


def _to_hashable_keys(class_images: Dict[str, List[Path]]) -> tuple:
    """Convert class_images keys to a hashable tuple for cache key."""
    return tuple(sorted((k, len(v)) for k, v in class_images.items()))


@st.cache_data(show_spinner=False)
def _cached_compute_mean_images(
    _class_images: Dict[str, List[Path]],
    _cache_key: tuple,
    img_size: Tuple[int, int],
    max_images: int,
) -> Dict[str, np.ndarray]:
    """Cached version of mean images computation. Returns numpy arrays."""
    mean_images = {}
    for class_name, images in _class_images.items():
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
            mean_images[class_name] = np.mean(imgs, axis=0).astype(np.uint8)
    return mean_images


@st.cache_data(show_spinner=False)
def _cached_compute_cosine_similarity_matrix(
    _class_images: Dict[str, List[Path]],
    _cache_key: tuple,
    n_samples: int,
    image_size: Tuple[int, int],
) -> pd.DataFrame:
    """Cached version of cosine similarity matrix computation."""
    class_vectors = {}
    for class_name, image_paths in _class_images.items():
        if not image_paths:
            continue
        sample_paths = random.sample(image_paths, min(n_samples, len(image_paths)))
        vectors = []
        for path in sample_paths:
            try:
                img = Image.open(path).convert("RGB").resize(image_size)
                arr = np.array(img, dtype=np.float32).flatten()
                arr /= np.linalg.norm(arr) + 1e-8
                vectors.append(arr)
            except Exception:
                continue
        if vectors:
            class_vectors[class_name] = np.stack(vectors)

    classes = list(class_vectors.keys())
    similarity_matrix = np.zeros((len(classes), len(classes)))
    for i, cls_i in enumerate(classes):
        for j, cls_j in enumerate(classes):
            sims = cosine_similarity(class_vectors[cls_i], class_vectors[cls_j])
            similarity_matrix[i, j] = sims.mean()
    return pd.DataFrame(similarity_matrix, index=classes, columns=classes)


@st.cache_data(show_spinner=False)
def _cached_compute_separability_index(
    _class_images: Dict[str, List[Path]],
    _cache_key: tuple,
    n_samples: int,
    image_size: Tuple[int, int],
) -> Dict[str, float]:
    """Cached version of separability index computation."""
    class_vectors = {}
    for class_name, image_paths in _class_images.items():
        if not image_paths:
            continue
        sample_paths = random.sample(image_paths, min(n_samples, len(image_paths)))
        vectors = []
        for path in sample_paths:
            try:
                img = Image.open(path).convert("RGB").resize(image_size)
                vectors.append(np.array(img, dtype=np.float32).flatten())
            except Exception:
                continue
        if vectors:
            class_vectors[class_name] = np.stack(vectors)

    if len(class_vectors) < 2:
        return {}

    all_vectors = np.vstack(list(class_vectors.values()))
    global_mean = all_vectors.mean(axis=0)

    scores: Dict[str, float] = {}
    for cls, vecs in class_vectors.items():
        class_mean = vecs.mean(axis=0)
        intra_var = float(np.mean(np.sum((vecs - class_mean) ** 2, axis=1)))
        inter_var = float(np.sum((class_mean - global_mean) ** 2))
        scores[cls] = inter_var / (intra_var + 1e-8)
    return scores


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
        max_images: int = 100,
    ) -> Dict[str, Image.Image]:
        """Calcule les images moyennes pour chaque classe (cached)."""
        cache_key = _to_hashable_keys(class_images)
        arrays = _cached_compute_mean_images(class_images, cache_key, img_size, max_images)
        return {k: Image.fromarray(v) for k, v in arrays.items()}

    def compute_cosine_similarity_matrix(
        self,
        class_images: Dict[str, List[Path]],
    ) -> pd.DataFrame:
        """Calcule la matrice de similarite cosinus entre classes (cached)."""
        cache_key = _to_hashable_keys(class_images)
        return _cached_compute_cosine_similarity_matrix(
            class_images, cache_key, self.n_samples, self.image_size,
        )

    def compute_separability_index(
        self,
        class_images: Dict[str, List[Path]],
    ) -> Dict[str, float]:
        """Indice de separabilite par classe (cached)."""
        cache_key = _to_hashable_keys(class_images)
        return _cached_compute_separability_index(
            class_images, cache_key, self.n_samples, self.image_size,
        )
