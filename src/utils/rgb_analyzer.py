"""Module pour l'analyse des distributions RGB des images."""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError
from scipy.stats import gaussian_kde


def _to_hashable_keys(class_images: Dict[str, List[Path]]) -> tuple:
    """Convert class_images keys to a hashable tuple for cache key."""
    return tuple(sorted((k, len(v)) for k, v in class_images.items()))


@st.cache_data(show_spinner=False)
def _cached_compute_rgb_distributions(
    _class_images: Dict[str, List[Path]],
    _cache_key: tuple,
    sample_size: int,
    max_images: int,
) -> Tuple[Dict[str, Tuple[np.ndarray, Dict[str, np.ndarray]]], float]:
    """Cached version of RGB distribution computation."""
    rgb_distributions: Dict[str, Tuple[np.ndarray, Dict[str, np.ndarray]]] = {}
    all_densities: list = []

    for cell_type, images in sorted(_class_images.items()):
        if not images:
            continue
        all_red, all_green, all_blue = [], [], []
        for img_path in images[:max_images]:
            try:
                img = Image.open(img_path).convert("RGB")
                arr = np.array(img)
                all_red.extend(arr[:, :, 0].flatten())
                all_green.extend(arr[:, :, 1].flatten())
                all_blue.extend(arr[:, :, 2].flatten())
            except UnidentifiedImageError:
                continue
        n = min(sample_size, len(all_red))
        if n == 0:
            continue
        indices = np.random.choice(len(all_red), n, replace=False)
        rgb_data = {
            "R": np.array(all_red)[indices],
            "G": np.array(all_green)[indices],
            "B": np.array(all_blue)[indices],
        }
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


@st.cache_data(show_spinner=False)
def _cached_compute_class_rgb_stats(
    _class_images: Dict[str, List[Path]],
    _cache_key: tuple,
    max_images: int,
) -> pd.DataFrame:
    """Cached version of class RGB stats computation."""
    rows = []
    for cell_type, images in sorted(_class_images.items()):
        if not images:
            continue
        all_r, all_g, all_b = [], [], []
        for img_path in images[:max_images]:
            try:
                img = Image.open(img_path).convert("RGB")
                arr = np.array(img)
                all_r.extend(arr[:, :, 0].flatten())
                all_g.extend(arr[:, :, 1].flatten())
                all_b.extend(arr[:, :, 2].flatten())
            except UnidentifiedImageError:
                continue
        if not all_r:
            continue
        r, g, b = np.array(all_r), np.array(all_g), np.array(all_b)
        mu_r, mu_g, mu_b = r.mean(), g.mean(), b.mean()
        luminosity = 0.299 * mu_r + 0.587 * mu_g + 0.114 * mu_b
        contrast = max(mu_r, mu_g, mu_b) - min(mu_r, mu_g, mu_b)
        rows.append({
            "Classe": cell_type,
            "μ(R)": round(mu_r, 1),
            "μ(G)": round(mu_g, 1),
            "μ(B)": round(mu_b, 1),
            "σ(R)": round(float(r.std()), 1),
            "σ(G)": round(float(g.std()), 1),
            "σ(B)": round(float(b.std()), 1),
            "Luminosité": round(luminosity, 1),
            "Contraste": round(contrast, 1),
        })
    return pd.DataFrame(rows).set_index("Classe") if rows else pd.DataFrame()


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
        class_images: Dict[str, List[Path]],
    ) -> Tuple[Dict[str, Tuple[np.ndarray, Dict[str, np.ndarray]]], float]:
        """Calcule les distributions RGB pour chaque classe (cached)."""
        cache_key = _to_hashable_keys(class_images)
        return _cached_compute_rgb_distributions(
            class_images, cache_key, self.sample_size, self.max_images,
        )

    def compute_class_rgb_stats(
        self,
        class_images: Dict[str, List[Path]],
    ) -> pd.DataFrame:
        """Calcule les statistiques RGB par classe (cached)."""
        cache_key = _to_hashable_keys(class_images)
        return _cached_compute_class_rgb_stats(
            class_images, cache_key, self.max_images,
        )
