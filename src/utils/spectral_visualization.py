"""
Module pour la visualisation spectrale des distributions RGB des cellules sanguines.

Ce module fournit des fonctions pour analyser et visualiser les distributions
de couleurs RGB dans les images de cellules sanguines.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from pathlib import Path
from typing import List, Union


def extract_pixel_data(image_path: Union[str, Path]) -> pd.DataFrame:
    """
    Extrait les données de pixels RGB d'une image.
    
    Args:
        image_path: Chemin vers l'image
    
    Returns:
        DataFrame avec colonnes x, y, red, green, blue
    """
    img = Image.open(image_path)
    
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    width, height = img.size
    pixels = list(img.getdata())
    
    pixel_list = []
    for index, pixel in enumerate(pixels):
        x = index % width
        y = index // width
        pixel_list.append({
            'x': x,
            'y': y,
            'red': pixel[0],
            'green': pixel[1],
            'blue': pixel[2]
        })
    
    df = pd.DataFrame(pixel_list)
    return df


def visualize_cell_types_distribution(
    base_dir: Union[str, Path], 
    cell_types: List[str],
    max_images_per_class: int = 30,
    sample_size: int = 10000
) -> plt.Figure:
    """
    Crée une visualisation des distributions RGB pour chaque type de cellule.
    
    Args:
        base_dir: Chemin vers le répertoire principal contenant les sous-dossiers de types de cellules
        cell_types: Liste des types de cellules à visualiser
        max_images_per_class: Nombre maximum d'images à analyser par classe (défaut: 30)
        sample_size: Taille de l'échantillon de pixels par classe (défaut: 10000)
    
    Returns:
        Figure matplotlib contenant les visualisations
    """
    base_dir = Path(base_dir)
    
    # Configuration du style
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['grid.color'] = '#cccccc'
    
    # Créer la grille de subplots
    n_types = len(cell_types)
    n_cols = 2
    n_rows = (n_types + 1) // 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 6 * n_rows))
    
    # Titre principal
    fig.suptitle('Distribution RGB par type de cellule', 
                 fontsize=20, 
                 y=0.995, 
                 fontweight='bold')
    
    # Aplatir le tableau d'axes
    if n_types > 1:
        axes = axes.ravel()
    else:
        axes = [axes]
    
    # Palette de couleurs
    colors = ['#FF6B6B', '#4ECB71', '#4A90E2']  # Rouge, Vert, Bleu
    labels = ['Rouge', 'Vert', 'Bleu']
    
    # Première passe : calculer les limites globales de l'axe y
    all_densities = []
    
    for idx, cell_type in enumerate(cell_types):
        cell_dir = base_dir / cell_type
        if not cell_dir.exists():
            continue
            
        # Récupérer les données RGB
        all_red, all_green, all_blue = [], [], []
        image_files = list(cell_dir.glob("*.jpg")) + list(cell_dir.glob("*.jpeg"))
        
        for img_path in image_files[:max_images_per_class]:
            try:
                df = extract_pixel_data(img_path)
                all_red.extend(df['red'])
                all_green.extend(df['green'])
                all_blue.extend(df['blue'])
            except Exception:
                continue
        
        if not all_red:
            continue
        
        # Échantillonnage pour la performance
        actual_sample_size = min(sample_size, len(all_red))
        indices = np.random.choice(len(all_red), actual_sample_size, replace=False)
        
        rgb_data = [
            np.array(all_red)[indices],
            np.array(all_green)[indices],
            np.array(all_blue)[indices]
        ]
        
        # Calculer KDE pour chaque canal de couleur
        for color_data in rgb_data:
            kde = sns.kdeplot(data=color_data, ax=axes[idx]).get_lines()[-1].get_data()
            all_densities.extend(kde[1])
    
    # Calculer les limites globales de l'axe y
    y_max = max(all_densities) if all_densities else 0.01
    
    # Deuxième passe : tracer avec l'axe y normalisé
    for idx, cell_type in enumerate(cell_types):
        cell_dir = base_dir / cell_type
        if not cell_dir.exists():
            axes[idx].text(0.5, 0.5, f'Dossier non trouvé\npour {cell_type}',
                          ha='center', va='center', transform=axes[idx].transAxes)
            continue
            
        # Effacer les tracés précédents
        axes[idx].clear()
        
        # Définir la couleur de fond
        axes[idx].set_facecolor('#f8f8f8')
        
        # Récupérer à nouveau les données RGB
        all_red, all_green, all_blue = [], [], []
        image_files = list(cell_dir.glob("*.jpg")) + list(cell_dir.glob("*.jpeg"))
        
        for img_path in image_files[:max_images_per_class]:
            try:
                df = extract_pixel_data(img_path)
                all_red.extend(df['red'])
                all_green.extend(df['green'])
                all_blue.extend(df['blue'])
            except Exception:
                continue
        
        if not all_red:
            axes[idx].text(0.5, 0.5, f'Aucune image valide\npour {cell_type}',
                          ha='center', va='center', transform=axes[idx].transAxes)
            continue
        
        actual_sample_size = min(sample_size, len(all_red))
        indices = np.random.choice(len(all_red), actual_sample_size, replace=False)
        
        rgb_data = [
            np.array(all_red)[indices],
            np.array(all_green)[indices],
            np.array(all_blue)[indices]
        ]
        
        # Tracer avec l'axe y normalisé
        for color_data, color, label in zip(rgb_data, colors, labels):
            sns.kdeplot(
                data=color_data,
                ax=axes[idx],
                color=color,
                label=label,
                fill=True,
                alpha=0.2,
                linewidth=2.5
            )
        
        # Définir les limites de l'axe y
        axes[idx].set_ylim(0, y_max * 1.1)  # Ajouter 10% de marge
        
        # Formatage du subplot
        axes[idx].set_xlabel('Valeur de pixel', fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('Densité', fontsize=12, fontweight='bold')
        axes[idx].tick_params(axis='both', labelsize=10)
        
        # Légende
        axes[idx].legend(fontsize=10, frameon=True, framealpha=0.9,
                        facecolor='white', edgecolor='gray')
        
        # Calculer les statistiques
        stats_text = (f'μ(R)={np.mean(rgb_data[0]):.1f}, '
                     f'μ(G)={np.mean(rgb_data[1]):.1f}, '
                     f'μ(B)={np.mean(rgb_data[2]):.1f}')
        
        # Titre amélioré
        axes[idx].set_title(f'Type: {cell_type}\n{stats_text}',
                          fontsize=14,
                          fontweight='bold',
                          pad=20)
        
        axes[idx].margins(0.1)
        axes[idx].grid(True, alpha=0.3)
    
    # Masquer les axes inutilisés
    for idx in range(len(cell_types), len(axes)):
        axes[idx].set_visible(False)
    
    # Ajuster la mise en page
    plt.tight_layout(h_pad=0.8, w_pad=0.8)
    fig.patch.set_facecolor('white')
    
    return fig
