import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union, Any
from pydantic import BaseModel, Field, field_validator, DirectoryPath, FilePath


class ImagePathModel(BaseModel):
    """
    Modèle Pydantic pour valider les chemins d'images.
    """
    base_path: DirectoryPath = Field(..., description="Chemin vers le répertoire principal du dataset")
    subdir_name: str = Field(..., description="Nom du sous-répertoire (type de cellule)")
    image_name: str = Field(..., description="Nom du fichier image")
    
    @field_validator('subdir_name')
    def validate_subdir_exists(cls, v, info):
        """Vérifie que le sous-répertoire existe."""
        values = info.data
        if 'base_path' in values:
            subdir_path = os.path.join(values['base_path'], v)
            if not os.path.isdir(subdir_path):
                raise ValueError(f"Le sous-répertoire '{v}' n'existe pas dans {values['base_path']}")
        return v
    
    @property
    def full_path(self) -> str:
        """Retourne le chemin complet vers l'image."""
        return os.path.join(self.base_path, self.subdir_name, self.image_name)
    
    def exists(self) -> bool:
        """Vérifie si le fichier image existe."""
        return os.path.isfile(self.full_path)


class ImageVisualizer:
    """
    Classe pour visualiser les images du dataset de cellules sanguines.
    Permet de mapper les sous-répertoires aux listes d'images, d'afficher des images
    individuelles ou des échantillons pour chaque type de cellule.
    """
    
    def __init__(self, base_path: Union[str, os.PathLike]):
        """
        Initialise le visualiseur d'images.
        
        Args:
            base_path: Chemin vers le répertoire principal du dataset.
        """
        self.base_path = os.path.abspath(base_path)
        if not os.path.isdir(self.base_path):
            raise ValueError(f"Le chemin '{self.base_path}' n'est pas un répertoire valide.")
        
        # Crée le mapping des sous-répertoires aux listes d'images
        self.dataset_map = self._map_subdirs_to_files()
    
    def _map_subdirs_to_files(self) -> Dict[str, List[str]]:
        """
        Crée un dictionnaire qui mappe chaque sous-répertoire à la liste de ses fichiers.
        
        Returns:
            Dict[str, List[str]]: Dictionnaire où les clés sont les noms des sous-répertoires
                                et les valeurs sont les listes de noms de fichiers.
        """
        direct_mapping = {}
        
        for item_name in os.listdir(self.base_path):
            item_path = os.path.join(self.base_path, item_name)
            
            # On s'assure que c'est bien un sous-répertoire
            if os.path.isdir(item_path):
                # Liste uniquement les fichiers dans chaque sous-répertoire
                files_in_subdir = [f for f in os.listdir(item_path) 
                                  if os.path.isfile(os.path.join(item_path, f))]
                direct_mapping[item_name] = files_in_subdir
                
        return direct_mapping
    
    def get_available_cell_types(self) -> List[str]:
        """
        Retourne la liste des types de cellules disponibles (noms des sous-répertoires).
        
        Returns:
            List[str]: Liste des noms des sous-répertoires.
        """
        return list(self.dataset_map.keys())
    
    def get_image_count(self, subdir_name: str) -> int:
        """
        Retourne le nombre d'images disponibles pour un type de cellule donné.
        
        Args:
            subdir_name: Nom du sous-répertoire (type de cellule).
            
        Returns:
            int: Nombre d'images dans le sous-répertoire.
            
        Raises:
            ValueError: Si le sous-répertoire n'existe pas.
        """
        if subdir_name not in self.dataset_map:
            raise ValueError(f"Le sous-répertoire '{subdir_name}' n'existe pas. "
                            f"Sous-répertoires disponibles : {self.get_available_cell_types()}")
        
        return len(self.dataset_map[subdir_name])
    
    def get_image_path(self, subdir_name: str, index_img: int) -> str:
        """
        Construit le chemin complet vers un fichier image.
        
        Args:
            subdir_name: Nom du sous-répertoire (type de cellule).
            index_img: Index de l'image dans la liste.
            
        Returns:
            str: Chemin complet vers l'image.
            
        Raises:
            ValueError: Si le sous-répertoire n'existe pas ou si l'index est invalide.
        """
        if subdir_name not in self.dataset_map:
            raise ValueError(f"Le sous-répertoire '{subdir_name}' n'existe pas. "
                            f"Sous-répertoires disponibles : {self.get_available_cell_types()}")
        
        images_list = self.dataset_map[subdir_name]
        
        if not 0 <= index_img < len(images_list):
            raise ValueError(f"L'index {index_img} est hors limites pour le sous-répertoire "
                            f"'{subdir_name}' (contient {len(images_list)} images).")
        
        image_name = images_list[index_img]
        
        # Utilise le modèle Pydantic pour valider et construire le chemin
        path_model = ImagePathModel(
            base_path=self.base_path,
            subdir_name=subdir_name,
            image_name=image_name
        )
        
        return path_model.full_path
    
    def show_image(self, subdir_name: str, index_img: int, display_mode: str = 'gray', 
                  ax: Optional[plt.Axes] = None) -> bool:
        """
        Affiche une image spécifique du dataset.
        
        Args:
            subdir_name: Nom du sous-répertoire (type de cellule).
            index_img: Index de l'image à afficher.
            display_mode: Mode d'affichage de l'image. Options:
                          'gray' - niveaux de gris
                          'color' - couleur RGB
                          'red' - canal rouge uniquement
                          'green' - canal vert uniquement
                          'blue' - canal bleu uniquement
            ax: L'axe sur lequel afficher l'image. Si None, crée une nouvelle figure.
            
        Returns:
            bool: True si l'image a été affichée avec succès, False sinon.
        """
        try:
            img_path = self.get_image_path(subdir_name, index_img)
        except ValueError as e:
            print(f"Erreur : {e}")
            return False
        
        if not os.path.exists(img_path):
            print(f"Erreur : Le fichier '{img_path}' n'existe pas.")
            return False
        
        # Si ax est None, crée une nouvelle figure
        if ax is None:
            plt.figure(figsize=(8, 5))
            ax = plt.gca()  # Get Current Axis
        
        # Charge l'image en couleur pour tous les modes (on extraira les canaux si nécessaire)
        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"Erreur : Impossible de charger l'image '{img_path}'.")
            return False
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Affiche l'image selon le mode demandé
        if display_mode == 'gray':
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            ax.imshow(img_gray, cmap='gray')
            title_mode = "(Niveaux de gris)"
        elif display_mode == 'color':
            ax.imshow(img_rgb)
            title_mode = "(Couleur)"
        elif display_mode == 'red':
            # Extrait le canal rouge (R)
            red_channel = np.zeros_like(img_rgb)
            red_channel[:, :, 0] = img_rgb[:, :, 0]  # Copie uniquement le canal rouge
            ax.imshow(red_channel)
            title_mode = "(Canal Rouge)"
        elif display_mode == 'green':
            # Extrait le canal vert (G)
            green_channel = np.zeros_like(img_rgb)
            green_channel[:, :, 1] = img_rgb[:, :, 1]  # Copie uniquement le canal vert
            ax.imshow(green_channel)
            title_mode = "(Canal Vert)"
        elif display_mode == 'blue':
            # Extrait le canal bleu (B)
            blue_channel = np.zeros_like(img_rgb)
            blue_channel[:, :, 2] = img_rgb[:, :, 2]  # Copie uniquement le canal bleu
            ax.imshow(blue_channel)
            title_mode = "(Canal Bleu)"
        else:
            print(f"Mode d'affichage '{display_mode}' non reconnu. Utilisation du mode 'color'.")
            ax.imshow(img_rgb)
            title_mode = "(Couleur)"
        
        ax.set_title(f"{subdir_name} {title_mode}\nindex: {index_img}")
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Si ax était None (nouvelle figure), affiche la figure
        if ax == plt.gca():
            plt.show()
        
        return True
    
    def get_random_indices(self, subdir_name: str, count: int) -> List[int]:
        """
        Génère une liste d'indices aléatoires pour un sous-répertoire donné.
        
        Args:
            subdir_name: Nom du sous-répertoire (type de cellule).
            count: Nombre d'indices à générer.
            
        Returns:
            List[int]: Liste d'indices aléatoires.
            
        Raises:
            ValueError: Si le sous-répertoire n'existe pas ou si count est supérieur 
                       au nombre d'images disponibles.
        """
        if subdir_name not in self.dataset_map:
            raise ValueError(f"Le sous-répertoire '{subdir_name}' n'existe pas. "
                            f"Sous-répertoires disponibles : {self.get_available_cell_types()}")
        
        images_list = self.dataset_map[subdir_name]
        total_images = len(images_list)
        
        if count > total_images:
            raise ValueError(f"Impossible de sélectionner {count} images aléatoires "
                            f"dans le sous-répertoire '{subdir_name}' qui n'en contient que {total_images}.")
        
        # Génère des indices aléatoires sans répétition
        return random.sample(range(total_images), count)
    
    def display_sample_images_per_subdir(self, num_samples_per_subdir: int = 3, 
                                        display_mode: str = 'gray', 
                                        randomize: bool = False) -> plt.Figure:
        """
        Crée une figure avec des échantillons d'images pour chaque sous-répertoire (type de cellule)
        en utilisant des subplots.
        
        Args:
            num_samples_per_subdir: Nombre d'images à afficher pour chaque sous-répertoire.
            display_mode: Mode d'affichage des images. Options:
                          'gray' - niveaux de gris
                          'color' - couleur RGB
                          'red' - canal rouge uniquement
                          'green' - canal vert uniquement
                          'blue' - canal bleu uniquement
            randomize: Si True, sélectionne des images aléatoires. Sinon, utilise les premières images.
            
        Returns:
            plt.Figure: La figure matplotlib contenant les subplots avec les images.
        """
        subdirs = self.get_available_cell_types()
        num_subdirs = len(subdirs)
        
        if num_subdirs == 0:
            print("Aucun sous-répertoire trouvé. Rien à afficher.")
            return
        
        # Crée une figure avec des subplots
        fig, axes = plt.subplots(
            nrows=num_subdirs,
            ncols=num_samples_per_subdir,
            figsize=(5 * num_samples_per_subdir, 4.5 * num_subdirs),
            squeeze=False
        )
        
        for i, subdir_name in enumerate(subdirs):
            # Détermine les indices des images à afficher
            if randomize:
                try:
                    indices = self.get_random_indices(subdir_name, num_samples_per_subdir)
                except ValueError as e:
                    print(f"Avertissement pour '{subdir_name}': {e}")
                    # Utilise les indices disponibles si pas assez d'images
                    indices = list(range(min(num_samples_per_subdir, self.get_image_count(subdir_name))))
            else:
                # Utilise les premiers indices
                indices = list(range(min(num_samples_per_subdir, self.get_image_count(subdir_name))))
            
            # Affiche chaque image
            for j, idx in enumerate(indices):
                if j < num_samples_per_subdir:  # Sécurité supplémentaire
                    success = self.show_image(
                        subdir_name=subdir_name,
                        index_img=idx,
                        display_mode=display_mode,
                        ax=axes[i, j]
                    )
                    
                    if not success:
                        axes[i, j].set_title(f"{subdir_name}\nNon disponible")
                        axes[i, j].axis('off')
            
            # Désactive les axes des subplots restants si pas assez d'images
            for j_empty in range(len(indices), num_samples_per_subdir):
                axes[i, j_empty].set_title(f"{subdir_name}\nPas assez d'images")
                axes[i, j_empty].axis('off')
        
        # Détermine le titre en fonction du mode d'affichage
        mode_title = {
            'gray': "en niveaux de gris",
            'color': "en couleur",
            'red': "canal rouge",
            'green': "canal vert",
            'blue': "canal bleu"
        }.get(display_mode, "")
        
        fig.suptitle(f"Échantillons d'images par type de cellule ({mode_title})", fontsize=16, y=1.01)
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        return fig


# Exemple d'utilisation:
if __name__ == "__main__":
    # Chemin vers le dataset
    data_path = '/home/anthony-sauvage/repo/BloodCellClassification/app/data/raw/bloodcells_dataset'
    
    try:
        # Initialise le visualiseur
        visualizer = ImageVisualizer(data_path)
        
        # Affiche les types de cellules disponibles
        print(f"Types de cellules disponibles : {visualizer.get_available_cell_types()}")
        
        # Exemples d'utilisation avec les différents modes d'affichage
        
        # Affiche une image spécifique en niveaux de gris
        # visualizer.show_image(subdir_name="EOSINOPHIL", index_img=0, display_mode='gray')
        
        # Affiche une image spécifique en couleur
        # visualizer.show_image(subdir_name="EOSINOPHIL", index_img=0, display_mode='color')
        
        # Affiche une image spécifique avec uniquement le canal rouge
        # visualizer.show_image(subdir_name="EOSINOPHIL", index_img=0, display_mode='red')
        
        # Affiche une image spécifique avec uniquement le canal vert
        # visualizer.show_image(subdir_name="EOSINOPHIL", index_img=0, display_mode='green')
        
        # Affiche une image spécifique avec uniquement le canal bleu
        # visualizer.show_image(subdir_name="EOSINOPHIL", index_img=0, display_mode='blue')
        
        # Affiche des échantillons pour chaque type de cellule (images aléatoires) en couleur
        # visualizer.display_sample_images_per_subdir(num_samples_per_subdir=3, display_mode='color', randomize=True)
        
        # Affiche des échantillons pour chaque type de cellule avec uniquement le canal rouge
        # visualizer.display_sample_images_per_subdir(num_samples_per_subdir=3, display_mode='red', randomize=True)
        
    except Exception as e:
        print(f"Une erreur est survenue : {e}")

