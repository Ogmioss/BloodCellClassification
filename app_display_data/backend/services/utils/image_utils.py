import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

def show_image(dataset_map, base_path, subdir_name, index_img, display_mode='gray', ax=None):
    """
    Affiche une image spécifique du dataset en utilisant le mapping direct.
    
    Args:
        dataset_map (dict): Le dictionnaire de mapping.
        base_path (str): Le chemin de base du dataset.
        subdir_name (str): Le nom du sous-répertoire.
        index_img (int): L'index de l'image à afficher.
        display_mode (str): Mode d'affichage de l'image. Options:
                          'gray' - niveaux de gris
                          'color' - couleur RGB
                          'red' - canal rouge uniquement
                          'green' - canal vert uniquement
                          'blue' - canal bleu uniquement
        ax (matplotlib.axes.Axes, optional): L'axe sur lequel afficher l'image. 
                                           Si None, crée une nouvelle figure.
    
    Returns:
        bool: True si l'image a été affichée avec succès, False sinon.
    """
    # Vérifie si le sous-répertoire existe dans le mapping
    if subdir_name not in dataset_map:
        print(f"Erreur : Le sous-répertoire '{subdir_name}' n'existe pas. Sous-répertoires disponibles : {list(dataset_map.keys())}")
        return False
    
    images_list = dataset_map[subdir_name]
    
    # Vérifie si l'index est valide
    if not 0 <= index_img < len(images_list):
        print(f"Erreur : L'index {index_img} est hors limites pour le sous-répertoire '{subdir_name}' (contient {len(images_list)} images).")
        return False
        
    image_name = images_list[index_img]
    img_path = os.path.join(base_path, subdir_name, image_name)
    
    if not os.path.exists(img_path):
        print(f"Erreur : Le fichier '{img_path}' n'existe pas.")
        return False
    
    # Si ax est None, crée une nouvelle figure (comportement original)
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
    
    ax.set_title(f"{subdir_name} {title_mode}" + "\n" + f"source: {image_name}")
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Si ax était None (nouvelle figure), affiche la figure
    if ax == plt.gca():
        plt.show()
    
    return True

# Fonction pour afficher des échantillons d'images pour chaque sous-répertoire
def display_sample_images_per_subdir(dataset_map, base_path, num_samples_per_subdir=3, display_mode='gray'):
    """
    Affiche un certain nombre d'images échantillons pour chaque sous-répertoire (type de cellule)
    en utilisant des subplots et la fonction show_image.

    Args:
        dataset_map (dict): Le dictionnaire mappant les sous-répertoires aux listes d'images.
        base_path (str): Le chemin de base du dataset.
        num_samples_per_subdir (int): Le nombre d'images à afficher pour chaque sous-répertoire.
        display_mode (str): Mode d'affichage des images. Options:
                          'gray' - niveaux de gris
                          'color' - couleur RGB
                          'red' - canal rouge uniquement
                          'green' - canal vert uniquement
                          'blue' - canal bleu uniquement
    """
    subdirs = list(dataset_map.keys())
    num_subdirs = len(subdirs)

    if num_subdirs == 0:
        print("Le dictionnaire dataset_map est vide. Aucune image à afficher.")
        return

    # Crée une figure avec des subplots
    fig, axes = plt.subplots(
        nrows=num_subdirs, 
        ncols=num_samples_per_subdir, 
        figsize=(5 * num_samples_per_subdir, 4.5 * num_subdirs),
        squeeze=False
    )

    for i, subdir_name in enumerate(subdirs):
        for j in range(num_samples_per_subdir):
            # Utilise show_image pour afficher l'image sur le subplot courant
            success = show_image(
                dataset_map=dataset_map,
                base_path=base_path,
                subdir_name=subdir_name,
                index_img=j,  # Utilise les j premières images
                display_mode=display_mode,
                ax=axes[i, j]  # Passe l'axe du subplot courant
            )
            
            # Si show_image échoue, désactive l'axe
            if not success:
                axes[i, j].set_title(f"{subdir_name}\nNon disponible")
                axes[i, j].axis('off')

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
    plt.show()

# Exemple d'utilisation
if __name__ == "__main__":
    # Assurez-vous que ces variables sont définies et contiennent les bonnes valeurs
    # data_path = '/home/anthony-sauvage/repo/BloodCellClassification/app/data/raw/bloodcells_dataset'
    # dataset_map = map_subdirs_to_files_direct(data_path)
    
    # Pour afficher des échantillons en différents modes
    # display_sample_images_per_subdir(dataset_map, data_path, num_samples_per_subdir=3, display_mode='gray')
    # display_sample_images_per_subdir(dataset_map, data_path, num_samples_per_subdir=3, display_mode='color')
    # display_sample_images_per_subdir(dataset_map, data_path, num_samples_per_subdir=3, display_mode='red')
    # display_sample_images_per_subdir(dataset_map, data_path, num_samples_per_subdir=3, display_mode='green')
    # display_sample_images_per_subdir(dataset_map, data_path, num_samples_per_subdir=3, display_mode='blue')
    pass
