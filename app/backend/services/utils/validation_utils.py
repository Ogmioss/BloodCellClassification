import cv2
import os
import numpy as np

def is_image_corrupted(image_path):
    """
    Détecte si une image JPG est corrompue ou non.
    
    Args:
        image_path (str): Chemin vers l'image à vérifier
        
    Returns:
        bool: True si l'image est corrompue, False sinon
    """
    # Vérifier si le fichier existe
    if not os.path.isfile(image_path):
        print(f"Erreur: Le fichier {image_path} n'existe pas.")
        return True
    
    # Vérifier si le fichier est vide
    if os.path.getsize(image_path) == 0:
        print(f"Erreur: Le fichier {image_path} est vide.")
        return True
    
    try:
        # Essayer de lire l'image avec OpenCV
        img = cv2.imread(image_path)
        
        # Vérifier si l'image a été correctement chargée
        if img is None:
            print(f"Erreur: Impossible de lire l'image {image_path}.")
            return True
        
        # Vérifier si l'image a des dimensions valides
        if img.shape[0] <= 0 or img.shape[1] <= 0:
            print(f"Erreur: L'image {image_path} a des dimensions invalides.")
            return True
            
        # Vérifier si l'image contient des données valides
        if np.sum(img) == 0:
            print(f"Erreur: L'image {image_path} ne contient que des pixels noirs.")
            return True
        
        # L'image semble valide
        return False
        
    except Exception as e:
        print(f"Erreur lors de la lecture de l'image {image_path}: {str(e)}")
        return True
