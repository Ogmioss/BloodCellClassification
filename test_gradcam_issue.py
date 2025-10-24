"""Script de test pour diagnostiquer le problème de GradCAM."""

from pathlib import Path
import torch
import numpy as np
from PIL import Image
from src.utils.gradcam_analyzer import (
    load_resnet_model,
    get_image_transform,
    gradcam_analysis
)

CLASS_NAMES = [
    'basophil', 'eosinophil', 'erythroblast', 'ig',
    'lymphocyte', 'monocyte', 'neutrophil', 'platelet'
]

def test_model_loading():
    """Test si le modèle se charge correctement."""
    print("=" * 60)
    print("TEST 1: Chargement du modèle")
    print("=" * 60)
    
    checkpoint_path = Path("models/checkpoints/best_model.pth")
    model = load_resnet_model(checkpoint_path, num_classes=8)
    
    print(f"✓ Modèle chargé depuis: {checkpoint_path}")
    print(f"✓ Type: {type(model)}")
    print(f"✓ En mode eval: {not model.training}")
    print(f"✓ Nombre de paramètres: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    return model


def test_image_loading():
    """Test si les images se chargent correctement."""
    print("=" * 60)
    print("TEST 2: Chargement des images")
    print("=" * 60)
    
    data_dir = Path("src/data/raw/bloodcells_dataset")
    
    # Trouver quelques images
    test_images = []
    for class_dir in data_dir.iterdir():
        if class_dir.is_dir():
            images = list(class_dir.glob("*.jpeg"))[:2]
            test_images.extend(images)
            if len(test_images) >= 4:
                break
    
    print(f"✓ Images trouvées: {len(test_images)}")
    for img_path in test_images:
        img = Image.open(img_path)
        print(f"  - {img_path.name}: {img.size}, mode={img.mode}")
    print()
    
    return test_images


def test_gradcam_variability(model, test_images):
    """Test si GradCAM produit des résultats différents pour différentes images."""
    print("=" * 60)
    print("TEST 3: Variabilité des heatmaps GradCAM")
    print("=" * 60)
    
    transform = get_image_transform()
    heatmaps = []
    
    for img_path in test_images[:4]:
        img, heatmap_colored, overlay, pred_label, pred_conf = gradcam_analysis(
            model, img_path, CLASS_NAMES, transform
        )
        
        # Convertir heatmap en grayscale pour analyse
        heatmap_gray = np.mean(heatmap_colored, axis=2)
        heatmaps.append(heatmap_gray)
        
        print(f"\nImage: {img_path.name}")
        print(f"  Prédiction: {pred_label} ({pred_conf:.1f}%)")
        print(f"  Heatmap - Min: {heatmap_gray.min():.3f}, Max: {heatmap_gray.max():.3f}, Mean: {heatmap_gray.mean():.3f}")
        print(f"  Heatmap - Std: {heatmap_gray.std():.3f}")
        
        # Analyser la distribution des valeurs
        high_activation = (heatmap_gray > 200).sum()
        medium_activation = ((heatmap_gray > 100) & (heatmap_gray <= 200)).sum()
        low_activation = (heatmap_gray <= 100).sum()
        total_pixels = heatmap_gray.size
        
        print(f"  Distribution des activations:")
        print(f"    - Haute (>200): {high_activation/total_pixels*100:.1f}%")
        print(f"    - Moyenne (100-200): {medium_activation/total_pixels*100:.1f}%")
        print(f"    - Basse (<100): {low_activation/total_pixels*100:.1f}%")
    
    # Comparer les heatmaps entre elles
    print("\n" + "=" * 60)
    print("Comparaison entre heatmaps:")
    print("=" * 60)
    
    for i in range(len(heatmaps)):
        for j in range(i+1, len(heatmaps)):
            # Calculer la corrélation
            corr = np.corrcoef(heatmaps[i].flatten(), heatmaps[j].flatten())[0, 1]
            # Calculer la différence absolue moyenne
            mae = np.abs(heatmaps[i] - heatmaps[j]).mean()
            print(f"Heatmap {i+1} vs {j+1}: Corrélation={corr:.3f}, MAE={mae:.3f}")
    
    print()


def test_model_predictions():
    """Test si le modèle fait des prédictions différentes."""
    print("=" * 60)
    print("TEST 4: Variabilité des prédictions")
    print("=" * 60)
    
    checkpoint_path = Path("models/checkpoints/best_model.pth")
    model = load_resnet_model(checkpoint_path, num_classes=8)
    transform = get_image_transform()
    
    data_dir = Path("src/data/raw/bloodcells_dataset")
    
    predictions = []
    for class_dir in data_dir.iterdir():
        if class_dir.is_dir():
            images = list(class_dir.glob("*.jpeg"))[:2]
            for img_path in images:
                img = Image.open(img_path).convert("RGB")
                input_tensor = transform(img).unsqueeze(0)
                
                with torch.no_grad():
                    output = model(input_tensor)
                    probs = torch.nn.functional.softmax(output, dim=1)
                    pred_class = probs.argmax(dim=1).item()
                    pred_conf = probs[0, pred_class].item()
                
                predictions.append({
                    'class': class_dir.name,
                    'pred': CLASS_NAMES[pred_class],
                    'conf': pred_conf,
                    'file': img_path.name
                })
                
                if len(predictions) >= 10:
                    break
        if len(predictions) >= 10:
            break
    
    print(f"Prédictions sur {len(predictions)} images:")
    for p in predictions:
        correct = "✓" if p['class'] == p['pred'] else "✗"
        print(f"  {correct} {p['class']:15s} -> {p['pred']:15s} ({p['conf']*100:.1f}%) [{p['file']}]")
    
    # Vérifier la diversité des prédictions
    unique_preds = len(set(p['pred'] for p in predictions))
    print(f"\nDiversité: {unique_preds}/{len(CLASS_NAMES)} classes prédites")
    print()


if __name__ == "__main__":
    print("\n🔬 DIAGNOSTIC GRADCAM\n")
    
    # Test 1: Chargement du modèle
    model = test_model_loading()
    
    # Test 2: Chargement des images
    test_images = test_image_loading()
    
    # Test 3: Variabilité GradCAM
    test_gradcam_variability(model, test_images)
    
    # Test 4: Variabilité des prédictions
    test_model_predictions()
    
    print("=" * 60)
    print("✓ Tests terminés")
    print("=" * 60)
