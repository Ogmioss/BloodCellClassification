"""Utility module for Grad-CAM analysis on blood cell images."""

from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from captum.attr import LayerAttribution, LayerGradCam
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms


def load_resnet_model(checkpoint_path: Path, num_classes: int = 8) -> torch.nn.Module:
    """
    Load a ResNet18 model from checkpoint.
    
    Args:
        checkpoint_path: Path to the model checkpoint file
        num_classes: Number of output classes
        
    Returns:
        Loaded and evaluated PyTorch model
    """
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()
    return model


def get_image_transform() -> transforms.Compose:
    """
    Get the standard image transformation pipeline for ResNet.
    
    Returns:
        Composed transforms for image preprocessing
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])


def load_dataset_images(data_dir: Path) -> Tuple[Dict[str, int], Dict[str, List[Path]]]:
    """
    Load and validate images from dataset directory.
    
    Args:
        data_dir: Path to the dataset directory
        
    Returns:
        Tuple of (class_counts, class_images) dictionaries
    """
    counts = {}
    class_images = {}
    
    for d in data_dir.iterdir():
        if d.is_dir():
            image_files = [
                f for f in list(d.glob("*.jpeg")) + list(d.glob("*.jpg")) 
                if not f.name.startswith(".")
            ]
            valid_images = []
            for f in image_files:
                try:
                    img = Image.open(f)
                    valid_images.append(f)
                except UnidentifiedImageError:
                    st.warning(f"Fichier ignoré (non-image ou corrompu) : {f.name}")
            counts[d.name] = len(valid_images)
            class_images[d.name] = valid_images
    
    return counts, class_images


def gradcam_analysis(
    model: torch.nn.Module,
    image_path: Path,
    class_names: List[str],
    transform: transforms.Compose
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str, float]:
    """
    Perform Grad-CAM analysis on a single image.
    
    Args:
        model: PyTorch model for inference
        image_path: Path to the input image
        class_names: List of class names
        transform: Image transformation pipeline
        
    Returns:
        Tuple of (original_image, heatmap, overlay, predicted_label, confidence)
    """
    # Ensure model is in eval mode and gradients are cleared
    model.eval()
    model.zero_grad()
    
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)
    
    # Prediction with no_grad context for efficiency
    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)
        pred_class = probs.argmax(dim=1).item()
        pred_conf = probs[0, pred_class].item() * 100
    
    # Clear any residual gradients before GradCAM
    model.zero_grad()
    
    # Grad-CAM computation (requires gradients)
    target_layer = model.layer4[-1]
    gradcam = LayerGradCam(model, target_layer)
    
    # Enable gradients only for attribution
    input_tensor.requires_grad = True
    attribution = gradcam.attribute(input_tensor, target=pred_class)
    
    # Clean up gradients after attribution
    model.zero_grad()
    
    upsampled_attr = LayerAttribution.interpolate(attribution, input_tensor.shape[2:])
    heatmap = upsampled_attr.squeeze().detach().cpu().numpy()
    heatmap = np.maximum(heatmap, 0)
    
    # Avoid division by zero
    if heatmap.max() > 0:
        heatmap /= heatmap.max()
    else:
        heatmap = np.zeros_like(heatmap)
    
    # Create overlay
    img = np.array(image)
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    overlay = np.uint8(0.6 * heatmap_colored + 0.4 * img)
    
    pred_label = class_names[pred_class] if pred_class < len(class_names) else f"Classe {pred_class}"
    
    return img, heatmap_colored, overlay, pred_label, pred_conf


def display_gradcam_results(
    img: np.ndarray,
    heatmap_colored: np.ndarray,
    overlay: np.ndarray,
    pred_label: str,
    pred_conf: float
) -> None:
    """
    Display Grad-CAM analysis results in Streamlit.
    
    Args:
        img: Original image array
        heatmap_colored: Colored heatmap array
        overlay: Overlay of heatmap on original image
        pred_label: Predicted class label
        pred_conf: Prediction confidence percentage
    """
    cols = st.columns(3)
    cols[0].image(img, caption="Image originale", use_container_width=True)
    cols[1].image(heatmap_colored, caption="Carte Grad-CAM", use_container_width=True)
    cols[2].image(overlay, caption="Overlay", use_container_width=True)
    
    st.markdown(
        f"<div style='text-align:center; font-size:16px; color:#333;'>"
        f"<b> Prédiction du modèle : </b> {pred_label} ({pred_conf:.1f}%)"
        f"</div>",
        unsafe_allow_html=True
    )
    st.markdown("---")
