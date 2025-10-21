"""
Inference Service

Single Responsibility: Handles model inference and prediction.
"""

from typing import Dict, List, Tuple
from pathlib import Path
import torch
from PIL import Image

from src.models.base_classifier import BaseClassifier
from src.services.data_transform_service import DataTransformService


class InferenceService:
    """
    Service responsible for model inference.
    Follows Single Responsibility Principle - handles predictions only.
    """
    
    def __init__(
        self, 
        model: BaseClassifier, 
        transform_service: DataTransformService,
        device: torch.device,
        class_names: List[str]
    ):
        """
        Initialize inference service.
        
        Args:
            model: Trained model for inference
            transform_service: Service for image transformations
            device: Device to run inference on
            class_names: List of class names in order
        """
        self.model = model
        self.transform_service = transform_service
        self.device = device
        self.class_names = class_names
        self.model.eval()
    
    def predict_image(
        self, 
        image: Image.Image
    ) -> Dict[str, any]:
        """
        Predict class for a single image.
        
        Args:
            image: PIL Image to classify
            
        Returns:
            Dictionary containing:
                - predicted_class: str
                - confidence: float
                - probabilities: Dict[str, float]
        """
        # Transform image
        transform = self.transform_service.get_val_test_transform()
        img_tensor = transform(image).unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            pred_idx = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][pred_idx].item()
        
        # Build probability dictionary
        prob_dict = {
            class_name: probabilities[0][i].item()
            for i, class_name in enumerate(self.class_names)
        }
        
        return {
            'predicted_class': self.class_names[pred_idx],
            'confidence': confidence,
            'probabilities': prob_dict
        }
    
    def predict_batch(
        self, 
        images: List[Image.Image]
    ) -> List[Dict[str, any]]:
        """
        Predict classes for a batch of images.
        
        Args:
            images: List of PIL Images to classify
            
        Returns:
            List of prediction dictionaries
        """
        return [self.predict_image(img) for img in images]
    
    @staticmethod
    def load_from_checkpoint(
        checkpoint_path: Path,
        config: Dict,
        device: torch.device,
        class_names: List[str]
    ) -> 'InferenceService':
        """
        Create InferenceService from a checkpoint file.
        
        Args:
            checkpoint_path: Path to model checkpoint
            config: Configuration dictionary
            device: Device to load model on
            class_names: List of class names
            
        Returns:
            InferenceService instance
        """
        from src.models.model_factory import ModelFactory
        
        # Create model
        model = ModelFactory.create_model(config, len(class_names), device)
        
        # Load checkpoint
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
        
        # Create transform service
        transform_service = DataTransformService(config)
        
        return InferenceService(model, transform_service, device, class_names)
