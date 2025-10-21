"""
Model Factory

Factory pattern for creating models.
Follows Open/Closed Principle and Dependency Inversion Principle.
"""

from typing import Dict, Any
import torch

from src.models.base_classifier import BaseClassifier
from src.models.resnet_classifier import ResNetClassifier


class ModelFactory:
    """
    Factory for creating classifier models.
    Encapsulates model creation logic and device placement.
    """
    
    @staticmethod
    def create_model(
        config: Dict[str, Any], 
        num_classes: int,
        device: torch.device
    ) -> BaseClassifier:
        """
        Create a classifier model based on configuration.
        
        Args:
            config: Configuration dictionary
            num_classes: Number of output classes
            device: Device to place the model on
            
        Returns:
            Classifier model on the specified device
        """
        model_config = config.get('model', {})
        model_name = model_config.get('name', 'resnet18')
        pretrained = model_config.get('pretrained', True)
        pretrained_weights = model_config.get('pretrained_weights', 'IMAGENET1K_V1')
        
        # Create model based on name
        if model_name.startswith('resnet'):
            model = ResNetClassifier(
                num_classes=num_classes,
                model_name=model_name,
                pretrained=pretrained,
                pretrained_weights=pretrained_weights if pretrained else None
            )
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Move model to device
        return model.to(device)
    
    @staticmethod
    def get_device() -> torch.device:
        """
        Get the best available device (MPS > CUDA > CPU).
        
        Returns:
            torch.device object
        """
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
