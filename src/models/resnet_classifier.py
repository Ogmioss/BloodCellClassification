"""
ResNet Classifier Implementation

Concrete implementation of ResNet-based classifier.
Follows Open/Closed Principle - extensible without modifying base.
"""

import torch.nn as nn
from torchvision import models
from typing import Optional

from src.models.base_classifier import BaseClassifier


class ResNetClassifier(BaseClassifier):
    """
    ResNet-based classifier implementation.
    Wraps torchvision ResNet models with a custom classification head.
    """
    
    def __init__(
        self, 
        num_classes: int,
        model_name: str = "resnet18",
        pretrained: bool = True,
        pretrained_weights: Optional[str] = "IMAGENET1K_V1"
    ):
        """
        Initialize ResNet classifier.
        
        Args:
            num_classes: Number of output classes
            model_name: Name of ResNet variant (resnet18, resnet34, resnet50, etc.)
            pretrained: Whether to use pretrained weights
            pretrained_weights: Name of pretrained weights to use
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.model_name = model_name
        
        # Create base model
        self.model = self._create_base_model(model_name, pretrained, pretrained_weights)
        
        # Replace classifier head
        self.set_classifier_head(num_classes)
    
    def _create_base_model(
        self, 
        model_name: str, 
        pretrained: bool,
        pretrained_weights: Optional[str]
    ) -> nn.Module:
        """
        Create the base ResNet model.
        
        Args:
            model_name: Name of the ResNet variant
            pretrained: Whether to use pretrained weights
            pretrained_weights: Name of pretrained weights
            
        Returns:
            Base ResNet model
        """
        model_factory = getattr(models, model_name)
        
        if pretrained and pretrained_weights:
            return model_factory(weights=pretrained_weights)
        else:
            return model_factory(weights=None)
    
    def get_num_features(self) -> int:
        """
        Get the number of input features to the final fully connected layer.
        
        Returns:
            Number of features
        """
        return self.model.fc.in_features
    
    def set_classifier_head(self, num_classes: int) -> None:
        """
        Replace the classifier head with a new linear layer.
        
        Args:
            num_classes: Number of output classes
        """
        num_features = self.get_num_features()
        self.model.fc = nn.Linear(num_features, num_classes)
        self.num_classes = num_classes
    
    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor
            
        Returns:
            Output logits
        """
        return self.model(x)
