"""
Base Classifier Interface

Defines the abstract interface for all classifiers.
Follows Interface Segregation Principle and Dependency Inversion Principle.
"""

from abc import ABC, abstractmethod
import torch.nn as nn


class BaseClassifier(ABC, nn.Module):
    """
    Abstract base class for all classifiers.
    Defines the interface that all concrete classifiers must implement.
    """
    
    @abstractmethod
    def get_num_features(self) -> int:
        """
        Get the number of features in the model's final layer.
        
        Returns:
            Number of features
        """
        pass
    
    @abstractmethod
    def set_classifier_head(self, num_classes: int) -> None:
        """
        Replace the classifier head with a new one for the given number of classes.
        
        Args:
            num_classes: Number of output classes
        """
        pass
