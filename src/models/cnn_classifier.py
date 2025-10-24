"""
CNN Classifier Implementation

Concrete implementation of a simple CNN classifier.
Extends BaseClassifier to integrate seamlessly with the model factory.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base_classifier import BaseClassifier


class CNNClassifier(BaseClassifier):
    """
    Simple CNN-based classifier.
    """

    def __init__(self, num_classes: int):
        """
        Initialize CNN classifier.
        
        Args:
            num_classes: Number of output classes
        """
        super().__init__()

        self.num_classes = num_classes

        # Define convolutional layers
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 1/2 spatial size

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Adaptive pooling to flatten
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected classifier head
        self.classifier = nn.Linear(128, num_classes)

    def get_num_features(self) -> int:
        return self.classifier.in_features

    def set_classifier_head(self, num_classes: int) -> None:
        self.classifier = nn.Linear(self.get_num_features(), num_classes)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
