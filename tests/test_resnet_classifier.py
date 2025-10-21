"""
Tests for ResNetClassifier
"""

import pytest
import torch
from src.models.resnet_classifier import ResNetClassifier


class TestResNetClassifier:
    """Test suite for ResNetClassifier."""
    
    def test_initialization(self):
        """Test model initialization."""
        num_classes = 8
        model = ResNetClassifier(
            num_classes=num_classes,
            model_name='resnet18',
            pretrained=False
        )
        
        assert model.num_classes == num_classes
        assert model.model_name == 'resnet18'
    
    def test_get_num_features(self):
        """Test getting number of features."""
        model = ResNetClassifier(num_classes=8, pretrained=False)
        num_features = model.get_num_features()
        
        assert isinstance(num_features, int)
        assert num_features > 0
    
    def test_set_classifier_head(self):
        """Test setting classifier head."""
        model = ResNetClassifier(num_classes=8, pretrained=False)
        new_num_classes = 10
        
        model.set_classifier_head(new_num_classes)
        
        assert model.num_classes == new_num_classes
        assert model.model.fc.out_features == new_num_classes
    
    def test_forward_pass(self):
        """Test forward pass through model."""
        num_classes = 8
        batch_size = 4
        
        model = ResNetClassifier(num_classes=num_classes, pretrained=False)
        model.eval()
        
        # Create dummy input
        x = torch.randn(batch_size, 3, 224, 224)
        
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (batch_size, num_classes)
    
    def test_pretrained_weights(self):
        """Test model with pretrained weights."""
        model = ResNetClassifier(
            num_classes=8,
            pretrained=True,
            pretrained_weights='IMAGENET1K_V1'
        )
        
        assert model is not None
        # Verify model can process input
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(x)
        assert output.shape == (1, 8)
