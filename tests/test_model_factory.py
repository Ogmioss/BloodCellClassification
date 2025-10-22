"""
Tests for ModelFactory
"""

import pytest
import torch
from src.models.model_factory import ModelFactory
from src.models.base_classifier import BaseClassifier


class TestModelFactory:
    """Test suite for ModelFactory."""
    
    @pytest.fixture
    def config(self):
        """Fixture providing test configuration."""
        return {
            'model': {
                'name': 'resnet18',
                'pretrained': False,  # Faster for testing
                'pretrained_weights': None
            }
        }
    
    def test_get_device(self):
        """Test device detection."""
        device = ModelFactory.get_device()
        
        assert isinstance(device, torch.device)
        assert device.type in ['cpu', 'cuda', 'mps']
    
    def test_create_resnet_model(self, config):
        """Test ResNet model creation."""
        device = torch.device('cpu')  # Use CPU for testing
        num_classes = 8
        
        model = ModelFactory.create_model(config, num_classes, device)
        
        assert isinstance(model, BaseClassifier)
        assert model.num_classes == num_classes
    
    def test_model_on_correct_device(self, config):
        """Test model is placed on correct device."""
        device = torch.device('cpu')
        num_classes = 8
        
        model = ModelFactory.create_model(config, num_classes, device)
        
        # Check first parameter's device
        first_param = next(model.parameters())
        assert first_param.device.type == device.type
    
    def test_invalid_model_name(self):
        """Test error handling for invalid model name."""
        config = {
            'model': {
                'name': 'invalid_model',
                'pretrained': False
            }
        }
        device = torch.device('cpu')
        
        with pytest.raises(ValueError):
            ModelFactory.create_model(config, 8, device)
