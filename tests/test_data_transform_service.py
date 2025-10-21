"""
Tests for DataTransformService
"""

import pytest
from torchvision import transforms
from src.services.data_transform_service import DataTransformService


class TestDataTransformService:
    """Test suite for DataTransformService."""
    
    @pytest.fixture
    def config(self):
        """Fixture providing test configuration."""
        return {
            'training': {
                'img_size': 224
            },
            'model': {
                'normalization': {
                    'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225]
                }
            },
            'augmentation': {
                'train': {
                    'horizontal_flip': True,
                    'vertical_flip': True,
                    'rotation_degrees': 30,
                    'color_jitter': {
                        'brightness': 0.2,
                        'contrast': 0.2,
                        'saturation': 0.2
                    }
                },
                'val_test': {
                    'enabled': False
                }
            }
        }
    
    def test_initialization(self, config):
        """Test service initialization."""
        service = DataTransformService(config)
        
        assert service.img_size == 224
        assert service.mean == [0.485, 0.456, 0.406]
        assert service.std == [0.229, 0.224, 0.225]
    
    def test_train_transform_creation(self, config):
        """Test training transformation pipeline creation."""
        service = DataTransformService(config)
        train_transform = service.get_train_transform()
        
        assert isinstance(train_transform, transforms.Compose)
        assert len(train_transform.transforms) > 0
    
    def test_val_test_transform_creation(self, config):
        """Test validation/test transformation pipeline creation."""
        service = DataTransformService(config)
        val_test_transform = service.get_val_test_transform()
        
        assert isinstance(val_test_transform, transforms.Compose)
        # Should have Resize, ToTensor, and Normalize only
        assert len(val_test_transform.transforms) == 3
    
    def test_transforms_different(self, config):
        """Test that train and val/test transforms are different."""
        service = DataTransformService(config)
        train_transform = service.get_train_transform()
        val_test_transform = service.get_val_test_transform()
        
        # Train should have more transforms (augmentation)
        assert len(train_transform.transforms) > len(val_test_transform.transforms)
