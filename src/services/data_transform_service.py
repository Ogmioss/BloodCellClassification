"""
Data Transformation Service

Single Responsibility: Handles creation of data transformations for training and validation/test sets.
"""

from typing import Dict, Any
from torchvision import transforms


class DataTransformService:
    """
    Service responsible for creating image transformations.
    Follows Single Responsibility Principle - only handles transform creation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the transformation service with configuration.
        
        Args:
            config: Configuration dictionary containing transformation parameters
        """
        self.img_size = config.get('training', {}).get('img_size', 224)
        self.normalization = config.get('model', {}).get('normalization', {})
        self.augmentation = config.get('augmentation', {})
        
        self.mean = self.normalization.get('mean', [0.485, 0.456, 0.406])
        self.std = self.normalization.get('std', [0.229, 0.224, 0.225])
    
    def get_train_transform(self) -> transforms.Compose:
        """
        Create transformation pipeline for training data with augmentation.
        
        Returns:
            Composed transformation pipeline for training
        """
        train_config = self.augmentation.get('train', {})
        
        transform_list = [
            transforms.Resize((self.img_size, self.img_size))
        ]
        
        # Add augmentations if enabled
        if train_config.get('horizontal_flip', False):
            transform_list.append(transforms.RandomHorizontalFlip())
        
        if train_config.get('vertical_flip', False):
            transform_list.append(transforms.RandomVerticalFlip())
        
        rotation_degrees = train_config.get('rotation_degrees', 0)
        if rotation_degrees > 0:
            transform_list.append(transforms.RandomRotation(rotation_degrees))
        
        color_jitter_config = train_config.get('color_jitter', {})
        if color_jitter_config:
            transform_list.append(
                transforms.ColorJitter(
                    brightness=color_jitter_config.get('brightness', 0),
                    contrast=color_jitter_config.get('contrast', 0),
                    saturation=color_jitter_config.get('saturation', 0)
                )
            )
        
        # Always add tensor conversion and normalization
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
        
        return transforms.Compose(transform_list)
    
    def get_val_test_transform(self) -> transforms.Compose:
        """
        Create transformation pipeline for validation/test data (no augmentation).
        
        Returns:
            Composed transformation pipeline for validation/test
        """
        return transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
