"""
Dataset Service

Single Responsibility: Handles dataset loading, splitting, and DataLoader creation.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, List
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torchvision.transforms import Compose
from sklearn.utils.class_weight import compute_class_weight


class DatasetService:
    """
    Service responsible for dataset operations.
    Follows Single Responsibility Principle - handles data loading and preparation.
    """
    
    def __init__(self, config: Dict[str, Any], dataset_path: Path):
        """
        Initialize dataset service with configuration.
        
        Args:
            config: Configuration dictionary
            dataset_path: Path to the dataset directory
        """
        self.config = config
        self.dataset_path = dataset_path
        
        training_config = config.get('training', {})
        self.batch_size = training_config.get('batch_size', 32)
        self.num_workers = training_config.get('num_workers', 2)
        self.seed = training_config.get('seed', 42)
        self.subset_size = training_config.get('subset_size', None)
        
        self.train_split = training_config.get('train_split', 0.7)
        self.val_split = training_config.get('val_split', 0.15)
        
        self.full_dataset = None
        self.class_names = None
    
    def load_dataset(
        self, 
        train_transform: Compose, 
        val_test_transform: Compose
    ) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
        """
        Load and prepare dataset with train/val/test splits.
        
        Args:
            train_transform: Transformation pipeline for training data
            val_test_transform: Transformation pipeline for validation/test data
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader, class_names)
        """
        # Load full dataset with train transform initially
        self.full_dataset = datasets.ImageFolder(
            str(self.dataset_path), 
            transform=train_transform
        )
        self.class_names = self.full_dataset.classes
        
        # Create subset if specified
        if self.subset_size is not None and self.subset_size < len(self.full_dataset):
            subset_dataset = self._create_subset()
        else:
            subset_dataset = self.full_dataset
        
        # Split dataset
        train_dataset, val_dataset, test_dataset = self._split_dataset(subset_dataset)
        
        # Apply appropriate transforms to val/test sets
        val_dataset.dataset.transform = val_test_transform
        test_dataset.dataset.transform = val_test_transform
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=self.num_workers
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False
        )
        
        return train_loader, val_loader, test_loader, self.class_names
    
    def _create_subset(self) -> Subset:
        """
        Create a random subset of the dataset.
        
        Returns:
            Subset of the dataset
        """
        np.random.seed(self.seed)
        subset_indices = np.random.choice(
            len(self.full_dataset), 
            self.subset_size, 
            replace=False
        )
        return Subset(self.full_dataset, subset_indices)
    
    def _split_dataset(
        self, 
        dataset: datasets.ImageFolder | Subset
    ) -> Tuple[Subset, Subset, Subset]:
        """
        Split dataset into train, validation, and test sets.
        
        Args:
            dataset: Dataset to split
            
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        n_train = int(self.train_split * len(dataset))
        n_val = int(self.val_split * len(dataset))
        n_test = len(dataset) - n_train - n_val
        
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, 
            [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(self.seed)
        )
        
        return train_dataset, val_dataset, test_dataset
    
    def compute_class_weights(self, train_dataset: Subset) -> torch.Tensor:
        """
        Compute class weights for handling class imbalance.
        
        Args:
            train_dataset: Training dataset subset
            
        Returns:
            Tensor of class weights
        """
        # Extract labels from training dataset
        labels = [y for _, y in train_dataset]
        
        # Compute balanced class weights
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.arange(len(self.class_names)),
            y=labels
        )
        
        return torch.tensor(class_weights, dtype=torch.float)
