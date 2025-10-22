"""
Dataset Service

Single Responsibility: Handles dataset loading, splitting, and DataLoader creation.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, List
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets
from torchvision.transforms import Compose
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split

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
        
        # Create weighted sampler for balanced training
        train_sampler = self._create_weighted_sampler(train_dataset)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            sampler=train_sampler,  # Use sampler instead of shuffle
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
        Split dataset into train, validation, and test sets using stratified split.
        Preserves class proportions across all splits.
        
        Args:
            dataset: Dataset to split
            
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        # Get indices and labels
        indices = list(range(len(dataset)))
        labels = [dataset[i][1] for i in indices]
        
        # First split: train vs (val + test) - stratified
        train_idx, temp_idx = train_test_split(
            indices,
            train_size=self.train_split,
            stratify=labels,
            random_state=self.seed
        )
        
        # Get labels for temp set
        temp_labels = [labels[i] for i in temp_idx]
        
        # Second split: val vs test - stratified
        val_ratio = self.val_split / (1.0 - self.train_split)
        val_idx, test_idx = train_test_split(
            temp_idx,
            train_size=val_ratio,
            stratify=temp_labels,
            random_state=self.seed
        )
        
        # Create Subset objects
        train_dataset = Subset(dataset, train_idx)
        val_dataset = Subset(dataset, val_idx)
        test_dataset = Subset(dataset, test_idx)
        
        return train_dataset, val_dataset, test_dataset
    
    def _create_weighted_sampler(self, train_dataset: Subset) -> WeightedRandomSampler:
        """
        Create a weighted sampler to balance classes during training.
        This ensures each class is sampled equally regardless of original distribution.
        
        Args:
            train_dataset: Training dataset subset
            
        Returns:
            WeightedRandomSampler for balanced sampling
        """
        # Get all labels from training set
        labels = [train_dataset[i][1] for i in range(len(train_dataset))]
        
        # Count samples per class
        class_counts = np.bincount(labels)
        
        # Compute weight for each class (inverse frequency)
        class_weights = 1.0 / class_counts
        
        # Assign weight to each sample based on its class
        sample_weights = [class_weights[label] for label in labels]
        
        # Create sampler
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True  # Allow sampling with replacement for balancing
        )
        
        return sampler
    
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
