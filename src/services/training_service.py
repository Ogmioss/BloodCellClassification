"""
Training Service

Single Responsibility: Handles model training logic.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.base_classifier import BaseClassifier


class TrainingService:
    """
    Service responsible for training models.
    Follows Single Responsibility Principle - handles training loop and optimization.
    """
    
    def __init__(
        self, 
        config: Dict[str, Any],
        model: BaseClassifier,
        device: torch.device,
        class_weights: Optional[torch.Tensor] = None
    ):
        """
        Initialize training service.
        
        Args:
            config: Configuration dictionary
            model: Model to train
            device: Device to train on
            class_weights: Optional class weights for loss function
        """
        self.config = config
        self.model = model
        self.device = device
        
        training_config = config.get('training', {})
        self.epochs = training_config.get('epochs', 20)
        self.learning_rate = training_config.get('learning_rate', 1e-3)
        
        # Setup loss and optimizer
        if class_weights is not None:
            class_weights = class_weights.to(device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        
        self.best_val_acc = 0.0
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        checkpoint_path: Optional[Path] = None
    ) -> Dict[str, float]:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            checkpoint_path: Optional path to save best model checkpoint
            
        Returns:
            Dictionary with final training metrics
        """
        for epoch in range(self.epochs):
            print(f"\nEpoch {epoch+1}/{self.epochs}")
            
            # Train phase
            train_loss, train_acc = self._train_epoch(train_loader)
            print(f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}")
            
            # Validation phase
            val_acc = self._validate_epoch(val_loader)
            print(f"Val Acc: {val_acc:.4f}")
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                if checkpoint_path:
                    self._save_checkpoint(checkpoint_path)
                    print(">>> Best model saved!")
        
        return {
            'best_val_acc': self.best_val_acc,
            'final_train_loss': train_loss,
            'final_train_acc': train_acc
        }
    
    def _train_epoch(self, train_loader: DataLoader) -> tuple[float, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Tuple of (epoch_loss, epoch_accuracy)
        """
        self.model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = self.criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.float() / len(train_loader.dataset)
        
        return epoch_loss, epoch_acc.item()
    
    def _validate_epoch(self, val_loader: DataLoader) -> float:
        """
        Validate for one epoch.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Validation accuracy
        """
        self.model.eval()
        val_corrects = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels.data)
        
        val_acc = val_corrects.float() / len(val_loader.dataset)
        return val_acc.item()
    
    def _save_checkpoint(self, checkpoint_path: Path) -> None:
        """
        Save model checkpoint.
        
        Args:
            checkpoint_path: Path to save checkpoint
        """
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        self.model.load_state_dict(torch.load(checkpoint_path))
