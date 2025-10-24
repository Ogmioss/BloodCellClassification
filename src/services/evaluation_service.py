"""
Evaluation Service

Single Responsibility: Handles model evaluation and metrics computation.
"""

from typing import Dict, List, Tuple
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix

from src.models.base_classifier import BaseClassifier


class EvaluationService:
    """
    Service responsible for model evaluation.
    Follows Single Responsibility Principle - handles evaluation and metrics.
    """
    
    def __init__(self, model: BaseClassifier, device: torch.device):
        """
        Initialize evaluation service.
        
        Args:
            model: Model to evaluate
            device: Device to run evaluation on
        """
        self.model = model
        self.device = device
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model on test data.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Dictionary containing evaluation metrics
        """
        self.model.eval()
        
        test_corrects = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                
                test_corrects += torch.sum(preds == labels.data)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        test_acc = test_corrects.float() / len(test_loader.dataset)
        
        return {
            'accuracy': test_acc.item(),
            'predictions': all_preds,
            'labels': all_labels
        }
    
    def get_predictions_and_labels(
        self, 
        test_loader: DataLoader
    ) -> Tuple[List[int], List[int]]:
        """
        Get predictions and ground truth labels.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Tuple of (predictions, labels)
        """
        results = self.evaluate(test_loader)
        return results['predictions'], results['labels']
    
    def compute_accuracy(self, predictions: List[int], labels: List[int]) -> float:
        """
        Compute accuracy from predictions and labels.
        
        Args:
            predictions: List of predicted class indices
            labels: List of ground truth class indices
            
        Returns:
            Accuracy score
        """
        predictions = np.array(predictions)
        labels = np.array(labels)
        return np.mean(predictions == labels)
    
    def compute_confusion_matrix(
        self, 
        predictions: List[int], 
        labels: List[int],
        num_classes: int
    ) -> np.ndarray:
        """
        Compute confusion matrix from predictions and labels.
        
        Args:
            predictions: List of predicted class indices
            labels: List of ground truth class indices
            num_classes: Number of classes
            
        Returns:
            Confusion matrix as numpy array of shape (num_classes, num_classes)
        """
        return confusion_matrix(labels, predictions, labels=list(range(num_classes)))
