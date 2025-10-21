"""
Tests for EvaluationService
"""

import pytest
import torch
from torch.utils.data import TensorDataset, DataLoader
from src.services.evaluation_service import EvaluationService
from src.models.resnet_classifier import ResNetClassifier


class TestEvaluationService:
    """Test suite for EvaluationService."""
    
    @pytest.fixture
    def model(self):
        """Fixture providing a test model."""
        model = ResNetClassifier(num_classes=8, pretrained=False)
        model.eval()
        return model
    
    @pytest.fixture
    def device(self):
        """Fixture providing device."""
        return torch.device('cpu')
    
    @pytest.fixture
    def test_loader(self):
        """Fixture providing a test data loader."""
        # Create dummy data
        batch_size = 4
        num_samples = 16
        
        images = torch.randn(num_samples, 3, 224, 224)
        labels = torch.randint(0, 8, (num_samples,))
        
        dataset = TensorDataset(images, labels)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    def test_initialization(self, model, device):
        """Test service initialization."""
        service = EvaluationService(model, device)
        
        assert service.model == model
        assert service.device == device
    
    def test_evaluate(self, model, device, test_loader):
        """Test evaluation method."""
        service = EvaluationService(model, device)
        results = service.evaluate(test_loader)
        
        assert 'accuracy' in results
        assert 'predictions' in results
        assert 'labels' in results
        assert 0 <= results['accuracy'] <= 1
        assert len(results['predictions']) == 16
        assert len(results['labels']) == 16
    
    def test_get_predictions_and_labels(self, model, device, test_loader):
        """Test getting predictions and labels."""
        service = EvaluationService(model, device)
        predictions, labels = service.get_predictions_and_labels(test_loader)
        
        assert len(predictions) == len(labels)
        assert len(predictions) == 16
    
    def test_compute_accuracy(self, model, device):
        """Test accuracy computation."""
        service = EvaluationService(model, device)
        
        # Perfect predictions
        predictions = [0, 1, 2, 3, 4]
        labels = [0, 1, 2, 3, 4]
        accuracy = service.compute_accuracy(predictions, labels)
        assert accuracy == 1.0
        
        # Half correct
        predictions = [0, 1, 2, 3, 4]
        labels = [0, 0, 2, 2, 4]
        accuracy = service.compute_accuracy(predictions, labels)
        assert accuracy == 0.6
