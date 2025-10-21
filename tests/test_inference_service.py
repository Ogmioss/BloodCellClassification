"""
Tests for InferenceService
"""

import pytest
import torch
from PIL import Image
import numpy as np
from pathlib import Path

from src.services.inference_service import InferenceService
from src.services.data_transform_service import DataTransformService
from src.models.resnet_classifier import ResNetClassifier


@pytest.fixture
def config():
    """Sample configuration for testing."""
    return {
        'training': {
            'img_size': 224
        },
        'model': {
            'name': 'resnet18',
            'normalization': {
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225]
            }
        }
    }


@pytest.fixture
def class_names():
    """Sample class names."""
    return ['class_a', 'class_b', 'class_c']


@pytest.fixture
def model(class_names):
    """Create a simple test model."""
    model = ResNetClassifier(
        num_classes=len(class_names),
        model_name='resnet18',
        pretrained=False
    )
    model.eval()
    return model


@pytest.fixture
def transform_service(config):
    """Create transform service."""
    return DataTransformService(config)


@pytest.fixture
def inference_service(model, transform_service, class_names):
    """Create inference service."""
    device = torch.device('cpu')
    return InferenceService(model, transform_service, device, class_names)


@pytest.fixture
def sample_image():
    """Create a sample PIL image."""
    # Create a random RGB image
    img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(img_array)


def test_inference_service_initialization(inference_service, class_names):
    """Test that inference service initializes correctly."""
    assert inference_service.model is not None
    assert inference_service.transform_service is not None
    assert inference_service.device == torch.device('cpu')
    assert inference_service.class_names == class_names


def test_predict_image(inference_service, sample_image, class_names):
    """Test single image prediction."""
    prediction = inference_service.predict_image(sample_image)
    
    assert 'predicted_class' in prediction
    assert 'confidence' in prediction
    assert 'probabilities' in prediction
    
    assert prediction['predicted_class'] in class_names
    assert 0.0 <= prediction['confidence'] <= 1.0
    assert len(prediction['probabilities']) == len(class_names)
    
    # Check that probabilities sum to 1
    prob_sum = sum(prediction['probabilities'].values())
    assert abs(prob_sum - 1.0) < 1e-5


def test_predict_batch(inference_service, sample_image, class_names):
    """Test batch prediction."""
    images = [sample_image, sample_image, sample_image]
    predictions = inference_service.predict_batch(images)
    
    assert len(predictions) == len(images)
    
    for prediction in predictions:
        assert 'predicted_class' in prediction
        assert prediction['predicted_class'] in class_names


def test_probability_dict_keys(inference_service, sample_image, class_names):
    """Test that probability dict has correct keys."""
    prediction = inference_service.predict_image(sample_image)
    prob_keys = set(prediction['probabilities'].keys())
    
    assert prob_keys == set(class_names)


def test_prediction_confidence_equals_max_probability(inference_service, sample_image):
    """Test that confidence equals the maximum probability."""
    prediction = inference_service.predict_image(sample_image)
    
    max_prob = max(prediction['probabilities'].values())
    assert abs(prediction['confidence'] - max_prob) < 1e-5
