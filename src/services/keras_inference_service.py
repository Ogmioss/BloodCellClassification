"""
Keras Inference Service

Single Responsibility: Handles model inference for Keras/TensorFlow models.
"""

from typing import Dict, List
from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf


class KerasInferenceService:
    """
    Service responsible for Keras model inference.
    Follows Single Responsibility Principle - handles predictions only.
    """
    
    def __init__(
        self, 
        model: tf.keras.Model,
        class_names: List[str],
        input_size: tuple = (224, 224)
    ):
        """
        Initialize Keras inference service.
        
        Args:
            model: Trained Keras model for inference
            class_names: List of class names in order
            input_size: Expected input size for the model (height, width)
        """
        self.model = model
        self.class_names = class_names
        self.input_size = input_size
    
    def _preprocess_image(self, image: Image.Image) -> np.ndarray:
        """
        Preprocess image for Keras model.
        
        Args:
            image: PIL Image to preprocess
            
        Returns:
            Preprocessed numpy array
        """
        # Resize image
        img = image.resize(self.input_size)
        
        # Convert to array and normalize to [0, 1]
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def predict_image(self, image: Image.Image) -> Dict[str, any]:
        """
        Predict class for a single image.
        
        Args:
            image: PIL Image to classify
            
        Returns:
            Dictionary containing:
                - predicted_class: str
                - confidence: float
                - probabilities: Dict[str, float]
        """
        # Preprocess image
        img_array = self._preprocess_image(image)
        
        # Make prediction
        predictions = self.model.predict(img_array, verbose=0)
        probabilities = predictions[0]
        
        # Get predicted class
        pred_idx = np.argmax(probabilities)
        confidence = float(probabilities[pred_idx])
        
        # Build probability dictionary
        prob_dict = {
            class_name: float(probabilities[i])
            for i, class_name in enumerate(self.class_names)
        }
        
        return {
            'predicted_class': self.class_names[pred_idx],
            'confidence': confidence,
            'probabilities': prob_dict
        }
    
    def predict_batch(self, images: List[Image.Image]) -> List[Dict[str, any]]:
        """
        Predict classes for a batch of images.
        
        Args:
            images: List of PIL Images to classify
            
        Returns:
            List of prediction dictionaries
        """
        return [self.predict_image(img) for img in images]
    
    @staticmethod
    def load_from_checkpoint(
        checkpoint_path: Path,
        class_names: List[str],
        input_size: tuple = (224, 224)
    ) -> 'KerasInferenceService':
        """
        Create KerasInferenceService from a checkpoint file.
        
        Args:
            checkpoint_path: Path to model checkpoint (.h5 or .keras)
            class_names: List of class names
            input_size: Expected input size for the model
            
        Returns:
            KerasInferenceService instance
        """
        # Load model
        model = tf.keras.models.load_model(checkpoint_path)
        
        return KerasInferenceService(model, class_names, input_size)
