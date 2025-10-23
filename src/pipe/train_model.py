"""
Main Training Script

Orchestrates all services to train a blood cell classification model.
"""

import json
from pathlib import Path
import torch

# Fix for "could not create a primitive" error in PyTorch 2.9.0+cpu
torch.backends.mkldnn.enabled = False

from src.services.yaml_loader import YamlLoader
from src.services.data_transform_service import DataTransformService
from src.services.dataset_service import DatasetService
from src.services.training_service import TrainingService
from src.services.evaluation_service import EvaluationService
from src.models.model_factory import ModelFactory


def main():
    """Main training pipeline."""
    
    # Load configuration
    print("Loading configuration...")
    loader = YamlLoader()
    config = loader.config
    
    # Get dataset path
    data_dir = loader.data_dir
    dataset_path = data_dir / "raw" / "bloodcells_dataset"
    
    print(f"Dataset path: {dataset_path}")
    
    # Get device
    device = ModelFactory.get_device()
    print(f"Using device: {device}")
    
    # Create data transforms
    print("\nCreating data transformations...")
    transform_service = DataTransformService(config)
    train_transform = transform_service.get_train_transform()
    val_test_transform = transform_service.get_val_test_transform()
    
    # Load dataset
    print("\nLoading dataset...")
    dataset_service = DatasetService(config, dataset_path)
    train_loader, val_loader, test_loader, class_names = dataset_service.load_dataset(
        train_transform, 
        val_test_transform
    )
    
    print(f"Classes: {class_names}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Compute class weights for imbalanced data
    print("\nComputing class weights...")
    class_weights = dataset_service.compute_class_weights(train_loader.dataset)
    print(f"Class weights: {class_weights}")
    
    # Create model
    print("\nCreating model...")
    model = ModelFactory.create_model(config, len(class_names), device)
    print(f"Model: {config['model']['name']}")
    
    # Setup checkpoint path
    checkpoint_dir = Path(loader.get_nested_value('paths.models.checkpoints', './models/checkpoints'))
    checkpoint_dir = loader._resolve_dir(checkpoint_dir)
    checkpoint_path = checkpoint_dir / "best_model.pth"
    
    # Train model
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50)
    
    training_service = TrainingService(config, model, device, class_weights)
    training_metrics = training_service.train(
        train_loader, 
        val_loader, 
        checkpoint_path
    )
    
    print("\n" + "="*50)
    print("Training completed!")
    print(f"Best validation accuracy: {training_metrics['best_val_acc']:.4f}")
    print("="*50)
    
    # Load best model and evaluate
    print("\nLoading best model for evaluation...")
    training_service.load_checkpoint(checkpoint_path)
    
    print("\nEvaluating on test set...")
    evaluation_service = EvaluationService(model, device)
    test_results = evaluation_service.evaluate(test_loader)
    
    print("\n" + "="*50)
    print("Test Results")
    print("="*50)
    print(f"Test Accuracy: {test_results['accuracy']:.4f}")
    print(f"Model saved to: {checkpoint_path}")
    
    # Compute confusion matrix
    print("\nComputing confusion matrix...")
    confusion_mat = evaluation_service.compute_confusion_matrix(
        test_results['predictions'],
        test_results['labels'],
        len(class_names)
    )
    
    # Save metrics to JSON file
    metrics_path = checkpoint_dir / "metrics.json"
    metrics_data = {
        'best_val_acc': training_metrics['best_val_acc'],
        'final_train_loss': training_metrics['final_train_loss'],
        'final_train_acc': training_metrics['final_train_acc'],
        'accuracy': test_results['accuracy'],
        'class_names': class_names,
        'confusion_matrix': confusion_mat.tolist()
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    
    print(f"Metrics saved to: {metrics_path}")
    
    return {
        'model': model,
        'training_metrics': training_metrics,
        'test_results': test_results,
        'class_names': class_names
    }


if __name__ == "__main__":
    results = main()
