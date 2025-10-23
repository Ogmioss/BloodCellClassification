"""
Evaluate Existing Model Script

Loads a trained model checkpoint and evaluates it on the test set,
generating metrics.json file for the Streamlit interface.
"""

import json
from pathlib import Path
import torch

# Fix for "could not create a primitive" error in PyTorch 2.9.0+cpu
torch.backends.mkldnn.enabled = False

from src.services.yaml_loader import YamlLoader
from src.services.data_transform_service import DataTransformService
from src.services.dataset_service import DatasetService
from src.services.evaluation_service import EvaluationService
from src.models.model_factory import ModelFactory


def main():
    """Evaluate existing model and generate metrics."""
    
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
    
    # Setup checkpoint path
    checkpoint_dir = Path(loader.get_nested_value('paths.models.checkpoints', './models/checkpoints'))
    checkpoint_dir = loader._resolve_dir(checkpoint_dir)
    checkpoint_path = checkpoint_dir / "best_model.pth"
    
    if not checkpoint_path.exists():
        print(f"❌ Error: Model checkpoint not found at {checkpoint_path}")
        print("Please train a model first using: uv run train-model")
        return None
    
    print(f"✅ Found model checkpoint: {checkpoint_path}")
    
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
    
    # Create model
    print("\nCreating model...")
    model = ModelFactory.create_model(config, len(class_names), device)
    print(f"Model: {config['model']['name']}")
    
    # Load checkpoint
    print(f"\nLoading checkpoint from {checkpoint_path}...")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Check if checkpoint has 'model.' prefix (from ResNetClassifier wrapper)
        # If not, we need to add it
        if not any(k.startswith('model.') for k in checkpoint.keys()):
            # Checkpoint was saved from the inner model, need to add 'model.' prefix
            checkpoint = {f'model.{k}': v for k, v in checkpoint.items()}
        
        model.load_state_dict(checkpoint)
        model.eval()
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return None
    
    # Evaluate on validation set
    print("\n" + "="*50)
    print("Evaluating on validation set...")
    print("="*50)
    evaluation_service = EvaluationService(model, device)
    val_results = evaluation_service.evaluate(val_loader)
    print(f"Validation Accuracy: {val_results['accuracy']:.4f}")
    
    # Evaluate on test set
    print("\n" + "="*50)
    print("Evaluating on test set...")
    print("="*50)
    test_results = evaluation_service.evaluate(test_loader)
    print(f"Test Accuracy: {test_results['accuracy']:.4f}")
    
    # Compute confusion matrix for test set
    print("\nComputing confusion matrix...")
    confusion_mat = evaluation_service.compute_confusion_matrix(
        test_results['predictions'],
        test_results['labels'],
        len(class_names)
    )
    print("✅ Confusion matrix computed")
    
    # Evaluate on training set (for completeness)
    print("\n" + "="*50)
    print("Evaluating on training set...")
    print("="*50)
    train_results = evaluation_service.evaluate(train_loader)
    print(f"Training Accuracy: {train_results['accuracy']:.4f}")
    
    # Save metrics to JSON file
    metrics_path = checkpoint_dir / "metrics.json"
    metrics_data = {
        'best_val_acc': val_results['accuracy'],
        'final_train_loss': 0.0,  # Not available from checkpoint
        'final_train_acc': train_results['accuracy'],
        'accuracy': test_results['accuracy'],
        'class_names': class_names,
        'confusion_matrix': confusion_mat.tolist()  # Convert numpy array to list for JSON
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    
    print("\n" + "="*50)
    print("Evaluation completed!")
    print("="*50)
    print(f"Test Accuracy: {test_results['accuracy']:.4f}")
    print(f"Validation Accuracy: {val_results['accuracy']:.4f}")
    print(f"Training Accuracy: {train_results['accuracy']:.4f}")
    print(f"Metrics saved to: {metrics_path}")
    print("\n💡 Refresh the Streamlit page to see the metrics!")
    
    return {
        'model': model,
        'val_results': val_results,
        'test_results': test_results,
        'train_results': train_results,
        'class_names': class_names
    }


if __name__ == "__main__":
    results = main()
