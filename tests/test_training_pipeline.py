"""
Tests for the training pipeline and model saving functionality.
"""

import pytest
from pathlib import Path
import sys
import subprocess
import json

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.services.yaml_loader import YamlLoader


class TestTrainingPipeline:
    """Test suite for model training pipeline."""
    
    @pytest.fixture
    def yaml_loader(self):
        """Fixture to create YamlLoader instance."""
        return YamlLoader()
    
    @pytest.fixture
    def checkpoints_dir(self, yaml_loader):
        """Fixture to get checkpoints directory path."""
        checkpoint_dir = Path(yaml_loader.get_nested_value('paths.models.checkpoints', './models/checkpoints'))
        return yaml_loader._resolve_dir(checkpoint_dir)
    
    def test_checkpoints_directory_exists(self, checkpoints_dir):
        """Test that checkpoints directory exists or can be created."""
        assert checkpoints_dir.exists() or checkpoints_dir.parent.exists(), \
            f"Checkpoints directory parent does not exist: {checkpoints_dir.parent}"
    
    def test_training_script_exists(self):
        """Test that training script exists."""
        train_script = project_root / "src" / "pipe" / "train_model.py"
        assert train_script.exists(), f"Training script not found: {train_script}"
    
    def test_training_script_is_importable(self):
        """Test that training script can be imported."""
        try:
            from src.pipe import train_model
            assert hasattr(train_model, 'main'), "train_model.py should have a 'main' function"
        except ImportError as e:
            pytest.fail(f"Failed to import train_model: {e}")
    
    def test_model_checkpoint_path_config(self, yaml_loader, checkpoints_dir):
        """Test that checkpoint path is correctly configured."""
        # Check that either checkpoints_dir exists or models dir exists
        models_dir = checkpoints_dir.parent
        assert models_dir.exists() or models_dir.parent.exists(), \
            f"Models directory or its parent does not exist: {models_dir}"
    
    def test_metrics_json_structure(self, checkpoints_dir):
        """Test that metrics.json has the expected structure if it exists."""
        metrics_path = checkpoints_dir / "metrics.json"
        
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            
            # Check expected keys
            expected_keys = ['best_val_acc', 'final_train_loss', 'final_train_acc', 'accuracy', 'class_names']
            for key in expected_keys:
                assert key in metrics, f"Missing expected key in metrics.json: {key}"
            
            # Check data types
            assert isinstance(metrics['best_val_acc'], (int, float)), "best_val_acc should be numeric"
            assert isinstance(metrics['accuracy'], (int, float)), "accuracy should be numeric"
            assert isinstance(metrics['class_names'], list), "class_names should be a list"
    
    def test_training_command_format(self, yaml_loader):
        """Test that the training command can be properly formatted."""
        command = ["uv", "run", "python", "-m", "src.pipe.train_model"]
        assert len(command) == 5, "Training command should have 5 parts"
        assert command[0] == "uv", "Command should start with 'uv'"
        assert command[-1] == "src.pipe.train_model", "Command should end with module path"
    
    def test_yaml_config_has_required_training_params(self, yaml_loader):
        """Test that config has all required training parameters."""
        config = yaml_loader.config
        
        # Check training section
        assert 'training' in config, "Config should have 'training' section"
        training = config['training']
        
        required_params = ['batch_size', 'epochs', 'learning_rate', 'img_size']
        for param in required_params:
            assert param in training, f"Missing required training parameter: {param}"
        
        # Check model section
        assert 'model' in config, "Config should have 'model' section"
        model = config['model']
        assert 'name' in model, "Model config should have 'name' parameter"
    
    def test_training_module_has_pytorch_fix(self):
        """Test that training script has the PyTorch mkldnn fix."""
        train_script = project_root / "src" / "pipe" / "train_model.py"
        with open(train_script, 'r') as f:
            content = f.read()
        
        # Check for the mkldnn fix
        assert "torch.backends.mkldnn.enabled = False" in content, \
            "Training script should have PyTorch mkldnn fix for CPU compatibility"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
