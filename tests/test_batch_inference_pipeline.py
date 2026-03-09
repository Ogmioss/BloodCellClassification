"""
Tests for batch inference pipeline.

TDD approach: tests for the reusable batch inference pipeline.
"""

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from PIL import Image


class TestFindImages:
    """Tests for find_images function."""

    def test_find_images_in_directory(self, tmp_path):
        """Should find all images in a directory."""
        from src.pipe.batch_inference_pipeline import find_images
        
        # Create test images
        for i in range(5):
            img = Image.new("RGB", (64, 64), color=(255, 0, 0))
            img.save(tmp_path / f"image_{i}.jpg")
        
        result = find_images(tmp_path)
        
        assert len(result) == 5
        assert all(p.endswith(".jpg") for p in result)

    def test_find_images_with_max_limit(self, tmp_path):
        """Should respect max_images limit."""
        from src.pipe.batch_inference_pipeline import find_images
        
        # Create test images
        for i in range(10):
            img = Image.new("RGB", (64, 64), color=(255, 0, 0))
            img.save(tmp_path / f"image_{i}.jpg")
        
        result = find_images(tmp_path, max_images=3)
        
        assert len(result) == 3

    def test_find_images_recursive(self, tmp_path):
        """Should find images in subdirectories."""
        from src.pipe.batch_inference_pipeline import find_images
        
        # Create subdirectory with images
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        
        img = Image.new("RGB", (64, 64), color=(255, 0, 0))
        img.save(tmp_path / "root.jpg")
        img.save(subdir / "sub.jpg")
        
        result = find_images(tmp_path, recursive=True)
        
        assert len(result) == 2

    def test_find_images_non_recursive(self, tmp_path):
        """Should not find images in subdirectories when recursive=False."""
        from src.pipe.batch_inference_pipeline import find_images
        
        # Create subdirectory with images
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        
        img = Image.new("RGB", (64, 64), color=(255, 0, 0))
        img.save(tmp_path / "root.jpg")
        img.save(subdir / "sub.jpg")
        
        result = find_images(tmp_path, recursive=False)
        
        assert len(result) == 1

    def test_find_images_missing_directory(self, tmp_path):
        """Should raise FileNotFoundError for missing directory."""
        from src.pipe.batch_inference_pipeline import find_images
        
        missing_dir = tmp_path / "nonexistent"
        
        with pytest.raises(FileNotFoundError):
            find_images(missing_dir)

    def test_find_images_multiple_extensions(self, tmp_path):
        """Should find images with different extensions."""
        from src.pipe.batch_inference_pipeline import find_images
        
        img = Image.new("RGB", (64, 64), color=(255, 0, 0))
        img.save(tmp_path / "image1.jpg")
        img.save(tmp_path / "image2.png")
        img.save(tmp_path / "image3.jpeg")
        
        result = find_images(tmp_path)
        
        assert len(result) == 3


class TestBatchInferenceResult:
    """Tests for BatchInferenceResult dataclass."""

    def test_to_dict(self):
        """Should convert result to dictionary."""
        from src.pipe.batch_inference_pipeline import BatchInferenceResult
        
        result = BatchInferenceResult(
            total_images=10,
            successful=8,
            failed=2,
            class_distribution={"lymphocyte": 5, "monocyte": 3},
            model_source="checkpoint:best_model.pth",
            predictions=[{"image_path": "test.jpg", "predicted_class": "lymphocyte"}],
            output_file=Path("/tmp/predictions.json"),
        )
        
        d = result.to_dict()
        
        assert d["summary"]["total_images"] == 10
        assert d["summary"]["successful"] == 8
        assert d["model_source"] == "checkpoint:best_model.pth"
        assert len(d["predictions"]) == 1


class TestRunBatchInference:
    """Tests for run_batch_inference function."""

    def test_run_batch_inference_with_mock_model(self, tmp_path):
        """Should run batch inference with mocked model."""
        from src.pipe.batch_inference_pipeline import run_batch_inference
        
        # Create test images
        for i in range(3):
            img = Image.new("RGB", (64, 64), color=(255, 0, 0))
            img.save(tmp_path / f"image_{i}.jpg")
        
        output_dir = tmp_path / "output"
        
        # Mock the ModelLoaderService
        mock_inference_service = MagicMock()
        mock_inference_service.predict_image.return_value = {
            "predicted_class": "lymphocyte",
            "confidence": 0.95,
            "probabilities": {"lymphocyte": 0.95, "monocyte": 0.05},
        }
        
        mock_load_result = MagicMock()
        mock_load_result.inference_service = mock_inference_service
        mock_load_result.source = "checkpoint:best_model.pth"
        
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_load_result
        
        with patch("src.pipe.batch_inference_pipeline.ModelLoaderService", return_value=mock_loader):
            result = run_batch_inference(
                input_dir=tmp_path,
                output_dir=output_dir,
                save_results=True,
                show_progress=False,
            )
        
        assert result.total_images == 3
        assert result.successful == 3
        assert result.failed == 0
        assert result.class_distribution["lymphocyte"] == 3
        assert result.output_file is not None
        assert result.output_file.exists()

    def test_run_batch_inference_no_images(self, tmp_path):
        """Should raise error when no images found."""
        from src.pipe.batch_inference_pipeline import run_batch_inference
        
        with pytest.raises(FileNotFoundError, match="No images found"):
            run_batch_inference(input_dir=tmp_path, save_results=False)

    def test_run_batch_inference_handles_errors(self, tmp_path):
        """Should handle errors during image processing."""
        from src.pipe.batch_inference_pipeline import run_batch_inference
        
        # Create test images
        for i in range(3):
            img = Image.new("RGB", (64, 64), color=(255, 0, 0))
            img.save(tmp_path / f"image_{i}.jpg")
        
        # Mock the ModelLoaderService to raise error on second image
        mock_inference_service = MagicMock()
        call_count = [0]
        
        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 2:
                raise ValueError("Test error")
            return {
                "predicted_class": "lymphocyte",
                "confidence": 0.95,
                "probabilities": {"lymphocyte": 0.95},
            }
        
        mock_inference_service.predict_image.side_effect = side_effect
        
        mock_load_result = MagicMock()
        mock_load_result.inference_service = mock_inference_service
        mock_load_result.source = "checkpoint:best_model.pth"
        
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_load_result
        
        with patch("src.pipe.batch_inference_pipeline.ModelLoaderService", return_value=mock_loader):
            result = run_batch_inference(
                input_dir=tmp_path,
                save_results=False,
                show_progress=False,
            )
        
        assert result.total_images == 3
        assert result.successful == 2
        assert result.failed == 1
