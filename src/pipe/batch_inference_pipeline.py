"""
Batch Inference Pipeline

Reusable pipeline for running batch inference on a directory of images.
Can be called from:
- CLI: uv run batch-inference
- Airflow DAG
- Python scripts
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from PIL import Image
from tqdm import tqdm

from src.core.constants import CLASS_NAMES
from src.services.yaml_loader import YamlLoader
from src.services.model_loader_service import ModelLoaderService


@dataclass
class BatchInferenceResult:
    """Result of batch inference pipeline."""
    
    # Summary statistics
    total_images: int = 0
    successful: int = 0
    failed: int = 0
    class_distribution: dict = field(default_factory=dict)
    
    # Model info
    model_source: str = ""
    
    # Predictions
    predictions: list = field(default_factory=list)
    
    # Output
    output_file: Optional[Path] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "summary": {
                "total_images": self.total_images,
                "successful": self.successful,
                "failed": self.failed,
                "class_distribution": self.class_distribution,
            },
            "model_source": self.model_source,
            "predictions": self.predictions,
            "output_file": str(self.output_file) if self.output_file else None,
        }


def find_images(
    input_dir: Path,
    extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tiff"),
    max_images: Optional[int] = None,
    recursive: bool = True,
) -> list[str]:
    """
    Find all image files in a directory.
    
    Args:
        input_dir: Directory to search
        extensions: Tuple of valid image extensions
        max_images: Maximum number of images to return (None for all)
        recursive: Search subdirectories
        
    Returns:
        List of image file paths as strings
    """
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    image_paths = []
    
    if recursive:
        for ext in extensions:
            image_paths.extend(input_dir.rglob(f"*{ext}"))
            image_paths.extend(input_dir.rglob(f"*{ext.upper()}"))
    else:
        for ext in extensions:
            image_paths.extend(input_dir.glob(f"*{ext}"))
            image_paths.extend(input_dir.glob(f"*{ext.upper()}"))
    
    # Convert to strings and sort
    image_paths = sorted([str(p) for p in image_paths])
    
    # Limit if specified
    if max_images is not None and len(image_paths) > max_images:
        image_paths = image_paths[:max_images]
    
    return image_paths


def run_batch_inference(
    input_dir: Path,
    output_dir: Optional[Path] = None,
    max_images: Optional[int] = None,
    prefer_mlflow: bool = True,
    save_results: bool = True,
    show_progress: bool = True,
) -> BatchInferenceResult:
    """
    Run batch inference on all images in a directory.
    
    Args:
        input_dir: Directory containing images to process
        output_dir: Directory to save results (default: models/predictions)
        max_images: Maximum number of images to process (None for all)
        prefer_mlflow: Try to load model from MLflow first
        save_results: Whether to save results to JSON file
        show_progress: Show progress bar
        
    Returns:
        BatchInferenceResult with predictions and summary
        
    Raises:
        FileNotFoundError: If input_dir doesn't exist or no model available
    """
    # Initialize
    loader = YamlLoader()
    config = loader.config
    
    # Resolve output directory
    if output_dir is None:
        output_dir = loader.project_root / "models" / "predictions"
    
    # Find images
    print(f"🔍 Searching for images in: {input_dir}")
    image_paths = find_images(input_dir, max_images=max_images)
    
    if not image_paths:
        raise FileNotFoundError(f"No images found in {input_dir}")
    
    print(f"📁 Found {len(image_paths)} images")
    
    # Load model
    print("📦 Loading model...")
    model_loader = ModelLoaderService(config)
    load_result = model_loader.load(prefer_mlflow=prefer_mlflow)
    inference_service = load_result.inference_service
    model_source = load_result.source
    
    print(f"✅ Model loaded from: {model_source}")
    
    # Run predictions
    print(f"🔮 Running inference on {len(image_paths)} images...")
    
    predictions = []
    class_counts = {name: 0 for name in CLASS_NAMES}
    
    iterator = tqdm(image_paths, desc="Processing") if show_progress else image_paths
    
    for img_path in iterator:
        try:
            image = Image.open(img_path).convert("RGB")
            prediction = inference_service.predict_image(image)
            
            result = {
                "image_path": img_path,
                "predicted_class": prediction["predicted_class"],
                "confidence": prediction["confidence"],
                "probabilities": prediction["probabilities"],
            }
            predictions.append(result)
            class_counts[prediction["predicted_class"]] += 1
            
        except Exception as e:
            print(f"⚠️ Error processing {img_path}: {e}")
            predictions.append({
                "image_path": img_path,
                "error": str(e),
            })
    
    # Build result
    successful = len([p for p in predictions if "predicted_class" in p])
    failed = len(predictions) - successful
    
    result = BatchInferenceResult(
        total_images=len(image_paths),
        successful=successful,
        failed=failed,
        class_distribution=class_counts,
        model_source=model_source,
        predictions=predictions,
    )
    
    # Save results
    if save_results:
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"predictions_{timestamp}.json"
        
        output_data = {
            "metadata": {
                "timestamp": timestamp,
                "model_source": model_source,
                "input_dir": str(input_dir),
            },
            "summary": {
                "total_images": result.total_images,
                "successful": result.successful,
                "failed": result.failed,
                "class_distribution": result.class_distribution,
            },
            "predictions": predictions,
        }
        
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)
        
        result.output_file = output_file
        print(f"💾 Results saved to: {output_file}")
    
    # Print summary
    print("\n✅ Batch inference completed!")
    print(f"   Total: {result.total_images}")
    print(f"   Successful: {result.successful}")
    print(f"   Failed: {result.failed}")
    print(f"   Class distribution: {result.class_distribution}")
    
    return result


def main() -> None:
    """CLI entry point for batch inference."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run batch inference on blood cell images"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="Directory containing images to process (default: dataset directory)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory to save results (default: models/predictions)",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Maximum number of images to process",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Don't try to load model from MLflow, use checkpoint only",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to file",
    )
    
    args = parser.parse_args()
    
    # Default input directory
    if args.input_dir is None:
        loader = YamlLoader()
        args.input_dir = loader.data_dir / "raw" / "bloodcells_dataset"
    
    result = run_batch_inference(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        max_images=args.max_images,
        prefer_mlflow=not args.no_mlflow,
        save_results=not args.no_save,
    )
    
    print(f"\n📊 Final result: {result.successful}/{result.total_images} successful")


if __name__ == "__main__":
    main()
