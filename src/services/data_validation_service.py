"""
Data Validation Service

Single Responsibility: Validates dataset quality and integrity before training.
Checks class distribution, image quality, and detects potential data issues.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from PIL import Image

from src.core.constants import CLASS_NAMES


@dataclass
class ValidationReport:
    """Report from data validation."""
    
    is_valid: bool = True
    dataset_path: str = ""
    total_images: int = 0
    class_distribution: dict = field(default_factory=dict)
    
    # Validation checks
    missing_classes: list = field(default_factory=list)
    imbalanced_classes: list = field(default_factory=list)
    corrupted_images: list = field(default_factory=list)
    invalid_size_images: list = field(default_factory=list)
    
    # Statistics
    min_class_count: int = 0
    max_class_count: int = 0
    imbalance_ratio: float = 0.0
    
    # Warnings and errors
    warnings: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert report to dictionary."""
        return {
            "is_valid": self.is_valid,
            "dataset_path": self.dataset_path,
            "total_images": self.total_images,
            "class_distribution": self.class_distribution,
            "missing_classes": self.missing_classes,
            "imbalanced_classes": self.imbalanced_classes,
            "corrupted_images": self.corrupted_images,
            "invalid_size_images": self.invalid_size_images,
            "statistics": {
                "min_class_count": self.min_class_count,
                "max_class_count": self.max_class_count,
                "imbalance_ratio": self.imbalance_ratio,
            },
            "warnings": self.warnings,
            "errors": self.errors,
        }
    
    def save(self, path: Path) -> None:
        """Save report to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class DataValidationService:
    """
    Service for validating dataset quality before training.
    
    Validates:
    - Class distribution (all classes present, balanced)
    - Image integrity (not corrupted, readable)
    - Image dimensions (minimum size requirements)
    """
    
    def __init__(
        self,
        min_samples_per_class: int = 10,
        max_imbalance_ratio: float = 10.0,
        min_image_size: tuple[int, int] = (32, 32),
        expected_classes: Optional[list[str]] = None,
    ):
        """
        Initialize DataValidationService.
        
        Args:
            min_samples_per_class: Minimum samples required per class
            max_imbalance_ratio: Maximum allowed ratio between largest and smallest class
            min_image_size: Minimum (width, height) for images
            expected_classes: List of expected class names (default: CLASS_NAMES)
        """
        self.min_samples_per_class = min_samples_per_class
        self.max_imbalance_ratio = max_imbalance_ratio
        self.min_image_size = min_image_size
        self.expected_classes = expected_classes or CLASS_NAMES
    
    def validate_dataset(self, dataset_path: Path) -> ValidationReport:
        """
        Validate a dataset and return a comprehensive report.
        
        Args:
            dataset_path: Path to dataset directory (with class subdirectories)
            
        Returns:
            ValidationReport with all validation results
        """
        report = ValidationReport(dataset_path=str(dataset_path))
        
        if not dataset_path.exists():
            report.is_valid = False
            report.errors.append(f"Dataset path does not exist: {dataset_path}")
            return report
        
        # Check class distribution
        class_dist = self.check_class_distribution(dataset_path)
        report.class_distribution = class_dist
        report.total_images = sum(class_dist.values())
        
        # Check for missing classes
        report.missing_classes = self._check_missing_classes(class_dist)
        if report.missing_classes:
            report.errors.append(f"Missing classes: {report.missing_classes}")
            report.is_valid = False
        
        # Check for imbalanced classes
        if class_dist:
            counts = [c for c in class_dist.values() if c > 0]
            if counts:
                report.min_class_count = min(counts)
                report.max_class_count = max(counts)
                report.imbalance_ratio = report.max_class_count / max(report.min_class_count, 1)
                
                report.imbalanced_classes = self._check_imbalanced_classes(class_dist)
                if report.imbalanced_classes:
                    report.warnings.append(
                        f"Imbalanced classes (ratio {report.imbalance_ratio:.1f}x): {report.imbalanced_classes}"
                    )
        
        # Check for classes with too few samples
        low_sample_classes = [
            name for name, count in class_dist.items()
            if 0 < count < self.min_samples_per_class
        ]
        if low_sample_classes:
            report.warnings.append(
                f"Classes with < {self.min_samples_per_class} samples: {low_sample_classes}"
            )
        
        return report
    
    def validate_images(
        self,
        dataset_path: Path,
        max_images_to_check: Optional[int] = None,
    ) -> ValidationReport:
        """
        Validate image integrity and dimensions.
        
        Args:
            dataset_path: Path to dataset directory
            max_images_to_check: Maximum images to validate (None for all)
            
        Returns:
            ValidationReport with image validation results
        """
        report = ValidationReport(dataset_path=str(dataset_path))
        
        if not dataset_path.exists():
            report.is_valid = False
            report.errors.append(f"Dataset path does not exist: {dataset_path}")
            return report
        
        # Find all images
        image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(dataset_path.rglob(f"*{ext}"))
            image_paths.extend(dataset_path.rglob(f"*{ext.upper()}"))
        
        # Limit if specified
        if max_images_to_check and len(image_paths) > max_images_to_check:
            import random
            image_paths = random.sample(image_paths, max_images_to_check)
        
        report.total_images = len(image_paths)
        
        # Validate each image
        for img_path in image_paths:
            try:
                with Image.open(img_path) as img:
                    # Check dimensions
                    if img.size[0] < self.min_image_size[0] or img.size[1] < self.min_image_size[1]:
                        report.invalid_size_images.append(str(img_path))
                    
                    # Verify image can be loaded
                    img.verify()
                    
            except Exception:
                report.corrupted_images.append(str(img_path))
        
        # Update validity
        if report.corrupted_images:
            report.errors.append(f"Found {len(report.corrupted_images)} corrupted images")
            report.is_valid = False
        
        if report.invalid_size_images:
            report.warnings.append(
                f"Found {len(report.invalid_size_images)} images smaller than {self.min_image_size}"
            )
        
        return report
    
    def check_class_distribution(self, dataset_path: Path) -> dict[str, int]:
        """
        Check class distribution in dataset.
        
        Expects dataset structure:
        dataset_path/
            class1/
                image1.jpg
                ...
            class2/
                ...
        
        Args:
            dataset_path: Path to dataset directory
            
        Returns:
            Dictionary mapping class names to sample counts
        """
        distribution = {name: 0 for name in self.expected_classes}
        
        if not dataset_path.exists():
            return distribution
        
        # Count images per class directory
        for class_dir in dataset_path.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name.lower()
                
                # Count images in this class
                image_count = 0
                for ext in (".jpg", ".jpeg", ".png", ".bmp", ".tiff"):
                    image_count += len(list(class_dir.glob(f"*{ext}")))
                    image_count += len(list(class_dir.glob(f"*{ext.upper()}")))
                
                # Map to expected class name if possible
                if class_name in distribution:
                    distribution[class_name] = image_count
                else:
                    # Try to find matching class
                    for expected in self.expected_classes:
                        if expected.lower() in class_name or class_name in expected.lower():
                            distribution[expected] = image_count
                            break
                    else:
                        # Unknown class
                        distribution[class_name] = image_count
        
        return distribution
    
    def _check_missing_classes(self, class_dist: dict[str, int]) -> list[str]:
        """Check for missing classes (zero samples)."""
        return [name for name in self.expected_classes if class_dist.get(name, 0) == 0]
    
    def _check_imbalanced_classes(self, class_dist: dict[str, int]) -> list[str]:
        """Check for severely imbalanced classes."""
        if not class_dist:
            return []
        
        counts = [c for c in class_dist.values() if c > 0]
        if not counts:
            return []
        
        avg_count = sum(counts) / len(counts)
        threshold = avg_count / self.max_imbalance_ratio
        
        return [
            name for name, count in class_dist.items()
            if 0 < count < threshold
        ]
    
    def compare_with_baseline(
        self,
        current_dist: dict[str, int],
        baseline_dist: dict[str, int],
        drift_threshold: float = 0.2,
    ) -> dict:
        """
        Compare current distribution with a baseline to detect data drift.
        
        Args:
            current_dist: Current class distribution
            baseline_dist: Baseline class distribution
            drift_threshold: Maximum allowed relative change (0.2 = 20%)
            
        Returns:
            Dictionary with drift analysis
        """
        drift_report = {
            "has_drift": False,
            "drifted_classes": [],
            "changes": {},
        }
        
        for class_name in self.expected_classes:
            current = current_dist.get(class_name, 0)
            baseline = baseline_dist.get(class_name, 0)
            
            if baseline > 0:
                relative_change = abs(current - baseline) / baseline
                drift_report["changes"][class_name] = {
                    "baseline": baseline,
                    "current": current,
                    "relative_change": relative_change,
                }
                
                if relative_change > drift_threshold:
                    drift_report["has_drift"] = True
                    drift_report["drifted_classes"].append(class_name)
        
        return drift_report
