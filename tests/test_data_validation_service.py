"""
Tests for DataValidationService.

TDD approach: tests for data validation functionality.
"""

from PIL import Image


class TestValidationReport:
    """Tests for ValidationReport dataclass."""

    def test_to_dict(self):
        """Should convert report to dictionary."""
        from src.services.data_validation_service import ValidationReport
        
        report = ValidationReport(
            is_valid=True,
            dataset_path="/data/test",
            total_images=100,
            class_distribution={"class1": 50, "class2": 50},
        )
        
        d = report.to_dict()
        
        assert d["is_valid"] is True
        assert d["total_images"] == 100
        assert d["class_distribution"]["class1"] == 50

    def test_save_report(self, tmp_path):
        """Should save report to JSON file."""
        from src.services.data_validation_service import ValidationReport
        
        report = ValidationReport(
            is_valid=True,
            total_images=100,
        )
        
        report_path = tmp_path / "report.json"
        report.save(report_path)
        
        assert report_path.exists()
        
        import json
        with open(report_path) as f:
            data = json.load(f)
        
        assert data["is_valid"] is True


class TestCheckClassDistribution:
    """Tests for check_class_distribution method."""

    def test_check_distribution_empty_dir(self, tmp_path):
        """Should return zeros for empty directory."""
        from src.services.data_validation_service import DataValidationService
        
        validator = DataValidationService(expected_classes=["class1", "class2"])
        dist = validator.check_class_distribution(tmp_path)
        
        assert dist["class1"] == 0
        assert dist["class2"] == 0

    def test_check_distribution_with_images(self, tmp_path):
        """Should count images per class."""
        from src.services.data_validation_service import DataValidationService
        
        # Create class directories with images
        class1_dir = tmp_path / "class1"
        class2_dir = tmp_path / "class2"
        class1_dir.mkdir()
        class2_dir.mkdir()
        
        # Create test images
        for i in range(5):
            img = Image.new("RGB", (64, 64), color=(255, 0, 0))
            img.save(class1_dir / f"img_{i}.jpg")
        
        for i in range(3):
            img = Image.new("RGB", (64, 64), color=(0, 255, 0))
            img.save(class2_dir / f"img_{i}.jpg")
        
        validator = DataValidationService(expected_classes=["class1", "class2"])
        dist = validator.check_class_distribution(tmp_path)
        
        assert dist["class1"] == 5
        assert dist["class2"] == 3

    def test_check_distribution_nonexistent_path(self, tmp_path):
        """Should return zeros for nonexistent path."""
        from src.services.data_validation_service import DataValidationService
        
        validator = DataValidationService(expected_classes=["class1"])
        dist = validator.check_class_distribution(tmp_path / "nonexistent")
        
        assert dist["class1"] == 0


class TestValidateDataset:
    """Tests for validate_dataset method."""

    def test_validate_missing_dataset(self, tmp_path):
        """Should report error for missing dataset."""
        from src.services.data_validation_service import DataValidationService
        
        validator = DataValidationService()
        report = validator.validate_dataset(tmp_path / "nonexistent")
        
        assert report.is_valid is False
        assert len(report.errors) > 0

    def test_validate_missing_classes(self, tmp_path):
        """Should detect missing classes."""
        from src.services.data_validation_service import DataValidationService
        
        # Create only one class
        class1_dir = tmp_path / "class1"
        class1_dir.mkdir()
        
        for i in range(10):
            img = Image.new("RGB", (64, 64))
            img.save(class1_dir / f"img_{i}.jpg")
        
        validator = DataValidationService(expected_classes=["class1", "class2"])
        report = validator.validate_dataset(tmp_path)
        
        assert report.is_valid is False
        assert "class2" in report.missing_classes

    def test_validate_valid_dataset(self, tmp_path):
        """Should pass for valid dataset."""
        from src.services.data_validation_service import DataValidationService
        
        # Create balanced classes
        for class_name in ["class1", "class2"]:
            class_dir = tmp_path / class_name
            class_dir.mkdir()
            
            for i in range(20):
                img = Image.new("RGB", (64, 64))
                img.save(class_dir / f"img_{i}.jpg")
        
        validator = DataValidationService(
            expected_classes=["class1", "class2"],
            min_samples_per_class=10,
        )
        report = validator.validate_dataset(tmp_path)
        
        assert report.is_valid is True
        assert report.total_images == 40
        assert len(report.errors) == 0

    def test_validate_imbalanced_classes(self, tmp_path):
        """Should warn about imbalanced classes."""
        from src.services.data_validation_service import DataValidationService
        
        # Create imbalanced classes
        class1_dir = tmp_path / "class1"
        class2_dir = tmp_path / "class2"
        class1_dir.mkdir()
        class2_dir.mkdir()
        
        # 100 images in class1, 5 in class2
        for i in range(100):
            img = Image.new("RGB", (64, 64))
            img.save(class1_dir / f"img_{i}.jpg")
        
        for i in range(5):
            img = Image.new("RGB", (64, 64))
            img.save(class2_dir / f"img_{i}.jpg")
        
        validator = DataValidationService(
            expected_classes=["class1", "class2"],
            max_imbalance_ratio=5.0,
        )
        report = validator.validate_dataset(tmp_path)
        
        # Should have warnings about imbalance
        assert len(report.warnings) > 0
        assert report.imbalance_ratio == 20.0  # 100/5


class TestValidateImages:
    """Tests for validate_images method."""

    def test_validate_valid_images(self, tmp_path):
        """Should pass for valid images."""
        from src.services.data_validation_service import DataValidationService
        
        # Create valid images
        for i in range(5):
            img = Image.new("RGB", (64, 64))
            img.save(tmp_path / f"img_{i}.jpg")
        
        validator = DataValidationService(min_image_size=(32, 32))
        report = validator.validate_images(tmp_path)
        
        assert report.is_valid is True
        assert len(report.corrupted_images) == 0

    def test_validate_small_images(self, tmp_path):
        """Should detect images smaller than minimum size."""
        from src.services.data_validation_service import DataValidationService
        
        # Create small images
        for i in range(3):
            img = Image.new("RGB", (16, 16))  # Smaller than minimum
            img.save(tmp_path / f"small_{i}.jpg")
        
        validator = DataValidationService(min_image_size=(32, 32))
        report = validator.validate_images(tmp_path)
        
        assert len(report.invalid_size_images) == 3
        assert len(report.warnings) > 0

    def test_validate_corrupted_images(self, tmp_path):
        """Should detect corrupted images."""
        from src.services.data_validation_service import DataValidationService
        
        # Create a corrupted image file
        corrupted_path = tmp_path / "corrupted.jpg"
        with open(corrupted_path, "wb") as f:
            f.write(b"not a valid image")
        
        validator = DataValidationService()
        report = validator.validate_images(tmp_path)
        
        assert len(report.corrupted_images) == 1
        assert report.is_valid is False


class TestCompareWithBaseline:
    """Tests for compare_with_baseline method."""

    def test_no_drift(self):
        """Should detect no drift when distributions are similar."""
        from src.services.data_validation_service import DataValidationService
        
        validator = DataValidationService(expected_classes=["class1", "class2"])
        
        current = {"class1": 100, "class2": 100}
        baseline = {"class1": 100, "class2": 100}
        
        result = validator.compare_with_baseline(current, baseline)
        
        assert result["has_drift"] is False
        assert len(result["drifted_classes"]) == 0

    def test_detect_drift(self):
        """Should detect drift when distributions differ significantly."""
        from src.services.data_validation_service import DataValidationService
        
        validator = DataValidationService(expected_classes=["class1", "class2"])
        
        current = {"class1": 100, "class2": 50}  # class2 dropped 50%
        baseline = {"class1": 100, "class2": 100}
        
        result = validator.compare_with_baseline(
            current, baseline, drift_threshold=0.2
        )
        
        assert result["has_drift"] is True
        assert "class2" in result["drifted_classes"]

    def test_drift_within_threshold(self):
        """Should not flag drift within threshold."""
        from src.services.data_validation_service import DataValidationService
        
        validator = DataValidationService(expected_classes=["class1", "class2"])
        
        current = {"class1": 100, "class2": 90}  # 10% change
        baseline = {"class1": 100, "class2": 100}
        
        result = validator.compare_with_baseline(
            current, baseline, drift_threshold=0.2  # 20% threshold
        )
        
        assert result["has_drift"] is False
