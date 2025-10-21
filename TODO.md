# TODO - Blood Cell Classification Project

## Completed ✅

### Code Refactoring (SOLID Architecture)
- [x] Refactored notebook training code into SOLID architecture
- [x] Created configuration-driven system (conf.yaml)
- [x] Implemented data transformation service (Single Responsibility)
- [x] Implemented dataset loading service (Single Responsibility)
- [x] Implemented model factory pattern (Open/Closed, Dependency Inversion)
- [x] Implemented training service (Single Responsibility)
- [x] Implemented evaluation service (Single Responsibility)
- [x] Implemented inference service (Single Responsibility)
- [x] Created orchestration script (train_model.py)
- [x] Created comprehensive test suite
- [x] Integrated backend services with Streamlit pages
- [x] Added model retraining button in Modele interface
- [x] Rédigé la documentation de `src/` dans `docs/src_documentation.md`

### Architecture Structure
```
src/
├── services/
│   ├── data_transform_service.py    # Handles data transformations
│   ├── dataset_service.py           # Handles dataset loading & splitting
│   ├── training_service.py          # Handles model training
│   ├── evaluation_service.py        # Handles model evaluation
│   ├── inference_service.py         # Handles model inference
│   └── yaml_loader.py               # Configuration management
├── models/
│   ├── base_classifier.py           # Abstract base class (Interface Segregation)
│   ├── resnet_classifier.py         # ResNet implementation (Open/Closed)
│   └── model_factory.py             # Model creation factory (Dependency Inversion)
├── pages/
│   ├── 3_Modele.py                  # Model info & metrics visualization
│   └── 4_Demo.py                    # Interactive demo using InferenceService
└── pipe/
    └── train_model.py               # Main training script

tests/
├── test_data_transform_service.py
├── test_model_factory.py
├── test_resnet_classifier.py
├── test_evaluation_service.py
└── test_inference_service.py
```

## Usage

### Run Training
```bash
# Using script entry point (recommended)
uv run train-model

# Or as a module
python3 -m src.pipe.train_model
```

### Run Tests
```bash
uv run pytest tests/
```

### Configuration
All parameters centralized in `src/conf.yaml`:
- Training parameters (batch_size, learning_rate, epochs, etc.)
- Model configuration (architecture, pretrained weights)
- Data augmentation settings
- Path management

## Next Steps

### Model Improvements
- [ ] Implement additional model architectures (EfficientNet, Vision Transformer)
- [ ] Add learning rate scheduler
- [ ] Implement early stopping
- [ ] Add gradient accumulation for larger effective batch sizes

### Monitoring & Logging
- [ ] Add tensorboard logging
- [ ] Implement MLflow experiment tracking
- [ ] Add progress visualization during training

### Evaluation & Analysis
- [ ] Generate confusion matrix visualization
- [ ] Implement per-class metrics reporting
- [ ] Add ROC curves and precision-recall curves
- [ ] Create model interpretability tools (Grad-CAM)

### Data Pipeline
- [ ] Implement data versioning (DVC)
- [ ] Add more sophisticated augmentation strategies
- [ ] Implement cross-validation
- [ ] Add stratified sampling

### Production
- [ ] Create model serving API
- [ ] Add model versioning
- [ ] Implement model deployment pipeline
- [ ] Add performance monitoring in production