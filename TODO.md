# TODO - Blood Cell Classification Project

## Completed ✅

### Model Evaluation & Metrics (Oct 23-24, 2024)
- [x] Fixed `YamlLoader` project_root path (was pointing to `src/` instead of project root)
- [x] Created `evaluate_model.py` script to evaluate existing models and generate metrics
- [x] Added `evaluate-model` entry point in `pyproject.toml`
- [x] Fixed checkpoint loading logic to handle models saved without 'model.' prefix
- [x] Generated `metrics.json` file with performance metrics (72.4% val accuracy, 71.6% test accuracy)
- [x] Updated `conf.yaml` paths to reflect actual project structure
- [x] Added confusion matrix computation in `EvaluationService`
- [x] Integrated confusion matrix in both `train_model.py` and `evaluate_model.py`
- [x] Added confusion matrix visualization in Streamlit page using Plotly heatmap
- [x] Added per-class accuracy table in Streamlit interface
- [x] Created `KerasInferenceService` for Keras/TensorFlow model inference (Oct 24)
- [x] Modified demo page to compare predictions from PyTorch and Keras models side-by-side (Oct 24)
- [x] Created `gradcam_analyzer.py` utility module for Grad-CAM analysis (Oct 24)
- [x] Refactored page 5 with tabs to compare ResNet with/without mask (Oct 24)
- [x] Implemented SOLID principles: extracted reusable functions to avoid code duplication (Oct 24)
- [x] Added masked models predictions in demo page (2-column layout: PyTorch models grouped, Keras models grouped) (Oct 24)
  - Column 1: ResNet + ResNet masked
  - Column 2: Keras Baseline PBC + Keras Baseline PBC masked
- [x] Fixed GradCAM implementation to avoid gradient pollution between images (Oct 24)
  - Added explicit `model.zero_grad()` calls before and after attribution
  - Added `torch.no_grad()` context for inference
  - Added `.cpu()` call to ensure proper tensor cleanup
  - Added division by zero protection in heatmap normalization
  - Improved model caching with clean state initialization
  - Added button to reload model and clear cache
  - Added seed-based random sampling with refresh button

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

### Dataset Exploration Enhancement (Oct 23, 2024)
- [x] Created `RGBAnalyzer` for RGB distribution analysis with KDE
- [x] Created `RGBChartGenerator` for Plotly RGB distribution charts
- [x] Created `ImageSimilarityCalculator` for cosine similarity and mean images
- [x] Added new tab "🎨 Couleurs RGB" with RGB distributions by class
- [x] Enhanced "🖼️ Exemples d'images" tab with mean images per class
- [x] Added cosine similarity matrix between classes (real images)
- [x] Integrated all new visualizations in Streamlit interface

### Application Orchestration
- [x] Created `start.sh` script with PID tracking and logging
- [x] Created `stop.sh` script for graceful shutdown
- [x] Added runtime configuration paths in `conf.yaml`
- [x] Implemented comprehensive tests for scripts (`test_scripts.py`)
- [x] Dockerized application with multi-stage build
- [x] Created `docker-compose.yml` for easy deployment
- [x] Created `.dockerignore` for optimized builds
- [x] Updated `README.md` with Docker instructions
- [x] Integrated Kaggle dataset download in Docker build process
- [x] Configured `kaggle.json` credentials in Dockerfile and docker-compose
- [x] Aligned data paths between `load_dataset.sh` and docker volumes

### Architecture Structure
```
src/
├── services/
│   ├── data_transform_service.py    # Handles data transformations
│   ├── dataset_service.py           # Handles dataset loading & splitting
│   ├── training_service.py          # Handles model training
│   ├── evaluation_service.py        # Handles model evaluation
│   ├── inference_service.py         # Handles PyTorch model inference
│   ├── keras_inference_service.py   # Handles Keras/TensorFlow model inference
│   └── yaml_loader.py               # Configuration management
├── models/
│   ├── base_classifier.py           # Abstract base class (Interface Segregation)
│   ├── resnet_classifier.py         # ResNet implementation (Open/Closed)
│   └── model_factory.py             # Model creation factory (Dependency Inversion)
├── utils/
│   ├── chart_generator.py           # General chart generation
│   ├── dataset_analyzer.py          # Dataset analysis utilities
│   ├── gradcam_analyzer.py          # Grad-CAM analysis utilities
│   ├── image_loader.py              # Image loading utilities
│   ├── image_similarity.py          # Cosine similarity & mean images
│   ├── rgb_analyzer.py              # RGB distribution analysis (KDE)
│   ├── rgb_chart_generator.py       # RGB distribution charts (Plotly)
│   ├── spectral_visualization.py    # Spectral visualization
│   ├── statistics_calculator.py     # Statistical calculations
│   └── streamlit_renderers.py       # Streamlit rendering components
├── pages/
│   ├── 2_Exploration_du_dataset.py  # Dataset exploration with visualizations
│   ├── 3_Modele.py                  # Model info & metrics visualization
│   ├── 4_Demo.py                    # Interactive demo: 4 models comparison (PyTorch & Keras, each with/without mask)
│   └── 5_Ameliorations_potentielles.py  # Grad-CAM analysis with/without mask comparison
└── pipe/
    ├── train_model.py               # Main training script
    └── evaluate_model.py            # Model evaluation script

scripts/
├── start.sh                         # Launch Streamlit with PID tracking
└── stop.sh                          # Stop Streamlit gracefully

tests/
├── test_data_transform_service.py
├── test_model_factory.py
├── test_resnet_classifier.py
├── test_evaluation_service.py
├── test_inference_service.py
└── test_scripts.py                  # Tests for start/stop scripts
```

## Usage

### Run Application

```bash
# Start Streamlit app
./start.sh

# Stop Streamlit app
./stop.sh

# View logs
tail -f logs/streamlit.log
```

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
- Runtime configuration (PID file, log file)

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
- [x] Generate confusion matrix visualization
- [x] Implement per-class metrics reporting
- [x] Create model interpretability tools (Grad-CAM)
- [ ] Add ROC curves and precision-recall curves

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