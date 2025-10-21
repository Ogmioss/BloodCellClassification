# BloodCellClassification
Application pour classer les photos d'imagerie médicale de cellules sanguines par Machine Learning

## 🐳 Démarrage rapide avec Docker (Recommandé)

### Prérequis
- Docker
- Docker Compose

### Lancement de l'application

```bash
# Construire et démarrer l'application
docker-compose up -d

# Voir les logs
docker-compose logs -f

# Arrêter l'application
docker-compose down
```

L'application sera accessible à l'adresse : **http://localhost:8501**

### Commandes Docker utiles

```bash
# Reconstruire l'image après modification du code
docker-compose up -d --build

# Accéder au conteneur
docker-compose exec bloodcell-app bash

# Voir l'état du conteneur
docker-compose ps

# Arrêter et supprimer les volumes
docker-compose down -v
```

## 💻 Installation locale (Alternative)

### Prérequis
- Python 3.11+
- uv package manager

### Installation

```bash
# Installer le projet en mode éditable (requis pour les imports)
uv pip install -e .
```

## Chargement jeu de données
```bash
./scripts/load_dataset.sh
```


### About Dataset

The dataset contains a total of 17,092 images of individual normal cells, which were acquired using the analyzer CellaVision DM96 in the Core Laboratory at the Hospital Clinic of Barcelona. The dataset is organized in the following eight groups: neutrophils, eosinophils, basophils, lymphocytes, monocytes, immature granulocytes (promyelocytes, myelocytes, and metamyelocytes), erythroblasts and platelets or thrombocytes. The size of the images is 360 x 363 pixels, in format JPG, and they were annotated by expert clinical pathologists. The images were captured from individuals without infection, hematologic or oncologic disease and free of any pharmacologic treatment at the moment of blood collection.

This high-quality labelled dataset may be used to train and test machine learning and deep learning models to recognize different types of normal peripheral blood cells. To our knowledge, this is the first publicly available set with large numbers of normal peripheral blood cells, so that it is expected to be a canonical dataset for model benchmarking.

## Entraînement du modèle

### Avec Docker

```bash
# Entraîner le modèle dans le conteneur
docker-compose exec bloodcell-app uv run train-model
```

### En local

```bash
# Recommandé avec uv
uv run train-model
```

Ou directement:
```bash
python3 -m src.pipe.train_model
```

## Prédiction
scripts/predict.py

## Interface utilisateur

### Avec Docker
L'interface est automatiquement lancée avec `docker-compose up -d`

### En local

```bash
# Avec uv (recommandé)
uv run streamlit run src/app.py

# Ou avec le script de lancement
./start.sh
```
