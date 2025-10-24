# Fonctionnalités d'exploration du dataset

## Vue d'ensemble

La page **"🔍 Exploration du dataset"** (`src/pages/2_Exploration_du_dataset.py`) offre une analyse complète et interactive du dataset de cellules sanguines.

## Onglets disponibles

### 1. 📊 Statistiques

Affiche les statistiques globales du dataset :
- Nombre total d'images
- Nombre de classes
- Classe la plus/moins représentée
- Dimensions moyennes, min et max des images
- Histogrammes des largeurs, hauteurs et ratios L/H

**Modules utilisés :**
- `StatisticsCalculator` : calculs statistiques
- `ChartGenerator` : génération des histogrammes Plotly

### 2. 📈 Distribution

Visualise la répartition des images par classe :
- Graphique en barres
- Diagramme circulaire (pie chart) interactif
- Tableau détaillé des classes

**Modules utilisés :**
- `ChartGenerator` : génération du pie chart

### 3. 🎨 Couleurs RGB

**NOUVELLE FONCTIONNALITÉ** - Analyse des distributions RGB par type de cellule :

- **Distributions KDE (Kernel Density Estimation)** : courbes de densité pour chaque canal RGB (Rouge, Vert, Bleu)
- **Organisation en 2 colonnes** : affichage optimisé pour comparer les classes
- **Légende commune** : facilite la lecture des graphiques
- **Graphiques interactifs Plotly** : hover pour voir les valeurs exactes

**Modules utilisés :**
- `RGBAnalyzer` : calcul des distributions RGB avec `scipy.stats.gaussian_kde`
- `RGBChartGenerator` : génération des graphiques Plotly

**Interprétation :**
Les courbes permettent de comparer la distribution des valeurs RGB entre classes. Les différences de densité peuvent indiquer :
- Des contrastes d'exposition différents
- Des variations de coloration propres à chaque type cellulaire
- Des caractéristiques discriminantes pour la classification

**Paramètres :**
- `sample_size` : 10 000 pixels échantillonnés par classe
- `max_images` : 30 images analysées par classe

### 4. 🖼️ Exemples d'images

Affiche des échantillons d'images avec plusieurs sections :

#### a) Sélection par classe
- Sélecteur interactif pour choisir une classe
- Affichage de 8 images aléatoires de la classe sélectionnée

#### b) Exemples aléatoires de plusieurs classes
- Affichage de 4 classes aléatoires
- Une image par classe

#### c) **NOUVELLE FONCTIONNALITÉ** - Images moyennes par classe
- Calcul de l'image moyenne pour chaque classe
- Affichage en grille 4 colonnes
- Permet de visualiser les caractéristiques typiques de chaque type de cellule

**Paramètres :**
- `img_size` : (128, 128) pixels
- `max_images` : 100 images moyennées par classe

#### d) **NOUVELLE FONCTIONNALITÉ** - Matrice de similarité cosinus
- Calcul de la similarité cosinus entre classes (images réelles)
- Heatmap interactive avec annotations
- Colormap "YlGnBu" pour une lecture optimale

**Modules utilisés :**
- `ImageSimilarityCalculator` : calcul des images moyennes et de la similarité cosinus
- `sklearn.metrics.pairwise.cosine_similarity` : calcul de similarité
- `seaborn` : génération de la heatmap

**Interprétation :**
- **Valeur proche de 1** : classes visuellement très similaires (couleurs, textures, structures)
- **Valeur plus faible** : différences marquées entre types de cellules
- Aide à identifier les classes potentiellement difficiles à distinguer

**Paramètres :**
- `n_samples` : 20 images échantillonnées par classe
- `image_size` : (64, 64) pixels pour performance

### 5. 🌈 Visualisation spectrale

Génération de visualisations spectrales détaillées (sur demande via bouton) :
- Histogrammes RGB par classe
- Statistiques μ(R), μ(G), μ(B)

**Module utilisé :**
- `spectral_visualization.visualize_cell_types_distribution`

## Architecture modulaire (SOLID)

Le code suit les principes SOLID avec une séparation claire des responsabilités :

### Modules d'analyse
- **`RGBAnalyzer`** : analyse des distributions RGB avec KDE
- **`ImageSimilarityCalculator`** : calcul de similarité et images moyennes
- **`DatasetAnalyzer`** : analyse globale du dataset
- **`StatisticsCalculator`** : calculs statistiques

### Modules de visualisation
- **`RGBChartGenerator`** : graphiques RGB Plotly
- **`ChartGenerator`** : graphiques généraux
- **`StreamlitRenderer`** : rendu des composants Streamlit

### Modules utilitaires
- **`ImageLoader`** : chargement des images
- **`YamlLoader`** : gestion de la configuration

## Dépendances

Les nouvelles fonctionnalités nécessitent :
- `scipy>=1.11.0` : pour `gaussian_kde` (KDE)
- `scikit-learn>=1.7.2` : pour `cosine_similarity`
- `seaborn>=0.13.2` : pour les heatmaps
- `plotly>=5.18.0` : pour les graphiques interactifs
- `matplotlib>=3.10.7` : pour les visualisations
- `numpy` : calculs numériques
- `pandas` : manipulation de données

## Performance

Les calculs sont optimisés pour la performance :
- **Échantillonnage** : limitation du nombre d'images et de pixels analysés
- **Redimensionnement** : images réduites pour les calculs de similarité
- **Normalisation L2** : pour la similarité cosinus
- **Caching potentiel** : possibilité d'ajouter `@st.cache_data` pour éviter les recalculs

## Utilisation

```bash
# Lancer l'application Streamlit
./start.sh

# Ou directement
uv run streamlit run src/pages/1_Presentation_du_projet.py
```

Naviguer vers l'onglet **"🔍 Exploration du dataset"** dans l'interface Streamlit.

## Évolutions futures possibles

- [ ] Ajout de caching Streamlit pour les calculs lourds
- [ ] Export des visualisations en PDF/PNG
- [ ] Analyse PCA (Principal Component Analysis) des images
- [ ] t-SNE pour visualiser les clusters de cellules
- [ ] Analyse de la texture (GLCM, LBP)
- [ ] Détection d'outliers visuels
