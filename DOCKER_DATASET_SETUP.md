# Docker Dataset Setup - Configuration Kaggle

## Vue d'ensemble

Le système Docker a été configuré pour télécharger automatiquement le dataset Kaggle lors du build de l'image.

## Modifications apportées

### 1. **Dockerfile** (`/Dockerfile`)

**Ajouts :**
- Installation de `unzip` pour décompresser le dataset
- Création des répertoires `src/data/raw` et `src/data/processed`
- Configuration du répertoire Kaggle credentials (`/root/.kaggle/`)
- Copie du fichier `kaggle.json` avec permissions `600`
- Exécution du script `load_dataset.sh` pendant le build

**Lignes clés :**
```dockerfile
# Install unzip for dataset extraction
RUN apt-get update && apt-get install -y unzip

# Setup Kaggle credentials
RUN mkdir -p /root/.kaggle
COPY kaggle.json /root/.kaggle/
RUN chmod 600 /root/.kaggle/kaggle.json

# Download dataset during build
RUN chmod +x scripts/load_dataset.sh
RUN bash scripts/load_dataset.sh || echo "Dataset download skipped"
```

### 2. **docker-compose.yml** (`/docker-compose.yml`)

**Modifications :**
- Volume `./src/data:/app/src/data` pour aligner avec la structure du script
- Volume `./kaggle.json:/root/.kaggle/kaggle.json:ro` pour permettre la mise à jour des credentials sans rebuild

**Lignes clés :**
```yaml
volumes:
  - ./src/data:/app/src/data
  - ./kaggle.json:/root/.kaggle/kaggle.json:ro
```

### 3. **.dockerignore** (`/.dockerignore`)

**Modifications :**
- Exclusion de `scripts/*` mais inclusion de `scripts/load_dataset.sh`
- Commentaire explicatif pour `src/data` (non exclu car nécessaire au build)

**Lignes clés :**
```
scripts/*
!scripts/load_dataset.sh
```

### 4. **TODO.md** (`/TODO.md`)

**Ajouts :**
- Documentation des tâches complétées pour l'intégration Kaggle

## Fonctionnement

### Build de l'image

1. Le `Dockerfile` copie `kaggle.json` dans `/root/.kaggle/`
2. Les permissions sont définies à `600` (requis par Kaggle API)
3. Le script `load_dataset.sh` est exécuté :
   - Installe `kaggle` via pip
   - Crée les répertoires nécessaires
   - Télécharge le dataset depuis Kaggle
   - Décompresse et nettoie les fichiers temporaires

### Runtime avec docker-compose

1. Le dataset est déjà présent dans l'image (téléchargé au build)
2. Le volume `./src/data:/app/src/data` permet la persistance des données
3. Le volume `./kaggle.json` permet de mettre à jour les credentials sans rebuild

## Prérequis

### Fichier kaggle.json

Le fichier `kaggle.json` doit être présent à la racine du projet avec la structure :

```json
{
  "username": "votre_username",
  "key": "votre_api_key"
}
```

**Obtenir les credentials Kaggle :**
1. Aller sur https://www.kaggle.com/
2. Cliquer sur votre profil → Account
3. Section "API" → "Create New API Token"
4. Télécharger le fichier `kaggle.json`

## Commandes

### Build et lancement

```bash
# Build l'image (télécharge le dataset)
docker-compose build

# Lance le container
docker-compose up -d

# Vérifier les logs
docker-compose logs -f
```

### Rebuild complet

```bash
# Rebuild sans cache (force le re-téléchargement)
docker-compose build --no-cache

# Rebuild et relance
docker-compose up -d --build
```

### Vérification du dataset

```bash
# Entrer dans le container
docker exec -it bloodcell-classification bash

# Vérifier la structure
ls -la /app/src/data/raw/
```

## Structure des données

```
src/data/
├── raw/                          # Dataset brut téléchargé
│   └── blood-cells-image-dataset/
└── processed/                    # Données transformées (générées à l'entraînement)
```

## Sécurité

⚠️ **IMPORTANT** : Le fichier `kaggle.json` contient des credentials sensibles.

**Bonnes pratiques :**
- Ne jamais commiter `kaggle.json` dans Git
- Ajouter `kaggle.json` au `.gitignore`
- Utiliser des secrets Docker en production
- Limiter les permissions du fichier (`chmod 600`)

## Troubleshooting

### Erreur "401 Unauthorized"

```bash
# Vérifier que kaggle.json existe et est valide
cat kaggle.json

# Vérifier les permissions
ls -la kaggle.json

# Rebuild avec les nouveaux credentials
docker-compose build --no-cache
```

### Dataset non téléchargé

```bash
# Vérifier les logs du build
docker-compose build 2>&1 | grep -A 10 "load_dataset"

# Télécharger manuellement dans le container
docker exec -it bloodcell-classification bash
cd /app
bash scripts/load_dataset.sh
```

### Espace disque insuffisant

Le dataset fait environ 300MB compressé. Vérifier l'espace disponible :

```bash
# Espace disque
df -h

# Nettoyer les images Docker inutilisées
docker system prune -a
```

## Notes

- Le dataset est téléchargé **une seule fois** lors du build
- Les volumes Docker permettent la persistance entre les redémarrages
- Le script `load_dataset.sh` peut être exécuté manuellement si nécessaire
- Le téléchargement prend environ 2-5 minutes selon la connexion
