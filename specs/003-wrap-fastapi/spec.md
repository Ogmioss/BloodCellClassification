# Feature Specification: Wrap FastAPI - MLOps Platform

**Feature Branch**: `003-wrap-fastapi`
**Created**: 2026-03-08
**Status**: Draft
**Input**: User description: "Refactoring du projet BloodCellClassification pour wrapper l'application avec FastAPI : restructuration de l'API, ajout de routers, métriques Prometheus, service de validation de données, batch inference pipeline, intégration Airflow, monitoring, et Docker Compose multi-profils."

## Clarifications

### Session 2026-03-08

- Q: Quel périmètre exact du refactoring ? → A: Ajouter une couche API REST (FastAPI) autour du projet existant de classification de cellules sanguines, avec orchestration ML (Airflow), tracking d'expériences (MLflow), stockage d'artefacts (MinIO S3), monitoring (Prometheus/Grafana), et déploiement conteneurisé (Docker Compose multi-profils)
- Q: Quelle stratégie de chargement de modèle ? → A: MLflow en priorité (model registry), fallback sur checkpoint local si MLflow indisponible
- Q: Quelle architecture pour les tâches ML longues ? → A: Background tasks FastAPI avec task store persistant (JSON), polling par le client ou par Airflow
- Q: Comment gérer les tâches ML concurrentes ? → A: **Une tâche GPU à la fois** — rejeter avec erreur 409 si un entraînement ou évaluation est déjà en cours. Les tâches CPU-only (batch inference, validation de données) restent parallèles
- Q: Quelle stratégie de stockage des données ? → A: **MinIO S3 unique** — tous les artefacts (modèles, métriques, datasets) passent par MinIO S3. Pas de fallback sur stockage local. MinIO est un prérequis obligatoire
- Q: Quelle stratégie de promotion de modèle ? → A: **Semi-automatique** — le DAG promeut en Staging si accuracy >= 90%. La promotion Staging → Production nécessite un appel API manuel explicite (POST /mlflow/promote)
- Q: Que contient le profil Docker Compose "light" ? → A: **API + Streamlit + MinIO + MLflow** — le profil light inclut les 4 services essentiels pour le développement et les démos, sans Airflow, Prometheus, Grafana ni PostgreSQL

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Prédire la classe d'une cellule sanguine via l'API (Priority: P1)

En tant qu'utilisateur ou système externe, je veux soumettre une image de cellule sanguine à l'API et recevoir une prédiction (classe + score de confiance), pour intégrer la classification dans un workflow automatisé.

**Why this priority**: C'est la fonctionnalité centrale — sans prédiction, aucun autre composant n'a de raison d'être. Le endpoint de prédiction est le point d'entrée principal de la valeur métier.

**Independent Test**: Envoyer une image via POST /predict/upload et vérifier que la réponse contient la classe prédite parmi les 8 classes connues, avec un score de confiance entre 0 et 1.

**Acceptance Scenarios**:

1. **Given** l'API est démarrée et le modèle est chargé, **When** j'envoie une image en base64 via POST /predict, **Then** je reçois une réponse JSON contenant la classe prédite, le score de confiance, et les probabilités par classe
2. **Given** l'API est démarrée, **When** j'upload un fichier image via POST /predict/upload, **Then** je reçois la même réponse de prédiction qu'avec l'endpoint base64
3. **Given** l'API est démarrée, **When** j'envoie un fichier non-image ou corrompu, **Then** je reçois une erreur 400 avec un message explicite
4. **Given** l'API tente de charger le modèle, **When** MLflow est indisponible, **Then** le modèle est chargé depuis le checkpoint local et la prédiction fonctionne normalement

---

### User Story 2 - Lancer et suivre un entraînement via l'API (Priority: P1)

En tant que data scientist, je veux déclencher un entraînement de modèle via l'API et suivre sa progression, pour automatiser le cycle d'entraînement sans accéder directement au serveur.

**Why this priority**: L'entraînement est la seconde fonction critique — il permet d'améliorer le modèle. Sans API d'entraînement, il faut un accès SSH au serveur.

**Independent Test**: Déclencher un entraînement via POST /ml/train, récupérer le task_id, et poller GET /ml/tasks/{task_id} jusqu'à la complétion avec métriques de résultat.

**Acceptance Scenarios**:

1. **Given** l'API est démarrée, **When** je POST /ml/train avec des paramètres optionnels (epochs, learning_rate), **Then** je reçois un task_id et le statut "pending"
2. **Given** un entraînement est en cours, **When** je GET /ml/tasks/{task_id}, **Then** je vois le statut courant (pending, running, completed, failed) et les métriques intermédiaires
3. **Given** un entraînement est terminé avec succès, **When** je consulte le résultat, **Then** je vois l'accuracy, le loss final, le run_id MLflow, et la version du modèle enregistré
4. **Given** un entraînement échoue, **When** je consulte le résultat, **Then** je vois le statut "failed" avec le message d'erreur détaillé

---

### User Story 3 - Orchestrer les pipelines ML via Airflow (Priority: P2)

En tant que ML engineer, je veux orchestrer les pipelines ML (entraînement, évaluation, batch inference, validation de données) via Airflow, pour planifier et automatiser les workflows ML avec gestion des dépendances et retry.

**Why this priority**: L'orchestration est essentielle pour la productionisation mais pas bloquante pour le MVP. Les pipelines peuvent être déclenchés manuellement via l'API en attendant.

**Independent Test**: Déclencher un DAG d'entraînement via le endpoint /pipelines/train, et vérifier que les étapes s'exécutent dans l'ordre (health check → entraînement → évaluation → promotion conditionnelle).

**Acceptance Scenarios**:

1. **Given** Airflow et l'API sont démarrés, **When** je POST /pipelines/train, **Then** un DAG run est créé et je reçois le dag_run_id
2. **Given** un DAG d'entraînement est en cours, **When** je GET /pipelines/status/{dag_id}/{dag_run_id}, **Then** je vois l'état de chaque étape du pipeline
3. **Given** un entraînement DAG est terminé avec accuracy >= 90%, **When** l'étape de promotion s'exécute, **Then** le modèle est promu en **Staging** dans le registre MLflow. La promotion Staging → Production nécessite un appel API manuel (POST /mlflow/promote)
4. **Given** Airflow est indisponible, **When** je POST /pipelines/train, **Then** je reçois une erreur explicite sans impact sur les autres fonctionnalités de l'API

---

### User Story 4 - Monitorer la santé et les performances du système (Priority: P2)

En tant qu'opérateur, je veux visualiser les métriques de santé du système (latence API, compteurs de prédictions, statut des services), pour détecter les anomalies et garantir la qualité de service.

**Why this priority**: Le monitoring est critique pour la production mais le système fonctionne sans. Il devient indispensable dès que le système est déployé en continu.

**Independent Test**: Vérifier que les métriques Prometheus sont exposées par l'API et que Grafana affiche les dashboards de prédiction.

**Acceptance Scenarios**:

1. **Given** l'API est démarrée, **When** j'accède à GET /metrics (Prometheus), **Then** je vois les compteurs de prédictions, l'histogramme de latence, et les métriques de confiance
2. **Given** Prometheus scrape l'API, **When** j'accède au dashboard Grafana, **Then** je visualise les métriques en temps réel avec des graphes pertinents
3. **Given** l'API est démarrée, **When** je GET /health, **Then** je reçois le statut du modèle (chargé/non chargé), le device (CPU/GPU), la source du modèle (MLflow/checkpoint), et la version

---

### User Story 5 - Valider les données avant entraînement (Priority: P3)

En tant que data scientist, je veux valider la qualité du dataset (distribution des classes, intégrité des images, détection de drift) avant de lancer un entraînement, pour éviter de gaspiller du compute sur des données corrompues.

**Why this priority**: La validation de données est une bonne pratique MLOps mais pas bloquante — un entraînement sur données corrompues échouera ou produira un mauvais modèle, mais ne cassera pas le système.

**Independent Test**: Lancer POST /ml/validate-data et vérifier que le rapport contient la distribution des classes, les éventuelles images corrompues, et les warnings de déséquilibre.

**Acceptance Scenarios**:

1. **Given** le dataset existe, **When** je POST /ml/validate-data, **Then** je reçois un rapport avec la distribution des classes, le nombre d'images par classe, et les warnings
2. **Given** des images sont corrompues dans le dataset, **When** la validation s'exécute, **Then** le rapport liste les fichiers corrompus avec leur chemin
3. **Given** une classe a significativement moins d'images que les autres, **When** la validation détecte le déséquilibre, **Then** un warning est inclus dans le rapport

---

### User Story 6 - Exécuter une inférence en batch (Priority: P3)

En tant que data scientist, je veux lancer une inférence sur un répertoire entier d'images et obtenir un fichier de résultats, pour classifier de grands volumes d'images sans appels unitaires.

**Why this priority**: Fonctionnalité de confort pour les analyses en masse. Les prédictions unitaires via /predict couvrent le besoin de base.

**Independent Test**: Placer des images dans un répertoire, lancer POST /ml/batch-inference, et vérifier que le fichier JSON de sortie contient une prédiction par image.

**Acceptance Scenarios**:

1. **Given** un répertoire contient des images, **When** je POST /ml/batch-inference avec le chemin du répertoire, **Then** je reçois un task_id pour suivre l'avancement
2. **Given** le batch inference est terminé, **When** je consulte le résultat, **Then** je vois le nombre total d'images traitées, la distribution des classes prédites, et le chemin du fichier de résultats

---

### User Story 7 - Déployer l'ensemble du système via Docker Compose (Priority: P2)

En tant que développeur ou opérateur, je veux déployer l'ensemble de la plateforme (API, Airflow, MLflow, MinIO, Prometheus, Grafana, Streamlit) via une seule commande Docker Compose, avec des profils pour différents niveaux de déploiement.

**Why this priority**: Le déploiement conteneurisé est essentiel pour la reproductibilité et le passage en production. Nécessaire dès que l'on sort du développement local.

**Independent Test**: Lancer docker compose up et vérifier que tous les services sont accessibles à leurs URLs respectives et que le health check de l'API répond.

**Acceptance Scenarios**:

1. **Given** Docker est installé, **When** je lance docker compose up, **Then** tous les services démarrent et sont accessibles (API, Airflow, MLflow, MinIO, Prometheus, Grafana, Streamlit)
2. **Given** je veux un déploiement léger, **When** j'utilise le profil "light", **Then** seuls l'API, le Streamlit, MinIO et MLflow démarrent (sans Airflow, Prometheus, Grafana ni PostgreSQL)
3. **Given** tous les services sont démarrés, **When** l'API est saine, **Then** Airflow peut déclencher des entraînements, MLflow enregistre les expériences, Prometheus collecte les métriques, et MinIO stocke les artefacts

---

### Edge Cases

- Que se passe-t-il si MLflow est indisponible au démarrage de l'API ? → Le modèle est chargé depuis le checkpoint local. Un log d'avertissement est émis. Les prédictions fonctionnent normalement.
- Que se passe-t-il si deux entraînements sont lancés simultanément ? → Le second est rejeté avec une erreur 409 (Conflict). Seule une tâche GPU (entraînement ou évaluation) peut s'exécuter à la fois. Les tâches CPU-only (batch inference, validation) restent parallèles.
- Que se passe-t-il si MinIO est indisponible ? → Les tâches ML nécessitant le stockage d'artefacts échouent avec une erreur explicite. MinIO est un prérequis obligatoire, pas un composant optionnel.
- Que se passe-t-il si le dataset n'existe pas au chemin configuré ? → Les endpoints de training/evaluation/validation retournent une erreur explicite avec le chemin attendu.
- Que se passe-t-il si l'API reçoit une image trop grande ? → L'image est redimensionnée à 224x224 par le pipeline de transformation. Pas de limite de taille côté upload.
- Que se passe-t-il si l'API redémarre pendant un entraînement en background ? → La tâche est perdue. Le task store persiste l'état "running" mais le processus n'est plus actif.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Le système DOIT exposer un endpoint REST de prédiction acceptant une image (base64 ou upload) et retournant la classe prédite parmi 8 classes de cellules sanguines, avec le score de confiance et les probabilités par classe
- **FR-002**: Le système DOIT charger le modèle de classification depuis le registre MLflow (stocké dans MinIO S3) en priorité, avec fallback automatique sur un checkpoint local uniquement pour le chargement du modèle en mémoire (pas pour le stockage d'artefacts)
- **FR-003**: Le système DOIT exposer des endpoints pour déclencher des tâches ML en arrière-plan (entraînement, évaluation, batch inference, validation de données) et retourner un identifiant de tâche pour le suivi. Une seule tâche GPU (entraînement, évaluation) peut s'exécuter à la fois ; les tâches CPU-only restent parallèles. Une tentative de lancer une seconde tâche GPU DOIT retourner une erreur 409
- **FR-004**: Le système DOIT permettre de consulter le statut et les résultats des tâches ML en cours ou terminées via leur identifiant
- **FR-005**: Le système DOIT persister l'état des tâches ML sur disque pour survivre aux redémarrages de l'API
- **FR-006**: Le système DOIT enregistrer automatiquement les résultats d'entraînement dans MLflow (paramètres, métriques, artefacts, modèle)
- **FR-007**: Le système DOIT exposer des endpoints pour déclencher les DAGs Airflow (entraînement, évaluation, batch inference, validation de données) et consulter leur statut
- **FR-008**: Les DAGs Airflow DOIVENT orchestrer les pipelines ML en appelant l'API FastAPI, sans embarquer de dépendances ML (PyTorch) dans le conteneur Airflow
- **FR-009**: Le système DOIT exposer des métriques Prometheus (compteurs de prédictions par classe, histogramme de latence, histogramme de confiance, compteurs d'erreurs, informations du modèle)
- **FR-010**: Le système DOIT exposer un endpoint de santé retournant le statut du modèle, le device, la source et la version du modèle
- **FR-011**: Le système DOIT valider la qualité du dataset (distribution des classes, intégrité des images, détection de déséquilibre) et produire un rapport structuré
- **FR-012**: Le système DOIT exécuter une inférence en batch sur un répertoire d'images et produire un fichier JSON de résultats avec statistiques de distribution
- **FR-013**: Le système DOIT être déployable via Docker Compose avec tous les services nécessaires (API, Airflow, MLflow, MinIO, Prometheus, Grafana, Streamlit, PostgreSQL)
- **FR-014**: Le système DOIT supporter des profils Docker Compose pour différents niveaux de déploiement : full (tous les services), light (API + Streamlit + MinIO + MLflow), airflow-only, monitoring
- **FR-015**: Le système DOIT stocker tous les artefacts ML (modèles, métriques, datasets) dans MinIO S3. MinIO est un prérequis obligatoire ; aucun fallback sur stockage local n'est supporté
- **FR-016**: L'API DOIT supporter le chargement différé des dépendances ML lourdes (PyTorch) pour un démarrage rapide et une faible consommation mémoire au repos

### Key Entities

- **Prediction**: Résultat d'une classification d'image (classe prédite, score de confiance, probabilités par classe, timestamp)
- **MLTask**: Tâche ML en arrière-plan (identifiant, type, statut, paramètres, résultat, timestamps début/fin). Persistée dans un task store JSON
- **Experiment**: Run MLflow (run_id, paramètres, métriques, artefacts, version du modèle). Persisté dans MLflow avec artefacts dans MinIO S3
- **ValidationReport**: Résultat d'une validation de dataset (distribution des classes, images corrompues, warnings, statistiques)
- **Model**: Modèle de classification (architecture, version, source MLflow/checkpoint, métriques de performance)

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Un utilisateur peut obtenir une prédiction de cellule sanguine en moins de 2 secondes après soumission de l'image via l'API
- **SC-002**: Le système charge le modèle et répond au premier appel de prédiction en moins de 30 secondes après démarrage
- **SC-003**: L'ensemble de la plateforme (8 services) démarre et est fonctionnel en moins de 5 minutes via docker compose up
- **SC-004**: 100% des entraînements déclenchés via l'API sont trackés dans MLflow avec paramètres, métriques et artefacts
- **SC-005**: Le modèle promu en Production maintient une accuracy >= 90% sur le jeu de test
- **SC-006**: Les métriques Prometheus sont accessibles et visualisables dans Grafana dans les 30 secondes suivant une prédiction
- **SC-007**: Les DAGs Airflow orchestrent les pipelines ML sans embarquer de dépendances ML dans le conteneur Airflow (taille image < 500 Mo)

## Assumptions

- Le projet cible Linux comme plateforme de déploiement
- Le modèle de base est un ResNet18 pré-entraîné sur ImageNet, fine-tuné pour 8 classes de cellules sanguines
- Le dataset est structuré en répertoires par classe (basophil, eosinophil, erythroblast, immature_granulocyte, lymphocyte, monocyte, neutrophil, platelet)
- MLflow utilise MinIO S3 comme artifact store. MinIO est un prérequis obligatoire (pas de fallback local)
- Les seuils de promotion de modèle : accuracy >= 90% sur le jeu de test → promotion automatique en Staging. Promotion Staging → Production manuelle via API
- L'API tourne en local ou sur un serveur dédié, pas d'authentification requise pour le MVP
- Airflow utilise PostgreSQL comme base de métadonnées
- La configuration centralisée est dans conf.yaml (hyperparamètres, chemins, paramètres MLflow)
- Le frontend Streamlit existant est conservé et enrichi avec les liens vers les services MLOps
- Python 3.11+ géré avec uv, PyTorch CPU-only pour l'image Docker (GPU optionnel en local)
