# Feature Specification: Propagation des parametres d'entrainement de bout en bout

**Feature Branch**: `004-training-config-propagation`
**Created**: 2026-03-12
**Status**: Draft
**Input**: Permettre le choix du modele et du dataset depuis Airflow et l'API, et corriger la propagation des hyperparametres (epochs, lr, batch_size) actuellement silencieusement ignores.

## Contexte et probleme

Le systeme accepte des parametres d'entrainement via l'API et Airflow, mais seul `dataset_path` est reellement propage. Les autres (`epochs`, `learning_rate`, `batch_size`) sont silencieusement ignores — la documentation et le schema API promettent une fonctionnalite qui n'existe pas. De plus, aucun moyen de choisir l'architecture du modele (`model_name`) n'existe.

**Problemes identifies** :
1. Les overrides API (epochs, lr, batch_size) sont acceptes mais jamais appliques
2. Le pipeline d'entrainement ne lit que `conf.yaml`, sans mecanisme d'override
3. Le DAG Airflow ne transmet que `dataset_path` a l'API
4. Aucun parametre `model_name` n'existe dans la chaine

## Analyse critique : Airflow et l'entrainement multi-modeles

### Ce qu'Airflow fait bien ici
- Orchestrer un pipeline lineaire (train → evaluate → promote)
- Declencher des entrainements manuels avec parametres via "Trigger DAG w/ config"
- Gerer les dependances entre taches et la logique de branchement (promotion conditionnelle)
- Fournir une UI de suivi, des logs, et un historique des runs

### Ce qu'Airflow ne fait PAS bien pour le multi-modeles
- **Pas de grid search natif** : Airflow n'est pas un framework d'optimisation d'hyperparametres (ce role revient a Optuna, Ray Tune, etc.)
- **Pas de comparaison automatique** : il ne sait pas comparer N modeles et choisir le meilleur — il faudrait une tache custom ou un outil externe
- **DAG statique** : le nombre de taches est defini a la creation du DAG, pas dynamiquement. Entrainer 5 modeles en parallele necessite soit 5 taches codees en dur, soit `Dynamic Task Mapping` (Airflow 2.3+)
- **Overhead de latence** : chaque tache a un overhead de scheduling (~5-10s). Pour tester 20 combinaisons, l'overhead devient significatif

### Recommandation
Pour le cas d'usage actuel (choisir UN modele et UN dataset par run), Airflow est parfaitement adapte. Le "Trigger DAG w/ config" est exactement le bon mecanisme : l'operateur choisit ses parametres dans le JSON et lance le pipeline.

Pour du multi-modeles compare (grid search, champion/challenger), il faudrait un outil dedie (Optuna, MLflow Experiments comparison) appele depuis une tache Airflow, pas Airflow seul.

**Perimetre de cette spec** : on se limite au cas "1 run = 1 modele + 1 dataset + hyperparametres", ce qui est le bon usage d'Airflow.

---

## User Scenarios & Testing

### User Story 1 - Corriger la propagation des hyperparametres existants (Priority: P1)

Un operateur MLOps lance un entrainement via l'API avec des hyperparametres personnalises (epochs, learning_rate, batch_size). Ces parametres doivent reellement etre appliques au lieu d'etre ignores.

**Why this priority** : C'est un bug — le systeme promet une fonctionnalite qui ne fonctionne pas. La documentation et le schema API sont trompeurs.

**Independent Test** : Lancer `POST /ml/train` avec `{"epochs": 5}`, verifier dans MLflow que le run a bien 5 epoques (pas 20).

**Acceptance Scenarios** :

1. **Given** l'API est demarree, **When** un utilisateur envoie `POST /ml/train` avec `{"epochs": 5, "learning_rate": 0.0005}`, **Then** l'entrainement utilise 5 epoques et un learning rate de 0.0005 (visible dans les metriques MLflow)
2. **Given** l'API est demarree, **When** un utilisateur envoie `POST /ml/train` avec `{}` (body vide), **Then** les valeurs par defaut de `conf.yaml` sont utilisees (epochs: 20, lr: 0.001, batch_size: 32)
3. **Given** l'API est demarree, **When** un utilisateur envoie `POST /ml/train` avec `{"batch_size": 64}` uniquement, **Then** seul batch_size est overriden, les autres gardent les valeurs de conf.yaml

---

### User Story 2 - Choix du modele via l'API (Priority: P2)

Un data scientist veut comparer ResNet18 et ResNet50 sur le meme dataset. Il lance deux entrainements successifs via l'API en specifiant un `model_name` different a chaque fois.

**Why this priority** : Fonctionnalite nouvelle demandee. Depend de P1 (propagation fonctionnelle).

**Independent Test** : Lancer `POST /ml/train` avec `{"model_name": "resnet50"}`, verifier dans MLflow que le run utilise ResNet50.

**Acceptance Scenarios** :

1. **Given** l'API est demarree, **When** un utilisateur envoie `POST /ml/train` avec `{"model_name": "resnet50"}`, **Then** l'entrainement utilise ResNet50 et le parametre est visible dans MLflow
2. **Given** l'API est demarree, **When** un utilisateur envoie `POST /ml/train` avec `{"model_name": "cnn"}`, **Then** l'entrainement utilise le CNN simple
3. **Given** l'API est demarree, **When** un utilisateur envoie `POST /ml/train` avec `{"model_name": "unknown_model"}`, **Then** le systeme renvoie une erreur claire indiquant les modeles supportes
4. **Given** l'API est demarree, **When** un utilisateur envoie `POST /ml/train` sans `model_name`, **Then** le modele par defaut de conf.yaml est utilise (resnet18)

---

### User Story 3 - Parametrage complet depuis Airflow (Priority: P3)

Un operateur MLOps declenche un entrainement depuis l'UI Airflow avec un JSON de configuration incluant le modele, le dataset, et les hyperparametres.

**Why this priority** : Extension naturelle de P1+P2 vers l'interface Airflow. Depend des deux stories precedentes.

**Independent Test** : Trigger le DAG depuis Airflow UI avec `{"model_name": "resnet50", "dataset_path": "raw/bloodcells_dataset", "epochs": 10}`, verifier dans MLflow.

**Acceptance Scenarios** :

1. **Given** Airflow et l'API sont demarres, **When** un operateur trigger le DAG `bloodcells_train_model_api` avec le config JSON `{"model_name": "resnet50", "epochs": 10, "dataset_path": "raw/bloodcells_dataset"}`, **Then** l'entrainement utilise ResNet50 avec 10 epoques sur le dataset specifie
2. **Given** Airflow et l'API sont demarres, **When** un operateur trigger le DAG sans config JSON, **Then** toutes les valeurs par defaut de conf.yaml sont utilisees
3. **Given** Airflow et l'API sont demarres, **When** un operateur trigger le DAG avec `{"learning_rate": 0.0001, "batch_size": 16}`, **Then** seuls lr et batch_size sont overrides

---

### Edge Cases

- Que se passe-t-il si `model_name` est un nom de modele non supporte (ex: "vgg16") ? → Erreur explicite avec liste des modeles valides
- Que se passe-t-il si `epochs` est negatif ou zero ? → Erreur de validation
- Que se passe-t-il si `batch_size` est trop grand pour la memoire ? → Erreur PyTorch a runtime, renvoyee dans le resultat de la tache
- Que se passe-t-il si un override partiel est fourni (ex: seulement `epochs`) ? → Les autres parametres gardent les valeurs conf.yaml
- Que se passe-t-il si conf.yaml est modifie entre deux runs ? → Chaque run charge conf.yaml au demarrage, les overrides s'appliquent par-dessus

## Requirements

### Functional Requirements

- **FR-001** : Le systeme DOIT propager les parametres `epochs`, `learning_rate`, `batch_size` depuis la requete API jusqu'au pipeline d'entrainement effectif
- **FR-002** : Le systeme DOIT accepter un parametre optionnel `model_name` dans la requete d'entrainement
- **FR-003** : Le systeme DOIT valider que `model_name` correspond a un modele supporte avant de lancer l'entrainement
- **FR-004** : Le systeme DOIT utiliser les valeurs de `conf.yaml` comme valeurs par defaut lorsqu'un parametre n'est pas fourni
- **FR-005** : Le systeme DOIT permettre un override partiel (ex: fournir seulement `epochs`, le reste vient de conf.yaml)
- **FR-006** : Le DAG Airflow DOIT lire et transmettre tous les parametres configurables (model_name, epochs, learning_rate, batch_size, dataset_path) depuis le JSON de declenchement
- **FR-007** : Le systeme DOIT enregistrer les parametres effectivement utilises (apres merge defaults + overrides) dans MLflow pour tracabilite
- **FR-008** : Le systeme DOIT renvoyer une erreur claire si un `model_name` non supporte est demande, incluant la liste des modeles valides

### Key Entities

- **TrainRequest** : requete d'entrainement avec parametres optionnels (dataset_path, epochs, learning_rate, batch_size, model_name)
- **Config Override** : mecanisme de merge entre les valeurs par defaut (conf.yaml) et les overrides fournis par l'utilisateur
- **DAG Config** : JSON passe lors du declenchement d'un DAG Airflow ("Trigger DAG w/ config")

## Assumptions

- Les modeles supportes sont ceux deja geres par ModelFactory : resnet18, resnet34, resnet50, cnn
- Aucun nouveau modele n'est ajoute dans cette spec (seule la possibilite de choisir parmi les existants)
- La validation de `model_name` est la responsabilite du pipeline, pas de l'API (l'API transmet, le pipeline valide)
- Le comportement de promotion automatique (accuracy >= 90% → Staging) reste inchange quel que soit le modele choisi

## Success Criteria

### Measurable Outcomes

- **SC-001** : Un utilisateur peut lancer un entrainement avec des hyperparametres personnalises via l'API et verifier dans MLflow que les parametres ont ete appliques — 100% des parametres fournis sont respectes
- **SC-002** : Un utilisateur peut choisir parmi les architectures disponibles (resnet18, resnet34, resnet50, cnn) via un seul parametre dans la requete
- **SC-003** : Un operateur peut configurer entierement un run d'entrainement depuis l'interface Airflow "Trigger DAG w/ config" sans modifier conf.yaml
- **SC-004** : Les requetes sans override continuent de fonctionner exactement comme avant (retro-compatibilite totale)
