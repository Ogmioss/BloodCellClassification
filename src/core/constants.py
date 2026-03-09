"""
Constantes partagées pour le projet BloodCellClassification.

Ce module centralise toutes les constantes utilisées à travers le projet
pour éviter la duplication et garantir la cohérence.
"""

# Classes de cellules sanguines (ordre alphabétique)
CLASS_NAMES: list[str] = [
    "basophil",
    "eosinophil",
    "erythroblast",
    "immature_granulocyte",
    "lymphocyte",
    "monocyte",
    "neutrophil",
    "platelet",
]

NUM_CLASSES: int = len(CLASS_NAMES)
