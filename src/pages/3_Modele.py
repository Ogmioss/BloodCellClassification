import streamlit as st


st.title("🧠 Modèle de classification")


st.markdown("""
Ici, on présente le modèle utilisé (par ex. **CNN** ou **Transfer Learning** via MobileNet, VGG16, etc.).


### Étapes :
1. Pré-traitement des images : redimensionnement, normalisation, augmentation
2. Architecture du modèle
3. Entraînement (dataset Mendeley)
4. Évaluation : accuracy, précision, rappel, F1-score
""")


st.info("💡 Cette section peut inclure des visualisations d'apprentissage, une matrice de confusion et les scores de test.")