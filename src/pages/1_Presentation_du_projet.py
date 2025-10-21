# 📘 Présentation du projet
import streamlit as st

st.set_page_config(page_title="Présentation du projet", layout="wide")

st.title("📘 Présentation du projet")

# Objectif
st.header("🎯 Objectif")
st.markdown("""
Le projet vise à **développer un modèle de Computer Vision** capable de reconnaître automatiquement les différents types de cellules sanguines normales à partir d'images de frottis microscopiques.  

Cette identification automatique permet de :  
- Faciliter l’**analyse et le classement des cellules normales** dans un frottis sanguin.  
- Fournir un **outil d’aide à la recherche et au développement de modèles de diagnostic**.  
- Réduire le temps et l’effort nécessaires pour analyser manuellement les images.
""")

# Contexte
st.header("📚 Contexte")
st.markdown("""
La classification des cellules sanguines normales est essentielle pour :  
- Développer des modèles fiables pour l’analyse des frottis.  
- Créer des références pour la **recherche et le benchmarking** en vision par ordinateur médicale.  

Actuellement, l'analyse des frottis est manuelle et longue. Automatiser la reconnaissance des cellules normales permet de standardiser les données et de faciliter la recherche biomédicale.
""")

# Données
st.header("📂 Données")
st.markdown("""
- **Source** : [Mendeley - Blood Cell Images Dataset](https://data.mendeley.com/datasets/snkd93bnjr/1)  
- **Images** : 17 092 images couleur (360×363 px)  
- **Types de cellules normales** : neutrophils, eosinophils, basophils, lymphocytes, monocytes, immature granulocytes (promyelocytes, myelocytes, metamyelocytes), erythroblasts, platelets  
- **Annotations** : réalisées par des pathologistes experts  
- **Particularité** : toutes les images proviennent d’individus sains, sans infection, maladie hématologique ou traitement pharmacologique.  

Ce dataset de haute qualité constitue une **référence pour entraîner et tester des modèles de reconnaissance des cellules sanguines normales**.
""")
