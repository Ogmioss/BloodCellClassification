import streamlit as st

# Configuration de la page
st.set_page_config(page_title="🩸 Blood Cells Classification", layout="wide")

# Sidebar avec titre en haut et message d’accueil
st.sidebar.title("💡 Navigation")
st.sidebar.markdown("Hello ! 👋 Utilise ce menu pour naviguer entre les pages et découvrir toutes les sections du projet !")

# Titre principal de la page
st.title("🩸 Blood Cells Classification Dashboard")

# Introduction avec colonnes pour texte et image
st.markdown("---")  # Ligne de séparation
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Sommaire")
    st.markdown("""
    Ce tableau de bord présente un projet de **classification des cellules sanguines** à partir d’images microscopiques.

    Les différentes parties du projet sont les suivantes :
    - 📝 Présentation du projet  
    - 🔍 Exploration du dataset  
    - 🧠 Modèle (architecture & entraînement)  
    - 🎯 Démo interactive  
    - 💡 Améliorations potentielles  
    - ✅ Conclusion
    """)

with col2:
    st.image(
        "src/assets/img_sang.jpg",
        width="stretch"
    )

st.markdown("---")
