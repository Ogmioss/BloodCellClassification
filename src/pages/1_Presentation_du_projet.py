"""Page de presentation du projet."""

import streamlit as st

from src.utils.streamlit_style import apply_global_style
from src.utils.streamlit_sidebar import render_mlops_sidebar

st.set_page_config(page_title="Presentation du projet", layout="wide")
apply_global_style()
render_mlops_sidebar()

st.title("Presentation du projet")

# ── KPIs en haut ──────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Images", "17 092")
c2.metric("Classes", "8")
c3.metric("Resolution", "360 x 363 px")
c4.metric("Source", "Mendeley")

st.divider()

# Objectif
st.header("Objectif")
st.markdown("""
Le projet vise a **developper un modele de Computer Vision** capable de reconnaitre
automatiquement les differents types de cellules sanguines normales a partir d'images
de frottis microscopiques.
""")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
**Analyse automatisee**

Faciliter l'analyse et le classement des cellules normales dans un frottis sanguin.
""")
with col2:
    st.markdown("""
**Aide a la recherche**

Fournir un outil d'aide a la recherche et au developpement de modeles de diagnostic.
""")
with col3:
    st.markdown("""
**Gain de temps**

Reduire le temps et l'effort necessaires pour analyser manuellement les images.
""")

st.divider()

# Contexte
st.header("Contexte")
st.markdown("""
La classification des cellules sanguines normales est essentielle pour :
- Developper des modeles fiables pour l'analyse des frottis.
- Creer des references pour la **recherche et le benchmarking** en vision par ordinateur medicale.

Actuellement, l'analyse des frottis est manuelle et longue. Automatiser la reconnaissance
des cellules normales permet de standardiser les donnees et de faciliter la recherche biomedicale.
""")

st.divider()

# Donnees
st.header("Donnees")
st.markdown(
    '- **Source** : <a href="https://data.mendeley.com/datasets/snkd93bnjr/1" target="_blank">'
    "Mendeley - Blood Cell Images Dataset</a>\n"
    "- **Images** : 17 092 images couleur (360x363 px)\n"
    "- **Types de cellules normales** : neutrophils, eosinophils, basophils, lymphocytes, monocytes, "
    "immature granulocytes (promyelocytes, myelocytes, metamyelocytes), erythroblasts, platelets\n"
    "- **Annotations** : realisees par des pathologistes experts\n"
    "- **Particularite** : toutes les images proviennent d'individus sains, sans infection, "
    "maladie hematologique ou traitement pharmacologique.\n\n"
    "Ce dataset de haute qualite constitue une **reference pour entrainer et tester des modeles "
    "de reconnaissance des cellules sanguines normales**.",
    unsafe_allow_html=True,
)

st.divider()
st.page_link("app.py", label="Retour a l'accueil")
