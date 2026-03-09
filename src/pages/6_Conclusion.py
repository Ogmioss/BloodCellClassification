"""Page de conclusion du projet."""

import streamlit as st
import httpx

from src.core.config import get_service_url
from src.utils.streamlit_style import apply_global_style
from src.utils.streamlit_sidebar import render_mlops_sidebar

st.set_page_config(page_title="Conclusion", layout="wide")
apply_global_style()
render_mlops_sidebar()

st.title("Conclusion")

# ── KPIs dynamiques (avec fallback) ──────────────────────────────────
api_url = get_service_url("api")
_metrics = None
try:
    resp = httpx.get(f"{api_url}/metrics", timeout=5.0)
    if resp.status_code == 200:
        _metrics = resp.json()
except Exception:
    pass

if _metrics:
    c1, c2, c3 = st.columns(3)
    c1.metric("Test Accuracy", f"{_metrics.get('accuracy', 0):.2%}")
    c2.metric("Best Val Accuracy", f"{_metrics.get('best_val_acc', 0):.2%}")
    c3.metric("Classes", len(_metrics.get("class_names", [])))

st.divider()

columns_config = [
    (
        "Resume",
        "#2ecc71",
        """
- Le modele ResNET de classification atteint une bonne performance sur le dataset Mendeley.
- Il est capable d'identifier la majorite des types de cellules avec une precision satisfaisante.
- Le modele baseline atteint une performance moyenne sur le dataset Mendeley.
    """,
    ),
    (
        "Limites",
        "#e74c3c",
        """
- Le modele ResNET n'a pas ete entraine sur l'ensemble du dataset par manque de puissance de calcul.
- Quelques classes rares (basophils, erythroblasts) restent difficiles a distinguer.
- Necessite d'un dataset plus equilibre et de techniques d'interpretation approfondies.
- Le modele peut avoir tendance a se focaliser sur l'arriere-plan de l'image.
    """,
    ),
    (
        "Perspectives",
        "#3498db",
        """
- Integrer du pre-processing dans le pipeline pour ameliorer la performance.
- Etendre l'application a la detection d'anomalies sur les cellules.
- Explorer des architectures plus avancees (EfficientNet, Vision Transformer).
- Ajouter des techniques d'augmentation de donnees ciblees.
    """,
    ),
]

col1, col2, col3 = st.columns(3)
for col, (title, color, content) in zip([col1, col2, col3], columns_config):
    with col:
        st.markdown(
            f'<div class="conclusion-card" style="border-left: 4px solid {color};">'
            f'<h3 style="color: {color}; margin-top: 0;">{title}</h3></div>',
            unsafe_allow_html=True,
        )
        st.markdown(content)

st.divider()
st.page_link("app.py", label="Retour a l'accueil")
