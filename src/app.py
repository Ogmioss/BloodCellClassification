import os
import streamlit as st

from src.utils.streamlit_style import apply_global_style
from src.utils.streamlit_sidebar import render_mlops_sidebar
from src.core.config import get_service_url

# Fix for "could not create a primitive" error in PyTorch 2.9.0+cpu
# Applied via env var instead of importing torch in Streamlit
os.environ.setdefault("MKLDNN_VERBOSE", "0")
os.environ.setdefault("DNNL_DEFAULT_FPMATH_MODE", "strict")

# Configuration de la page
st.set_page_config(page_title="Blood Cells Classification", layout="wide")
apply_global_style()
render_mlops_sidebar()

# Sidebar
st.sidebar.title("Navigation")
st.sidebar.markdown("Utilise ce menu pour naviguer entre les pages.")

# ── Hero section ──────────────────────────────────────────────────────
st.markdown(
    '<div class="hero-section">'
    "<h1>Blood Cells Classification</h1>"
    "<p>Classification automatique des cellules sanguines par deep learning</p>"
    "</div>",
    unsafe_allow_html=True,
)

# ── KPIs systeme ─────────────────────────────────────────────────────
api_url = get_service_url("api")
_health = None
try:
    import httpx

    resp = httpx.get(f"{api_url}/health", timeout=3.0)
    if resp.status_code == 200:
        _health = resp.json()
except Exception:
    pass

if _health:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("API", "Connectee")
    model_name = _health.get("model_name", "?")
    c2.metric("Modele", model_name.upper())
    device = _health.get("device", "cpu")
    c3.metric("Device", "GPU" if "cuda" in str(device) else "CPU")
    loaded = _health.get("model_loaded", False)
    c4.metric("Etat modele", "Charge" if loaded else "Non charge")
else:
    st.info(
        "API non connectee — les metriques seront disponibles au lancement de l'API."
    )

# ── Image + intro ─────────────────────────────────────────────────────
col_img, col_intro = st.columns([1, 2])

with col_img:
    st.image("src/assets/img_sang.jpg", use_container_width=True)

with col_intro:
    st.markdown(
        '<div class="info-card">'
        '<h3 style="margin-top:0; color: #2c3e50;">A propos du projet</h3>'
        '<p style="color: #555; line-height: 1.7; margin-bottom: 0;">'
        "Ce tableau de bord presente un projet de <strong>classification des cellules "
        "sanguines</strong> a partir d'images de frottis microscopiques. "
        "Il couvre l'exploration du dataset (<strong>17 092 images, 8 classes</strong>), "
        "l'entrainement d'un modele <strong>ResNet18</strong>, "
        "une demo de prediction en temps reel, et l'analyse d'interpretabilite via Grad-CAM."
        "</p></div>",
        unsafe_allow_html=True,
    )

st.markdown("")

# ── Navigation cards ──────────────────────────────────────────────────
_NAV_CARDS = [
    (
        "pages/1_Presentation_du_projet.py",
        "Presentation",
        "Objectif, contexte et donnees",
        "#2ecc71",
    ),
    (
        "pages/2_Exploration_du_dataset.py",
        "Exploration",
        "Statistiques et analyse RGB",
        "#3498db",
    ),
    ("pages/3_Modele.py", "Modele", "Architecture et evaluation", "#9b59b6"),
    ("pages/4_Demo.py", "Demo", "Prediction en temps reel", "#e67e22"),
    ("pages/5_Interpretabilite.py", "Interpretabilite", "Analyse Grad-CAM", "#1abc9c"),
    ("pages/6_Conclusion.py", "Conclusion", "Bilan et perspectives", "#e74c3c"),
]

cols = st.columns(3)
for i, (page_path, title, description, color) in enumerate(_NAV_CARDS):
    with cols[i % 3]:
        st.markdown(
            f'<div class="nav-card" style="border-left: 5px solid {color};">'
            f'<strong style="font-size:1.05rem;">{title}</strong>'
            f'<br><span style="color:#7f8c8d; font-size:0.88rem;">{description}</span>'
            f"</div>",
            unsafe_allow_html=True,
        )
        st.page_link(page_path, label="Ouvrir")

st.divider()
