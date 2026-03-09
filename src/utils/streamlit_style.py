"""Style global pour l'application Streamlit."""

import streamlit as st

_CSS = """
<style>
/* ── Métriques : effet card avec bordure gauche ─────────────── */
[data-testid="stMetric"] {
    background: #ffffff;
    border-left: 4px solid #c0392b;
    border-radius: 8px;
    padding: 12px 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
}
[data-testid="stMetricLabel"] {
    font-size: 0.85rem;
    color: #7f8c8d;
}
[data-testid="stMetricValue"] {
    font-size: 1.6rem;
    font-weight: 700;
}

/* ── Tabs : style plus affirmé ──────────────────────────────── */
button[data-baseweb="tab"] {
    font-size: 0.95rem;
    font-weight: 600;
    padding: 10px 20px;
}

/* ── Plotly : supprimer marges excessives ────────────────────── */
[data-testid="stPlotlyChart"] {
    border-radius: 8px;
}

/* ── DataFrames : headers + bordures ────────────────────────── */
[data-testid="stDataFrame"] {
    border-radius: 8px;
    overflow: hidden;
}

/* ── Sidebar : style ────────────────────────────────────────── */
section[data-testid="stSidebar"] > div {
    padding-top: 1.5rem;
}
section[data-testid="stSidebar"] a {
    color: #2c3e50;
    text-decoration: none;
    font-size: 0.9rem;
}
section[data-testid="stSidebar"] a:hover {
    color: #c0392b;
    text-decoration: underline;
}

/* ── Expanders : style discret ──────────────────────────────── */
details[data-testid="stExpander"] {
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    background: #ffffff;
}

/* ── Containers généraux ────────────────────────────────────── */
.block-container {
    padding-top: 2rem;
    max-width: 1200px;
}

/* ── Hero section ──────────────────────────────────────────── */
.hero-section {
    background: linear-gradient(135deg, #c0392b 0%, #e74c3c 100%);
    padding: 2.5rem 2rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    text-align: center;
}
.hero-section h1 { color: white; margin: 0; font-size: 2.4rem; }
.hero-section p { color: rgba(255,255,255,0.85); font-size: 1.1rem; margin: 0.5rem 0 0 0; }

/* ── Cards generiques ──────────────────────────────────────── */
.info-card {
    border: 1px solid #e0e0e0;
    border-radius: 10px;
    padding: 1.5rem;
    background: #ffffff;
    box-shadow: 0 2px 6px rgba(0,0,0,0.06);
}
.nav-card {
    border: 1px solid #e0e0e0;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    background: #ffffff;
    box-shadow: 0 2px 6px rgba(0,0,0,0.06);
    min-height: 90px;
    margin-bottom: 6px;
}
.conclusion-card {
    border-radius: 8px;
    padding: 8px 16px;
    background: #ffffff;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
}
</style>
"""


def apply_global_style() -> None:
    """Injecte le CSS global dans la page Streamlit (appeler une seule fois dans app.py)."""
    st.markdown(_CSS, unsafe_allow_html=True)
