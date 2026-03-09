"""Sidebar MLOps globale pour l'application Streamlit.

Affiche les liens vers les outils externes (MLflow, Airflow, Grafana, API Docs)
avec des indicateurs de sante (health checks) et l'etat du systeme.

Usage dans chaque page :
    from src.utils.streamlit_sidebar import render_mlops_sidebar
    render_mlops_sidebar()
"""

import streamlit as st
import httpx

from src.core.config import get_all_external_urls, get_all_service_urls


@st.cache_data(ttl=30)
def _check_service_health(url: str, path: str = "/health") -> bool:
    """Verifie si un service est accessible."""
    try:
        resp = httpx.get(f"{url}{path}", timeout=3.0)
        return resp.status_code < 500
    except Exception:
        return False


def _status_icon(healthy: bool) -> str:
    return "\U0001f7e2" if healthy else "\U0001f534"


def render_mlops_sidebar() -> None:
    """Affiche la sidebar MLOps avec liens externes et etat du systeme."""
    # URLs internes pour les health checks (reseau Docker)
    urls = get_all_service_urls()
    # URLs externes pour les liens navigateur (localhost)
    ext = get_all_external_urls()

    health_api = _check_service_health(urls["api"], "/health")
    health_mlflow = _check_service_health(urls["mlflow"], "/health")
    health_airflow = _check_service_health(urls["airflow"], "/health")
    health_grafana = _check_service_health(urls["grafana"], "/api/health")

    with st.sidebar:
        st.markdown("---")
        st.markdown("**Outils MLOps**")

        st.markdown(
            f'{_status_icon(health_mlflow)} <a href="{ext["mlflow"]}" target="_blank">MLflow</a>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'{_status_icon(health_airflow)} <a href="{ext["airflow"]}" target="_blank">Airflow</a>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'{_status_icon(health_grafana)} <a href="{ext["grafana"]}" target="_blank">Grafana</a>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'{_status_icon(health_api)} <a href="{ext["api"]}/docs" target="_blank">API Docs</a>',
            unsafe_allow_html=True,
        )

        # Etat du systeme
        st.markdown("---")
        st.markdown("**Systeme**")

        if health_api:
            try:
                resp = httpx.get(f"{urls['api']}/health", timeout=3.0)
                if resp.status_code == 200:
                    info = resp.json()
                    model_name = info.get("model_name", "?")
                    device = info.get("device", "?")
                    loaded = info.get("model_loaded", False)
                    status = f"{_status_icon(loaded)} {model_name}"
                    device_label = "GPU" if "cuda" in str(device) else "CPU"
                    st.caption(f"Modele : {status}")
                    st.caption(f"Device : {device_label}")
                    return
            except Exception:
                pass

        st.caption(f"API : {_status_icon(False)} Indisponible")
