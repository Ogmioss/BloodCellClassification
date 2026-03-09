"""Model page — display model info, metrics and trigger training via FastAPI."""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go
import httpx

from src.core.config import get_service_url
from src.utils.streamlit_style import apply_global_style
from src.utils.streamlit_sidebar import render_mlops_sidebar

FASTAPI_URL = get_service_url("api")
MLFLOW_URL = get_service_url("mlflow")

st.set_page_config(page_title="Modele", layout="wide")
apply_global_style()
render_mlops_sidebar()

st.title("Modele de classification")


# ── Helpers ───────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def _api_get_cached(path: str, timeout: float = 10.0) -> dict | None:
    """GET helper with cache for stable endpoints."""
    try:
        resp = httpx.get(f"{FASTAPI_URL}{path}", timeout=timeout)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def _api_get(path: str, timeout: float = 10.0) -> dict | None:
    """GET helper returning JSON or None on failure."""
    try:
        resp = httpx.get(f"{FASTAPI_URL}{path}", timeout=timeout)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


# ── API health check ──────────────────────────────────────────────────
health = _api_get("/health")
api_available = health is not None

if not api_available:
    st.warning(
        "API non disponible. Les sections interactives (entrainement, metriques) "
        "sont desactivees. Lancez `uv run start-api` pour les activer."
    )


# ── Section 1: Model info ────────────────────────────────────────────
st.header("1. Architecture du modele")

if api_available:
    model_info = _api_get_cached("/model/info")
    if model_info:
        model_name = model_info.get("model_name", "resnet18")
        class_names = model_info.get("class_names", [])
        norm = model_info.get("normalization", {})

        st.markdown(f"""
### Modele utilise: **{model_name.upper()}**

- **Type:** Transfer Learning (ResNet)
- **Pre-entraine:** {"Oui" if model_info.get("pretrained", True) else "Non"}
- **Poids:** {model_info.get("pretrained_weights", "IMAGENET1K_V1")}
- **Nombre de classes:** {model_info.get("num_classes", len(class_names))}
- **Source du modele:** {model_info.get("model_source", "?")}
""")

        st.subheader("Normalisation")
        st.write(f"- **Mean:** {norm.get('mean', [0.485, 0.456, 0.406])}")
        st.write(f"- **Std:** {norm.get('std', [0.229, 0.224, 0.225])}")

        with st.expander("Voir les classes"):
            cols = st.columns(4)
            for i, cn in enumerate(class_names):
                with cols[i % 4]:
                    st.write(f"{i + 1}. {cn}")

        # MLflow info with contextual link
        mlflow_info = model_info.get("mlflow", {})
        if mlflow_info.get("available"):
            mlflow_model_name = mlflow_info.get("model_name", "")
            mlflow_version = mlflow_info.get("latest_version", "")
            tracking_uri = mlflow_info.get("tracking_uri", "")

            st.info(
                f"MLflow: model *{mlflow_model_name}* "
                f"v{mlflow_version} ({tracking_uri})"
            )
            st.markdown(
                f'<a href="{MLFLOW_URL}" target="_blank">'
                f"Voir les experiences dans MLflow</a>",
                unsafe_allow_html=True,
            )
    else:
        st.warning("Impossible de charger les infos du modele.")
else:
    st.info("Connectez l'API pour afficher les informations du modele.")

st.divider()

# ── Section 2: Training trigger ───────────────────────────────────────
st.header("2. Re-entrainement du modele")

if api_available:
    st.markdown("""
Cliquez sur le bouton ci-dessous pour lancer l'entrainement.
Le modele sera sauvegarde dans le repertoire `checkpoints`.
""")

    if "training_task_id" not in st.session_state:
        st.session_state.training_task_id = None

    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button(
            "Lancer l'entrainement",
            type="primary",
            disabled=st.session_state.training_task_id is not None,
        ):
            try:
                resp = httpx.post(
                    f"{FASTAPI_URL}/ml/train",
                    json={},
                    timeout=10.0,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    st.session_state.training_task_id = data["task_id"]
                    st.rerun()
                else:
                    st.error(f"Erreur ({resp.status_code}): {resp.text}")
            except Exception as e:
                st.error(f"Erreur: {e}")

    with col2:
        if st.session_state.training_task_id:
            task = _api_get(f"/ml/tasks/{st.session_state.training_task_id}")
            if task:
                status = task.get("status", "unknown")
                if status in ("pending", "running"):
                    st.info(f"Entrainement en cours (status: {status})...")
                elif status == "completed":
                    st.success("Entrainement termine!")
                    if task.get("result"):
                        st.json(task["result"])
                    st.session_state.training_task_id = None
                elif status == "failed":
                    st.error(f"Echec: {task.get('error', '?')}")
                    st.session_state.training_task_id = None
else:
    st.info("Connectez l'API pour lancer un entrainement.")

st.divider()

# ── Section 3: Evaluation metrics ─────────────────────────────────────
st.header("3. Evaluation")

if api_available:
    metrics = _api_get("/metrics")

    if metrics:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Test Accuracy", f"{metrics.get('accuracy', 0):.2%}")
        with col2:
            st.metric("Best Val Accuracy", f"{metrics.get('best_val_acc', 0):.2%}")
        with col3:
            st.metric(
                "Final Train Accuracy", f"{metrics.get('final_train_acc', 0):.2%}"
            )

        # Confusion matrix
        cm_data = metrics.get("confusion_matrix")
        cm_classes = metrics.get("class_names", [])
        if cm_data and cm_classes:
            st.divider()
            st.subheader("Matrice de confusion")

            confusion_mat = np.array(cm_data)
            fig = ff.create_annotated_heatmap(
                z=confusion_mat,
                x=cm_classes,
                y=cm_classes,
                colorscale="Blues",
                showscale=True,
                annotation_text=confusion_mat.astype(str),
            )
            fig.update_layout(
                title="Matrice de confusion (Test Set)",
                xaxis_title="Predictions",
                yaxis_title="Vraies etiquettes",
                height=600,
                xaxis={"side": "bottom"},
                yaxis={"autorange": "reversed"},
            )
            for annotation in fig.layout.annotations:
                annotation.font.size = 10
            st.plotly_chart(fig, use_container_width=True)

            # Per-class accuracy — table + bar chart
            st.subheader("Exactitude par classe")
            rows = []
            for i, cn in enumerate(cm_classes):
                total = int(confusion_mat[i].sum())
                correct = int(confusion_mat[i, i])
                acc = (correct / total * 100) if total > 0 else 0
                rows.append(
                    {
                        "Classe": cn,
                        "Correct": correct,
                        "Total": total,
                        "Exactitude (%)": round(acc, 1),
                    }
                )

            acc_df = pd.DataFrame(rows)
            col_table, col_chart = st.columns([1, 2])
            with col_table:
                st.dataframe(acc_df, use_container_width=True, hide_index=True)
            with col_chart:
                sorted_acc = acc_df.sort_values("Exactitude (%)", ascending=False)
                colors = [
                    "#2ecc71" if v >= 90 else ("#f39c12" if v >= 70 else "#e74c3c")
                    for v in sorted_acc["Exactitude (%)"]
                ]
                fig_acc = go.Figure(
                    go.Bar(
                        x=sorted_acc["Classe"],
                        y=sorted_acc["Exactitude (%)"],
                        marker_color=colors,
                        text=[f"{v:.1f}%" for v in sorted_acc["Exactitude (%)"]],
                        textposition="outside",
                    )
                )
                fig_acc.update_layout(
                    title="Exactitude par classe",
                    yaxis_title="%",
                    height=400,
                    yaxis=dict(range=[0, 105]),
                )
                st.plotly_chart(fig_acc, use_container_width=True)

        with st.expander("Details complets des metriques", expanded=False):
            display = {k: v for k, v in metrics.items() if k != "confusion_matrix"}
            st.json(display)
    else:
        st.warning("Aucune metrique disponible. Entrainez un modele d'abord.")
        st.info("Lancez l'entrainement ci-dessus ou utilisez `uv run train-model`.")
else:
    st.info("Connectez l'API pour afficher les metriques d'evaluation.")

st.divider()
st.page_link("app.py", label="Retour a l'accueil")
