"""Demo page — interactive blood cell classification via the FastAPI backend."""

import streamlit as st
import plotly.graph_objects as go
from PIL import Image
import httpx

from src.core.config import get_service_url
from src.utils.streamlit_style import apply_global_style
from src.utils.streamlit_sidebar import render_mlops_sidebar

FASTAPI_URL = get_service_url("api")

st.set_page_config(page_title="Demo", layout="wide")
apply_global_style()
render_mlops_sidebar()

st.title("Demonstration interactive")
st.markdown("Upload une image de frottis sanguin pour obtenir la prediction du modele.")


# Check API health
@st.cache_data(ttl=30)
def check_api_health() -> dict | None:
    try:
        resp = httpx.get(f"{FASTAPI_URL}/health", timeout=5.0)
        if resp.status_code == 200:
            return resp.json()
    except httpx.ConnectError:
        return None
    return None


health = check_api_health()

if health is None:
    st.warning("API non disponible. Lancez `uv run start-api` pour utiliser la demo.")
    st.info(
        "Cette page permet d'uploader une image de frottis sanguin "
        "et d'obtenir une prediction du modele en temps reel."
    )
else:
    st.success(
        f"API connectee — modele: {health.get('model_name', '?')}, device: {health.get('device', '?')}"
    )
    st.divider()

    st.header("Upload et prediction")

    uploaded = st.file_uploader(
        "Choisir une image (.jpg / .png)", type=["jpg", "jpeg", "png"]
    )

    if uploaded:
        img = Image.open(uploaded).convert("RGB")

        # Call /predict/upload
        with st.spinner("Prediction en cours..."):
            try:
                uploaded.seek(0)
                files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
                resp = httpx.post(
                    f"{FASTAPI_URL}/predict/upload",
                    files=files,
                    timeout=30.0,
                )

                if resp.status_code == 200:
                    prediction = resp.json()

                    # Layout 2 colonnes : image a gauche, resultats a droite
                    col_img, col_res = st.columns([1, 2])

                    with col_img:
                        st.image(
                            img, caption="Image uploadee", use_container_width=True
                        )

                    with col_res:
                        st.subheader(
                            f"Classe predite : {prediction['predicted_class']}"
                        )
                        st.metric("Confiance", f"{prediction['confidence']:.2%}")

                        confidence = prediction["confidence"]
                        if confidence >= 0.80:
                            st.success("Prediction fiable (confiance >= 80%)")
                        elif confidence >= 0.50:
                            st.warning("Prediction incertaine (confiance 50-80%)")
                        else:
                            st.error("Prediction peu fiable (confiance < 50%)")

                    st.divider()

                    # Bar chart horizontal des probabilites
                    st.subheader("Probabilites par classe")
                    sorted_probs = sorted(
                        prediction["probabilities"].items(),
                        key=lambda x: x[1],
                    )
                    classes = [item[0] for item in sorted_probs]
                    probs = [item[1] for item in sorted_probs]

                    fig = go.Figure(
                        go.Bar(
                            x=probs,
                            y=classes,
                            orientation="h",
                            marker_color=[
                                "#c0392b"
                                if c == prediction["predicted_class"]
                                else "#bdc3c7"
                                for c in classes
                            ],
                            text=[f"{p:.1%}" for p in probs],
                            textposition="outside",
                        )
                    )
                    fig.update_layout(
                        xaxis_title="Probabilite",
                        height=max(300, len(classes) * 40),
                        margin=dict(l=120, r=40, t=20, b=40),
                        xaxis=dict(range=[0, 1.05], tickformat=".0%"),
                    )
                    st.plotly_chart(fig, use_container_width=True)

                elif resp.status_code == 503:
                    st.warning("Modele non disponible. Entrainez un modele d'abord.")
                else:
                    st.error(f"Erreur API ({resp.status_code}): {resp.text}")
            except httpx.ConnectError:
                st.error("Connexion a l'API perdue.")
            except Exception as e:
                st.error(f"Erreur: {e}")

st.divider()
st.page_link("app.py", label="Retour a l'accueil")
