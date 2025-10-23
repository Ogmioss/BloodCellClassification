# ============================
# pages/2_Exploration_du_dataset.py (version finale)
# ============================

import streamlit as st
import pandas as pd
from PIL import Image, UnidentifiedImageError
from pathlib import Path
import random
import plotly.express as px
from src.services.yaml_loader import YamlLoader

loader = YamlLoader()

st.title("🔍 Exploration du dataset")

# Chemin du dataset
DATA_DIR = loader.data_raw_dir / "bloodcells_dataset"

if not DATA_DIR.exists():
    st.warning("⚠️ Dossier introuvable. Vérifie le chemin du dataset.")
else:
    st.success("Dataset détecté ✅")

    counts = {}
    widths, heights = [], []
    class_images = {}

    # Parcours des sous-dossiers
    for d in DATA_DIR.iterdir():
        if d.is_dir():
            image_files = [f for f in list(d.glob("*.jpeg")) + list(d.glob("*.jpg")) if not f.name.startswith(".")]
            valid_images = []
            for f in image_files:
                try:
                    img = Image.open(f)
                    widths.append(img.width)
                    heights.append(img.height)
                    valid_images.append(f)
                except UnidentifiedImageError:
                    st.warning(f"Fichier ignoré (non-image ou corrompu) : {f.name}")
            counts[d.name] = len(valid_images)
            class_images[d.name] = valid_images

    if counts:
        df = pd.DataFrame.from_dict(counts, orient='index', columns=['Nombre d\'images'])
        df = df.sort_values(by='Nombre d\'images', ascending=False)

        # Onglets Streamlit
        # tab1, tab2, tab3 = st.tabs(["📊 Statistiques", "📈 Distribution", "🖼️ Exemples d'images"])
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Statistiques",
            "📈 Distribution",
            "🎨 Couleurs RGB",
            "🖼️ Exemples d'images"
        ])
        # ---------------- Statistiques ----------------
        with tab1:
            st.subheader("📋 Statistiques globales")
            total_images = df["Nombre d'images"].sum()
            st.write(f"Nombre total d'images : {total_images}")
            st.write(f"Nombre de classes : {len(df)}")
            st.write(f"Classe la plus représentée : {df.index[0]} ({df.iloc[0,0]} images)")
            st.write(f"Classe la moins représentée : {df.index[-1]} ({df.iloc[-1,0]} images)")
            if widths and heights:
                st.write(f"Taille moyenne des images : {sum(widths)/len(widths):.1f} x {sum(heights)/len(heights):.1f}")
                st.write(f"Taille min : {min(widths)} x {min(heights)}, Taille max : {max(widths)} x {max(heights)}")

            # Histogrammes dimensions
            st.subheader("Histogrammes des dimensions")
            hist_w = px.histogram(widths, nbins=30, labels={'value':'Largeur (px)'}, title="Distribution des largeurs")
            hist_h = px.histogram(heights, nbins=30, labels={'value':'Hauteur (px)'}, title="Distribution des hauteurs")
            st.plotly_chart(hist_w)
            st.plotly_chart(hist_h)

            # Ratio largeur/hauteur
            ratios = [w/h for w,h in zip(widths, heights) if h != 0]
            hist_ratio = px.histogram(ratios, nbins=30, labels={'value':'Ratio L/H'}, title="Distribution du ratio L/H")
            st.plotly_chart(hist_ratio)

        # ---------------- Distribution ----------------
        with tab2:
            st.subheader("Distribution des classes")
            st.bar_chart(df)

            # Pie chart interactif
            fig = px.pie(df, names=df.index, values='Nombre d\'images', title="Répartition en pourcentage des classes")
            st.plotly_chart(fig)

            # Tableau
            st.subheader("Table des classes et nombre d'images")
            st.dataframe(df)
        
                # ---------------- Distributions RGB par classe (version Streamlit + multi-graphes) ----------------
                # ---------------- Distributions RGB par classe (version Streamlit + 2 colonnes + légende unique) ----------------
        with tab3:
            import numpy as np
            import plotly.graph_objects as go
            from scipy.stats import gaussian_kde

            st.subheader("🎨 Distribution des canaux RGB par type de cellule")

            cell_types = sorted(class_images.keys())
            if not cell_types:
                st.info("Aucune classe trouvée pour l'analyse RGB.")
            else:
                all_densities = []
                rgb_distributions = {}

                # --- Première passe : calcul des densités pour normaliser l'axe Y ---
                for cell_type in cell_types:
                    images = class_images[cell_type]
                    if not images:
                        continue

                    all_red, all_green, all_blue = [], [], []
                    for img_path in images[:30]:
                        try:
                            img = Image.open(img_path).convert("RGB")
                            arr = np.array(img)
                            all_red.extend(arr[:, :, 0].flatten())
                            all_green.extend(arr[:, :, 1].flatten())
                            all_blue.extend(arr[:, :, 2].flatten())
                        except UnidentifiedImageError:
                            continue

                    sample_size = min(10000, len(all_red))
                    if sample_size == 0:
                        continue
                    indices = np.random.choice(len(all_red), sample_size, replace=False)
                    rgb_data = {
                        "R": np.array(all_red)[indices],
                        "G": np.array(all_green)[indices],
                        "B": np.array(all_blue)[indices],
                    }

                    x_vals = np.linspace(0, 255, 256)
                    densities = {}
                    for c in ["R", "G", "B"]:
                        kde = gaussian_kde(rgb_data[c])
                        y_vals = kde(x_vals)
                        densities[c] = y_vals
                        all_densities.extend(y_vals)
                    rgb_distributions[cell_type] = (x_vals, densities)

                y_max = max(all_densities) if all_densities else 0.01

                # --- Couleurs et légende commune ---
                colors = {"R": "#FF6B6B", "G": "#4ECB71", "B": "#4A90E2"}
                labels = {"R": "Rouge", "G": "Vert", "B": "Bleu"}

                # ✅ Légende commune (affichée une seule fois au-dessus)
                legend_fig = go.Figure()
                for c in ["R", "G", "B"]:
                    legend_fig.add_trace(
                        go.Scatter(
                            x=[None],
                            y=[None],
                            mode="lines",
                            line=dict(color=colors[c], width=3),
                            name=labels[c],
                        )
                    )
                legend_fig.update_layout(
                    showlegend=True,
                    height=50,
                    margin=dict(l=0, r=0, t=0, b=0),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.0,
                        xanchor="center",
                        x=0.5,
                        bgcolor="rgba(255,255,255,0.7)",
                        bordercolor="rgba(180,180,180,0.4)",
                        borderwidth=1,
                        font=dict(size=12)
                    ),
                )
                st.plotly_chart(legend_fig, use_container_width=True)

                # --- Organisation en deux colonnes ---
                col1, col2 = st.columns(2)
                cols = [col1, col2]

                for i, cell_type in enumerate(cell_types):
                    if cell_type not in rgb_distributions:
                        continue

                    x_vals, densities = rgb_distributions[cell_type]
                    fig = go.Figure()

                    for c in ["R", "G", "B"]:
                        fig.add_trace(
                            go.Scatter(
                                x=x_vals,
                                y=densities[c],
                                mode="lines",
                                line=dict(color=colors[c], width=3),
                                fill="tozeroy",
                                opacity=0.3,
                                showlegend=False  # ❌ Légende désactivée ici
                            )
                        )

                    # --- Mise en page du graphique individuel ---
                    fig.update_layout(
                        title=dict(
                            text=f"<b>{cell_type}</b>",
                            font=dict(size=20, color="#222", family="Arial Black"),
                            x=0.5,
                            y=0.93,  # 🔹 titre plus haut pour éviter la superposition
                        ),
                        xaxis=dict(
                            title="Valeur de pixel (0–255)",
                            showgrid=True,
                            gridcolor="rgba(220,220,220,0.3)",
                        ),
                        yaxis=dict(
                            title="Densité",
                            showgrid=True,
                            gridcolor="rgba(220,220,220,0.3)",
                            range=[0, y_max * 1.1],
                        ),
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        margin=dict(l=40, r=20, t=60, b=40),  # 🔹 espace augmenté en haut
                        hovermode="x unified",
                    )

                    cols[i % 2].plotly_chart(fig, use_container_width=True)

            st.caption("🔎 Ces courbes permettent de comparer la distribution des valeurs RGB entre classes. "
           "Les différences de densité peuvent indiquer des contrastes d’exposition ou des variations de coloration "
           "propres à chaque type cellulaire.")
            

            

        # ---------------- Exemples d'images ----------------
        with tab4:
            st.subheader("Sélection par classe")
            selected_class = st.selectbox("Choisis une classe pour voir des images", df.index)
            if selected_class in class_images and class_images[selected_class]:
                sample_images = random.sample(class_images[selected_class], min(4, len(class_images[selected_class])))
                cols = st.columns(4)
                for i, img_path in enumerate(sample_images):
                    try:
                        img = Image.open(img_path)
                        cols[i % 4].image(img, caption=selected_class, use_container_width=True)
                    except UnidentifiedImageError:
                        st.warning(f"Impossible d'afficher l'image {img_path.name}")
            else:
                st.info("Aucune image disponible pour cette classe.")

            st.subheader("Exemples aléatoires de plusieurs classes")
            valid_classes = [c for c in class_images if class_images[c]]
            if valid_classes:
                selected_classes = random.sample(valid_classes, min(4, len(valid_classes)))
                cols = st.columns(4)
                for i, c in enumerate(selected_classes):
                    try:
                        sample_img = random.choice(class_images[c])
                        img = Image.open(sample_img)
                        cols[i].image(img, caption=c, use_container_width=True)
                    except UnidentifiedImageError:
                        st.warning(f"Impossible d'afficher l'image {sample_img.name}")
            else:
                st.info("Aucune image trouvée dans les sous-dossiers.")
            
            st.markdown("---")
            st.subheader("Image moyenne par classe")

            import numpy as np

            img_size = (128, 128)  # Taille standardisée pour le calcul
            mean_images = {}

            for class_name, images in class_images.items():
                if not images:
                    continue

                imgs = []
                for img_path in images[:100]:  # Limite à 30 images pour la performance
                    try:
                        img = Image.open(img_path).convert("RGB").resize(img_size)
                        imgs.append(np.array(img, dtype=np.float32))
                    except UnidentifiedImageError:
                        continue

                if imgs:
                    mean_img = np.mean(imgs, axis=0).astype(np.uint8)
                    mean_images[class_name] = Image.fromarray(mean_img)

            if not mean_images:
                st.info("Aucune image moyenne n’a pu être calculée.")
            else:
                # Affichage en grille 4x2
                n_cols = 4
                cols = st.columns(n_cols)
                for i, (class_name, mean_img) in enumerate(mean_images.items()):
                    with cols[i % n_cols]:
                        st.image(mean_img, caption=class_name, use_container_width=True)
            import numpy as np
            import pandas as pd
            import streamlit as st
            from sklearn.metrics.pairwise import cosine_similarity
            import seaborn as sns
            import matplotlib.pyplot as plt
            import numpy as np
            from PIL import Image


            avg_images = {}

            for class_name, image_paths in class_images.items():
                if not image_paths:
                    continue

                # Limite à 30 images par classe pour performance
                sample_paths = image_paths[:30]
                imgs = []

                for p in sample_paths:
                    try:
                        img = Image.open(p).convert("RGB").resize((128, 128))
                        imgs.append(np.array(img, dtype=np.float32))
                    except Exception:
                        continue

                if imgs:
                    avg_image = np.mean(imgs, axis=0)
                    avg_images[class_name] = avg_image / 255.0

            # st.markdown("---")
            # st.subheader("Similarité entre les images moyennes (distance cosinus)")

            # # Aplatir toutes les images moyennes pour comparaison
            # flattened_images = {cls: avg_images[cls].flatten() for cls in avg_images.keys()}

            # # Convertir en DataFrame
            # image_vectors = pd.DataFrame(flattened_images).T  # chaque ligne = une classe
            # image_vectors.columns = [f"pixel_{i}" for i in range(image_vectors.shape[1])]

            # # Calcul de la similarité cosinus
            # cosine_sim_matrix = cosine_similarity(image_vectors)
            # cosine_df = pd.DataFrame(cosine_sim_matrix, index=avg_images.keys(), columns=avg_images.keys())
            # # # Normalisation min-max pour mieux visualiser les écarts
            # # min_val, max_val = cosine_df.min().min(), cosine_df.max().max()
            # # normalized_df = (cosine_df - min_val) / (max_val - min_val)

            # # # Plot heatmap normalisée
            # # fig, ax = plt.subplots(figsize=(8, 6))
            # # sns.heatmap(
            # #     normalized_df,
            # #     annot=True,
            # #     cmap="viridis",
            # #     fmt=".3f",
            # #     linewidths=0.5,
            # #     cbar_kws={"label": "Similarité cosinus (normalisée)"},
            # # )
            # # ax.set_title("Similarité cosinus normalisée entre images moyennes", fontsize=14, fontweight="bold")
            # # st.pyplot(fig)



            # # Plot heatmap stylée
            # fig, ax = plt.subplots(figsize=(8, 6))
            # sns.heatmap(
            #     cosine_df, 
            #     annot=True, 
            #     cmap="YlGnBu", 
            #     fmt=".3f", 
            #     linewidths=0.5, 
            #     cbar_kws={"label": "Similarité cosinus"}
            # )
            # ax.set_title("Matrice de similarité entre images moyennes", fontsize=14, fontweight="bold")
            # st.pyplot(fig)
            # st.markdown("""
            #     **Interprétation :**  
            #     La matrice ci-dessus montre la similarité cosinus entre les images moyennes de chaque type de cellule.  
            #     Une valeur proche de **1** indique que deux classes présentent une **structure et une répartition colorimétrique globalement similaires**,  
            #     tandis qu’une valeur plus basse traduit des **différences plus marquées** dans l’organisation visuelle.  

            #     On peut ainsi identifier les classes de cellules qui sont **visuellement proches** les unes des autres,  
            #     et celles qui présentent une **signature visuelle distincte**.
            #     """)
            import numpy as np
            import pandas as pd
            import streamlit as st
            from sklearn.metrics.pairwise import cosine_similarity
            from PIL import Image
            import random
            import seaborn as sns
            import matplotlib.pyplot as plt

            st.markdown("---")
            st.subheader("Similarité cosinus moyenne entre classes (images réelles)")

            # Limiter à N images par classe pour performance
            N_SAMPLES = 20
            IMAGE_SIZE = (64, 64)

            # Dictionnaire pour stocker les vecteurs d’images
            class_vectors = {}

            # Chargement et vectorisation des images
            for class_name, image_paths in class_images.items():
                if not image_paths:
                    continue

                sample_paths = random.sample(image_paths, min(N_SAMPLES, len(image_paths)))
                vectors = []

                for path in sample_paths:
                    try:
                        img = Image.open(path).convert("RGB").resize(IMAGE_SIZE)
                        arr = np.array(img, dtype=np.float32).flatten()
                        arr /= np.linalg.norm(arr) + 1e-8  # normalisation L2
                        vectors.append(arr)
                    except Exception:
                        continue

                if vectors:
                    class_vectors[class_name] = np.stack(vectors)

            # Calcul de la similarité moyenne entre classes
            classes = list(class_vectors.keys())
            similarity_matrix = np.zeros((len(classes), len(classes)))

            for i, cls_i in enumerate(classes):
                for j, cls_j in enumerate(classes):
                    sims = cosine_similarity(class_vectors[cls_i], class_vectors[cls_j])
                    similarity_matrix[i, j] = sims.mean()  # moyenne des similarités

            # Conversion en DataFrame pour visualisation
            cosine_df = pd.DataFrame(similarity_matrix, index=classes, columns=classes)

            # Heatmap stylisée
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(
                cosine_df,
                annot=True,
                cmap="YlGnBu",
                fmt=".3f",
                linewidths=0.5,
                cbar_kws={"label": "Similarité cosinus moyenne"},
            )
            # ax.set_title("Similarité cosinus moyenne entre classes (images réelles)", fontsize=14, fontweight="bold")
            st.pyplot(fig)

            # Interprétation
            st.markdown("""
            **Interprétation :**  
            Ce graphique montre la similarité cosinus moyenne entre les images réelles de chaque classe.  
            Une valeur élevée (proche de 1) indique que les images de deux classes présentent des caractéristiques visuelles très proches  
            (couleurs, textures ou structures globales similaires).  
            Des valeurs plus faibles traduisent des différences plus nettes entre les types de cellules.
            """)


     


    else:
        st.info("Aucun sous-dossier contenant des images trouvé.")
