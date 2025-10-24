import streamlit as st


st.title("🧾 Conclusion")


st.markdown("""
### Résumé
- Le modèle ResNET de classification atteint une bonne performance sur le dataset Mendeley.
- Il est capable d’identifier la majorité des types de cellules avec une précision satisfaisante.
- Le modèle baseline atteint une performance moyenne sur le dataset Mendeley.


### Limites
- Le modèle ResNET n'a pas été entrainé sur l'ensemble du dataset par manque de puissance de calcul.


- Quelques classes rares (basophils, erythroblasts) restent difficiles à distinguer.
- Nécessité d’un dataset plus équilibré et de techniques d’interprétation approfondies.
- Le modèle peut avoir tendance à se focaliser sur l'arrière plan de l'image pour les prédictions.


### Perspectives
- Intégrer du pré-processing dans le pipeline pour améliorer la performance.
- Étendre l’application à la détection d'anomalies sur les cellules.
""")