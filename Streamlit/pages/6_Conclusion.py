import streamlit as st


st.title("🧾 Conclusion")


st.markdown("""
### Résumé
- Le modèle de classification atteint une bonne performance sur le dataset Mendeley.
- Il est capable d’identifier la majorité des types de cellules avec une précision satisfaisante.


### Limites
- Quelques classes rares (basophils, erythroblasts) restent difficiles à distinguer.
- Nécessité d’un dataset plus équilibré et de techniques d’interprétation approfondies.


### Perspectives
- Intégrer le modèle dans un outil clinique assisté par IA.
- Étendre l’étude à d'autres pathologies (ex : leucémie).
""")