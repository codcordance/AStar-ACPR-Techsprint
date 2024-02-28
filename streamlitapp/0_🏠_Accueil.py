import streamlit as st
import os
from openai import AzureOpenAI

if "client" not in st.session_state:
    os.environ["AZURE_OPENAI_KEY"] = ""  #push public sur github  
    os.environ["AZURE_OPENAI_ENDPOINT"] = ""  #push public sur github 
    st.session_state["client"] = AzureOpenAI(
      api_key = os.getenv("AZURE_OPENAI_KEY"),  
      api_version = "2023-05-15",
      azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    )

st.set_page_config(
    page_title="VEGA - Accueil",
    page_icon="🏠",
)

st.write("# Bienvenue sur VéGa! 👋")

st.markdown(
    """Bienvenue sur *VéGa*, l'outil de _Visualisation, d’évaluation de Graduation 
et d’analyse_ de l'ACPR."""
)

st.sidebar.image("vega.png")
st.info("Sélectionnez une application dans le menu à gauche.")