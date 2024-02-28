import streamlit as st
from streamlit_modal import Modal
import os
import re
import json
import pandas as pd
import numpy as np
from openai import AzureOpenAI
import fitz

def generate_embeddings(text, model="text-embedding-3-large"):
    return st.session_state["client"].embeddings.create(input = [text], model=model).data[0].embedding

def group_paragraphs(pars, min_l=50):
    g_pars = []
    curr = ""

    for p in pars:
        curr += ("   " if curr else "") + p.strip()
        if len(curr) > min_l:
            g_pars.append(curr)
            curr = ""
    if curr:
        g_pars.append(curr)
    
    return g_pars

st.set_page_config(page_title="VéGa - VeGanalyst", page_icon="🔬")

st.write("# 🔬 VeGanalyst")

st.write("VeGanalyst est un outil d'analyse de conformité.")


uploaded_file = st.file_uploader("Sélectionnez un fichier (PDF) dont vous souhaitez analyser la conformité.")
if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    pdf = fitz.open("pdf", bytes_data)
    pages = [page for page in pdf]
    page_select = st.slider("Pages à analyser", 1, len(pages), (1, len(pages)))

    open_modal = st.columns((4,1))[1].button("Suivant", type="primary", use_container_width=True)
    if open_modal:
        start, stop = page_select
        pars = []

        for i, page in enumerate(pages[start-1:stop]):
            pars.extend([b[4].replace('\n', ' ') for b in page.get_text("blocks") if b[4] and len(b[4]) > 6])

        g_pars = group_paragraphs(pars)
        df = pd.DataFrame(g_pars, columns=['Text'])
        df2 = df.shift(2, fill_value="") + df.shift(1, fill_value="") + df + df.shift(-1, fill_value="") + df.shift(-2, fill_value="")
        df = df2.copy()
        st.dataframe(df)