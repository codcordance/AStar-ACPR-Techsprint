import streamlit as st
from streamlit_modal import Modal
import os
import re
import json
import pandas as pd
import numpy as np
from openai import AzureOpenAI
import fitz

class VegaAnalyst:
  def __init__(self, system_prompt):
    os.environ["AZURE_OPENAI_DeploymentId"] = "gpt-4-turbo"
    self.client = st.session_state["client"]

    self.embeddings = df
    self.system_prompt = system_prompt

  def _cosine_similarity(self, a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
  
  def _get_embedding(self, text, model="ada-002"): # model = "deployment_name"
    return self.client.embeddings.create(input = [text], model=model).data[0].embedding

  def search_paragraphs(self, _embeddings: pd.DataFrame, user_query, top_n=4):
    embedding = self._get_embedding(
        user_query,
        model="ada-002" # model should be set to the deployment name you chose when you deployed the text-embedding-ada-002 (Version 2) model
    )
    embeddings = _embeddings.copy()
    embeddings["similarities"] = embeddings["embed"].apply(lambda x: self._cosine_similarity(x, embedding))

    res = (
        embeddings.sort_values("similarities", ascending=False)
        .head(top_n)
    )
    return res

  def generate_answer(self, prompt, system_prompt):
    res = self.client.chat.completions.create(
      model=os.getenv("AZURE_OPENAI_DeploymentId"),
      response_format={"type": "json_object"},
      messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
      ]
    ).choices[0].message
    
    return res.content

  def analyse(self, prompt):
    res = self.search_paragraphs(self.embeddings, prompt, top_n=2)
    sys_prompt = self.system_prompt + "\n".join(res["Text"])
    answer = self.generate_answer(prompt, sys_prompt)
    return ({"answer": answer, "docs": res })

def parse_object_to_md(obj):
    md = """"""
    md += f"# Point de contrôle\n{obj['name']}\n\n\n"
    md += f"## Catégorie\n {obj['category_name']}\n\n"
    md += f"## Description\n{obj['desc']}\n\n"
    
    base_leg = """"""
    for leg in obj['base_leg']:
      base_leg += f"- {leg}\n"
    md += f"## Base légale\n{base_leg}\n\n"

    methodology = """"""
    for i, meth in enumerate(obj['methodology']):
      methodology += f"{i+1}. {meth}\n"
    md += f"## Méthodologie\n{methodology}\n\n"

    md += f"## Exemple de conformité\n{obj['conform_example']}\n\n"
    md += f"## Exemple de non-conformité\n{obj['non_conform_example']}\n\n"
    md += f"## Procédure\n{obj['proc']}"
    return md

def generate_embeddings(text, model="ada-002"):
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

if 'embed-analyst' not in st.session_state:
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
            with st.spinner("Embedding du document..."):
                df['embed'] = df["Text"].apply(generate_embeddings) 
            st.session_state['embed-analyst'] = df
            st.rerun()
else:
    df = st.session_state['embed-analyst']
    btn = st.columns((4, 1))[1].button("❌ Annuler", use_container_width=True)
    if btn:
        st.session_state.pop('embed-analyst')
        st.rerun()

    walk = [dir for dir in os.walk('pdc_content')]
    cats = []
    controlpointcol = []
    for dir in walk[0][1]:
        with open(f'pdc_content/{dir}/info.json') as f:
            d = json.load(f)
            cats.append(d)
 
    for i, cat in enumerate(cats):
        code = cat["code"]
        
        jsons = [f for f in walk[i+1][2] if f != "info.json"]
        for fname in jsons:
            with open(f'pdc_content/{code}/{fname}') as f:
                dataf = json.load(f)
                controlpointcol.append(dataf)
    #st.write(controlpointcol)

    cps = st.multiselect("Sélectionnez les points de contrôle d'intérêt", controlpointcol,
     format_func= lambda c: f"{c['name']} ({c['category_name']})")

    cpsmd = [parse_object_to_md(cp) for cp in cps]
    analbtn = st.button("Analyser", use_container_width=True)

    if analbtn:
        for cp in cps:
            st.write(f"### Point de contrôle: {cp['name']}")

            st.write("Détail")
        

        st.write("Analyse en cours !")