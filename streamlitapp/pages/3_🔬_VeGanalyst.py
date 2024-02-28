import streamlit as st
from streamlit_modal import Modal
import os
import re
import json
import pandas as pd
import numpy as np
from openai import AzureOpenAI
import fitz
from concurrent.futures import ThreadPoolExecutor, as_completed

class VegaAnalyst:
  def __init__(self, df, system_prompt):
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
st.sidebar.image("vega.png")

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

    def onchangecps():
        st.session_state.pop("displayanalysis")
    cps = st.multiselect("Sélectionnez les points de contrôle d'intérêt", controlpointcol,
     format_func = lambda c: f"{c['name']} ({c['category_name']})", 
     on_change =onchangecps)

    cpsmd = [parse_object_to_md(cp) for cp in cps]
    analbtn = st.button("Analyser", use_container_width=True)

    if "displayanalysis" not in st.session_state:
        st.session_state["displayanalysis"] = False

    if analbtn | st.session_state["displayanalysis"]:
        st.session_state["displayanalysis"] = True
        analyst = VegaAnalyst(df,"""Objectif Principal : Tu es un expert de l'Authorité de Contrôle Prudentiel de Résolution rattaché à la banque de France. Ta mission est de juger de la conformité d'une pratique relativement à un corpus de textes règlementaires et normatifs qui t'es fourni. Ce que tu écris doit pouvoir être présenté dans des rapports officiels. Tu n'es pas un agent de conversation. 

1. Accès aux Informations :
- Tu dois extraire de manière précise les informations pertinentes des documents normatifs pour juger de la conformité d'une pratique.
- Tu dois restituer le contexte entourant une information pour une meilleure compréhension. Fais bien attention, à séparer la citation du texte normatif des informations contextuelles. Le texte normatif doit être clairement identifiable et ne doit pas avoir été modifié. 

2. Pré-évaluation de la conformité :
- Tu dois utiliser la méthodologie décrite dans les points de contrôle afin de vérifier la conformité.
- Pour pré-évaluer la conformité, tu dois aussi te baser sur les exemples de conformité et les exemples de non-conformité donnés dans les points de contrôle.

3. Gestion des Erreurs et Ambiguïtés :
- Tu dois gérer les situations où une requête est ambiguë ou incomplète en demandant des clarifications.
- Si l’utilisateur fait une faute de frappe, corrige-la en citant la correction et réponds à la question posée par la phrase corrigée. 
- Si tu ne connais pas la réponse, dis-le et ne cherche pas à rajouter d’autres éléments à part des questions pour plus de précision.
- Si tu lis une abréviation dans un document ou si tu lis une abréviation dans la phrase de l’utilisateur et que tu ne comprends pas l’abréviation, cite l’abréviation et demande à l’utilisateur de la définir. 

4. Mises à Jour Légales :
- Tu dois informer les utilisateurs des modifications récentes dans la législation financière en précisant les dates. 

5. Résultats :
- Une fois que tu as analysé la conformité de la pratique, tu dois fournir une réponse dans un fichier JSON. Le fichier doit être structuré ainsi : 
{
"category" : "émission de dette",
"code" : "5mat"
"name" : "AT1 Own Funds Instruments  Monitoring",
"desc" : "Les dettes doivent être émises publiquement et être disponibles à tout le monde",
"base_leg" : [
"1. Article 1", 
"2. Article 2 du Code monétaire"],
"practise_summary" : "<insère un résumé synthétique de la pratique évaluée>",
"summary": "<insère ta réponse en langage naturel et explique ton raisonnement>",
"conform" : "oui / non / nsp / N/A",
"procedure" : "En cas de non-conformité, il faut informer l’établissement de la non-conformité et demander les éléments de remédiation dans un délai de 3 mois / Aucune procédure à suivre / Je ne parviens pas à juger de la conformité de cette pratique, une analyse humaine est peut-être requise, sinon veuillez reformuler la demande / Cette pratique semble ne pas être en relation avec mes compétences et donc mon analyse n'est pas applicable"
}
- Si un champ du JSON n'est pas applicable, remplis le avec "N/A"
- Si tu n'est pas sûr de la conformité de la procédure vis à vis du point de contrôle, remplis le champ "conform" avec la valeur "nsp"

6. Procédure à évaluer :
- Voici des paragraphes de la procédure qui sont en rapport avec le point de contrôle, dont tu dois évaluer la conformité :
""")
        comp = st.empty()
        with comp.container(border=True):
            for i, cp in enumerate(cps):
                st.write(f"#### {cp['name']}")
                with st.spinner("Analyse en cours"):
                    res = analyst.analyse(f"Analyse la conformité du texte qui t'es fourni vis à vis du point de contrôle suivant :\n{cpsmd[i]}")
                    resobj = json.loads(res["answer"])
                    if resobj["conform"] == "oui":
                        st.success("✅ Conforme !")
                    elif resobj["conform"] == "non":
                        st.error("❌ Non conforme !")
                        st.markdown("*Procédure à suivre*")
                        st.markdown(resobj["procedure"])
                    elif resobj["conform"] == "nsp":
                        st.warning("⚠️ Ambigü")
                        st.markdown("*Procédure à suivre*")
                        st.markdown(resobj["procedure"])
                    else:
                        st.info("⚫ Non applicable")
                    
                    with st.expander("Détail"):
                        st.markdown("#### Analyse")
                        st.markdown(resobj["summary"])
                        st.markdown("### Base légale")
                        st.markdown("\n".join(resobj["base_leg"]))
                        st.markdown("---")
                        st.markdown("#### Paragraphes sources")
                        st.markdown(res["docs"]["Text"].values[0])

                    cols_rating = st.columns((9, 1, 1))
                    cols_rating[0].markdown("Notez la réponse du modèle :")
                    badbtn = cols_rating[1].empty().button("👎", key=f"rank-{i}-pos")
                    goodbtn = cols_rating[2].empty().button("👍", key=f"rank-{i}-neg")
                    if badbtn:
                        st.session_state[f"analyst-rate-{i}"] = True
                        st.toast('Votre avis négatif a été enregistré !')
                    if goodbtn:
                        st.session_state[f"analyst-rate-{i}"] = True
                        st.toast('Votre avis positif a été enregistré !')