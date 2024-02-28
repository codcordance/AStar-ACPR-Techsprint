import streamlit as st
from streamlit_modal import Modal
import os
import re
import json
import pandas as pd
import numpy as np
from openai import AzureOpenAI


class VegaAssistant:
  def __init__(self, embeddings_source, system_prompt):
    os.environ["AZURE_OPENAI_DeploymentId"] = "gpt-4-turbo"
    self.client = st.session_state["client"]
    contents = []
    obj = json.load(open(embeddings_source, "r"))
    for i in range(len(obj)):
      contents.append({'content': obj[i]["content"], "embedding": obj[i]["embedding"]})
    
    self.embeddings = pd.DataFrame(contents, columns=['content', 'embedding'])
    self.system_prompt = system_prompt
    self.trace = []

  def _cosine_similarity(self, a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
  
  def _get_embedding(self, text, model="text-embedding-3-large"): # model = "deployment_name"
    return self.client.embeddings.create(input = [text], model=model).data[0].embedding

  def search_docs(self, _embeddings: pd.DataFrame, user_query, top_n=4):
      embedding = self._get_embedding(
          user_query,
          model="text-embedding-3-large" # model should be set to the deployment name you chose when you deployed the text-embedding-text-embedding-3-large (Version 2) model
      )
      embeddings = _embeddings.copy()
      embeddings["similarities"] = embeddings["embedding"].apply(lambda x: self._cosine_similarity(x, embedding))

      res = (
          embeddings.sort_values("similarities", ascending=False)
          .head(top_n)
      )
      return res

  def generate_answer(self, prompt, system_prompt):
    trace = [
        {"role": "system", "content": system_prompt},
      ]
    trace.extend(st.session_state.vegassistmsg)
    res = self.client.chat.completions.create(
      model=os.getenv("AZURE_OPENAI_DeploymentId"),
      #response_format={"type": "json_object"},
      messages=trace,
      stream=True
    )
    #self.trace.append({
    #  "role": res.role,
    #  "content": res.content
    #})
    
    return res

  def chat(self, prompt):
    res = self.search_docs(self.embeddings, prompt, top_n=2)

    sys_prompt = self.system_prompt + "\n".join(res["content"])
    answer = self.generate_answer(prompt, sys_prompt)
    return answer

st.set_page_config(page_title="VéGa - VeGassist", page_icon="📄")

st.write("# 📄 VeGassist")

system_prompt = """
Objectif Principal : Tu es un assistant expert de l'Autorité de Contrôle Prudentiel et de Résolution rattaché à la Banque de France. Ton nom est VéGassist. Tu dois permettre aux utilisateurs de l'ACPR d'accéder facilement aux documents normatifs relatifs à la supervision des activités des établissements financiers et de vérifier la conformité des phrases relatives aux produits financiers aux règlements en vigueur.
1. Recherche Documentaire :
- Si l’utilisateur te demande de rechercher des documents normatifs en utilisant des mots-clés, des catégories spécifiques ou des références règlementaires, tu dois uniquement citer les documents faisant référence à ces éléments dans une liste ordonnée.
- Si la formulation de l’utilisateur est trop complexe, demande-lui de segmenter ses requêtes.
- Si l’utilisateur ne parvient pas à trouver ce qu’il cherche, donne-lui des exemples de phrases à renseigner. Par exemple, donne-lui des mots clés ou des catégories. 
- Si tu lis une abréviation dans un document ou si tu lis une abréviation dans la phrase de l’utilisateur et que tu ne comprends pas l’abréviation, cite l’abréviation et demande à l’utilisateur de la définir. 
- Tu ne dois jamais utiliser d'information externe aux points de contrôle. Tu ne dois jamais inventer des réponses. Tu ne dois jamais imaginer des réponses. 
2. Accès aux Informations :
- Tu dois extraire de manière précise les informations pertinentes des points de contrôle en réponse aux requêtes de l'utilisateur.
- Tu dois restituer le contexte entourant une information pour une meilleure compréhension. Fais bien attention, à séparer la citation du point de contrôle des informations contextuelles. Le point de contrôle doit être clairement identifiable et ne doit pas avoir été modifié. 
3. Interaction Naturelle :
- Favorise une interaction conversationnelle naturelle avec l'utilisateur en comprenant le langage courant et en fournissant des réponses compréhensibles.
- Réponds avec un langage formel est clair. Ce que tu écris doit pouvoir être présenté dans des rapports officiels. 
- Ne cite pas le point de contrôle entier sauf si on te le demande. Typiquement, ne donne pas la base légale si on ne le demande pas, et ne donne pas la procédure sauf dans le cas d'une analyse négative.
4. Conformité des Phrases aux Règlements :
- Si la phrase de l'utilisateur est jugée conforme, dis uniquement qu'elle est conforme. Si l'utilisateur te demande de citer la base légale, tu dois la lui citer. 
- Si la phrase de l'utilisateur est jugée non conforme, tu dois expliquer pourquoi elle n'est pas conforme. Dans ce cas, écris aussi la procédure à suivre. Fais en sorte que l'affichage de tes réponses soit clair. Et tu dois citer la base légale si on te le demande.
- Si tu n'as pas les documents permettant de vérifier la conformité de la phrase, dis uniquement qu'il te manque des documents. 
- Si la phrase de l'utilisateur n'a aucun lien avec la vérification de conformité, dis-lui que l'analyse de conformité n'est pas applicable. 
5. Gestion des Erreurs et Ambiguïtés :
- Tu dois gérer les situations où une requête est ambiguë ou incomplète en demandant des clarifications.
- Si l’utilisateur fait une faute de frappe, corrige-la en citant la correction et réponds à la question posée par la phrase corrigée. 
- Si tu ne connais pas la réponse, dis-le et ne cherche pas à rajouter d’autres éléments à part des questions pour plus de précision.
6. Mises à Jour Légales :
- Tu dois informer les utilisateurs des modifications récentes dans la législation financière en précisant les dates. 
7. Assistance et Support :
- Fournis un support contextuel pour aider les utilisateurs à formuler des requêtes de manière efficace. Attention, cela ne doit jamais modifier les points de contrôle dans tes réponses.
8. Points de contrôle à utiliser :
- Tu dois baser l'entièreté de tes réponses sur les documents suivant : 
"""

assist = VegaAssistant("dump_controlpoint_embeddings.json", system_prompt)

def resetmsg():
    st.session_state.vegassistmsg = [{"role": "assistant", "content": "_Salut !_ Comment puis-je t'aider ?"}]

if "vegassistmsg" not in st.session_state:
    resetmsg()

st.write("VeGassist est votre assistant de vérification de conformité. Possez-lui une question et il vous répondra !")
st.columns((3,2))[1].button("🔁 Réinitialiser la conversation", use_container_width=True, on_click=resetmsg)

chat = st.container(border=True)
for message in st.session_state.vegassistmsg:
    with chat.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Posez une question ici")
if prompt:
    st.session_state.vegassistmsg.append({"role": "user", "content": prompt})
    with chat.chat_message("user"):
        st.markdown(prompt)
    with chat.chat_message("assistant"):
        stream = assist.chat(prompt)
        response = st.write_stream(stream)
    st.session_state.vegassistmsg.append({"role": "assistant", "content": response})