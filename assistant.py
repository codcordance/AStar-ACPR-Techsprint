import pandas as pd
import numpy as np
import json
import os
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
    if (len(self.trace) == 0):
      self.trace = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
      ]
    else:
      self.trace.append({"role": "user", "content": prompt})
    res = self.client.chat.completions.create(
      model=os.getenv("AZURE_OPENAI_DeploymentId"),
      response_format={"type": "json_object"},
      messages=self.trace
    ).choices[0].message
    self.trace.append({
      "role": res.role,
      "content": res.content
    })
    
    return res.content

  def chat(self, prompt):
    res = self.search_docs(self.embeddings, prompt, top_n=2)
    sys_prompt = self.system_prompt + "\n".join(res["content"])
    answer = self.generate_answer(prompt, sys_prompt)
    return ({"answer": answer, "docs": res })