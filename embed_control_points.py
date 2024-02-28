import pandas as pd
import numpy as np
import re
import os
import json
from openai import AzureOpenAI

# file_name = "raw_control_points.json"

# try:
#   with open(file_name, 'r', encoding="utf-8") as file:
#     content = file.read()
#     contents.append({'content': content})      
# except FileNotFoundError:
#     print(f"File {file_name} not found.")

pd.options.mode.chained_assignment = None 

class ControlPointsEmbedder:
  def __init__(self):
    os.environ["AZURE_OPENAI_KEY"] = "7e93421f46cd4680831023addcb0f42d"
    os.environ["AZURE_OPENAI_ENDPOINT"] = "https://francecentral-openai.openai.azure.com"
    self.client = AzureOpenAI(
      api_key = os.getenv("AZURE_OPENAI_KEY"),  
      api_version = "2023-05-15",
      azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    )


  def embed_control_points_from_file(self, file_name, target_file_name="control_points_with_embeddings.json"):
    df = self._read_json_to_df(file_name)
    df['content'] = df['content'].apply(self._parse_object_to_md)
    df['content'] = df['content'].apply(self._normalize_text)
    df['embedding'] = df['content'].apply(self._generate_embeddings)
    df.to_json(target_file_name, orient="records")
    return df


  # def embed_control_points_from_df(self, _df: pd.DataFrame):
  #   df = _df.copy()
  #   df['content'] = df['content'].apply(self._parse_object_to_md)
  #   df['content'] = df['content'].apply(self._normalize_text)
  #   df['embedding'] = df['content'].apply(self._generate_embeddings)
  #   df.to_json("control_points_with_embeddings.json", orient="records")
  #   return df


  def embed_control_points_from_objs(self, obj: list, target_file_name="control_points_with_embeddings.json"):
    contents = []
    for i in range(len(obj)):
      contents.append({'content': obj[i]})
    df = pd.DataFrame(contents, columns=['content'])
    df['content'] = df['content'].apply(self._parse_object_to_md)
    df['content'] = df['content'].apply(self._normalize_text)
    df['embedding'] = df['content'].apply(self._generate_embeddings)
    df.to_json(target_file_name, orient="records")
    return df


  def _read_json_to_df(self, file_name):
    try:
      with open(file_name, 'r', encoding="utf-8") as file:
        obj = json.load(file)
        contents = []
        for i in range(len(obj)):
          contents.append({'content': obj[i]})
    except FileNotFoundError:
        print(f"File {file_name} not found.")
    return pd.DataFrame(contents, columns=['content'])


  def _parse_object_to_md(self, obj):
    md = """"""
    md += f"# {obj['name']}\n=======\n"
    md += f"## Catégorie {obj['category_name']}\n\n"
    md += f"## Description\n{obj['desc']}\n\n"
    
    base_leg = """"""
    for leg in obj['base_leg']:
      base_leg += f"- {leg}\n"
    md += f"## Base légale\n{base_leg}\n\n"

    methodology = """"""
    for meth in obj['methodology']:
      methodology += f"- {meth}\n"
    md += f"## Méthodologie\n{methodology}\n\n"

    md += f"## Exemple de conformité\n{obj['conform_example']}\n\n"
    md += f"## Exemple de non-conformité\n{obj['non_conform_example']}\n\n"
    md += f"## Procédure\n{obj['proc']}\n\n"
    return md


  # s is input text
  def _normalize_text(self, s, sep_token = " \n "):
      s = re.sub(r'\s+',  ' ', s).strip()
      s = re.sub(r". ,","",s)
      # remove all instances of multiple spaces
      s = s.replace("..",".")
      s = s.replace(". .",".")
      # s = s.replace("\n", "")
      s = s.strip()
      
      return s


  def _generate_embeddings(self, text, model="text-embedding-3-large"): # model = "deployment_name"
      return self.client.embeddings.create(input = [text], model=model).data[0].embedding