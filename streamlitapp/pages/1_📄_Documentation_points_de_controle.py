import streamlit as st
from streamlit_modal import Modal
import os
import re
import json
import pandas as pd
import numpy as np
from openai import AzureOpenAI

st.set_page_config(page_title="VéGa - Documentation points de contrôle", page_icon="📄")

walk = [dir for dir in os.walk('pdc_content')]
cats = []
for dir in walk[0][1]:
    with open(f'pdc_content/{dir}/info.json') as f:
        d = json.load(f)
        cats.append(d)


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
      methodology += f"{i}. {meth}\n"
    md += f"## Méthodologie\n{methodology}\n\n"

    md += f"## Exemple de conformité\n{obj['conform_example']}\n\n"
    md += f"## Exemple de non-conformité\n{obj['non_conform_example']}\n\n"
    md += f"## Procédure\n{obj['proc']}"
    return md


def generate_embeddings(text, model="ada-002"):
    return st.session_state["client"].embeddings.create(input = [text], model=model).data[0].embedding


def normalize_text(s, sep_token = " \n "):
    s = re.sub(r'\s+',  ' ', s).strip()
    s = re.sub(r". ,","",s)
    # remove all instances of multiple spaces
    s = s.replace("..",".")
    s = s.replace(". .",".")
    s = s.replace("\n", "")
    s = s.strip()
    
    return s


def addcp():
    import streamlit as st
    st.sidebar.image("vega.png")

    cat = st.session_state['state-doccp-cat-ct']
    st.write(f"# ➕ Ajout d'un point de contrôle à la catégorie {cat['name']}")

    retour = st.columns((3,1))[1].button("🔙 Retour", type="secondary", use_container_width=True)
    if retour:
        st.session_state['state-doccp'] = "enum"
        st.rerun()

    st.write("#### Général")

    st.text_input("Catégorie", value=cat['name'], disabled=True)

    code = st.text_input("Code", placeholder="dispo",
            help="Code court, sans accents, chiffres ou caractères spéciaux, par exemple \"dispo\" ")

    name = st.text_input("Nom du point de contrôle", placeholder="Disponibilité",
            help="Nom court mais complet, par exemple \"Disponibilité\" ")

    desc = st.text_area("Description", placeholder="L’API doit être tout le temps disponible.",
            help="Description détaillée")

    st.write("#### Base légale")
    st.markdown("Listes des références qui servent de base légale au point de contrôle, séparée par des virgules.")
    st.markdown("_Double cliquer sur une ligne pour l'éditer._")

    base_leg = st.data_editor(pd.DataFrame(["Premier article..."], columns=["Etape"]), key="baseleg", use_container_width=True,
     hide_index=True, num_rows="dynamic")

    st.write("#### Méthodologie")
    st.markdown("Etapes à suivre par le programme pour déterminer la conformité à ce point de contrôle.")
    st.markdown("_Double cliquer sur une ligne pour l'éditer._")
    methodo = st.data_editor(pd.DataFrame(["Première étape..."], columns=["Etape"]), key="methodo", use_container_width=True,
     hide_index=True, num_rows="dynamic")

    st.write("#### Exemples")
    
    exconf = st.text_area("Exemple de conformité", placeholder="Détail d'un exemple de conformité au point de contrôle",
            help="Donnez ici le détail d'un exemple de conformité au point de contrôle")

    exnonconf = st.text_area("Exemple de non conformité", placeholder="Détail d'un exemple de non conformité au point de contrôle",
            help="Donnez ici le détail d'un exemple de non conformité au point de contrôle")

    st.write("#### Procédure")

    proc = st.text_area("Procédure à suivre en cas de non-conformité", placeholder="Il faut prendre contact avec l’établissement pour comprendre l’origine de l’indisponibilité.",
            help="Procédure à suivre en cas de non-conformité")

    st.markdown('##')
    save = st.button("✨ Créer", type="primary", use_container_width=True)

    if save:
        data = {
            "category_name": cat["name"],
            "name": name,
            "code": code,
            "desc": desc,
            "base_leg": base_leg.values.ravel().tolist(),
            "methodology": methodo.values.ravel().tolist(),
            "conform_example": exconf,
            "non_conform_example": exnonconf,
            "proc": proc
        }
        with open(f'pdc_content/{cat["code"]}/{code}.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        st.session_state['state-doccp'] = "enum"
        st.rerun()

def editcp():
    import streamlit as st
    st.sidebar.image("vega.png")

    cat = st.session_state['state-doccp-cat-ct']
    fname = st.session_state['state-doccp-fname']

    data = None
    with open(f'pdc_content/{cat["code"]}/{fname}.json') as f:
        data = json.load(f)

    while data == None:
        st.write("chargement...")
    
    st.write(f"# ✏️ Edition du point de contrôle {data['name']}")
        
    colps = st.columns((1,1,2))
    delete = colps[1].button("⚠️ Supprimer", type="primary", use_container_width=True)
    retour = colps[2].button("🔙 Retour", type="secondary", use_container_width=True)
    if retour:
        st.session_state['state-doccp'] = "enum"
        st.rerun()
    if delete:
        os.remove(f'pdc_content/{cat["code"]}/{fname}.json')
        st.session_state['state-doccp'] = "enum"
        st.rerun()

    st.write("#### Général")

    st.text_input("Catégorie", value=cat['name'], disabled=True)

    code = st.text_input("Code", value=data['code'], disabled=True)

    name = st.text_input("Nom du point de contrôle", value=data["name"],
            help="Nom court mais complet, par exemple \"Disponibilité\" ")

    desc = st.text_area("Description", value=data["desc"],
            help="Description détaillée")

    st.write("#### Base légale")
    st.markdown("Listes des références qui servent de base légale au point de contrôle, séparée par des virgules.")
    st.markdown("_Double cliquer sur une ligne pour l'éditer._")

    base_leg = st.data_editor(pd.DataFrame(data["base_leg"], columns=["Etape"]), key="baseleg", use_container_width=True,
    hide_index=True, num_rows="dynamic")

    st.write("#### Méthodologie")
    st.markdown("Etapes à suivre par le programme pour déterminer la conformité à ce point de contrôle.")
    st.markdown("_Double cliquer sur une ligne pour l'éditer._")
    methodo = st.data_editor(pd.DataFrame(data["methodology"], columns=["Etape"]), key="methodo", use_container_width=True,
    hide_index=True, num_rows="dynamic")

    st.write("#### Exemples")
    
    exconf = st.text_area("Exemple de conformité", value=data["conform_example"], placeholder="Détail d'un exemple de conformité au point de contrôle",
            help="Donnez ici le détail d'un exemple de conformité au point de contrôle")

    exnonconf = st.text_area("Exemple de non conformité", value=data["non_conform_example"], placeholder="Détail d'un exemple de non-conformité au point de contrôle",
            help="Donnez ici le détail d'un exemple de non conformité au point de contrôle")

    st.write("#### Procédure")

    proc = st.text_area("Procédure à suivre en cas de non-conformité",  value=data["proc"], placeholder="Il faut prendre contact avec l’établissement pour comprendre l’origine de l’indisponibilité.",
            help="Procédure à suivre en cas de non-conformité")

    st.markdown('##')
    save = st.button("💾 Enregistrer", type="primary", use_container_width=True)

    if save:
        data = {
            "category_name": cat["name"],
            "name": name,
            "code": code,
            "desc": desc,
            "base_leg": base_leg.values.ravel().tolist(),
            "methodology": methodo.values.ravel().tolist(),
            "conform_example": exconf,
            "non_conform_example": exnonconf,
            "proc": proc
        }
        with open(f'pdc_content/{cat["code"]}/{code}.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        st.session_state['state-doccp'] = "enum"
        st.rerun()

def enum():
    import streamlit as st

    st.write("# Documentation points de contrôle 📄")
    st.sidebar.image("vega.png")

    st.markdown(
        """Ci-dessous sont listées les catégories de points de contrôle, et dans chacune les points associés."""
    )

    modal = Modal("Création d'une catégorie 📚", key="create-category")

    open_modal = st.columns((3,1))[1].button("Ajouter une catégorie", type="primary", use_container_width=True)
    if open_modal:
        modal.open()

    if modal.is_open():
        with modal.container():
            code = st.text_input("Code", placeholder="openfinance",
            help="Code court, sans accents, chiffres ou caractères spéciaux, par exemple \"openfinance\" ")

            name = st.text_input("Nom complet", placeholder="Open Finance",
            help="Nom complet, détaillé, par exemple \"Open Finance\" ")

            create = st.button("✨ Créer", type="primary")
            if create:
                os.mkdir(f'pdc_content/{code}')
                with open(f'pdc_content/{code}/info.json', 'w', encoding='utf-8') as f:
                    json.dump({"code": code, "name": name}, f, ensure_ascii=False, indent=4)
                modal.close()

    tabs = st.tabs([data["name"] for data in cats])

    controlpointcol = []

    for i, tab in enumerate(tabs):
        code = cats[i]["code"]
        cols = tab.columns((1,1)) 
        create = cols[0].button("➕ Ajouter un point de contrôle", key=f"addto-cat-{i}", type="secondary", use_container_width=True)
        delete = cols[1].button("🗑️ Supprimer la catégorie", key=f"delete-cat-{i}", type="primary", use_container_width=True)
        if delete:
            for i in walk[i+1][2]:
                os.remove(f'pdc_content/{code}/{i}')

            os.rmdir(f'pdc_content/{code}')
            st.rerun()
        if create:
            st.session_state['state-doccp-cat-ct'] = cats[i]
            st.session_state['state-doccp'] = "addcp"
            st.rerun()
        
        jsons = [f for f in walk[i+1][2] if f != "info.json"]
        for fname in jsons:
            with open(f'pdc_content/{code}/{fname}') as f:
                dataf = json.load(f)
                controlpointcol.append(dataf)
                colstab = tab.columns((4,1))
                colstab[0].markdown(f"#### {dataf['name']}")
                colstab[0].markdown(f"{dataf['desc']}")
                openbtn = colstab[1].button("Modifier ✏️", key=f"open-{code}-{fname}")

                if openbtn:
                    st.session_state['state-doccp-cat-ct'] = cats[i]
                    st.session_state['state-doccp-fname'] = dataf['code']
                    st.session_state['state-doccp'] = "editcp"
                    st.rerun()
                tab.markdown("""---""")

    regemb = st.button("🔄 Régénérer les embeddings des points de contrôle 🔄", use_container_width = True)
    if regemb:
        with st.status("Tritement initial...", expanded=True) as status:
            for cp in controlpointcol:
                st.write(f"Point de contrôle {cp['category_name']} > {cp['name']}")
            
            df = pd.DataFrame(np.array(controlpointcol), columns=['content'])

            st.write("Parsing en markdown")
            df['content'] = df['content'].apply(parse_object_to_md)
            st.write("Normalisation")
            df['content'] = df['content'].apply(normalize_text)
            status.update(label="Génération des embeddings", state="running", expanded=True)
            st.write("Génération des embeddings")
            df['embedding'] = df['content'].apply(generate_embeddings)
            st.write("Enregistrement")
            df.to_json("dump_controlpoint_embeddings.json", orient="records")
            st.write("🎉🎉 Terminé ! Aperçu du tableau avec les embeddings :")
            st.write(df)
            status.update(label="🎉 Génération terminée !", state="complete", expanded=True)
                

states = {
    "enum": enum,
    "addcp": addcp,
    "editcp": editcp
}

for cat in cats:
    states[f"addcp-{cat['code']}"] = lambda: addcp(cat)

if 'state-doccp' not in st.session_state:
    st.session_state['state-doccp'] = 'enum'

states[st.session_state['state-doccp']]()