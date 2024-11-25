import streamlit as st
import pandas as pd
import numpy as np
import json
import requests
import digitalhub as dh
import os
import streamlit

@st.cache_data
def load_mappings(lang):
    res = {}
    with open(f"src/config/label_mappings_do/{lang}.json") as f:
        d = json.load(f)
        res.update(d)
    with open(f"src/config/label_mappings_mt/{lang}.json") as f:
        d = json.load(f)
        res.update(d)
    with open(f"src/config/label_mappings_tc/{lang}.json") as f:
        d = json.load(f)
        res.update(d)

    return res

def annotate(text):
    body = {"text": text}
    preds = requests.post(f"http://{service_url}/", json=body).json()["predictions"][0:20]
    return preds


mappings = load_mappings("it")


st.title('EUR Lex text classifier')

service_url = os.environ.get("SERVICE_URL", "")

if service_url == None or service_url == "":
    service_url = st.text_input("Service Endpoint", value="", placeholder="host:port")


ta = st.text_area("Testo", value="", placeholder="Fornisci il testo da classificare", height=340)

if st.button("Annota"):
    predictions = annotate(ta)
    labels = list(map(lambda x: mappings[x["label"]] if x["label"] in mappings else x["label"], predictions))
    for l in labels:
        st.text(l)