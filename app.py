"""
app.py - Streamlit demo (same engine as app_gradio.py: src/agri_llm.py).

Streamlit secrets / env vars:
  AGRI_ADAPTERS  JSON {"lora": "user/repo-lora", "qlora": "user/repo-qlora"}
  AGRI_BASE_MODEL, AGRI_4BIT, HF_TOKEN   (see app_gradio.py)

Reality check: Streamlit Community Cloud's free tier (~1 GB RAM) cannot hold a
1.5B model. Deploy this file to a Hugging Face Space (Streamlit SDK) instead -
see GUIDE.md section 7.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, os.path.join(HERE, "..", "src")):
    if os.path.exists(os.path.join(_p, "agri_llm.py")):
        sys.path.insert(0, _p)
        break

import streamlit as st
from agri_llm import AgriLLM, DEFAULT_BASE_MODEL

st.set_page_config(page_title="Agri Q&A - LoRA/QLoRA demo", page_icon="\U0001F33E", layout="centered")


def _cfg(key, default=""):
    return os.environ.get(key) or (st.secrets.get(key, default) if hasattr(st, "secrets") else default)


BASE = _cfg("AGRI_BASE_MODEL", DEFAULT_BASE_MODEL)
ADAPTERS = json.loads(_cfg("AGRI_ADAPTERS", "{}") or "{}")
FOUR_BIT = str(_cfg("AGRI_4BIT", "0")) == "1"


@st.cache_resource(show_spinner="Loading model (first run downloads weights)...")
def load_llm():
    return AgriLLM(BASE, ADAPTERS, load_in_4bit=FOUR_BIT, hf_token=_cfg("HF_TOKEN", None) or None)


st.title("\U0001F33E Agricultural Q&A - fine-tuned Qwen2.5-1.5B")
st.caption("Research prototype - not a substitute for professional agricultural advice.")

llm = load_llm()
question = st.text_input("Ask a farming question", placeholder="How do I control aphids on cotton?")
facts_text = st.text_area("(Optional) verified facts, one per line - RAG-style answers", height=80)
choice = st.radio("Model", ["Compare all"] + llm.modes, horizontal=True)
max_new = st.slider("Max answer length (tokens)", 40, 300, 160, 20)

if st.button("Answer", type="primary", disabled=not question.strip()):
    facts = [l.strip(" -\u2022") for l in facts_text.splitlines() if l.strip()] or None
    for m in (llm.modes if choice == "Compare all" else [choice]):
        with st.spinner(f"Generating ({m})..."):
            ans = llm.generate(question, mode=m, facts=facts, max_new_tokens=max_new)
        st.subheader(m)
        st.write(ans)
