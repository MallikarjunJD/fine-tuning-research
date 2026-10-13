"""
app_gradio.py - Gradio demo for the agricultural Q&A fine-tunes.

Works on: Hugging Face Spaces (Gradio SDK), Kaggle/Colab (share=True), your laptop.

Configure with environment variables (no code edits needed):
  AGRI_ADAPTERS   JSON  {"lora": "<repo-or-folder>", "qlora": "<repo-or-folder>"}
  AGRI_BASE_MODEL base model id (default Qwen/Qwen2.5-1.5B-Instruct)
  AGRI_4BIT       "1" -> load base in 4-bit (needs an NVIDIA GPU + bitsandbytes;
                  ~1.2 GB of weights, fits a 4 GB laptop GPU)
  HF_TOKEN        only for private repos
Place this file next to agri_llm.py (or keep the repo layout: ../src/agri_llm.py).
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, os.path.join(HERE, "..", "src")):
    if os.path.exists(os.path.join(_p, "agri_llm.py")):
        sys.path.insert(0, _p)
        break

import gradio as gr
from agri_llm import AgriLLM, DEFAULT_BASE_MODEL

BASE = os.environ.get("AGRI_BASE_MODEL", DEFAULT_BASE_MODEL)
ADAPTERS = json.loads(os.environ.get("AGRI_ADAPTERS", "{}"))
FOUR_BIT = os.environ.get("AGRI_4BIT", "0") == "1"
DISCLAIMER = ("Research prototype - fine-tuned on community Q&A data, **not** a substitute "
              "for professional agricultural advice.")
EXAMPLES = ["How do I control aphids on cotton?", "Can maize grow in clay soil?",
            "Which pest feeds on the sap of bean plants and distorts the leaves?",
            "What causes shriveling and rotting of bean seeds?"]

_llm = None


def get_llm() -> AgriLLM:
    global _llm
    if _llm is None:
        _llm = AgriLLM(BASE, ADAPTERS, load_in_4bit=FOUR_BIT, hf_token=os.environ.get("HF_TOKEN"))
    return _llm


def parse_facts(text):
    lines = [l.strip(" -\u2022\t") for l in (text or "").splitlines()]
    return [l for l in lines if l] or None


def respond(question, mode, facts_text, max_new_tokens):
    if not question or not question.strip():
        return "Please type a question."
    llm = get_llm()
    facts = parse_facts(facts_text)
    modes = llm.modes if mode == "Compare all" else [mode]
    parts = []
    for m in modes:
        ans = llm.generate(question, mode=m, facts=facts, max_new_tokens=int(max_new_tokens))
        parts.append(f"### {m}\n{ans}")
    return "\n\n".join(parts)


def build_ui():
    choices = ["Compare all", "base"] + list(ADAPTERS.keys())
    with gr.Blocks(title="Agri Q&A - LoRA / QLoRA demo") as demo:
        gr.Markdown(f"# Agricultural Q&A - fine-tuned Qwen2.5-1.5B\n{DISCLAIMER}")
        q = gr.Textbox(label="Farming question", lines=2)
        mode = gr.Radio(choices, value=choices[0], label="Model")
        facts = gr.Textbox(label="(Optional) paste verified facts - one per line - to try RAG-style answers",
                           lines=3)
        n = gr.Slider(40, 300, value=160, step=20, label="Max answer length (tokens)")
        out = gr.Markdown()
        gr.Button("Answer", variant="primary").click(respond, [q, mode, facts, n], out)
        gr.Examples(EXAMPLES, inputs=q)
    return demo


if __name__ == "__main__":
    build_ui().queue().launch(share=os.environ.get("GRADIO_SHARE", "0") == "1")
