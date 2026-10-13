"""
inference.py - ask the base model and/or the fine-tuned adapters a question.

    python src/inference.py --question "How do I control aphids on cotton?" \
        --adapters lora=outputs/lora qlora=outputs/qlora --mode all
    python src/inference.py --question "..." --adapters qlora=user/agri-qa-qlora --mode qlora --4bit

--mode: base | <adapter name> | all.   (All modes share ONE base model - see agri_llm.py.)
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agri_llm import AgriLLM, DEFAULT_BASE_MODEL


def parse_adapters(items):
    out = {}
    for it in items or []:
        name, _, path = it.partition("=")
        if not path:
            raise SystemExit(f"--adapters expects name=path, got '{it}'")
        out[name] = path
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", required=True)
    ap.add_argument("--adapters", nargs="*", default=[], help="name=path_or_repo ...")
    ap.add_argument("--mode", default="all")
    ap.add_argument("--base_model", default=DEFAULT_BASE_MODEL)
    ap.add_argument("--max_new_tokens", type=int, default=160)
    ap.add_argument("--4bit", dest="four_bit", action="store_true")
    ap.add_argument("--facts", nargs="*", default=None, help="optional retrieved facts (RAG-style)")
    a = ap.parse_args()

    llm = AgriLLM(a.base_model, parse_adapters(a.adapters), load_in_4bit=a.four_bit)
    for m in (llm.modes if a.mode == "all" else [a.mode]):
        print(f"\n[{m}]\n{llm.generate(a.question, mode=m, facts=a.facts, max_new_tokens=a.max_new_tokens)}")
