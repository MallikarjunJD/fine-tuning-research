"""
evaluate.py - fair, isolated evaluation of base vs LoRA vs QLoRA on the held-out test split.

    python src/evaluate.py --adapters lora=outputs/lora qlora=outputs/qlora --n 200

What makes it fair:
  * ONE base model, adapters switched by name (no shared-object contamination).
  * The self-test must pass first (modes must produce different logits).
  * Every mode uses the same prompt builder, decoding settings and test questions.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agri_llm import AgriLLM, score_predictions, quality_flags
from data import build_splits, load_config
from inference import parse_adapters

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", nargs="*", default=[], help="name=path_or_repo ...")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--no_bertscore", action="store_true")
    ap.add_argument("--out", default="results/eval_results.json")
    a = ap.parse_args()

    cfg = load_config()
    test = build_splits(cfg)["test"].select(range(a.n))
    questions, refs = list(test["question"]), list(test["answer"])

    llm = AgriLLM(cfg["model"]["base_model_id"], parse_adapters(a.adapters))
    st = llm.self_test()
    print("self-test:", st)
    if not st["passed"]:
        raise SystemExit("Modes are NOT isolated (identical logits). Refusing to evaluate.")

    g = cfg["generation"]
    results, samples = {}, {}
    for m in llm.modes:
        preds = llm.generate_batch(questions, mode=m, max_new_tokens=g["max_new_tokens"],
                                   repetition_penalty=g["repetition_penalty"])
        results[m] = score_predictions(preds, refs, use_bertscore=not a.no_bertscore)
        samples[m] = [{"q": q, "ref": r, "pred": p, "flags": quality_flags(p, r)}
                      for q, r, p in zip(questions, refs, preds)][:20]
        print(m, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in results[m].items()})

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"results": results, "samples": samples}, f, indent=2)
    print("saved", a.out)
