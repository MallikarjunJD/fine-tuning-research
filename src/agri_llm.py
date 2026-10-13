"""
agri_llm.py - inference layer for the agricultural Q&A fine-tunes (LoRA / QLoRA).

WHY THIS FILE EXISTS
--------------------
PEFT modifies the model object you hand it IN PLACE. If you do

    lora_model  = PeftModel.from_pretrained(base_model, lora_dir)
    qlora_model = PeftModel.from_pretrained(base_model, qlora_dir)   # same base_model!

then "base_model", "lora_model" and "qlora_model" are all views of ONE set of
layers, and the last adapter loaded wins for all three. Every model then gives
the same answers. This module avoids that by design:

    * ONE base model is loaded once.
    * Each adapter gets its own NAME inside that single PeftModel.
    * "base" = adapters switched off (peft's disable_adapter()).
    * A logit-level self-test proves the modes really differ.

It also holds the single source of truth for the prompt format, so training,
evaluation, the demo apps and the future RAG pipeline cannot drift apart.

torch / transformers / peft are imported lazily so the pure-text helpers can be
used (and unit-tested) on machines without a GPU stack.
"""
from __future__ import annotations

import contextlib
import gc
import os
import string
from collections import Counter
from typing import Callable, Dict, List, Optional, Sequence

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# MUST stay byte-identical to the system prompt used during training.
SYSTEM_PROMPT = (
    "You are an agricultural assistant. Answer the farmer's question "
    "clearly, correctly, and concisely, in plain English."
)
IM_START, IM_END = "<|im_start|>", "<|im_end|>"


# --------------------------------------------------------------------------
# 1. Prompt format (single source of truth)
# --------------------------------------------------------------------------
def build_prompt(question: str, facts: Optional[Sequence[str]] = None,
                 system: str = SYSTEM_PROMPT) -> str:
    """Prompt that ends with an OPEN assistant turn (what we feed at inference).

    With facts=None this is byte-identical to the format used in training.
    With facts, the retrieved facts are placed in the user turn (RAG).
    NOTE: the released adapters were trained WITHOUT facts in the prompt, so
    Arm D (fine-tuned + RAG) is only as good as the base model's ability to
    follow the facts block. Re-train on facts-in-prompt data to improve it.
    """
    user = question.strip()
    if facts:
        cleaned = [f.strip() for f in facts if f and f.strip()]
        if cleaned:
            block = "\n".join(f"- {f}" for f in cleaned)
            user = (
                "Use ONLY the verified facts below to answer. If they do not "
                "contain the answer, say you are not sure.\n\n"
                f"Facts:\n{block}\n\nQuestion: {question.strip()}"
            )
    return (
        f"{IM_START}system\n{system}{IM_END}\n"
        f"{IM_START}user\n{user}{IM_END}\n"
        f"{IM_START}assistant\n"
    )


def build_train_text(question: str, answer: str, facts: Optional[Sequence[str]] = None,
                     system: str = SYSTEM_PROMPT) -> str:
    """Full training example = prompt + answer + end-of-turn token."""
    return build_prompt(question, facts, system) + answer.strip() + IM_END


# --------------------------------------------------------------------------
# 2. Pure-text metrics / flags (no heavy dependencies)
# --------------------------------------------------------------------------
_PUNCT = str.maketrans("", "", string.punctuation)


def normalize_text(s: str) -> str:
    return " ".join(str(s).lower().translate(_PUNCT).split())


def token_f1(pred: str, ref: str) -> float:
    """SQuAD-style token F1. Good for short factoid answers like 'Flint maize'."""
    p, r = normalize_text(pred).split(), normalize_text(ref).split()
    if not p or not r:
        return float(p == r)
    overlap = sum((Counter(p) & Counter(r)).values())
    if overlap == 0:
        return 0.0
    prec, rec = overlap / len(p), overlap / len(r)
    return 2 * prec * rec / (prec + rec)


def contains_reference(pred: str, ref: str) -> bool:
    r = normalize_text(ref)
    return bool(r) and r in normalize_text(pred)


def repeat_ratio(text: str, n: int = 3) -> float:
    """Share of repeated word n-grams (0 = no repetition, ->1 = loops)."""
    w = text.split()
    if len(w) < n + 1:
        return 0.0
    grams = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def quality_flags(pred: str, ref: str = "") -> List[str]:
    flags = []
    if not pred.strip():
        flags.append("empty")
    elif len(pred.split()) < 3:
        flags.append("very_short")
    if repeat_ratio(pred) > 0.3:
        flags.append("repetitive")
    if ref and token_f1(pred, ref) == 0.0 and pred.strip():
        flags.append("no_overlap_with_reference")
    if ref and len(pred.split()) > 6 * max(1, len(ref.split())) and len(pred.split()) > 60:
        flags.append("much_longer_than_reference")
    return flags


def score_predictions(preds: Sequence[str], refs: Sequence[str],
                      use_bertscore: bool = True) -> Dict[str, object]:
    """token_f1 / ref_contained / exact_match / lengths always; ROUGE-L and
    BERTScore when the `evaluate` package is available."""
    import numpy as np

    out: Dict[str, object] = {"n": len(preds)}
    out["token_f1"] = float(np.mean([token_f1(p, r) for p, r in zip(preds, refs)]))
    out["ref_contained"] = float(np.mean([contains_reference(p, r) for p, r in zip(preds, refs)]))
    out["exact_match"] = float(np.mean([normalize_text(p) == normalize_text(r) for p, r in zip(preds, refs)]))
    out["avg_words_pred"] = float(np.mean([len(p.split()) for p in preds]))
    out["avg_words_ref"] = float(np.mean([len(r.split()) for r in refs]))
    try:
        import evaluate as hf_evaluate
        out["rouge_l"] = float(hf_evaluate.load("rouge").compute(
            predictions=list(preds), references=list(refs))["rougeL"])
    except Exception as e:  # keep going - other metrics are still valid
        out["rouge_l"], out["rouge_error"] = None, str(e)[:200]
    if use_bertscore:
        try:
            import evaluate as hf_evaluate
            b = hf_evaluate.load("bertscore").compute(
                predictions=list(preds), references=list(refs), lang="en")
            out["bertscore_f1"] = float(np.mean(b["f1"]))
        except Exception as e:
            out["bertscore_f1"], out["bertscore_error"] = None, str(e)[:200]
    return out


# --------------------------------------------------------------------------
# 3. Adapter integrity checks (does the saved adapter actually contain training?)
# --------------------------------------------------------------------------
def _load_adapter_tensors(adapter_dir: str):
    from safetensors.torch import load_file
    return load_file(os.path.join(adapter_dir, "adapter_model.safetensors"))


def inspect_adapter(adapter_dir: str) -> Dict[str, float]:
    """LoRA initialises B = 0, so an adapter whose B matrices are all ~0 was
    never trained. Trained adapters have clearly non-zero B."""
    import torch
    sd = _load_adapter_tensors(adapter_dir)
    A = [v.float() for k, v in sd.items() if "lora_A" in k]
    B = [v.float() for k, v in sd.items() if "lora_B" in k]
    return {
        "tensors": len(sd),
        "lora_B_tensors": len(B),
        "lora_B_mean_abs": float(torch.stack([t.abs().mean() for t in B]).mean()) if B else float("nan"),
        "lora_B_all_zero_tensors": int(sum(bool((t == 0).all()) for t in B)),
        "lora_A_mean_abs": float(torch.stack([t.abs().mean() for t in A]).mean()) if A else float("nan"),
    }


def compare_adapters(dir_a: str, dir_b: str) -> Dict[str, float]:
    """Relative L2 difference of the lora_B weights of two adapters.
    ~0 means the two adapters are (almost) the same weights."""
    import torch
    a, b = _load_adapter_tensors(dir_a), _load_adapter_tensors(dir_b)
    keys = sorted(k for k in a if "lora_B" in k and k in b)
    if not keys:
        return {"shared_B_tensors": 0, "rel_l2_diff_B": float("nan")}
    num = sum(float((a[k].float() - b[k].float()).pow(2).sum()) for k in keys)
    den = sum(float(a[k].float().pow(2).sum()) for k in keys) or 1e-12
    return {"shared_B_tensors": len(keys), "rel_l2_diff_B": (num / den) ** 0.5}


# --------------------------------------------------------------------------
# 4. The model wrapper
# --------------------------------------------------------------------------
class AgriLLM:
    """One base model + any number of NAMED adapters, with a clean `mode` switch.

        llm = AgriLLM(adapters={"lora": "/path/lora_final", "qlora": "/path/qlora_final"})
        llm.generate("How do I control aphids?", mode="base")
        llm.generate("How do I control aphids?", mode="qlora")
        llm.generate("...", mode="qlora", facts=["Spray neem oil 5 ml/L ..."])   # RAG

    mode == "base" switches adapters OFF; any other mode must be an adapter name.
    Adapters can be local folders OR Hugging Face repo ids.
    """

    def __init__(self, base_model_id: str = DEFAULT_BASE_MODEL,
                 adapters: Optional[Dict[str, str]] = None,
                 load_in_4bit: bool = False, dtype: Optional[str] = None,
                 tokenizer_id: Optional[str] = None, hf_token: Optional[str] = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        cuda = torch.cuda.is_available()
        if dtype is None:
            dtype = "bfloat16" if (cuda and torch.cuda.is_bf16_supported()) else ("float16" if cuda else "float32")
        self.torch_dtype = getattr(torch, dtype)
        self.base_model_id = base_model_id
        self.load_in_4bit = load_in_4bit

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id or base_model_id, token=hf_token)
        self.tokenizer.padding_side = "left"          # required for batched generation
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        stops = {self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids(IM_END)}
        self.stop_ids = [i for i in stops if i is not None and i >= 0]

        kwargs = {"token": hf_token}
        if cuda:
            kwargs["device_map"] = {"": 0}            # whole model on GPU 0 (no hidden sharding)
        if load_in_4bit:
            if not cuda:
                raise RuntimeError("4-bit loading needs a CUDA GPU (bitsandbytes).")
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=self.torch_dtype)
        else:
            kwargs["torch_dtype"] = self.torch_dtype
        base = AutoModelForCausalLM.from_pretrained(base_model_id, **kwargs)
        base.eval()

        self.model = base
        self.adapter_names: List[str] = []
        for name, path in (adapters or {}).items():
            self.add_adapter(name, path, hf_token=hf_token)
        self.model.eval()

    # ---- adapters ---------------------------------------------------------
    def add_adapter(self, name: str, path: str, hf_token: Optional[str] = None):
        from peft import PeftModel
        if name == "base":
            raise ValueError("'base' is reserved for 'adapters off'.")
        if name in self.adapter_names:
            raise ValueError(f"adapter '{name}' already loaded")
        if not self.adapter_names:
            # first adapter: wrap the base ONCE
            self.model = PeftModel.from_pretrained(self.model, path, adapter_name=name, token=hf_token)
        else:
            self.model.load_adapter(path, adapter_name=name, token=hf_token)
        self.adapter_names.append(name)

    @property
    def modes(self) -> List[str]:
        return ["base"] + self.adapter_names

    @property
    def device(self):
        return next(self.model.parameters()).device

    def model_size_gb(self) -> float:
        """Approx. weight memory (works for 4-bit too: packed bytes are counted)."""
        return sum(p.numel() * p.element_size() for p in self.model.parameters()) / 1e9

    def _activate(self, mode: Optional[str]):
        if mode in (None, "base"):
            return self.model.disable_adapter() if self.adapter_names else contextlib.nullcontext()
        if mode not in self.adapter_names:
            raise KeyError(f"unknown mode '{mode}'. Available: {self.modes}")
        self.model.set_adapter(mode)
        return contextlib.nullcontext()

    # ---- generation -------------------------------------------------------
    def generate_batch(self, questions: Sequence[str], mode: str = "base",
                       facts_list: Optional[Sequence[Optional[Sequence[str]]]] = None,
                       batch_size: int = 8, max_new_tokens: int = 160,
                       repetition_penalty: float = 1.1, do_sample: bool = False,
                       temperature: float = 0.7, top_p: float = 0.9) -> List[str]:
        import torch
        outs: List[str] = []
        gen = dict(max_new_tokens=max_new_tokens, repetition_penalty=repetition_penalty,
                   pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.stop_ids,
                   do_sample=do_sample)
        if do_sample:
            gen.update(temperature=temperature, top_p=top_p)
        with self._activate(mode), torch.inference_mode():
            for i in range(0, len(questions), batch_size):
                chunk = list(questions[i:i + batch_size])
                prompts = [build_prompt(q, facts_list[i + j] if facts_list else None)
                           for j, q in enumerate(chunk)]
                enc = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
                out = self.model.generate(**enc, **gen)
                new_tokens = out[:, enc["input_ids"].shape[1]:]      # drop the prompt
                outs.extend(t.strip() for t in self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
        return outs

    def generate(self, question: str, mode: str = "base",
                 facts: Optional[Sequence[str]] = None, **kw) -> str:
        return self.generate_batch([question], mode=mode,
                                   facts_list=[facts] if facts else None, batch_size=1, **kw)[0]

    # ---- proof that modes are really different -----------------------------
    def logit_fingerprint(self, question: str, mode: str = "base"):
        import torch
        enc = self.tokenizer(build_prompt(question), return_tensors="pt").to(self.device)
        with self._activate(mode), torch.inference_mode():
            return self.model(**enc).logits[0, -1].float().cpu()

    def self_test(self, question: str = "How do I control aphids on cotton?",
                  min_diff: float = 1e-2) -> Dict[str, object]:
        """Max |logit difference| between every pair of modes on one prompt.
        If any pair is ~0 the modes are NOT isolated (the bug this module prevents)."""
        fps = {m: self.logit_fingerprint(question, m) for m in self.modes}
        diffs, names = {}, self.modes
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                diffs[f"{a} vs {b}"] = float((fps[a] - fps[b]).abs().max())
        return {"diffs": diffs, "passed": all(d > min_diff for d in diffs.values()),
                "threshold": min_diff}

    def unload(self):
        import torch
        self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# --------------------------------------------------------------------------
# 5. RAG hooks (the seam where the future retriever plugs in)
# --------------------------------------------------------------------------
ARMS = {
    "A": {"mode": "base", "use_rag": False, "desc": "Base LLM"},
    "B": {"mode": "base", "use_rag": True,  "desc": "Base LLM + RAG"},
    "C": {"mode": "ft",   "use_rag": False, "desc": "Fine-tuned"},
    "D": {"mode": "ft",   "use_rag": True,  "desc": "Fine-tuned + RAG"},
    # "E" (hybrid: + rerank + context filter + freshness) is built in the RAG phase.
}

Retriever = Callable[..., List[str]]   # retriever(question, k=3) -> list of fact strings


def run_arm(llm: AgriLLM, arm: str, question: str, retriever: Optional[Retriever] = None,
            ft_adapter: str = "qlora", top_k: int = 3, **gen_kwargs) -> Dict[str, object]:
    cfg = ARMS[arm]
    facts = None
    if cfg["use_rag"]:
        if retriever is None:
            raise ValueError(f"arm {arm} needs a retriever")
        facts = retriever(question, k=top_k)
    mode = ft_adapter if cfg["mode"] == "ft" else "base"
    return {"arm": arm, "desc": cfg["desc"], "mode": mode, "question": question,
            "facts": facts, "answer": llm.generate(question, mode=mode, facts=facts, **gen_kwargs)}


class TfidfRetriever:
    """Tiny stand-in retriever for testing the plumbing. Replace with the
    FAISS + BGE-M3 hybrid retriever later - keep the call signature."""

    def __init__(self, docs: Sequence[str]):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.docs = list(docs)
        self.vec = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2))
        self.mat = self.vec.fit_transform(self.docs)

    def __call__(self, query: str, k: int = 3) -> List[str]:
        import numpy as np
        sims = (self.mat @ self.vec.transform([query]).T).toarray().ravel()
        idx = np.argsort(-sims)[:k]
        return [self.docs[i] for i in idx if sims[i] > 0]


# --------------------------------------------------------------------------
# 6. Export / publish helpers
# --------------------------------------------------------------------------
def export_merged(adapter_dir: str, out_dir: str, base_model_id: str = DEFAULT_BASE_MODEL,
                  dtype: str = "float16") -> str:
    """Bake an adapter into a standalone model folder (no peft needed to serve it).
    Loads its OWN fresh base on CPU, so it never touches a live AgriLLM."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    base = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype=getattr(torch, dtype))
    merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
    os.makedirs(out_dir, exist_ok=True)
    merged.save_pretrained(out_dir, safe_serialization=True)
    AutoTokenizer.from_pretrained(base_model_id).save_pretrained(out_dir)
    return out_dir


def push_folder_to_hub(folder: str, repo_id: str, token: Optional[str] = None,
                       private: bool = False, repo_type: str = "model") -> str:
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type=repo_type, private=private, exist_ok=True)
    api.upload_folder(folder_path=folder, repo_id=repo_id, repo_type=repo_type)
    return f"https://huggingface.co/{repo_id}"
