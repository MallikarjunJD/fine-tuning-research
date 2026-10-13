"""Tests that run WITHOUT a GPU / torch / transformers.
Run:  python -m pytest -q tests/        (or)   python tests/test_agri_llm.py
"""
import os, sys, types, contextlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import agri_llm as A

# ---- 1. prompt parity with the exact training-time format --------------------
TRAIN_SYSTEM = ("You are an agricultural assistant. Answer the farmer's question "
                "clearly, correctly, and concisely, in plain English.")

def _training_text(q, a):   # copied from the training notebook (cell 14)
    return (f"<|im_start|>system\n{TRAIN_SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\n{q}<|im_end|>\n"
            f"<|im_start|>assistant\n{a}<|im_end|>")

def _training_prompt_only(q):
    return (f"<|im_start|>system\n{TRAIN_SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\n{q}<|im_end|>\n"
            f"<|im_start|>assistant\n")

def test_prompt_parity():
    q, a = "what is seed dormancy", "Seed dormancy is the condition in which a seed fails to germinate"
    assert A.SYSTEM_PROMPT == TRAIN_SYSTEM
    assert A.build_prompt(q) == _training_prompt_only(q)
    assert A.build_train_text(q, a) == _training_text(q, a)
    assert A.build_train_text(q, a).startswith(A.build_prompt(q))   # prompt is a strict prefix

def test_rag_prompt_contains_facts_and_question():
    p = A.build_prompt("How much neem oil?", facts=["Use 5 ml/L neem oil.", "  ", "Spray in the evening."])
    assert "- Use 5 ml/L neem oil." in p and "- Spray in the evening." in p
    assert "Question: How much neem oil?" in p and p.endswith("<|im_start|>assistant\n")
    assert p.count("- ") == 2                                        # blank fact dropped
    assert A.build_prompt("q", facts=[]) == A.build_prompt("q")      # empty facts -> plain prompt

# ---- 2. metrics --------------------------------------------------------------
def test_metrics():
    assert A.token_f1("Flint maize", "Flint maize") == 1.0
    assert A.token_f1("Dent maize", "Flint maize") == 0.5
    assert A.token_f1("", "x") == 0.0 and A.token_f1("", "") == 1.0
    assert A.contains_reference("The answer is Bean aphids (Aphis fabae).", "Bean aphids (Aphis fabae)")
    assert not A.contains_reference("Bean leaf beetle", "Bean aphids")
    assert A.repeat_ratio("a b c a b c a b c a b c") > 0.3 and A.repeat_ratio("one two three four") == 0.0
    assert "empty" in A.quality_flags("") and "repetitive" in A.quality_flags("a b c " * 10)
    assert "no_overlap_with_reference" in A.quality_flags("something else entirely", "Flint maize")

def test_score_predictions_without_optional_deps():
    s = A.score_predictions(["Flint maize", "x y"], ["Flint maize", "Dent maize"], use_bertscore=False)
    assert s["exact_match"] == 0.5 and abs(s["token_f1"] - 0.5) < 1e-9 and s["n"] == 2

def test_tfidf_retriever():
    r = A.TfidfRetriever(["neem oil controls aphids on cotton", "wheat is sown in november",
                          "clay soil holds water for maize"])
    assert r("how to control aphids", k=1) == ["neem oil controls aphids on cotton"]
    assert r("zzzz unrelated", k=3) == []                            # no overlap -> nothing returned

# ---- 3. mode switching / isolation logic with fake torch+transformers+peft ----
class _Vec:                                   # tiny stand-in for a logits tensor
    def __init__(s, v): s.v = list(v)
    def __sub__(s, o): return _Vec(a - b for a, b in zip(s.v, o.v))
    def abs(s): return _Vec(abs(a) for a in s.v)
    def max(s): return max(s.v)
    def __float__(s): return float(s.v)

class _Enc(dict):
    def to(self, d): return self

class _Arr:                                   # supports out[:, n:] and iteration
    def __init__(s, rows): s.rows = rows
    def __getitem__(s, idx):
        _, sl = idx; return _Arr([r[sl] for r in s.rows])

def _install_fakes(state):
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False, is_bf16_supported=lambda: False,
                                       empty_cache=lambda: None)
    torch.float32 = "float32"; torch.float16 = "float16"; torch.bfloat16 = "bfloat16"
    torch.inference_mode = lambda: contextlib.nullcontext()

    class Tok:
        pad_token = None; eos_token = "<eos>"; eos_token_id = 7; pad_token_id = 7; padding_side = "right"
        @classmethod
        def from_pretrained(cls, *a, **k): return cls()
        def convert_tokens_to_ids(self, t): return 9
        def __call__(self, prompts, return_tensors=None, padding=False):
            prompts = [prompts] if isinstance(prompts, str) else prompts
            state["prompts"].extend(prompts)
            return _Enc(input_ids=types.SimpleNamespace(shape=(len(prompts), 5)))
        def batch_decode(self, arr, skip_special_tokens=True):
            return [f"  ans[{state['active']}]  " for _ in arr.rows]

    class Base:
        @classmethod
        def from_pretrained(cls, *a, **k): return cls()
        def eval(self): return self
        def parameters(self): yield types.SimpleNamespace(device="cpu", numel=lambda: 10, element_size=lambda: 2)
    tf = types.ModuleType("transformers"); tf.AutoTokenizer = Tok; tf.AutoModelForCausalLM = Base

    class Peft(Base):
        def __init__(s): s.loaded = {}; s.cur = None; s.disabled = False
        @classmethod
        def from_pretrained(cls, model, path, adapter_name="default", **k):
            o = cls(); o.loaded[adapter_name] = path; o.cur = adapter_name
            state["wrap_calls"] += 1; return o
        def load_adapter(s, path, adapter_name, **k): s.loaded[adapter_name] = path
        def set_adapter(s, n): s.cur = n; state["active"] = n
        @contextlib.contextmanager
        def disable_adapter(s):
            state["active"] = "base"; s.disabled = True
            try: yield
            finally: s.disabled = False
        def generate(s, **kw): return _Arr([[1, 2, 3, 4, 5, 6, 7]] * kw_rows(kw))
    def kw_rows(kw): return kw["input_ids"].shape[0]
    peft = types.ModuleType("peft"); peft.PeftModel = Peft
    sys.modules.update(torch=torch, transformers=tf, peft=peft)

def test_isolated_modes_and_single_wrap():
    state = {"prompts": [], "active": None, "wrap_calls": 0}
    _install_fakes(state)
    llm = A.AgriLLM(adapters={"lora": "/x/lora", "qlora": "/x/qlora"})
    assert state["wrap_calls"] == 1, "base must be wrapped by PeftModel exactly ONCE"
    assert llm.modes == ["base", "lora", "qlora"]
    assert llm.model.loaded == {"lora": "/x/lora", "qlora": "/x/qlora"}
    assert llm.tokenizer.padding_side == "left" and llm.tokenizer.pad_token == "<eos>"
    assert set(llm.stop_ids) == {7, 9}
    assert llm.generate("q1", mode="base") == "ans[base]"
    assert llm.generate("q1", mode="lora") == "ans[lora]"
    assert llm.generate("q1", mode="qlora") == "ans[qlora]"
    assert llm.model.disabled is False                              # adapters re-enabled after "base"
    outs = llm.generate_batch(["a", "b", "c"], mode="qlora", batch_size=2)
    assert len(outs) == 3 and all(o == "ans[qlora]" for o in outs)  # batching + prompt-slicing
    try: llm.generate("q", mode="nope"); raise SystemExit("should have raised")
    except KeyError: pass
    try: llm.add_adapter("base", "/x"); raise SystemExit("should have raised")
    except ValueError: pass
    try: llm.add_adapter("lora", "/x"); raise SystemExit("should have raised")
    except ValueError: pass

def test_rag_arms_and_self_test_logic():
    state = {"prompts": [], "active": None, "wrap_calls": 0}
    _install_fakes(state)
    llm = A.AgriLLM(adapters={"lora": "/x/l", "qlora": "/x/q"})
    retr = A.TfidfRetriever(["neem oil controls aphids", "wheat sown november"])
    r = A.run_arm(llm, "D", "how to control aphids", retriever=retr, ft_adapter="qlora")
    assert r["mode"] == "qlora" and r["facts"] == ["neem oil controls aphids"]
    assert any("Facts:" in p and "neem oil controls aphids" in p for p in state["prompts"])
    assert A.run_arm(llm, "A", "q")["mode"] == "base" and A.run_arm(llm, "C", "q", ft_adapter="lora")["mode"] == "lora"
    try: A.run_arm(llm, "B", "q"); raise SystemExit("B without retriever must raise")
    except ValueError: pass
    # self_test: distinct logits -> pass ; identical logits (the user's bug) -> fail
    vecs = {"base": _Vec([0, 0]), "lora": _Vec([1, 0]), "qlora": _Vec([0, 2])}
    llm.logit_fingerprint = lambda q, m: vecs[m]
    assert llm.self_test()["passed"] is True
    vecs["lora"] = vecs["qlora"] = vecs["base"]                       # contaminated: all identical
    res = llm.self_test(); assert res["passed"] is False and all(d == 0 for d in res["diffs"].values())

if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("PASS", n)
