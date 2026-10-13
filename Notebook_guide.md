# Notebook Guide — LoRA vs QLoRA Fine-Tuning of Qwen2.5-1.5B-Instruct on Agricultural Q&A

This is the main learning artifact of the project. Part 1 walks through the
Kaggle notebook section by section with code and short explanations. Part 2
is the deep dive: the theory you should be able to explain in an interview,
independent of this specific codebase.

Legend used throughout:
> **Interview takeaway:** the one sentence you should be able to say out loud.
> **Engineering decision:** why we chose X over Y, explicitly.
> **Common mistake:** what beginners get wrong here.
> **Deep dive:** optional deeper theory.
> **Reality check:** an honest limitation or gotcha.

---

# PART 1 — NOTEBOOK CODE + EXPLANATION

## 1. Project Introduction

**Objective:** determine whether LoRA/QLoRA fine-tuning measurably improves
a small instruction-tuned LLM's answers on agricultural Q&A, and what each
method costs in memory, time, and adapter size.

**Research questions:**
1. Does fine-tuning improve answer quality (ROUGE-L, BERTScore) over the base model?
2. How much does QLoRA reduce GPU memory / adapter size vs LoRA, and at what cost?
3. How does LoRA rank affect trainable parameters and performance?
4. What failures remain, and are they fine-tuning problems or knowledge problems (→ RAG)?

**Task:** open-ended free-text question answering in the agricultural domain.
**Model:** `Qwen/Qwen2.5-1.5B-Instruct`.
**Dataset:** `KisanVaani/agriculture-qa-english-only` (~22.6k QA pairs; 4k/500/500 train/val/test subsample used here).
**Expected result:** fine-tuned models should score higher on ROUGE-L/BERTScore than the base model on in-distribution questions, with QLoRA using substantially less peak GPU memory than LoRA at a small, measurable quality cost. **(This is a hypothesis, not yet a measured result — see `results/`.)**

---

## 2. Environment Setup

```python
# Kaggle ships torch + CUDA pre-installed — do NOT reinstall torch, it can break the CUDA build.
!pip install -q -U transformers datasets peft trl accelerate bitsandbytes evaluate bert_score sentencepiece
```

- `transformers`: model/tokenizer loading, generation.
- `datasets`: streaming/loading the Hub dataset, map/filter/split.
- `peft`: LoRA adapter construction (`LoraConfig`, `get_peft_model`, `prepare_model_for_kbit_training`).
- `trl`: `SFTTrainer`/`SFTConfig` — a thin, well-tested wrapper around `transformers.Trainer` built for instruction fine-tuning.
- `accelerate`: device placement / mixed precision plumbing used internally by `Trainer`.
- `bitsandbytes`: the 4-bit NF4 quantization kernels QLoRA depends on.
- `evaluate` + `bert_score`: ROUGE and BERTScore metric implementations.

> **Engineering decision:** we use the modern `SFTTrainer(model, train_dataset, peft_config=...)` calling convention (current as of the TRL documentation verified for this project), not the older tutorials' `tokenizer=` keyword or manual `Trainer` + `DataCollatorForLanguageModeling` setup. Older tutorials online are a common source of `TypeError: unexpected keyword argument` — always check `trl`'s installed version against its current docs before copying a snippet.

## 3. GPU Verification

```python
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))
    print("VRAM (GB):", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))
    print("bf16 supported:", torch.cuda.is_bf16_supported())
```

- **CUDA** is NVIDIA's parallel-compute platform; PyTorch dispatches tensor ops to it instead of the CPU.
- **VRAM** (GPU memory) is the hard ceiling on how large a model, batch, and sequence length you can train — see the Part 2-F memory deep dive; it is usually the binding constraint before "compute speed" is.

## 4. Reproducibility

```python
import random, numpy as np, torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
```

Reproducibility matters because without a fixed seed you cannot tell whether
a performance difference between LoRA and QLoRA is real or just sampling
noise from a different train/val split or weight initialization order.

## 5. Dataset Loading

```python
from datasets import load_dataset

raw = load_dataset("KisanVaani/agriculture-qa-english-only", split="train")
print(raw)
print(raw[0])
```

Inspect size, fields, and a few examples before touching the data — never
assume a dataset card's description matches the actual columns.

```python
import pandas as pd
df = raw.to_pandas()
print(df.shape)
print(df.isna().sum())
print(df.sample(5, random_state=SEED))
```

## 6. Dataset Preprocessing

```python
def clean(example):
    return {
        "question": str(example["question"]).strip(),
        "answer": str(example["answers"]).strip(),
    }

cleaned = raw.map(clean)
cleaned = cleaned.filter(
    lambda ex: len(ex["question"]) > 5 and len(ex["answer"]) > 5
               and ex["question"].lower() != ex["answer"].lower()
)
cleaned = cleaned.shuffle(seed=SEED)

train_ds = cleaned.select(range(0, 4000))
val_ds   = cleaned.select(range(4000, 4500))
test_ds  = cleaned.select(range(4500, 5000))
```

> **Common mistake:** shuffling *after* splitting (or splitting before
> dedup/cleaning) can leak near-duplicate rows across splits. Here we clean
> and dedupe-filter first, shuffle once with a fixed seed, then slice — so
> the same row can never land in two splits.

**Leakage prevention:** the 20 examples used later for the qualitative
base-vs-fine-tuned table are drawn from `test_ds`, never from `train_ds`.

## 7. Prompt / Instruction Formatting

```python
SYSTEM = ("You are an agricultural assistant. Answer the farmer's question "
          "clearly, correctly, and concisely, in plain English.")

def to_train_text(example):
    text = (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\n{example['question']}<|im_end|>\n"
            f"<|im_start|>assistant\n{example['answer']}<|im_end|>")
    return {"text": text}

def to_prompt_only(question):
    return (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\n{question}<|im_end|>\n"
            f"<|im_start|>assistant\n")

train_sft = train_ds.map(to_train_text)
val_sft   = val_ds.map(to_train_text)
```

- **Instruction:** the system message, fixed across all examples.
- **Input:** the farmer's question.
- **Response:** the answer, present during training, withheld (generated) at inference.
- **Labels:** `SFTTrainer` computes the language-modeling loss over the whole
  formatted string by default; because we always use the same fixed system
  prompt and the answer is short relative to the prompt, we accept this
  rather than masking the prompt tokens out of the loss — see Part 2-B.

## 8. Base Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    device_map="auto",
)
print(base_model.config)
print("Params:", sum(p.numel() for p in base_model.parameters()) / 1e6, "M")
```

- **Architecture:** decoder-only Transformer (Qwen2 architecture), grouped-query attention, RMSNorm, SwiGLU MLPs.
- **Parameter count:** ~1.5B.
- **Context length:** up to 32,768 tokens natively (we cap sequences at 512 tokens here to keep training fast; the model is not stressed near its real limit).
- **Tokenizer:** Qwen's byte-level BPE tokenizer, shared with the ChatML-style `<|im_start|>`/`<|im_end|>` chat template.
- **Why this model:** see Decision Log below.

## 9. BASELINE (before any fine-tuning)

```python
@torch.no_grad()
def generate(model, question, max_new_tokens=200):
    prompt = to_prompt_only(question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                          do_sample=False, repetition_penalty=1.1,
                          pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

for ex in test_ds.select(range(3)):
    print("Q:", ex["question"])
    print("Base model:", generate(base_model, ex["question"]))
    print("Reference :", ex["answer"])
    print("---")
```

> **Interview takeaway:** "We need a baseline because otherwise we cannot
> determine whether fine-tuning actually helped — an improvement is only
> meaningful relative to a measured starting point, not an assumed one."

## 10. Tokenization

```python
sample = train_sft[0]["text"]
tokens = tokenizer.tokenize(sample)
ids = tokenizer(sample)["input_ids"]
print("First 20 tokens:", tokens[:20])
print("First 20 ids:", ids[:20])
print("Sequence length:", len(ids))

lengths = [len(tokenizer(t)["input_ids"]) for t in train_sft.select(range(200))["text"]]
import numpy as np
print("mean/median/95th pct length:", np.mean(lengths), np.median(lengths), np.percentile(lengths, 95))
```

Padding pads shorter sequences up to the batch's max length so tensors are
rectangular; truncation cuts sequences longer than `max_seq_length` (512
here, chosen from the 95th-percentile check above) so a single long outlier
doesn't blow up memory for the whole batch.

## 11. LoRA

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

lora_model = get_peft_model(base_model, lora_config)
lora_model.print_trainable_parameters()
```

Typical output shape: `trainable params: ~4.5M || all params: ~1.55B || trainable%: ~0.29%`
(exact numbers depend on the installed model revision — read them from your
own run rather than trusting a number typed in a guide).

- **r (rank):** the bottleneck dimension of the low-rank update; higher r = more capacity, more trainable parameters.
- **alpha:** a scaling factor applied to the LoRA update (`scaling = alpha / r`); keeps the update's effective magnitude stable as r changes.
- **dropout:** regularizes the adapter path only, not the frozen base.
- **target_modules:** which linear layers get an adapter attached — here, the four attention projections, a common and cheap default; MLP projections could be added for more capacity at more parameter cost.

> **Interview takeaway:** freezing the base model and training only the
> adapters means we do NOT need to store gradients or optimizer states for
> ~99.7% of the parameters — this, not the adapter's small file size, is
> where most of LoRA's memory savings actually come from.

## 12. QLoRA

```python
from transformers import BitsAndBytesConfig
from peft import prepare_model_for_kbit_training

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

qlora_base = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, quantization_config=bnb_config, device_map="auto"
)
qlora_base = prepare_model_for_kbit_training(qlora_base)

qlora_model = get_peft_model(qlora_base, lora_config)  # same LoraConfig as above
qlora_model.print_trainable_parameters()
```

QLoRA is **not** "LoRA but better" — it is the same LoRA adapter design
applied on top of a base model whose *frozen* weights are stored in 4-bit
NF4 instead of bf16/fp32. What changes:
- **Quantization:** each weight is mapped to one of 16 representable 4-bit
  levels using the NF4 (NormalFloat4) codebook, which is tuned for
  normally-distributed weights (typical of trained neural nets), not a naive
  uniform 4-bit grid.
- **Double quantization:** the quantization *constants* themselves are
  further quantized, saving a small additional amount of memory per weight.
- **Compute dtype:** matmuls are de-quantized on the fly to `bfloat16` for
  the actual forward/backward compute; the 4-bit storage is a memory
  optimization, not a computation format.
- **Frozen quantized weights + trainable LoRA adapters:** the adapters
  themselves are still stored and trained in full precision (or bf16) —
  quantization is applied only to the frozen base.

> **Common mistake:** saying "QLoRA trains a 4-bit model." It doesn't — it
> keeps a 4-bit *frozen* base and trains full-precision adapters on top.

## 13. Training Configuration

```python
from trl import SFTConfig, SFTTrainer

sft_config = SFTConfig(
    output_dir="outputs/lora",          # or outputs/qlora
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,       # effective batch size = 16
    learning_rate=2e-4,                  # ~10x a typical full-FT LR — PEFT convention
    warmup_ratio=0.03,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
    save_total_limit=2,
    bf16=torch.cuda.is_bf16_supported(),
    optim="paged_adamw_8bit",            # QLoRA run only; use "adamw_torch" for plain LoRA
    report_to="none",
    dataset_text_field="text",
    max_length=512,
    packing=False,
    seed=SEED,
)
```

| Parameter | Too low | Too high | Why we picked ours |
|---|---|---|---|
| learning_rate | underfits, loss barely moves | loss spikes/diverges, adapters overwrite useful base behavior | 2e-4 is the standard PEFT starting point (~10x full-FT LR since only a tiny parameter subset is updated) |
| batch_size / grad_accum | noisy gradients, slow | out-of-memory | 4×4=16 fits Kaggle's ~16GB comfortably at seq len 512 |
| epochs | underfits | overfits/memorizes | 3 epochs on 4k examples is enough to see a clear before/after without heavy memorization |
| warmup_ratio | unstable early steps | wastes early training budget | 3% is a standard light warmup for short runs |
| weight_decay | no regularization | can suppress the adapter's useful signal since it's already tiny | 0.01 is a mild, standard default |
| max_length | truncates real answers | wastes memory on padding | set from the measured 95th-percentile token length in Section 10 |

## 14. Training

```python
trainer = SFTTrainer(
    model=lora_model,          # or qlora_model
    args=sft_config,
    train_dataset=train_sft,
    eval_dataset=val_sft,
    peft_config=None,          # already applied via get_peft_model above
    processing_class=tokenizer,
)

# REQUIRED for the QLoRA run specifically (harmless no-op for plain LoRA):
# SFTTrainer detects `isinstance(model, PeftModel)` and internally re-runs
# peft.prepare_model_for_kbit_training() on quantized models — see the
# "Common mistake" callout below for exactly why. That re-freezes every
# parameter, including the adapters, and nothing re-enables them since
# peft_config=None. Restore it explicitly before training:
for name, param in trainer.model.named_parameters():
    if "lora_" in name:
        param.requires_grad = True
trainer.model.print_trainable_parameters()  # confirm it matches Section 11/12's count

trainer.train()
```

> **Common mistake:** passing an already-PEFT-wrapped model to `SFTTrainer`
> without also passing `peft_config` looks harmless — and for plain LoRA it
> is. For QLoRA it silently breaks training. `SFTTrainer.__init__` checks
> `isinstance(model, PeftModel)` regardless of whether `peft_config` is also
> given; for a quantized model, that triggers an internal call to
> `peft.prepare_model_for_kbit_training()` on the model a *second* time.
> That function's first action is `for p in model.parameters(): p.requires_grad
> = False` — it freezes everything, including your adapters, and because
> `peft_config=None` here, nothing re-enables them afterward. The result:
> training runs with zero trainable parameters, silently — no error, just a
> validation loss that never moves. This is version-specific TRL internal
> behavior (verified by reading `trl/trainer/sft_trainer.py`'s
> `_prepare_peft_model`), not something obvious from the public API, which
> is exactly why it's worth knowing about rather than just patching around.

> **Engineering decision:** we call `get_peft_model` ourselves and pass the
> already-wrapped model, rather than passing `peft_config=lora_config`
> directly to `SFTTrainer`. Both work *for LoRA*; for QLoRA, calling
> `get_peft_model` yourself requires the explicit `requires_grad` restore
> above. The alternative — passing the quantized base model plus
> `peft_config=qlora_lora_config` and letting `SFTTrainer` build the
> `PeftModel` itself — sidesteps this entirely, at the cost of not being
> able to call `print_trainable_parameters()` before training starts.
> Wrapping explicitly lets us call
> `print_trainable_parameters()` and inspect the adapter *before* training
> starts, which is worth the extra line for a learning-focused project.

Watch training loss trend downward and eval loss track it without
diverging upward (a sign of overfitting on this small dataset).

## 15. Training Curves

```python
import matplotlib.pyplot as plt

history = trainer.state.log_history
train_loss = [(h["step"], h["loss"]) for h in history if "loss" in h]
eval_loss = [(h["step"], h["eval_loss"]) for h in history if "eval_loss" in h]

plt.plot(*zip(*train_loss), label="train loss")
plt.plot(*zip(*eval_loss), label="eval loss")
plt.xlabel("step"); plt.ylabel("loss"); plt.legend(); plt.title("LoRA training curve")
plt.show()
```

- **Underfitting:** both curves stay high and flat.
- **Overfitting:** train loss keeps falling while eval loss flattens or rises.
- **Convergence:** both curves fall together and plateau.

## 16. Fine-Tuned Inference

```python
from peft import PeftModel

# Reload cleanly to make sure we're testing the saved artifact, not the live training object
inference_base = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")
lora_inference_model = PeftModel.from_pretrained(inference_base, "outputs/lora")

for ex in test_ds.select(range(5, 8)):   # examples NOT used in training
    print("Q:", ex["question"])
    print("Fine-tuned:", generate(lora_inference_model, ex["question"]))
    print("Reference :", ex["answer"])
    print("---")
```

## 17. Evaluation

```python
import evaluate as hf_evaluate

rouge = hf_evaluate.load("rouge")
bertscore = hf_evaluate.load("bertscore")

def evaluate_model(model, dataset):
    preds = [generate(model, ex["question"]) for ex in dataset]
    refs = [ex["answer"] for ex in dataset]
    r = rouge.compute(predictions=preds, references=refs)
    b = bertscore.compute(predictions=preds, references=refs, lang="en")
    return {
        "rouge_l": r["rougeL"],
        "bertscore_f1": sum(b["f1"]) / len(b["f1"]),
        "exact_match_rate": sum(p.strip().lower() == r_.strip().lower() for p, r_ in zip(preds, refs)) / len(preds),
    }

base_scores = evaluate_model(base_model, test_ds)
lora_scores = evaluate_model(lora_inference_model, test_ds)
print("Base:", base_scores)
print("LoRA:", lora_scores)
```

**Why ROUGE-L and BERTScore, not BLEU:** this is open-ended free-text QA
with many valid phrasings of a correct answer. BLEU's n-gram precision
penalizes valid paraphrases too harshly and rewards verbatim overlap; ROUGE-L
(longest common subsequence) tolerates reordering better, and BERTScore
captures semantic similarity even when the wording is completely different.
Exact match is reported only as a diagnostic — near-0 is expected and fine
for this task; near-100% would actually be a red flag for memorization.

## 18. Base vs Fine-Tuned Comparison

```python
import pandas as pd

rows = []
for ex in test_ds.select(range(10)):
    rows.append({
        "question": ex["question"],
        "base_model": generate(base_model, ex["question"]),
        "fine_tuned": generate(lora_inference_model, ex["question"]),
        "reference": ex["answer"],
    })
comparison_df = pd.DataFrame(rows)
comparison_df
```

Populate `results/fine_tuned_results.md` with the real rows this prints —
do not hand-write example outputs.

## 19. LoRA vs QLoRA Comparison

```python
import time

def measure(mode, model_fn):
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    model = model_fn()
    trainer = SFTTrainer(model=model, args=sft_config, train_dataset=train_sft, eval_dataset=val_sft, processing_class=tokenizer)
    trainer.train()
    elapsed = time.time() - t0
    peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9
    return {"mode": mode, "peak_mem_gb": peak_mem_gb, "train_seconds": elapsed}

# Run once per mode with matching sft_config.output_dir, then tabulate results
```

Compare: peak GPU memory, wall-clock training time, trainable-parameter
count (from Section 11/12 printouts), adapter file size on disk
(`du -sh outputs/lora outputs/qlora`), and the quantitative scores from
Section 17. If both are implemented on the same data/seed/steps, the
comparison is fair; state explicitly if any of those was not held constant.

## 20. Ablation Experiment: LoRA Rank

```python
results = {}
for r in [4, 16, 64]:
    cfg = LoraConfig(r=r, lora_alpha=r * 2, lora_dropout=0.05, bias="none",
                      task_type="CAUSAL_LM", target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    model_r = get_peft_model(
        AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"),
        cfg,
    )
    ablation_config = sft_config  # same config, output_dir changed per run, num_train_epochs=1
    trainer_r = SFTTrainer(model=model_r, args=ablation_config, train_dataset=train_sft, eval_dataset=val_sft, processing_class=tokenizer)
    trainer_r.train()
    results[r] = evaluate_model(model_r, test_ds.select(range(100)))
print(results)
```

Rank was chosen over dataset-size/epoch/LR ablations because it's the
cheapest axis to vary fairly at fixed compute — same data, same steps — while
answering a genuinely useful engineering question: how much adapter capacity
does this task need?

## 21. Failure Analysis (Required)

```python
for ex in test_ds.select(range(20)):
    pred = generate(lora_inference_model, ex["question"])
    # Flag candidates for manual review: very short, very long, or off-topic answers
    if len(pred) < 15 or len(pred) > 600:
        print("FLAGGED:", ex["question"], "->", pred)
```

Manually review flagged and a few random non-flagged examples for:
hallucinated specifics (invented dosages/dates), verbatim echoes of training
answers on similar-but-not-identical questions (overfitting), malformed or
truncated output, and degraded quality on crops/regions underrepresented in
the training subsample. Record real examples in `results/fine_tuned_results.md`
along with a hypothesis for each failure — do not invent examples here.

## 22. Final Conclusions

Answer each research question from Section 1 using the actual numbers from
Sections 17-20 once they exist. Until then, this section in the notebook
should explicitly read **"pending experiment results"** rather than a
pre-written conclusion — writing the conclusion before the data exists is
exactly the kind of fabrication this project is designed to avoid.

---

# PART 2 — DEEP DIVE

## A. What Is Fine-Tuning?

```text
Pre-training              → learn general language structure/knowledge from massive unlabeled text
Instruction tuning        → teach the model to follow instructions / hold a conversation format
Fine-tuning               → adapt weights further toward a specific task/domain using labeled examples
Parameter-efficient FT    → do the above by updating a small added parameter subset, not all weights
Prompting                 → no weight updates at all; shape behavior purely via the input text
RAG                       → no weight updates; supply relevant external documents at inference time
```

Fine-tuning sits between "prompting" (cheap, no persistence, limited by
context window) and "training from scratch" (extremely expensive, full
control). It changes the model's actual weights, so the adaptation persists
without needing to re-supply instructions every time.

## B. What Actually Changes During Fine-Tuning?

For each training batch: a forward pass computes predicted next-token
probabilities; a loss (cross-entropy against the true next tokens) measures
how wrong those predictions were; backpropagation computes the gradient of
that loss with respect to every trainable parameter; the optimizer (AdamW
here) uses those gradients (plus its own running moment estimates) to update
the trainable parameters by a small step scaled by the learning rate. In
full fine-tuning, "trainable parameters" means the entire model. In LoRA, it
means only the injected low-rank matrices — the base weights receive no
gradient updates at all (they're marked `requires_grad=False`).

## C. Why Not Fine-Tune Every Parameter?

- **Computational cost:** gradients must be computed and stored for every
  trainable parameter — for a 1.5B model, that's 1.5B extra tensors just for
  gradients, in addition to the weights themselves.
- **VRAM:** AdamW keeps two extra floating-point states per trainable
  parameter (first and second moment estimates) — full fine-tuning of a 1.5B
  model in fp32 can require on the order of 20+ GB of VRAM just for
  weights+gradients+optimizer states, before activations.
- **Catastrophic forgetting:** updating all weights on a small, narrow
  dataset can overwrite general capabilities the base model had (broad
  language understanding, other domains) in favor of overfitting the narrow
  target task.
- **Storage:** a full fine-tuned checkpoint is the same size as the entire
  base model (several GB); a LoRA adapter is typically tens of MB.

## D. LoRA Deeply

The core idea: instead of directly updating a weight matrix `W` (shape
`d × k`), freeze `W` and learn an additive low-rank update:

```text
W' = W + ΔW
ΔW = BA
```

where `B` is `d × r`, `A` is `r × k`, and `r ≪ min(d, k)`. Why this works:
empirically, the *useful* update direction for adapting a pre-trained model
to a new task tends to have a low "intrinsic rank" — most of the update's
effect can be captured by a much lower-dimensional subspace than the full
weight matrix. Instead of learning `d × k` new numbers, you learn
`r × (d + k)` numbers, which for `r=16` on a matrix like `d=k=1536` is a
~48x reduction in trainable parameters for that matrix alone.

- **rank (r):** the bottleneck width of `B`/`A`; larger r = more expressive updates, more parameters, more VRAM.
- **alpha:** the update is scaled by `alpha / r` before being added to `W`, so `alpha` controls the *magnitude* of the adapter's contribution independent of `r`.
- **target_modules:** which weight matrices get a `BA` adapter attached — commonly attention projections, sometimes also MLP projections for more capacity.
- **dropout:** applied to the adapter's input during training only, regularizing the low-rank path specifically.

**Small numerical example:** if `W` is 4×4 and we pick `r=1`, `ΔW = BA` where
`B` is 4×1 and `A` is 1×4 — instead of learning 16 independent numbers for
`ΔW`, you learn 4 + 4 = 8 numbers, and every column of `ΔW` is forced to be a
scalar multiple of the same underlying direction (since `ΔW`'s rank is at
most 1). That constraint is exactly the "low-rank" assumption — it's a
strong inductive bias, and increasing `r` relaxes it.

## E. QLoRA Deeply

QLoRA = LoRA fine-tuning applied on top of a **quantized frozen base**.
Precision ladder: `fp32` (32 bits/weight, training-grade precision) →
`fp16`/`bf16` (16 bits, standard inference/training precision, half the
memory of fp32) → `int8` (8 bits, a coarser but usable inference format) →
`4-bit NF4` (QLoRA's format, roughly 1/8th the memory of fp32 per weight).
NF4 (NormalFloat4) is a 4-bit format whose 16 representable values are
spaced to match the distribution of typical trained neural network weights
(approximately normal, concentrated near zero), rather than a naive uniform
4-bit grid — this preserves more information per bit for realistic weight
distributions. **Double quantization** additionally quantizes the per-block
scaling constants used during 4-bit quantization, saving a further small
amount of memory per parameter. The **compute dtype** (bf16 here) is the
precision used when the 4-bit weights are de-quantized on the fly for the
actual matrix multiplication — the model is never *computed* in 4-bit, only
*stored* in 4-bit. The LoRA adapters themselves remain in full/half
precision and are the only thing actually trained. This combination is what
makes it possible to fine-tune models that wouldn't otherwise fit in a given
GPU's memory at all, at some (small, task-dependent) cost in achievable
quality versus a full-precision base.

## F. Memory Analysis

GPU memory during training is spent on:

```text
Model weights        — frozen (LoRA) or quantized+frozen (QLoRA), or full (full FT)
Gradients            — one per TRAINABLE parameter only
Optimizer states     — AdamW keeps ~2 extra values per trainable parameter
Activations          — intermediate forward-pass tensors kept for backprop
KV cache             — relevant at generation/inference time, not training
Temporary tensors    — quantization/dequantization buffers, CUDA workspace, etc.
```

> **Reality check:** "the model is only ~1 GB in 4-bit" does **not** mean
> "I only need 1 GB of VRAM." Training also needs memory for activations
> (which scale with batch size × sequence length × hidden size × number of
> layers) and, for the trainable adapter parameters, gradients and optimizer
> states. This is precisely why a 4 GB local laptop GPU can plausibly run
> **inference** with a 4-bit 1.5B model but cannot realistically **train**
> it, even with LoRA — the activation and optimizer-state overhead alone can
> exceed what's left after loading the quantized weights.

## G. Dataset Design

Quality usually matters more than raw size because: duplicate or
near-duplicate examples waste training steps without adding new signal;
noisy examples (wrong answers, mismatched Q/A pairs) directly teach the
model incorrect associations; train/test contamination (identical or
near-identical examples appearing in both splits) inflates measured
performance without reflecting real generalization; and inconsistent
formatting (missing punctuation, HTML remnants, inconsistent answer length)
makes it harder for the model to learn a stable output pattern. A smaller,
carefully cleaned dataset routinely beats a larger, noisier one for this
kind of instruction-style task.

## H. Hyperparameters

| Hyperparameter | Too low → | Too high → | Practical starting point | Why we chose ours |
|---|---|---|---|---|
| Learning rate | underfitting, near-zero loss movement | loss diverges/spikes, adapters overwrite useful behavior | 2e-4 for LoRA/QLoRA (vs ~2e-5 for full FT) | Standard PEFT convention: ~10x full-FT LR since only a tiny fraction of parameters is updated |
| Batch size | noisy, unstable gradients | out-of-memory | as large as VRAM allows, then use grad accumulation | 4 (micro-batch) chosen to fit Kaggle's ~16GB at seq_len 512 |
| Gradient accumulation | small effective batch, noisy | slower wall-clock per effective step | 4 steps → effective batch 16 | balances stability against Kaggle session time limits |
| Epochs | underfitting | overfitting/memorization on a small dataset | 1-3 for PEFT on a few thousand examples | 3 chosen to see a clear signal without heavy memorization on 4k examples |
| Warmup ratio | unstable first steps | wastes early training budget doing nothing useful | 3-10% of total steps | 3% (short run, cosine schedule handles the rest) |
| Weight decay | no regularization, can overfit | can suppress the (already tiny) useful adapter signal | 0.0-0.01 for PEFT | 0.01, a mild standard default |
| LoRA rank (r) | not enough adapter capacity for the task | more parameters/VRAM than needed, no better performance | 8-64 depending on task complexity | 16 as the primary run; 4/16/64 ablated explicitly |
| LoRA alpha | weak adapter contribution | adapter can dominate/destabilize the base model's behavior | often set to 2x r | 32 (2x our r=16) |
| Dropout | can overfit on small data | can underfit / slow convergence | 0.0-0.1 | 0.05, standard for small PEFT runs |
| Max sequence length | truncates real answers | wastes memory/time on padding | set from measured token-length percentiles | 512, from the 95th-percentile check in Section 10 |

## I. Evaluation

Evaluating open-ended LLM generation is hard because there is rarely one
"correct" string — many different phrasings can be equally right, and a
metric that only rewards exact wording will punish good answers. **Exact
match / accuracy** work well for classification-style tasks with a fixed
answer set, poorly for free text. **F1** (token overlap) is a step up but
still surface-level. **ROUGE/BLEU** are n-gram overlap metrics — useful,
cheap, and standard, but blind to correct paraphrases and can reward
fluent-but-wrong text that happens to share vocabulary with the reference.
**Semantic similarity metrics (BERTScore, embedding cosine similarity)**
compare meaning rather than surface tokens, catching valid paraphrases that
n-gram metrics miss, but they can also reward text that is topically similar
but factually wrong. **Human evaluation** is the gold standard for
correctness and usefulness but doesn't scale. **LLM-as-a-judge** (prompting
a strong separate model to score/compare outputs) is a scalable middle
ground, but inherits the judge model's own biases and blind spots and
should be treated as a proxy, disclosed as such, not as ground truth.

## J. Fine-Tuning vs RAG

**Fine-tuning is useful for:** behavior (how the model responds — tone,
verbosity, refusal patterns), style (format, structure, language register),
task adaptation (e.g., always answering in a Q&A style specific to this
domain), and domain-specific *patterns* the model can learn to produce
reliably (e.g., typical structure of an agronomic answer).

**RAG is useful for:** supplying external knowledge the model wasn't trained
on, information that changes frequently (prices, weather-dependent
recommendations, current advisories), citations/grounding (pointing to the
actual source document), and private or proprietary documents that
shouldn't be baked into model weights.

**Why fine-tuning is usually the wrong tool for "putting a PDF into the
model":** fine-tuning on a document doesn't reliably teach the model to
recall that document's specific facts on demand — it's expensive, doesn't
scale to frequently-updated documents, provides no verifiable citation back
to the source, and risks the model confidently paraphrasing/hallucinating
details rather than retrieving them exactly. Retrieval instead fetches the
actual relevant text at answer time and gives the model something concrete
to ground its answer in. This is exactly why the upcoming agricultural
advisory project combines both: fine-tuning for consistent, well-formatted,
domain-appropriate *behavior*, and RAG for correct, sourced, up-to-date
*facts*.

## K. Deployment

Realistic options for a project at this scale: a **Gradio** demo (base vs
adapter side-by-side) is cheap to build and can run on a free Hugging Face
Space CPU tier for a 1.5B model at reduced speed, or on a small GPU Space
for responsive generation. **Streamlit** is an equally viable alternative
for the same kind of demo. A hosted **inference endpoint** (paid) would be
appropriate for a production system, not a portfolio experiment.

```text
Training hardware   → Kaggle GPU (T4 x2 / P100), needed for LoRA and especially QLoRA fine-tuning
Inference hardware  → local RTX 2050 (4GB) is workable for 4-bit inference only, at modest speed
Deployment hardware → free-tier CPU/GPU Space for a demo; a paid endpoint only if this became a real product
```

**CPU inference** of the 1.5B model in 4-bit is realistic for a low-traffic
demo (expect several seconds per response, not real-time chat speed) — not
recommended for anything latency-sensitive.

Given this project's scope, an actual deployed Space is not part of the
deliverable; a lightweight local Gradio demo script is a reasonable
lightweight substitute if one is wanted later.

## L. Hugging Face Hub

- **Model repository:** hosts full model weights — not what we push here, since we only trained adapters.
- **Adapter repository:** hosts just the LoRA adapter files (`adapter_config.json`, `adapter_model.safetensors` — typically tens of MB, not gigabytes) plus the tokenizer files.
- **Model card:** the repo's `README.md` on the Hub — should state the base model, the dataset, training hyperparameters, intended use, and known limitations (mirroring this project's own README).
- **Uploading:** `model.push_to_hub("username/repo-name")` and `tokenizer.push_to_hub("username/repo-name")` after training, where `model` is the `PeftModel`.
- **Loading later:** load the base model fresh, then `PeftModel.from_pretrained(base_model, "username/repo-name")` — the adapter repo alone is not a runnable model; it always requires the same base model ID it was trained against.

---

# DECISION LOG — Why We Chose This Approach

| Decision | Selected | Alternatives considered | Why selected |
|---|---|---|---|
| Base model | Qwen2.5-1.5B-Instruct | TinyLlama-1.1B, Phi-3.5-mini (3.8B), Llama-3.2-1B/3B | Small enough for LoRA fine-tuning on a free Kaggle GPU in hours, large enough that the LoRA vs QLoRA memory gap is clearly visible, already instruction-tuned (fair baseline), strong `transformers`/`peft`/`trl` support |
| Fine-tuning method | LoRA + QLoRA (compared) | Full fine-tuning | Full FT of even a 1.5B model needs far more VRAM (weights+grads+optimizer states for 100% of parameters) than Kaggle's free tier comfortably offers, and doesn't teach the PEFT skills this project targets |
| Quantization | 4-bit NF4 (QLoRA) | 8-bit | 4-bit gives a starker, more interview-relevant memory comparison against full-precision LoRA; NF4 specifically (vs plain int4) is the QLoRA-paper-standard format supported natively by `bitsandbytes` |
| Trainer | TRL `SFTTrainer` | Custom PyTorch training loop | `SFTTrainer` is a thin, well-maintained wrapper around `transformers.Trainer` purpose-built for instruction fine-tuning — using it doesn't hide anything (we still inspect trainable params, loss curves, and generations manually), and a hand-rolled loop would mostly reimplement the same logic with more bug surface |
| Dataset | KisanVaani/agriculture-qa-english-only | b-mc2/sql-create-context (text-to-SQL), generic instruction datasets (Alpaca-style) | Directly relevant to the upcoming Agricultural AI Advisory (Fine-Tuning + RAG) project, giving continuity of domain knowledge; public, sizeable, and well-documented enough for a leakage-safe split |
| Evaluation metric | ROUGE-L + BERTScore F1 | BLEU, exact match only | Open-ended free-text answers make BLEU's n-gram precision overly harsh on valid paraphrases; ROUGE-L and BERTScore better tolerate reordering and semantic equivalence |
| Ablation axis | LoRA rank (4/16/64) | Dataset size, epoch count, learning rate | Cheapest to hold fair (same data/steps/seed) while answering a genuinely useful engineering question about adapter capacity |

---

# INTERVIEW PREPARATION

## Beginner

1. **What is fine-tuning?** Continuing to train a pre-trained model's weights on a smaller, task-specific labeled dataset so it adapts to that task.
2. **What is LoRA?** A parameter-efficient fine-tuning method that freezes the base model and injects small trainable low-rank matrices (`ΔW = BA`) into selected weight matrices.
3. **What is QLoRA?** LoRA fine-tuning performed on top of a base model whose frozen weights are stored in 4-bit quantized (NF4) format, drastically reducing memory needed to hold the base model.
4. **What does LoRA rank mean?** The bottleneck dimension `r` of the low-rank update matrices — it controls how much capacity the adapter has and how many parameters it adds.
5. **Why freeze the base model?** To avoid recomputing/storing gradients and optimizer states for the vast majority of parameters, and to avoid catastrophic forgetting of the model's general capabilities.
6. **What happens during backpropagation?** The loss's gradient is computed with respect to each trainable parameter via the chain rule, then used by the optimizer to update those parameters.
7. **Why does fine-tuning require more memory than inference?** Training additionally needs gradients and optimizer states for trainable parameters, and must retain more activations for the backward pass; inference only needs a forward pass and (for generation) a KV cache.
8. **What is a tokenizer?** The component that converts raw text into integer token IDs the model's embedding layer understands, and back again for decoding.
9. **What's the difference between pre-training and fine-tuning?** Pre-training learns general language structure/knowledge from massive unlabeled text; fine-tuning adapts an already-pretrained model to a narrower task using labeled examples.
10. **Why use a validation set?** To monitor generalization during training and detect overfitting before touching the untouched test set.

## Intermediate

11. **What is the difference between LoRA and QLoRA?** LoRA trains low-rank adapters on top of a full-precision (fp16/bf16) frozen base; QLoRA trains the same kind of adapters on top of a 4-bit-quantized frozen base — the adapter design is identical, only the base weight storage/precision differs.
12. **What does `lora_alpha` do?** Scales the LoRA update's magnitude (`scaling = alpha / r`) independently of the rank, controlling how strongly the adapter influences the frozen base's output.
13. **Why does PEFT use a higher learning rate than full fine-tuning?** Because only a small fraction of parameters is being updated, a proportionally larger step size is needed to achieve a comparable overall change in model behavior.
14. **What is NF4 quantization?** A 4-bit numeric format whose representable values are spaced to match the (roughly normal) distribution of typical trained neural network weights, preserving more information per bit than a naive uniform 4-bit grid.
15. **What is double quantization?** Additionally quantizing the scaling constants used during 4-bit quantization, saving a further small amount of memory per parameter.
16. **Why might increasing epochs hurt performance?** Beyond a point, the model starts memorizing training-specific phrasing rather than learning generalizable patterns, which shows up as falling train loss but rising eval loss.
17. **What's the effect of increasing LoRA rank?** More trainable parameters and adapter capacity, at the cost of more VRAM and compute; performance gains typically diminish past a task-appropriate rank.
18. **Why is ROUGE preferred over BLEU here?** BLEU's n-gram precision penalizes valid paraphrases too harshly for open-ended QA; ROUGE-L (LCS-based) tolerates reordering better.
19. **What is `target_modules` in LoraConfig?** The list of linear layers (e.g., attention projections) that get a LoRA adapter attached; expanding it (e.g., to MLP layers) adds capacity and parameters.
20. **Why use gradient accumulation?** To simulate a larger effective batch size than would otherwise fit in memory, by accumulating gradients over several micro-batches before an optimizer step.

## Advanced

21. **Why does QLoRA use a bf16 compute dtype with 4-bit stored weights?** Because the matrix multiplication itself is de-quantized on the fly to bf16 for numerically stable, hardware-supported compute — the model is stored in 4-bit but never actually computed in 4-bit.
22. **What is `prepare_model_for_kbit_training` doing?** Casting norm layers to fp32 for stability, enabling gradient checkpointing hooks, and making frozen input embeddings require gradients so backprop can flow into the LoRA adapters attached downstream.
23. **Why can a low-rank update be a good approximation of a full fine-tuning update?** Empirically, the useful direction of change needed to adapt a pre-trained model to a new task tends to have low "intrinsic rank" — most of its effect lies in a much lower-dimensional subspace than the full weight matrix.
24. **What are the main components of GPU memory during training?** Model weights, gradients (trainable params only), optimizer states, activations, and various temporary/workspace tensors.
25. **Why might a LoRA adapter underperform a full fine-tune on a very complex task?** The low-rank constraint limits how much the model's behavior can change; a task requiring broad, high-rank changes across many weight matrices may exceed what a modest-rank adapter can express.
26. **What's a risk specific to catastrophic forgetting in full fine-tuning that LoRA mitigates?** Because the base weights never change, the model's original general capabilities remain intact even if the adapter is later removed or replaced.
27. **Why is `assistant_only_loss` (loss masking) relevant to instruction fine-tuning?** Without it, the model is also trained to predict the fixed system/user prompt tokens, which wastes learning signal on text it doesn't need to generate; masking focuses the loss on the assistant's response only.
28. **How would you detect if a metric improvement is just an artifact of the split rather than real learning?** Check for train/test leakage, verify metric consistency across multiple seeds/re-runs, and manually inspect a sample of predictions rather than trusting the aggregate number alone.

## Project-Specific

29. **Why did you choose Qwen2.5-1.5B-Instruct specifically?** Small enough for LoRA fine-tuning on a free Kaggle GPU, large enough that LoRA vs QLoRA memory differences are visible and meaningful, already instruction-tuned for a fair baseline, and well-supported by the `transformers`/`peft`/`trl` stack.
30. **Why the agriculture Q&A dataset?** It's public, reasonably sized, directly relevant to a larger agricultural advisory project I'm building next, and its free-text answer format supports a meaningful before/after comparison.
31. **How did you evaluate the model?** ROUGE-L and BERTScore F1 as primary metrics for open-ended text, exact match as a diagnostic only, plus a manual qualitative comparison table on held-out examples.
32. **How do you know fine-tuning actually improved performance, and not just that the model memorized training answers?** By evaluating exclusively on a held-out test split never seen during training, and by checking that exact-match rate stays low (near-total exact matches would indicate memorization rather than generalization).
33. **What would happen if the dataset size doubled?** Likely marginal quality improvement given the domain is fairly narrow, but training time and compute cost would roughly double too — a candidate for a future ablation rather than an assumption.

## Failure / Debugging

34. **What caused your model's failures, based on your failure analysis?** [To be filled with the actual observed causes from `results/fine_tuned_results.md` once training has run — e.g., hallucinated specifics on underrepresented crops, echoed training phrasing on near-duplicate questions.]
35. **Why not use RAG instead of fine-tuning here?** RAG solves a knowledge-grounding problem (supplying current, verifiable facts); this project specifically isolates and studies the fine-tuning half of a system that will use both, so RAG was deliberately left for the next project.
36. **When would you choose RAG instead of fine-tuning?** When the need is for up-to-date, verifiable, or private/document-specific information rather than a change in the model's behavior or style.
37. **What would happen if the learning rate were set 10x too high?** Loss would likely spike or diverge instead of decreasing smoothly, and the adapter could learn a degenerate, unstable transformation of the base model's behavior.
38. **How would you deploy this if asked to?** A Gradio demo comparing base vs adapter responses on a free-tier Hugging Face Space, given the project's scale; not a paid production endpoint.
39. **What would you change with an A100/H100 available?** Increase batch size and sequence length, fine-tune a larger base model, run full fine-tuning as an additional comparison point, and expand the training set closer to the dataset's full ~22.6k examples.
40. **What would you change with only 4GB VRAM available?** Restrict to QLoRA-only, use a smaller max sequence length and micro-batch size with heavier gradient accumulation, and expect training (not just inference) to still likely be infeasible at this model size — 4-bit inference only would be the realistic ceiling.

---

# Before You Start Training

Checklist:
- [ ] Dataset verified (`load_dataset` runs, fields match `data.py` expectations, no unexpected nulls)
- [ ] Model verified (`AutoModelForCausalLM.from_pretrained` loads without error, parameter count matches expectation)
- [ ] Tokenizer verified (pad token set, chat template produces the expected string)
- [ ] GPU verified (`torch.cuda.is_available()` is True, VRAM and bf16 support printed)
- [ ] Package versions verified (`pip show transformers peft trl bitsandbytes accelerate`, cross-checked against current docs)
- [ ] Baseline completed (base model evaluated on the test split before any training)
- [ ] Training configuration checked (`training_config.yaml` values match what's actually passed to `SFTConfig`)
- [ ] Experiment logging ready (loss history captured for the training-curve plot)
- [ ] Evaluation ready (`evaluate.load("rouge")` and `evaluate.load("bertscore")` both import cleanly — the latter needs internet access to fetch its underlying model)
- [ ] Hugging Face account/repository ready if pushing adapters (`huggingface_hub` login token available in Kaggle secrets)

Step-by-step execution order:

```text
Step 1: Environment setup + GPU verification
   ↓
Step 2: Load and inspect raw dataset
   ↓
Step 3: Clean, filter, split (train/val/test), format prompts
   ↓
Step 4: Load base model + tokenizer, run BASELINE evaluation
   ↓
Step 5: Configure and train LoRA
   ↓
Step 6: Configure and train QLoRA (same data/seed/steps)
   ↓
Step 7: Evaluate base, LoRA, and QLoRA on the test split (quantitative + qualitative)
   ↓
Step 8: Run the LoRA-rank ablation (r = 4 / 16 / 64)
   ↓
Step 9: Failure analysis on held-out examples
   ↓
Step 10: Fill in results/*.md and README.md with real, measured numbers
   ↓
Step 11: (Optional) push the best adapter to the Hugging Face Hub
```
