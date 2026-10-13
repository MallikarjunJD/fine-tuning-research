"""
push_adapter_to_hub.py — push a trained LoRA/QLoRA adapter (from src/train.py's
--output_dir) to the Hugging Face Hub as its own small adapter repository.

This does NOT push the base model. It pushes only:
  - adapter_config.json
  - adapter_model.safetensors   (typically tens of MB, not GB)
  - tokenizer files
  - an auto-generated model card

Prerequisites:
  1. huggingface-cli login   (or set the HF_TOKEN env var)
  2. A completed training run, e.g. outputs/qlora/ from src/train.py

Usage:
    python scripts/push_adapter_to_hub.py \
        --adapter_dir outputs/qlora \
        --repo_id your-username/qwen2.5-1.5b-agri-qa-qlora \
        --base_model Qwen/Qwen2.5-1.5B-Instruct
"""

import argparse
from huggingface_hub import HfApi, create_repo
from transformers import AutoTokenizer
from peft import PeftModel, AutoPeftModelForCausalLM


MODEL_CARD_TEMPLATE = """---
base_model: {base_model}
library_name: peft
tags:
- lora
- qlora
- agriculture
- question-answering
license: mit
---

# {repo_name}

A LoRA/QLoRA adapter fine-tuned on top of `{base_model}` for agricultural
question answering, trained on a subsample of
[`KisanVaani/agriculture-qa-english-only`](https://huggingface.co/datasets/KisanVaani/agriculture-qa-english-only).

This is an **adapter**, not a standalone model — it must be loaded on top of
the base model listed above.

## Usage

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_id = "{base_model}"
tokenizer = AutoTokenizer.from_pretrained(base_id)
base_model = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.float32)
model = PeftModel.from_pretrained(base_model, "{repo_id}")

prompt = (
    "<|im_start|>system\\nYou are an agricultural assistant. Answer the farmer's "
    "question clearly, correctly, and concisely, in plain English.<|im_end|>\\n"
    "<|im_start|>user\\nHow do I control aphids on cotton?<|im_end|>\\n"
    "<|im_start|>assistant\\n"
)
inputs = tokenizer(prompt, return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

## Training details

See the full project (notebook, decision log, evaluation results) at:
`<link to your GitHub repo>`

## Limitations

Community-curated training data, not agronomist-verified. See the project
README's "Limitations" section for the full disclosure.
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_dir", required=True, help="e.g. outputs/qlora")
    parser.add_argument("--repo_id", required=True, help="e.g. your-username/qwen2.5-1.5b-agri-qa-qlora")
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--private", action="store_true", help="Create a private repo instead of public")
    args = parser.parse_args()

    api = HfApi()
    create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)

    # Load the adapter fresh from disk (sanity check that it actually loads before pushing)
    model = AutoPeftModelForCausalLM.from_pretrained(args.adapter_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir)

    model.push_to_hub(args.repo_id, private=args.private)
    tokenizer.push_to_hub(args.repo_id, private=args.private)

    card = MODEL_CARD_TEMPLATE.format(
        base_model=args.base_model,
        repo_name=args.repo_id.split("/")[-1],
        repo_id=args.repo_id,
    )
    api.upload_file(
        path_or_fileobj=card.encode(),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="model",
    )

    print(f"Pushed adapter to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
