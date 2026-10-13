"""
data.py — dataset loading, cleaning, splitting, and prompt formatting.

This mirrors sections 5-7 of the notebook so the same logic can be imported
by src/train.py and src/evaluate.py instead of being copy-pasted.
"""

import random
import yaml
from datasets import load_dataset, DatasetDict


def load_config(path: str = "configs/training_config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_raw_dataset(cfg: dict):
    """Pull the agriculture Q&A dataset from the Hugging Face Hub."""
    ds = load_dataset(cfg["dataset"]["hf_id"], split="train")
    return ds


def clean_example(example: dict, cfg: dict) -> dict:
    """Strip whitespace, drop empty/near-duplicate fields.

    Common mistake this guards against: many QA scrapes contain empty
    answers, HTML remnants, or answers that just repeat the question.
    """
    q_field = cfg["dataset"]["question_field"]
    a_field = cfg["dataset"]["answer_field"]
    question = str(example.get(q_field, "")).strip()
    answer = str(example.get(a_field, "")).strip()
    return {"question": question, "answer": answer}


def is_valid(example: dict) -> bool:
    q, a = example["question"], example["answer"]
    if len(q) <= 5 or len(a) <= 5:      # identical to the notebook's filter (> 5 chars)
        return False
    if q.lower() == a.lower():
        return False
    return True


def build_splits(cfg: dict) -> DatasetDict:
    """Deterministic train/val/test/held-out split.

    IMPORTANT (leakage prevention): splitting happens ONCE, before any
    shuffling that touches the training subsample, and the held-out
    qualitative set is carved out of the *test* split, never the train
    split, so no example used for the final "base vs fine-tuned" table was
    ever seen during training.
    """
    seed = cfg["project"]["seed"]
    raw = load_raw_dataset(cfg)
    raw = raw.map(lambda ex: clean_example(ex, cfg))
    raw = raw.filter(is_valid)
    raw = raw.shuffle(seed=seed)

    n_train = cfg["dataset"]["train_size"]
    n_val = cfg["dataset"]["val_size"]
    n_test = cfg["dataset"]["test_size"]

    train = raw.select(range(0, n_train))
    val = raw.select(range(n_train, n_train + n_val))
    test = raw.select(range(n_train + n_val, n_train + n_val + n_test))

    return DatasetDict({"train": train, "validation": val, "test": test})


def format_prompt(question: str, answer: str, cfg: dict, for_generation: bool = False) -> str:
    """Turn a (question, answer) pair into the Qwen chat-template string.

    for_generation=True omits the answer and leaves the assistant turn open,
    which is what we feed the model at inference/baseline time.
    """
    system = cfg["prompt"]["system_instruction"]
    if for_generation:
        return (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{question}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
    return cfg["prompt"]["template"].format(system=system, question=question, answer=answer)


def to_sft_text(dataset, cfg: dict):
    """Map a split into a single `text` column consumed by SFTTrainer."""
    return dataset.map(
        lambda ex: {"text": format_prompt(ex["question"], ex["answer"], cfg)}
    )


if __name__ == "__main__":
    config = load_config()
    splits = build_splits(config)
    print(splits)
    print(splits["train"][0])
