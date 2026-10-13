"""
train.py — LoRA / QLoRA fine-tuning entry point.

Usage (on Kaggle, inside the notebook, or as a script with a GPU attached):
    python src/train.py --mode lora   --output_dir outputs/lora
    python src/train.py --mode qlora  --output_dir outputs/qlora

Only ONE flag changes between the two runs: --mode. Everything else (data,
seed, LoRA r/alpha, epochs, effective batch size) is held constant on purpose
so that the LoRA vs QLoRA comparison in results/ is fair.
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer

from data import load_config, build_splits, to_sft_text


def build_model_and_tokenizer(cfg: dict, mode: str):
    model_id = cfg["model"]["base_model_id"]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if mode == "qlora":
        q = cfg["qlora"]
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=q["load_in_4bit"],
            bnb_4bit_quant_type=q["bnb_4bit_quant_type"],
            bnb_4bit_use_double_quant=q["bnb_4bit_use_double_quant"],
            bnb_4bit_compute_dtype=getattr(torch, q["bnb_4bit_compute_dtype"]),
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=bnb_config, device_map="auto"
        )
        # Required before adding LoRA adapters on top of a quantized base:
        # casts norms to fp32, enables gradient checkpointing hooks, and
        # makes the (frozen) input embeddings require grad so gradients can
        # flow into the adapters.
        model = prepare_model_for_kbit_training(model)
    elif mode == "lora":
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            device_map="auto",
        )
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'lora' or 'qlora'.")

    return model, tokenizer


def build_lora_config(cfg: dict) -> LoraConfig:
    l = cfg["lora"]
    return LoraConfig(
        r=l["r"],
        lora_alpha=l["lora_alpha"],
        lora_dropout=l["lora_dropout"],
        bias=l["bias"],
        task_type=l["task_type"],
        target_modules=l["target_modules"],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["lora", "qlora"], required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--config", default="configs/training_config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    torch.manual_seed(cfg["project"]["seed"])

    splits = build_splits(cfg)
    train_ds = to_sft_text(splits["train"], cfg)
    val_ds = to_sft_text(splits["validation"], cfg)

    model, tokenizer = build_model_and_tokenizer(cfg, args.mode)
    peft_config = build_lora_config(cfg)

    t = {**cfg["training"], **cfg.get("training_overrides", {}).get(args.mode, {})}
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=t["num_train_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=t["learning_rate"],
        warmup_ratio=t["warmup_ratio"],
        weight_decay=t["weight_decay"],
        lr_scheduler_type=t["lr_scheduler_type"],
        logging_steps=t["logging_steps"],
        eval_strategy=t["eval_strategy"],
        eval_steps=t["eval_steps"],
        save_strategy=t["save_strategy"],
        save_steps=t["save_steps"],
        save_total_limit=t["save_total_limit"],
        bf16=t["bf16"],
        fp16=t["fp16"],
        optim=t["optim"] if args.mode == "qlora" else "adamw_torch",
        report_to=t["report_to"],
        dataset_text_field="text",
        max_length=cfg["model"]["max_seq_length"],
        packing=False,
        seed=cfg["project"]["seed"],
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    trainer.model.print_trainable_parameters()
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved {args.mode} adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
