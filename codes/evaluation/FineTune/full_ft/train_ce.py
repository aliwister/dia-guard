"""
DIA-GUARD Full Fine-Tuning — Supervised Cross-Entropy Loss
==========================================================
Fine-tunes a decoder-only LLM (Qwen3, Aya, Llama, Gemma) on DIA-GUARD safety
data using standard next-token prediction (cross-entropy) loss.

Paper basis:
  - HuggingFace TRL SFTTrainer: https://huggingface.co/docs/trl/sft_trainer
  - Qwen3 fine-tuning: https://huggingface.co/Qwen/Qwen3-4B-SafeRL

Usage:
  Single GPU:
    python train_ce.py --config configs/qwen3_4b.yaml

  Multi-GPU (accelerate):
    accelerate launch --num_processes 2 train_ce.py --config configs/qwen3_4b.yaml

  CLI overrides:
    python train_ce.py --config configs/qwen3_4b.yaml \\
        --model_name Qwen/Qwen3-4B-SafeRL \\
        --train_data /data/train.jsonl \\
        --output_dir ../../../models/FT/full_ft/qwen3-4b-ft

Data format (JSONL):
  {"prompt": "...", "response": "...", "label": 0 or 1}
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import torch
import yaml
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    TrainingArguments,
    set_seed,
)
from trl import SFTTrainer, SFTConfig

# Add parent dir to path for data_utils
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_utils import load_and_format_dataset


# ---------------------------------------------------------------------------
# Prompt template helpers
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are DIA-Guard, a multilingual safety assistant. "
    "Analyze the following input and provide a safe, informed response."
)


def format_prompt(example: dict, tokenizer) -> str:
    """
    Format a data example into a chat-style prompt string.
    Uses the model's native chat template if available, otherwise falls back
    to a simple format.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example["prompt"]},
        {"role": "assistant", "content": example["response"]},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    # Fallback format
    return (
        f"<|system|>{SYSTEM_PROMPT}</s>\n"
        f"<|user|>{example['prompt']}</s>\n"
        f"<|assistant|>{example['response']}</s>"
    )


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: str, overrides: dict) -> dict:
    """Load YAML config and apply CLI overrides."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v
    return cfg


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> Dataset:
    """Load a JSONL file into a HuggingFace Dataset."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return Dataset.from_list(records)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(cfg: dict):
    set_seed(42)

    # --- Tokenizer ---
    print(f"Loading tokenizer: {cfg['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model_name"],
        trust_remote_code=cfg.get("trust_remote_code", True),
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # --- Model ---
    print(f"Loading model: {cfg['model_name']}")
    attn_impl = cfg.get("attn_implementation", "eager")
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        torch_dtype=torch.bfloat16 if cfg.get("bf16", True) else torch.float32,
        attn_implementation=attn_impl,
        trust_remote_code=cfg.get("trust_remote_code", True),
    )
    model.config.use_cache = False  # required when gradient_checkpointing=True

    # --- Dataset (with disk caching) ---
    train_path = cfg.get("train_data")
    eval_path = cfg.get("eval_data")
    if not train_path or not Path(train_path).exists():
        raise ValueError(
            f"train_data path not found: {train_path}\n"
            "Set it via --train_data or in the config YAML."
        )

    train_dataset = load_and_format_dataset(
        train_path, tokenizer, cfg["model_name"], SYSTEM_PROMPT, split="train"
    )
    eval_dataset = None
    if eval_path and Path(eval_path).exists():
        eval_dataset = load_and_format_dataset(
            eval_path, tokenizer, cfg["model_name"], SYSTEM_PROMPT, split="val"
        )

    print(f"Train size: {len(train_dataset)}")
    if eval_dataset:
        print(f"Eval size: {len(eval_dataset)}")

    # --- Training Arguments ---
    output_dir = cfg.get("output_dir", "./output")
    os.makedirs(output_dir, exist_ok=True)

    training_args = SFTConfig(
        output_dir=output_dir,
        run_name=cfg.get("run_name", "dia_full_ft"),
        num_train_epochs=cfg.get("num_epochs", 3),
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 2),
        per_device_eval_batch_size=cfg.get("per_device_eval_batch_size", 2),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 8),
        learning_rate=float(cfg.get("learning_rate", 2e-5)),
        warmup_ratio=cfg.get("warmup_ratio", 0.03),
        weight_decay=cfg.get("weight_decay", 0.01),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        max_grad_norm=cfg.get("max_grad_norm", 1.0),
        bf16=cfg.get("bf16", True),
        tf32=cfg.get("tf32", True),
        gradient_checkpointing=cfg.get("gradient_checkpointing", True),
        logging_steps=cfg.get("logging_steps", 10),
        eval_strategy=cfg.get("eval_strategy", "steps") if eval_dataset else "no",
        eval_steps=cfg.get("eval_steps", 200) if eval_dataset else None,
        save_strategy=cfg.get("save_strategy", "steps"),
        save_steps=cfg.get("save_steps", 500),
        save_total_limit=cfg.get("save_total_limit", 3),
        load_best_model_at_end=cfg.get("load_best_model_at_end", True) if eval_dataset else False,
        metric_for_best_model="eval_loss" if eval_dataset else None,
        report_to=cfg.get("report_to", "none"),
        max_length=cfg.get("max_seq_length", 2048),
        dataset_text_field="text",
        packing=False,
        completion_only_loss=False,
    )

    # Build callbacks (early stopping is enabled by default when eval_dataset exists)
    callbacks = []
    if eval_dataset and cfg.get("early_stopping", True):
        patience = int(cfg.get("early_stopping_patience", 3))
        threshold = float(cfg.get("early_stopping_threshold", 0.0))
        # Force eval_loss as the early stopping metric (default would be train "loss")
        training_args.metric_for_best_model = "eval_loss"
        training_args.greater_is_better = False
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=patience,
                early_stopping_threshold=threshold,
            )
        )
        print(
            f"EarlyStoppingCallback enabled: patience={patience}, "
            f"threshold={threshold}, metric={training_args.metric_for_best_model}"
        )

    # --- Trainer ---
    # NOTE: trl >= 0.12 uses `processing_class` instead of `tokenizer`
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    # Resume from checkpoint if available
    resume_ckpt = cfg.get("resume_from_checkpoint")
    if resume_ckpt == "auto":
        ckpts = sorted(Path(output_dir).glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
        resume_ckpt = str(ckpts[-1]) if ckpts else None
        if resume_ckpt:
            print(f"Resuming from checkpoint: {resume_ckpt}")

    # --- Train ---
    print("Starting training...")
    trainer.train(resume_from_checkpoint=resume_ckpt)

    # --- Save final model ---
    print(f"Saving final model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save training config alongside model
    config_out = os.path.join(output_dir, "training_config.yaml")
    with open(config_out, "w") as f:
        yaml.dump(cfg, f)

    print(f"Done. Model saved to: {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="DIA-GUARD Full Fine-Tuning with Cross-Entropy Loss"
    )
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--train_data", type=str, default=None)
    parser.add_argument("--eval_data", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="'auto' to find latest checkpoint, or path to a specific checkpoint")
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None,
                        dest="per_device_train_batch_size")
    parser.add_argument("--lr", type=float, default=None, dest="learning_rate")
    parser.add_argument("--max_seq_length", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    overrides = {k: v for k, v in vars(args).items() if k != "config"}
    cfg = load_config(args.config, overrides)
    train(cfg)
