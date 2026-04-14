"""
DIA-GUARD PEFT Fine-Tuning — LoRA / QLoRA + Cross-Entropy Loss
===============================================================
Fine-tunes a decoder-only LLM using LoRA or QLoRA adapters with standard
supervised cross-entropy (next-token prediction) loss.

LoRA injects trainable rank-decomposition matrices into the attention and MLP
layers of the frozen base model: W' = W + BA, where B ∈ R^(d×r), A ∈ R^(r×k).
This reduces trainable parameters from ~4B to ~50M while preserving performance.

References:
  - LoRA (Hu et al., 2022): https://arxiv.org/abs/2106.09685
  - QLoRA (Dettmers et al., 2023): https://arxiv.org/abs/2305.14314
  - HuggingFace PEFT: https://github.com/huggingface/peft
  - TRL SFTTrainer: https://huggingface.co/docs/trl/sft_trainer

Usage:
  python train_ce_lora.py --config configs/qwen3_4b_lora.yaml \\
      --train_data /data/train.jsonl

  # QLoRA (4-bit)
  python train_ce_lora.py --config configs/qwen3_4b_lora.yaml \\
      --use_qlora true --train_data /data/train.jsonl

Data format (JSONL):
  {"prompt": "...", "response": "...", "label": 0 or 1}
"""

import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np

import torch
import torch.serialization
import yaml


# Patch Trainer._load_rng_state to skip loading RNG state from checkpoint.
# Transformers calls torch.load(..., weights_only=True) on the RNG file, but the
# file contains numpy objects that fail the weights-only check across torch/numpy
# version combinations. RNG state only affects random-seed reproducibility on
# resume — training correctness and loss curves are unaffected.
from transformers import Trainer as _Trainer
_Trainer._load_rng_state = lambda self, checkpoint: None


from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    set_seed,
)
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

# Add parent dir to path for data_utils
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_utils_new import load_and_format_dataset

os.environ['WANDB_API_KEY'] = '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'
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
os.environ.setdefault("RAYON_NUM_THREADS", "24")
# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are DIA-Guard, a multilingual safety assistant. "
    "Analyze the following input and provide a safe, informed response."
)


def format_prompt(example: dict, tokenizer) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example["prompt"]},
        {"role": "assistant", "content": example["response"]},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    return (
        f"<|system|>{SYSTEM_PROMPT}</s>\n"
        f"<|user|>{example['prompt']}</s>\n"
        f"<|assistant|>{example['response']}</s>"
    )


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(config_path: str, overrides: dict) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v
    return cfg


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> Dataset:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return Dataset.from_list(records)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train(cfg: dict):
    set_seed(42)

    use_qlora = cfg.get("use_qlora", False) or cfg.get("load_in_4bit", False)

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

    # --- Quantization config (QLoRA) ---
    bnb_config = None
    if use_qlora:
        print("Using QLoRA (4-bit quantization)")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=cfg.get("bnb_4bit_use_double_quant", True),
        )

    # --- Base model ---
    # Pin QLoRA to a single GPU per process so accelerate's prepare() doesn't
    # fail when device_map="auto" splits 4-bit layers across multiple GPUs.
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device_map = {"": local_rank} if use_qlora else None

    print(f"Loading model: {cfg['model_name']}")
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16 if not use_qlora else None,
        attn_implementation=cfg.get("attn_implementation", "eager"),
        trust_remote_code=cfg.get("trust_remote_code", True),
        device_map=device_map,
    )
    model.config.use_cache = False

    # Prepare for k-bit training (required for QLoRA)
    if use_qlora:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=cfg.get("gradient_checkpointing", True),
        )

    # --- LoRA config ---
    target_modules = cfg.get("lora_target_modules", [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.get("lora_r", 64),
        lora_alpha=cfg.get("lora_alpha", 128),
        lora_dropout=cfg.get("lora_dropout", 0.05),
        bias=cfg.get("lora_bias", "none"),
        target_modules=target_modules,
        # Rank-Stabilized LoRA: scales by alpha/sqrt(r) — recommended for r >= 16
        use_rslora=cfg.get("use_rslora", True),
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Register gradient hooks on all trainable parameters to replace NaN
    # gradients with zero during backprop, before they reach the optimizer.
    # This prevents a single unstable batch from corrupting Adam's state.
    _nan_grad_count = [0]
    for p in model.parameters():
        if p.requires_grad:
            def _nan_hook(grad, counter=_nan_grad_count):
                if torch.isnan(grad).any():
                    counter[0] += 1
                    return torch.zeros_like(grad)
                return grad
            p.register_hook(_nan_hook)

    # --- Dataset (with disk caching) ---
    train_path = cfg.get("train_data")
    eval_path = cfg.get("eval_data")
    if not train_path or not Path(train_path).exists():
        raise ValueError(f"train_data not found: {train_path}")

    max_seq_length = cfg.get("max_seq_length", 2048)
    train_dataset = load_and_format_dataset(
        train_path, tokenizer, cfg["model_name"], SYSTEM_PROMPT, split="train",
        max_length=max_seq_length,
    )
    eval_dataset = None
    if eval_path and Path(eval_path).exists():
        eval_dataset = load_and_format_dataset(
            eval_path, tokenizer, cfg["model_name"], SYSTEM_PROMPT, split="val",
            max_length=max_seq_length,
        )

    print(f"Train: {len(train_dataset)} | Eval: {len(eval_dataset) if eval_dataset else 'N/A'}")

    # --- SFTTrainer ---
    output_dir = cfg.get("output_dir", "./output")
    os.makedirs(output_dir, exist_ok=True)

    training_args = SFTConfig(
        output_dir=output_dir,
        run_name=cfg.get("run_name", "dia_peft"),
        num_train_epochs=cfg.get("num_epochs", 3),
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 4),
        per_device_eval_batch_size=cfg.get("per_device_eval_batch_size", 4),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 4),
        learning_rate=float(cfg.get("learning_rate", 2e-4)),
        warmup_ratio=cfg.get("warmup_ratio", 0.03),
        weight_decay=cfg.get("weight_decay", 0.01),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        max_grad_norm=cfg.get("max_grad_norm", 0.3),
        bf16=cfg.get("bf16", True) and not use_qlora,
        bf16_full_eval=cfg.get("bf16_full_eval", False),
        tf32=cfg.get("tf32", True),
        gradient_checkpointing=cfg.get("gradient_checkpointing", True),
        logging_steps=cfg.get("logging_steps", 10),
        eval_strategy=cfg.get("eval_strategy", "steps") if eval_dataset else "no",
        eval_steps=cfg.get("eval_steps", 200) if eval_dataset else None,
        save_strategy=cfg.get("save_strategy", "steps"),
        save_steps=cfg.get("save_steps", 500),
        save_total_limit=cfg.get("save_total_limit", 3),
        load_best_model_at_end=cfg.get("load_best_model_at_end", True) if eval_dataset else False,
        report_to=cfg.get("report_to", "none"),
        max_length=cfg.get("max_seq_length", 2048),
        dataset_text_field="text",
        packing=False,
        completion_only_loss=False,  # handled manually via DataCollatorForCompletionOnlyLM
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
    # Build completion-only data collator.
    # For Qwen3Guard, add_generation_prompt=True appends:
    #   <|im_start|>assistant\n<think>\n\n</think>\n\n
    # before the actual completion. We use "</think>\n\n" as the response
    # template so loss is computed only on "Safety: Safe/Unsafe..." tokens.
    is_guard_model = "guard" in cfg["model_name"].lower()
    if is_guard_model:
        # Extract token IDs from a real formatted sequence so they match what the
        # tokenizer produces in context (BPE splits differ when tokenizing in isolation).
        _dummy = tokenizer.apply_chat_template(
            [{"role": "user", "content": "x"}],
            tokenize=False,
            add_generation_prompt=True,
        )
        _dummy_ids = tokenizer.encode(_dummy, add_special_tokens=False)
        _suffix_len = len(tokenizer.encode("</think>\n\n", add_special_tokens=False))
        response_template = _dummy_ids[-_suffix_len:]
    else:
        response_template = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)

    data_collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
    )
    print(f"[DataCollator] response_template={repr(tokenizer.decode(response_template))}")

    _train_check_done = False
    _eval_check_done = False

    class DiagnosticTrainer(SFTTrainer):
        def compute_loss(self, model, inputs, **kwargs):
            loss = super().compute_loss(model, inputs, **kwargs)
            if not torch.isfinite(loss):
                print(f"[LossGuard] non-finite loss ({loss.item()}) — skipping backward")
                return torch.zeros((), requires_grad=True, device=loss.device, dtype=loss.dtype)
            return loss

        def training_step(self, model, inputs, *args, **kwargs):
            nonlocal _train_check_done
            if "labels" in inputs:
                if not _train_check_done:
                    first_labels = inputs["labels"][0]
                    active_ids = first_labels[first_labels != -100].tolist()
                    print(f"[LabelCheck train] {repr(tokenizer.decode(active_ids))}")
                    _train_check_done = True
                if (inputs["labels"] == -100).all():
                    print("[NaNGuard] all-masked batch — skipping")
                    return torch.tensor(0.0, device=next(model.parameters()).device)
            return super().training_step(model, inputs, *args, **kwargs)

        def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
            nonlocal _eval_check_done
            if "labels" in inputs and not _eval_check_done:
                first_labels = inputs["labels"][0]
                active_ids = first_labels[first_labels != -100].tolist()
                print(f"[LabelCheck eval]  {repr(tokenizer.decode(active_ids))}")
                _eval_check_done = True
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

    # NOTE: trl >= 0.12 uses `processing_class` instead of `tokenizer`
    trainer = DiagnosticTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    # Resume from checkpoint if available
    resume_ckpt = cfg.get("resume_from_checkpoint")
    if resume_ckpt == "auto":
        # Find latest checkpoint in output_dir
        ckpts = sorted(Path(output_dir).glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
        resume_ckpt = str(ckpts[-1]) if ckpts else None
        if resume_ckpt:
            print(f"Resuming from checkpoint: {resume_ckpt}")

    print("Starting LoRA training...")
    trainer.train(resume_from_checkpoint=resume_ckpt)

    # --- Save adapter ---
    print(f"Saving LoRA adapter to {output_dir}")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    with open(os.path.join(output_dir, "training_config.yaml"), "w") as f:
        yaml.dump(cfg, f)

    print(f"\nDone. Adapter saved to: {output_dir}")
    print(
        "To merge adapter into base model for distillation:\n"
        "  from peft import PeftModel\n"
        "  base = AutoModelForCausalLM.from_pretrained(...)\n"
        "  model = PeftModel.from_pretrained(base, output_dir)\n"
        "  merged = model.merge_and_unload()\n"
        "  merged.save_pretrained(output_dir + '-merged')"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="DIA-GUARD PEFT LoRA/QLoRA + Cross-Entropy Training"
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--train_data", type=str, default=None)
    parser.add_argument("--eval_data", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="'auto' to find latest checkpoint, or path to a specific checkpoint")
    parser.add_argument("--use_qlora", type=lambda x: x.lower() == "true", default=None)
    parser.add_argument("--lora_r", type=int, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None, dest="learning_rate")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    overrides = {k: v for k, v in vars(args).items() if k != "config"}
    cfg = load_config(args.config, overrides)
    train(cfg)