"""
DIA-GUARD Full Fine-Tuning — Contrastive Loss
=============================================
Fine-tunes a decoder-only LLM on DIA-GUARD safety data using a combined
Cross-Entropy + Contrastive loss. The contrastive loss pushes safe response
representations close to the prompt anchor while pushing harmful responses away.

Loss formulation:
  L_total = alpha * L_CE + (1 - alpha) * L_contrastive

  L_CE = -1/T * Σ_t log p_θ(y_t | x, y_{<t})         [next-token prediction]

  L_contrastive = Σ max(0, margin                       [triplet loss on last
                         - sim(h_anc, h_pos)             hidden states]
                         + sim(h_anc, h_neg))

  sim(a, b) = cosine_similarity(a, b)

Reference:
  SimCSE (Gao et al., 2021): https://arxiv.org/abs/2104.08821
  Adapted for decoder safety task.

Data format (JSONL with triplets):
  {"prompt": "...", "response": "safe response", "neg_response": "unsafe response", "label": 1}

Usage:
  python train_contrastive.py --config configs/qwen3_4b.yaml \\
      --train_data /data/triplets.jsonl \\
      --alpha 0.7 --margin 0.3
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
    set_seed,
)
from accelerate import Accelerator
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are DIA-Guard, a multilingual safety assistant. "
    "Analyze the following input and provide a safe, informed response."
)


def build_prompt(prompt: str, response: str, tokenizer) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    return f"<|system|>{SYSTEM_PROMPT}</s>\n<|user|>{prompt}</s>\n<|assistant|>{response}</s>"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> List[Dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


class TripletDataset(torch.utils.data.Dataset):
    """Dataset for triplet contrastive training.

    Each example contains:
      - anchor: the input prompt
      - positive: safe response
      - negative: unsafe response
    """

    def __init__(self, records: List[Dict], tokenizer, max_length: int = 2048):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        prompt = rec["prompt"]
        safe_resp = rec["response"]
        unsafe_resp = rec.get("neg_response", "")

        # Tokenize anchor (prompt only, for representation)
        anchor_text = f"<|system|>{SYSTEM_PROMPT}</s>\n<|user|>{prompt}</s>\n<|assistant|>"
        pos_text = build_prompt(prompt, safe_resp, self.tokenizer)
        neg_text = build_prompt(prompt, unsafe_resp, self.tokenizer) if unsafe_resp else pos_text

        def tokenize(text):
            return self.tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )

        anchor_enc = tokenize(anchor_text)
        pos_enc = tokenize(pos_text)
        neg_enc = tokenize(neg_text)

        return {
            "anchor_input_ids": anchor_enc["input_ids"].squeeze(0),
            "anchor_attention_mask": anchor_enc["attention_mask"].squeeze(0),
            "pos_input_ids": pos_enc["input_ids"].squeeze(0),
            "pos_attention_mask": pos_enc["attention_mask"].squeeze(0),
            "neg_input_ids": neg_enc["input_ids"].squeeze(0),
            "neg_attention_mask": neg_enc["attention_mask"].squeeze(0),
            # For CE loss: use positive sequence as labels
            "ce_input_ids": pos_enc["input_ids"].squeeze(0),
            "ce_labels": pos_enc["input_ids"].squeeze(0).clone(),
        }


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def get_last_hidden_state(model, input_ids, attention_mask):
    """Extract the last non-padding hidden state as the sequence representation."""
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )
    hidden_states = outputs.hidden_states[-1]  # (B, T, H)
    # Get last non-padding position for each sequence
    seq_lengths = attention_mask.sum(dim=1) - 1  # (B,)
    batch_size = hidden_states.size(0)
    last_hidden = hidden_states[
        torch.arange(batch_size, device=hidden_states.device), seq_lengths
    ]  # (B, H)
    return last_hidden


def triplet_contrastive_loss(h_anchor, h_pos, h_neg, margin: float = 0.3):
    """
    Triplet loss using cosine similarity.
    L = max(0, margin - cos(anchor, pos) + cos(anchor, neg))
    """
    sim_pos = F.cosine_similarity(h_anchor, h_pos, dim=-1)   # (B,)
    sim_neg = F.cosine_similarity(h_anchor, h_neg, dim=-1)   # (B,)
    loss = F.relu(margin - sim_pos + sim_neg)
    return loss.mean()


def ce_loss_from_logits(model, input_ids, labels):
    """Standard cross-entropy loss (next-token prediction), masking padding."""
    # Shift labels: predict token t+1 given tokens 0..t
    logits = model(input_ids=input_ids).logits  # (B, T, V)
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    # Mask padding
    shift_labels[shift_labels == model.config.pad_token_id] = -100
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    return loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(cfg: dict):
    set_seed(42)
    accelerator = Accelerator(
        mixed_precision="bf16" if cfg.get("bf16", True) else "no",
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 8),
    )

    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model_name"],
        trust_remote_code=cfg.get("trust_remote_code", True),
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # --- Model ---
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        torch_dtype=torch.bfloat16 if cfg.get("bf16", True) else torch.float32,
        attn_implementation=cfg.get("attn_implementation", "eager"),
        trust_remote_code=cfg.get("trust_remote_code", True),
    )
    model.config.use_cache = False
    if cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()

    # --- Dataset ---
    train_path = cfg.get("train_data")
    if not train_path or not Path(train_path).exists():
        raise ValueError(f"train_data not found: {train_path}")

    records = load_jsonl(train_path)
    dataset = TripletDataset(records, tokenizer, cfg.get("max_seq_length", 2048))
    loader = DataLoader(
        dataset,
        batch_size=cfg.get("per_device_train_batch_size", 2),
        shuffle=True,
        num_workers=2,
    )

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.get("learning_rate", 2e-5)),
        weight_decay=cfg.get("weight_decay", 0.01),
    )
    total_steps = (
        len(loader)
        * cfg.get("num_epochs", 3)
        // cfg.get("gradient_accumulation_steps", 8)
    )
    warmup_steps = int(total_steps * cfg.get("warmup_ratio", 0.03))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # --- Accelerate ---
    model, optimizer, loader, scheduler = accelerator.prepare(
        model, optimizer, loader, scheduler
    )

    alpha = cfg.get("alpha", 0.7)
    margin = cfg.get("margin", 0.3)
    output_dir = cfg.get("output_dir", "./output")
    os.makedirs(output_dir, exist_ok=True)
    grad_accum = cfg.get("gradient_accumulation_steps", 8)

    # --- Training loop ---
    global_step = 0
    for epoch in range(cfg.get("num_epochs", 3)):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}", disable=not accelerator.is_main_process)

        for step, batch in enumerate(pbar):
            with accelerator.accumulate(model):
                # CE Loss on positive (safe) sequences
                l_ce = ce_loss_from_logits(
                    model, batch["ce_input_ids"], batch["ce_labels"]
                )

                # Contrastive: extract last hidden states
                h_anchor = get_last_hidden_state(
                    model, batch["anchor_input_ids"], batch["anchor_attention_mask"]
                )
                h_pos = get_last_hidden_state(
                    model, batch["pos_input_ids"], batch["pos_attention_mask"]
                )
                h_neg = get_last_hidden_state(
                    model, batch["neg_input_ids"], batch["neg_attention_mask"]
                )

                l_contrastive = triplet_contrastive_loss(h_anchor, h_pos, h_neg, margin)

                loss = alpha * l_ce + (1.0 - alpha) * l_contrastive

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), cfg.get("max_grad_norm", 1.0))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            epoch_loss += loss.item()
            if accelerator.is_main_process:
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "l_ce": f"{l_ce.item():.4f}",
                    "l_contra": f"{l_contrastive.item():.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                })

            # Save checkpoint
            if global_step % cfg.get("save_steps", 500) == 0 and accelerator.is_main_process:
                ckpt_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
                unwrapped = accelerator.unwrap_model(model)
                unwrapped.save_pretrained(ckpt_dir)
                tokenizer.save_pretrained(ckpt_dir)
                print(f"Saved checkpoint to {ckpt_dir}")

        avg_loss = epoch_loss / len(loader)
        if accelerator.is_main_process:
            print(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")

    # --- Save final model ---
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        with open(os.path.join(output_dir, "training_config.yaml"), "w") as f:
            yaml.dump(cfg, f)
        print(f"Final model saved to: {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def load_config(config_path: str, overrides: dict) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v
    return cfg


def parse_args():
    parser = argparse.ArgumentParser(
        description="DIA-GUARD Full Fine-Tuning with Contrastive Loss"
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--train_data", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--alpha", type=float, default=None,
                        help="Weight for CE loss (default 0.7)")
    parser.add_argument("--margin", type=float, default=None,
                        help="Triplet contrastive margin (default 0.3)")
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None, dest="learning_rate")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    overrides = {k: v for k, v in vars(args).items() if k != "config"}
    cfg = load_config(args.config, overrides)
    train(cfg)
