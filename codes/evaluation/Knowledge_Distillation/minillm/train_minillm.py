"""
DIA-GUARD MINILLM — Reverse KL Knowledge Distillation
======================================================
Implements MINILLM (Gu et al., ICLR 2024) for distilling a fine-tuned teacher
decoder LLM into a smaller student decoder LLM using reverse KL divergence.

Core algorithm:
  1. Student generates sequences y_s ~ p_student(·|x)
  2. Compute per-sequence reward: r(y_s) = log p_teacher(y_s|x) - log p_student(y_s|x)
  3. Estimate baseline V(x) via EMA to reduce variance
  4. Policy gradient update: L_PG = -(r(y_s) - V(x)) * sum_t log p_student(y_s_t|x, y_s_{<t})
  5. SFT regularization: L_SFT = -sum_t log p_student(y_ref_t|x, y_ref_{<t})
  6. Total loss: L = L_PG + beta * L_SFT

Paper: Gu et al. (2024). Knowledge Distillation of Large Language Models. ICLR 2024.
       https://arxiv.org/abs/2306.08543
Official code: https://github.com/microsoft/LMOps/tree/main/minillm

Usage:
  python train_minillm.py \\
      --teacher_model ../../../../models/FT/full_ft/qwen3-4b-ft \\
      --student_model Qwen/Qwen3Guard-Gen-0.6B \\
      --train_data /path/to/train.jsonl \\
      --output_dir ../../../../models/KD/minillm/qwen3-guard-0.6b

Data format (JSONL):
  {"prompt": "...", "response": "...", "label": 0 or 1}
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Optional

import torch
import torch.nn.functional as F
import yaml
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
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


def build_prompt_only(prompt: str, tokenizer) -> str:
    """Build prompt-only string (no response) for student generation."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return f"<|system|>{SYSTEM_PROMPT}</s>\n<|user|>{prompt}</s>\n<|assistant|>"


def build_full_text(prompt: str, response: str, tokenizer) -> str:
    """Build full prompt+response string for SFT loss."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    return (
        f"<|system|>{SYSTEM_PROMPT}</s>\n"
        f"<|user|>{prompt}</s>\n"
        f"<|assistant|>{response}</s>"
    )


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


class MINILLMDataset(torch.utils.data.Dataset):
    """Each example provides a prompt (for generation) and a reference response (for SFT reg)."""

    def __init__(self, records: List[Dict], tokenizer, max_prompt_length: int = 512,
                 max_ref_length: int = 2048):
        self.records = records
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.max_ref_length = max_ref_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        prompt_text = build_prompt_only(rec["prompt"], self.tokenizer)
        ref_text = build_full_text(rec["prompt"], rec["response"], self.tokenizer)

        prompt_enc = self.tokenizer(
            prompt_text, max_length=self.max_prompt_length, truncation=True,
            padding="max_length", return_tensors="pt",
        )
        ref_enc = self.tokenizer(
            ref_text, max_length=self.max_ref_length, truncation=True,
            padding="max_length", return_tensors="pt",
        )

        return {
            "prompt_input_ids": prompt_enc["input_ids"].squeeze(0),
            "prompt_attention_mask": prompt_enc["attention_mask"].squeeze(0),
            "ref_input_ids": ref_enc["input_ids"].squeeze(0),
            "ref_attention_mask": ref_enc["attention_mask"].squeeze(0),
            "ref_labels": ref_enc["input_ids"].squeeze(0).clone(),
        }


# ---------------------------------------------------------------------------
# Sequence log-probability computation
# ---------------------------------------------------------------------------

def compute_sequence_logprob(model, input_ids, attention_mask):
    """
    Compute per-sequence log probability under the model:
      log p(y|x) = Σ_t log p(y_t | x, y_{<t})

    Returns:
      seq_logprob: (B,) tensor
      token_logprobs: (B, T-1) tensor
    """
    with torch.no_grad() if model.training is False else torch.enable_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # (B, T, V)

    # Shift: predict t+1 from t
    shift_logits = logits[:, :-1, :].contiguous()   # (B, T-1, V)
    shift_labels = input_ids[:, 1:].contiguous()     # (B, T-1)

    log_probs = F.log_softmax(shift_logits, dim=-1)  # (B, T-1, V)
    token_lp = log_probs.gather(
        2, shift_labels.unsqueeze(-1)
    ).squeeze(-1)  # (B, T-1)

    # Mask padding (pad_token_id → 0 contribution)
    pad_id = model.config.pad_token_id if model.config.pad_token_id is not None else 0
    mask = (shift_labels != pad_id).float()
    token_lp = token_lp * mask

    seq_logprob = token_lp.sum(dim=-1)  # (B,)
    return seq_logprob, token_lp


def compute_sft_loss(model, input_ids, labels):
    """Standard cross-entropy SFT loss (with -100 masking)."""
    outputs = model(input_ids=input_ids)
    logits = outputs.logits
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    pad_id = model.config.pad_token_id if model.config.pad_token_id is not None else 0
    shift_labels[shift_labels == pad_id] = -100

    import torch.nn as nn
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    return loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )


# ---------------------------------------------------------------------------
# MINILLM Trainer
# ---------------------------------------------------------------------------

class MINILLMTrainer:
    """
    MINILLM training loop with:
      - Policy gradient (REINFORCE) using reverse KL reward
      - EMA baseline for variance reduction
      - SFT regularization on reference responses
    """

    def __init__(self, teacher, student, tokenizer, cfg, accelerator):
        self.teacher = teacher
        self.student = student
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.accelerator = accelerator

        self.beta = cfg.get("beta", 0.5)
        self.baseline_ema = cfg.get("baseline_ema", 0.99)
        self.max_new_tokens = cfg.get("max_new_tokens", 256)
        self.temperature = cfg.get("temperature", 1.0)
        self.top_p = cfg.get("top_p", 0.9)
        self.baseline = 0.0  # EMA baseline value

    def generate_student_responses(self, prompt_input_ids, prompt_attention_mask):
        """
        Sample sequences from student: y_s ~ p_student(·|x)

        The dataset right-pads prompts (cheaper for SFT loss). For generation
        we need pads on the LEFT, so we re-arrange the batch in-place: every
        row's pad tokens are rolled to the left and the real tokens to the
        right. This is equivalent to left-padding without re-tokenizing.
        """
        pad_id = self.tokenizer.pad_token_id
        # Re-pad: left-shift each row so pads are on the left.
        B, T = prompt_input_ids.shape
        new_ids = torch.full_like(prompt_input_ids, pad_id)
        new_mask = torch.zeros_like(prompt_attention_mask)
        for i in range(B):
            valid = prompt_attention_mask[i].sum().item()
            new_ids[i, T - valid:] = prompt_input_ids[i, :valid]
            new_mask[i, T - valid:] = 1
        self.student.eval()
        with torch.no_grad():
            generated = self.student.generate(
                input_ids=new_ids,
                attention_mask=new_mask,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                pad_token_id=pad_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        self.student.train()
        return generated

    def compute_reward(self, generated_ids, generated_attention_mask):
        """
        r(y_s|x) = (log p_teacher(y_s|x) - log p_student(y_s|x)) / |y_s|

        Per-token mean (length-normalized) to keep reward magnitudes stable;
        equivalent to a per-token reverse-KL estimate. Without this the PG
        loss explodes to ~1e6 because both logprob sums scale with sequence
        length.
        """
        # Teacher log-prob (no grad)
        self.teacher.eval()
        with torch.no_grad():
            teacher_logprob, _ = compute_sequence_logprob(
                self.teacher, generated_ids, generated_attention_mask
            )

        # Student log-prob (with grad)
        student_logprob, token_lp = compute_sequence_logprob(
            self.student, generated_ids, generated_attention_mask
        )

        # Length normalization (number of non-pad response tokens, min 1)
        pad_id = self.tokenizer.pad_token_id
        seq_len = (generated_ids != pad_id).sum(dim=-1).clamp_min(1).float()

        reward = (teacher_logprob.detach() - student_logprob.detach()) / seq_len
        # Clip reward to bound PG loss magnitude during early training when
        # untrained students produce very low-prob sequences.
        reward = reward.clamp(-5.0, 5.0)
        return reward, student_logprob / seq_len, token_lp

    def policy_gradient_loss(self, reward, student_logprob):
        """
        L_PG = -E[(r(y) - V(x)) * log p_student(y)]
        Update baseline V via EMA.
        """
        advantage = reward - self.baseline
        # Update EMA baseline
        batch_mean_reward = reward.mean().item()
        self.baseline = (
            self.baseline_ema * self.baseline
            + (1 - self.baseline_ema) * batch_mean_reward
        )
        loss = -(advantage.detach() * student_logprob).mean()
        return loss

    def step(self, batch):
        """Single training step. Returns (total_loss, pg_loss, sft_loss, mean_reward)."""
        # 1. Generate student responses
        generated_ids = self.generate_student_responses(
            batch["prompt_input_ids"], batch["prompt_attention_mask"]
        )
        # Build attention mask for generated sequences
        pad_id = self.tokenizer.pad_token_id
        gen_attention_mask = (generated_ids != pad_id).long()

        # 2. Compute reward
        reward, student_logprob, _ = self.compute_reward(generated_ids, gen_attention_mask)

        # 3. Policy gradient loss
        l_pg = self.policy_gradient_loss(reward, student_logprob)

        # 4. SFT regularization loss
        l_sft = compute_sft_loss(
            self.student, batch["ref_input_ids"], batch["ref_labels"]
        )

        # 5. Total loss
        loss = l_pg + self.beta * l_sft

        return loss, l_pg, l_sft, reward.mean()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train(cfg: dict):
    set_seed(42)
    accelerator = Accelerator(
        mixed_precision="bf16" if cfg.get("bf16", True) else "no",
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 4),
    )

    # --- Tokenizer (shared) ---
    # NOTE: Use left-padding for model.generate() (batched decoding),
    # right-padding for SFT loss computation. We set "right" as default and
    # switch in generate_student_responses() below.
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["student_model"],
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # --- Teacher (frozen, optionally quantized) ---
    print(f"Loading teacher: {cfg['teacher_model']}")
    teacher_bnb = None
    quant_mode = None
    if cfg.get("teacher_load_in_4bit", False):
        teacher_bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        quant_mode = "4bit"
    elif cfg.get("teacher_load_in_8bit", False):
        teacher_bnb = BitsAndBytesConfig(load_in_8bit=True)
        quant_mode = "8bit"
    teacher = AutoModelForCausalLM.from_pretrained(
        cfg["teacher_model"],
        quantization_config=teacher_bnb,
        torch_dtype=torch.bfloat16 if quant_mode is None else None,
        attn_implementation=cfg.get("attn_implementation", "eager"),
        trust_remote_code=True,
        device_map="auto" if quant_mode is not None else None,
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    print(f"Teacher loaded and frozen.")

    # --- Student ---
    print(f"Loading student: {cfg['student_model']}")
    student = AutoModelForCausalLM.from_pretrained(
        cfg["student_model"],
        torch_dtype=torch.bfloat16 if cfg.get("bf16", True) else torch.float32,
        attn_implementation=cfg.get("attn_implementation", "eager"),
        trust_remote_code=True,
    )
    student.config.use_cache = False
    if cfg.get("gradient_checkpointing", True):
        student.gradient_checkpointing_enable()

    # --- Dataset ---
    train_path = cfg.get("train_data")
    if not train_path or not Path(train_path).exists():
        raise ValueError(f"train_data not found: {train_path}")

    records = load_jsonl(train_path)
    dataset = MINILLMDataset(
        records, tokenizer,
        max_prompt_length=cfg.get("max_prompt_length", 512),
        max_ref_length=cfg.get("max_seq_length", 1024),
    )
    loader = DataLoader(
        dataset, batch_size=cfg.get("per_device_train_batch_size", 2),
        shuffle=True, num_workers=2,
    )
    print(f"Training samples: {len(dataset)}")

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=float(cfg.get("learning_rate", 1e-5)),
        weight_decay=cfg.get("weight_decay", 0.01),
    )
    total_steps = (
        len(loader) * cfg.get("num_epochs", 3)
        // cfg.get("gradient_accumulation_steps", 4)
    )
    warmup_steps = int(total_steps * cfg.get("warmup_ratio", 0.03))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # --- Accelerate (student only) ---
    student, optimizer, loader, scheduler = accelerator.prepare(
        student, optimizer, loader, scheduler
    )
    if not cfg.get("teacher_load_in_4bit"):
        teacher = teacher.to(accelerator.device)

    # --- MINILLM Trainer ---
    minillm = MINILLMTrainer(teacher, student, tokenizer, cfg, accelerator)

    output_dir = cfg.get("output_dir", "./output")
    os.makedirs(output_dir, exist_ok=True)
    global_step = 0

    # --- Training loop ---
    for epoch in range(cfg.get("num_epochs", 3)):
        student.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}", disable=not accelerator.is_main_process)
        epoch_loss = 0.0

        for batch in pbar:
            with accelerator.accumulate(student):
                loss, l_pg, l_sft, mean_reward = minillm.step(batch)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(student.parameters(), cfg.get("max_grad_norm", 1.0))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            epoch_loss += loss.item()
            if accelerator.is_main_process:
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "pg": f"{l_pg.item():.4f}",
                    "sft": f"{l_sft.item():.4f}",
                    "reward": f"{mean_reward.item():.4f}",
                    "baseline": f"{minillm.baseline:.4f}",
                })

            if global_step % cfg.get("save_steps", 500) == 0 and accelerator.is_main_process:
                ckpt_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
                unwrapped = accelerator.unwrap_model(student)
                unwrapped.save_pretrained(ckpt_dir)
                tokenizer.save_pretrained(ckpt_dir)
                print(f"Saved checkpoint: {ckpt_dir}")

        if accelerator.is_main_process:
            print(f"Epoch {epoch+1} avg loss: {epoch_loss / len(loader):.4f}")

    # --- Save final student ---
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(student)
        unwrapped.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        with open(os.path.join(output_dir, "training_config.yaml"), "w") as f:
            yaml.dump(cfg, f)
        print(f"Student model saved to: {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def load_config(path: str, overrides: dict) -> dict:
    cfg = {}
    if path and Path(path).exists():
        with open(path) as f:
            cfg = yaml.safe_load(f) or {}
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v
    return cfg


def parse_args():
    parser = argparse.ArgumentParser(
        description="DIA-GUARD MINILLM — Reverse KL Knowledge Distillation"
    )
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config")
    parser.add_argument("--teacher_model", type=str, required=True,
                        help="Path or HF ID of fine-tuned teacher")
    parser.add_argument("--student_model", type=str, required=True,
                        help="Path or HF ID of student base model")
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2,
                        dest="per_device_train_batch_size")
    parser.add_argument("--lr", type=float, default=1e-5, dest="learning_rate")
    parser.add_argument("--beta", type=float, default=0.5,
                        help="SFT regularization weight")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--teacher_load_in_8bit", action="store_true",
                        help="Load teacher in 8-bit (bnb) — less accurate than bf16 but more than 4bit")
    parser.add_argument("--teacher_load_in_4bit", action="store_true",
                        help="Load teacher in 4-bit to save memory")
    parser.add_argument("--bf16", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--gradient_checkpointing", type=lambda x: x.lower() == "true",
                        default=True)
    parser.add_argument("--save_steps", type=int, default=500)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config_path = args.config
    overrides = {k: v for k, v in vars(args).items() if k != "config"}
    cfg = load_config(config_path, overrides)
    train(cfg)
