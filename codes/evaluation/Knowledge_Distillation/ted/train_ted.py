"""
DIA-GUARD TED — Task-Aware Layer-wise Knowledge Distillation
=============================================================
Implements TED (Liang et al., ICML 2023) for distilling a fine-tuned teacher
decoder LLM into a smaller student decoder LLM via layer-wise hidden state alignment.

Key difference from MINILLM/GKD: TED aligns BOTH output distributions AND
intermediate hidden states using learned task-aware filters (W_l).

Loss (three terms):
  L_TED = α * L_CE + β * L_KD + γ * L_layer

  L_CE    = -Σ_t log p_student(y_t | x, y_{<t})
          (standard cross-entropy on safe responses)

  L_KD    = KL(softmax(z_teacher/τ) || softmax(z_student/τ)) * τ²
          (output logit-level KD with temperature τ)

  L_layer = (1/|L|) * Σ_{l ∈ L} (1/T) * Σ_t || W_l · h_s_l(t) - h_t_l'(t) ||²_F
          where W_l ∈ R^(d_teacher × d_student) is a learnable task-aware filter
          and l' is the teacher layer aligned to student layer l

Paper: Liang et al. (2023). Less is More: Task-aware Layer-wise Distillation
       for Language Model Compression. ICML 2023.
       https://arxiv.org/abs/2210.01351

Usage:
  python train_ted.py \\
      --teacher_model ../../../../models/FT/full_ft/qwen3-4b-ft \\
      --student_model Qwen/Qwen3Guard-Gen-0.6B \\
      --train_data /path/to/train.jsonl \\
      --output_dir ../../../../models/KD/ted/qwen3-guard-0.6b \\
      --lam 0.5

Data format (JSONL):
  {"prompt": "...", "response": "...", "label": 0 or 1}
"""

import argparse
import json
import math
import os
from pathlib import Path
from typing import List, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
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


def build_full_text(prompt: str, response: str, tokenizer) -> str:
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


class TEDDataset(torch.utils.data.Dataset):
    def __init__(self, records: List[Dict], tokenizer, max_length: int = 1024):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        text = build_full_text(rec["prompt"], rec["response"], self.tokenizer)
        enc = self.tokenizer(
            text, max_length=self.max_length, truncation=True,
            padding="max_length", return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": enc["input_ids"].squeeze(0).clone(),
        }


# ---------------------------------------------------------------------------
# Task-Aware Filter Module
# ---------------------------------------------------------------------------

class TaskAwareFilters(nn.Module):
    """
    Learnable linear projections W_l that map student hidden states
    to teacher hidden dimension at each aligned layer.

    W_l ∈ R^(d_teacher × d_student)

    These are trained jointly with the student during distillation.
    """

    def __init__(self, num_layers: int, student_hidden: int, teacher_hidden: int):
        super().__init__()
        self.filters = nn.ModuleList([
            nn.Linear(student_hidden, teacher_hidden, bias=False)
            for _ in range(num_layers)
        ])
        # Initialize with small random values for stable training
        for filt in self.filters:
            nn.init.normal_(filt.weight, mean=0.0, std=0.02)

    def forward(self, student_hidden_states: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
          student_hidden_states: list of (B, T, d_student) tensors
        Returns:
          list of (B, T, d_teacher) projected tensors
        """
        return [filt(hs) for filt, hs in zip(self.filters, student_hidden_states)]


# ---------------------------------------------------------------------------
# Layer alignment mapping
# ---------------------------------------------------------------------------

def get_layer_alignment(num_student_layers: int, num_teacher_layers: int,
                        num_align_layers: int = -1):
    """
    Map student layers to teacher layers (uniform spacing).

    If num_align_layers == -1, align all student layers.
    Otherwise, select num_align_layers evenly spaced student layers.

    Returns: list of (student_layer_idx, teacher_layer_idx) tuples
    """
    if num_align_layers == -1 or num_align_layers >= num_student_layers:
        student_layers = list(range(num_student_layers))
    else:
        # Evenly space the alignment layers
        student_layers = [
            round(i * (num_student_layers - 1) / (num_align_layers - 1))
            for i in range(num_align_layers)
        ]

    # Map each student layer to the corresponding teacher layer
    alignment = []
    for s_l in student_layers:
        t_l = round(s_l * (num_teacher_layers - 1) / (num_student_layers - 1))
        t_l = min(t_l, num_teacher_layers - 1)
        alignment.append((s_l, t_l))

    return alignment


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def layer_alignment_loss(
    projected_student_states: List[torch.Tensor],
    teacher_states: List[torch.Tensor],
    attention_mask: torch.Tensor,
    loss_type: str = "mse",
) -> torch.Tensor:
    """
    Compute layer-wise alignment loss between projected student and teacher states.

    L_layer = (1/|L|) * Σ_l (1/T) * Σ_t loss(W_l·h_s_l(t), h_t_l'(t))

    Args:
      projected_student_states: list of (B, T, d_teacher) tensors
      teacher_states: list of (B, T, d_teacher) tensors
      attention_mask: (B, T) padding mask
      loss_type: "mse" or "cosine"
    """
    total_loss = 0.0
    mask = attention_mask.float().unsqueeze(-1)  # (B, T, 1)

    for proj_s, t_s in zip(projected_student_states, teacher_states):
        if loss_type == "mse":
            l = ((proj_s - t_s) ** 2).mean(dim=-1, keepdim=True)  # (B, T, 1)
        elif loss_type == "cosine":
            # 1 - cosine similarity (loss form)
            l = 1.0 - F.cosine_similarity(proj_s, t_s, dim=-1).unsqueeze(-1)
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

        l = (l * mask).sum() / mask.sum().clamp(min=1)
        total_loss = total_loss + l

    return total_loss / max(len(projected_student_states), 1)


def ce_loss(logits, labels, pad_id: int = 0) -> torch.Tensor:
    """Standard cross-entropy with padding mask."""
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_labels[shift_labels == pad_id] = -100
    return nn.CrossEntropyLoss(ignore_index=-100)(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )


def kd_logit_loss(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    attention_mask: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    """
    Token-level KD loss between teacher and student output distributions.
    KL(softmax(z_t/τ) || softmax(z_s/τ)) * τ²

    Operates on the shifted (next-token prediction) logits and masks padding.
    """
    # Shift to next-token prediction positions
    t_logits = teacher_logits[:, :-1, :].contiguous()   # (B, T-1, V)
    s_logits = student_logits[:, :-1, :].contiguous()   # (B, T-1, V)
    mask = attention_mask[:, 1:].float()                  # (B, T-1)

    # Softmax / log-softmax with temperature
    t_probs = F.softmax(t_logits / temperature, dim=-1)           # (B, T-1, V)
    s_log_probs = F.log_softmax(s_logits / temperature, dim=-1)   # (B, T-1, V)

    # KL divergence per token: (B, T-1)
    kl = (t_probs * (t_probs.log().clamp(min=-1e9) - s_log_probs)).sum(dim=-1)
    kl = (kl * mask).sum() / mask.sum().clamp(min=1)

    return kl * (temperature ** 2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train(cfg: dict):
    set_seed(42)
    accelerator = Accelerator(
        mixed_precision="bf16" if cfg.get("bf16", True) else "no",
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 4),
    )

    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["student_model"], trust_remote_code=True, padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id or 0

    # --- Teacher (frozen, optionally quantized) ---
    print(f"Loading teacher: {cfg['teacher_model']}")
    bnb_cfg = None
    quant_mode = None
    if cfg.get("teacher_load_in_4bit", False):
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        quant_mode = "4bit"
    elif cfg.get("teacher_load_in_8bit", False):
        bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
        quant_mode = "8bit"
    teacher = AutoModelForCausalLM.from_pretrained(
        cfg["teacher_model"],
        quantization_config=bnb_cfg,
        torch_dtype=torch.bfloat16 if quant_mode is None else None,
        attn_implementation=cfg.get("attn_implementation", "eager"),
        trust_remote_code=True,
        output_hidden_states=True,
        device_map="auto" if quant_mode is not None else None,
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # Get teacher hidden dim and number of layers
    teacher_hidden = teacher.config.hidden_size
    teacher_num_layers = teacher.config.num_hidden_layers
    print(f"Teacher: hidden={teacher_hidden}, layers={teacher_num_layers}")

    # --- Student ---
    print(f"Loading student: {cfg['student_model']}")
    student = AutoModelForCausalLM.from_pretrained(
        cfg["student_model"],
        torch_dtype=torch.bfloat16 if cfg.get("bf16", True) else torch.float32,
        attn_implementation=cfg.get("attn_implementation", "eager"),
        trust_remote_code=True,
        output_hidden_states=True,
    )
    student.config.use_cache = False
    if cfg.get("gradient_checkpointing", True):
        student.gradient_checkpointing_enable()

    student_hidden = student.config.hidden_size
    student_num_layers = student.config.num_hidden_layers
    print(f"Student: hidden={student_hidden}, layers={student_num_layers}")

    # --- Layer alignment ---
    num_align = cfg.get("num_align_layers", -1)
    alignment = get_layer_alignment(student_num_layers, teacher_num_layers, num_align)
    actual_align = len(alignment)
    print(f"Aligning {actual_align} layer pairs: {alignment[:3]}{'...' if actual_align > 3 else ''}")

    # --- Task-aware filters ---
    filters = TaskAwareFilters(
        num_layers=actual_align,
        student_hidden=student_hidden,
        teacher_hidden=teacher_hidden,
    )
    if cfg.get("bf16", True):
        filters = filters.to(torch.bfloat16)

    # --- Dataset ---
    train_path = cfg.get("train_data")
    if not train_path or not Path(train_path).exists():
        raise ValueError(f"train_data not found: {train_path}")

    records = load_jsonl(train_path)
    dataset = TEDDataset(records, tokenizer, cfg.get("max_seq_length", 1024))
    loader = DataLoader(
        dataset,
        batch_size=cfg.get("per_device_train_batch_size", 2),
        shuffle=True,
        num_workers=2,
    )
    print(f"Training samples: {len(dataset)}")

    # --- Optimizer (student params + filter params) ---
    all_params = list(student.parameters()) + list(filters.parameters())
    optimizer = torch.optim.AdamW(
        all_params,
        lr=float(cfg.get("learning_rate", 2e-5)),
        weight_decay=cfg.get("weight_decay", 0.01),
    )
    total_steps = (
        len(loader) * cfg.get("num_epochs", 3)
        // cfg.get("gradient_accumulation_steps", 4)
    )
    warmup_steps = int(total_steps * cfg.get("warmup_ratio", 0.03))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    student, filters, optimizer, loader, scheduler = accelerator.prepare(
        student, filters, optimizer, loader, scheduler
    )
    if quant_mode is None:
        teacher = teacher.to(accelerator.device)

    alpha = cfg.get("alpha", 1.0)       # CE loss weight
    beta = cfg.get("beta", 1.0)         # KD logit loss weight
    lam = cfg.get("lam", 0.5)           # layer alignment loss weight (γ in paper)
    temperature = cfg.get("temperature", 2.0)
    loss_type = cfg.get("layer_loss_type", "mse")
    output_dir = cfg.get("output_dir", "./output")
    os.makedirs(output_dir, exist_ok=True)
    global_step = 0

    # --- Training loop ---
    for epoch in range(cfg.get("num_epochs", 3)):
        student.train()
        filters.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}", disable=not accelerator.is_main_process)

        for batch in pbar:
            with accelerator.accumulate(student):
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                labels = batch["labels"]

                # --- Teacher forward (no grad, collect hidden states) ---
                with torch.no_grad():
                    teacher_out = teacher(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                    )
                # teacher_out.hidden_states: tuple of (B, T, d_teacher), len = num_layers+1
                # Index 0 = embedding, 1..N = transformer layers
                teacher_hidden_states = teacher_out.hidden_states[1:]  # layers only

                # --- Student forward (with grad, collect hidden states) ---
                student_out = student(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                student_hidden_states = student_out.hidden_states[1:]

                # --- CE Loss ---
                l_ce = ce_loss(student_out.logits, labels, pad_id)

                # --- KD logit loss (teacher vs student output distributions) ---
                l_kd = kd_logit_loss(
                    teacher_out.logits, student_out.logits, attention_mask, temperature
                )

                # --- Layer alignment loss ---
                # Collect aligned hidden states
                aligned_student = []
                aligned_teacher = []
                for s_l, t_l in alignment:
                    aligned_student.append(student_hidden_states[s_l])
                    aligned_teacher.append(teacher_hidden_states[t_l])

                # Apply task-aware filters: W_l · h_s_l
                projected_student = accelerator.unwrap_model(filters)(aligned_student)

                l_layer = layer_alignment_loss(
                    projected_student, aligned_teacher, attention_mask, loss_type
                )

                # --- Total loss: α·L_CE + β·L_KD + γ·L_layer ---
                loss = alpha * l_ce + beta * l_kd + lam * l_layer

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        all_params, cfg.get("max_grad_norm", 1.0)
                    )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            if accelerator.is_main_process:
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "ce": f"{l_ce.item():.4f}",
                    "kd": f"{l_kd.item():.4f}",
                    "layer": f"{l_layer.item():.4f}",
                })

            if global_step % cfg.get("save_steps", 500) == 0 and accelerator.is_main_process:
                ckpt_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
                unwrapped_student = accelerator.unwrap_model(student)
                unwrapped_student.save_pretrained(ckpt_dir)
                tokenizer.save_pretrained(ckpt_dir)
                # Also save filters
                torch.save(
                    accelerator.unwrap_model(filters).state_dict(),
                    os.path.join(ckpt_dir, "task_aware_filters.pt"),
                )
                print(f"Saved checkpoint: {ckpt_dir}")

    # --- Save final student and filters ---
    if accelerator.is_main_process:
        unwrapped_student = accelerator.unwrap_model(student)
        unwrapped_student.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        torch.save(
            accelerator.unwrap_model(filters).state_dict(),
            os.path.join(output_dir, "task_aware_filters.pt"),
        )
        # Save alignment metadata
        alignment_meta = {
            "student_num_layers": student_num_layers,
            "teacher_num_layers": teacher_num_layers,
            "student_hidden": student_hidden,
            "teacher_hidden": teacher_hidden,
            "alignment": alignment,
            "num_align_layers": actual_align,
        }
        with open(os.path.join(output_dir, "alignment_metadata.json"), "w") as f:
            json.dump(alignment_meta, f, indent=2)
        with open(os.path.join(output_dir, "training_config.yaml"), "w") as f:
            yaml.dump(cfg, f)
        print(f"Student model saved to: {output_dir}")
        print(f"Task-aware filters saved to: {output_dir}/task_aware_filters.pt")


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
        description="DIA-GUARD TED — Task-Aware Layer-wise Knowledge Distillation"
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--teacher_model", type=str, required=True)
    parser.add_argument("--student_model", type=str, required=True)
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Weight for CE task loss (default 1.0)")
    parser.add_argument("--beta", type=float, default=1.0,
                        help="Weight for KD logit loss (default 1.0)")
    parser.add_argument("--lam", type=float, default=0.5,
                        help="Weight for layer alignment loss γ (default 0.5)")
    parser.add_argument("--temperature", type=float, default=2.0,
                        help="Temperature τ for KD logit loss (default 2.0)")
    parser.add_argument("--num_align_layers", type=int, default=-1,
                        help="Number of layers to align (-1 = all layers)")
    parser.add_argument("--layer_loss_type", type=str, default="mse",
                        choices=["mse", "cosine"])
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2,
                        dest="per_device_train_batch_size")
    parser.add_argument("--lr", type=float, default=2e-5, dest="learning_rate")
    parser.add_argument("--teacher_load_in_8bit", action="store_true",
                        help="Load teacher in 8-bit (bnb)")
    parser.add_argument("--teacher_load_in_4bit", action="store_true",
                        help="Quantize teacher to 4-bit to reduce VRAM")
    parser.add_argument("--bf16", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--gradient_checkpointing", type=lambda x: x.lower() == "true",
                        default=True)
    parser.add_argument("--save_steps", type=int, default=500)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    overrides = {k: v for k, v in vars(args).items() if k != "config"}
    cfg = load_config(args.config, overrides)
    train(cfg)
