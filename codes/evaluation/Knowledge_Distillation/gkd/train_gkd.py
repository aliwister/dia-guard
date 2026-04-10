"""
DIA-GUARD GKD — On-Policy Generalized Knowledge Distillation
=============================================================
Implements GKD (Agarwal et al., ICLR 2024) for distilling a fine-tuned teacher
decoder LLM into a smaller student decoder LLM.

Key idea: Train the student on a *mixture* of student-generated and
teacher-generated sequences, using the teacher's soft token-level probabilities
as targets (on-policy training).

Algorithm per batch:
  With prob λ: sample y_s ~ p_student(·|x)   [on-policy: student generation]
  With prob (1-λ): use reference y from data   [off-policy: teacher/data distribution]
  Minimize D(p_teacher(·|x,y_{<t}) || p_student(·|x,y_{<t})) at each position t

Divergence options: Forward KL, Reverse KL, JSD, TVD

Paper: Agarwal et al. (2024). On-Policy Distillation of Language Models. ICLR 2024.
       https://arxiv.org/abs/2306.13649

Note on TRL integration:
  TRL >= 0.9.0 ships a native GKDTrainer that wraps this logic. If you prefer the
  high-level API, use:
    from trl import GKDTrainer, GKDConfig
  This script provides a transparent, research-friendly implementation without TRL
  as a hard dependency on that specific version.

Usage:
  python train_gkd.py \\
      --teacher_model ../../../../models/FT/full_ft/qwen3-4b-ft \\
      --student_model Qwen/Qwen3Guard-Gen-0.6B \\
      --train_data /path/to/train.jsonl \\
      --output_dir ../../../../models/KD/gkd/qwen3-guard-0.6b \\
      --lam 0.5 \\
      --divergence jsd

Data format (JSONL):
  {"prompt": "...", "response": "...", "label": 0 or 1}
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import List, Dict, Literal

import torch
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
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are DIA-Guard, a multilingual safety assistant. "
    "Analyze the following input and provide a safe, informed response."
)


def build_prompt_only(prompt: str, tokenizer) -> str:
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


class GKDDataset(torch.utils.data.Dataset):
    def __init__(self, records: List[Dict], tokenizer,
                 max_prompt_length: int = 512, max_seq_length: int = 1024):
        self.records = records
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        prompt_text = build_prompt_only(rec["prompt"], self.tokenizer)
        ref_text = build_full_text(rec["prompt"], rec["response"], self.tokenizer)

        def tok(text, max_len):
            enc = self.tokenizer(
                text, max_length=max_len, truncation=True,
                padding="max_length", return_tensors="pt",
            )
            return enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0)

        p_ids, p_mask = tok(prompt_text, self.max_prompt_length)
        r_ids, r_mask = tok(ref_text, self.max_seq_length)

        return {
            "prompt_input_ids": p_ids,
            "prompt_attention_mask": p_mask,
            "ref_input_ids": r_ids,
            "ref_attention_mask": r_mask,
            "prompt_text": rec["prompt"],
        }


# ---------------------------------------------------------------------------
# Divergence functions (token-level, shape: (B, T, V))
# ---------------------------------------------------------------------------

def forward_kl(p_teacher_logits, p_student_logits, temperature: float = 1.0):
    """
    Forward KL: KL(p_T || p_S) = Σ p_T * (log p_T - log p_S)
    Stable: avoids explicit p_T log computation by using logsumexp.
    """
    log_pt = F.log_softmax(p_teacher_logits / temperature, dim=-1)
    log_ps = F.log_softmax(p_student_logits / temperature, dim=-1)
    pt = log_pt.exp()
    return (pt * (log_pt - log_ps)).sum(dim=-1)  # (B, T)


def reverse_kl(p_teacher_logits, p_student_logits, temperature: float = 1.0):
    """
    Reverse KL: KL(p_S || p_T) = Σ p_S * (log p_S - log p_T)
    """
    log_pt = F.log_softmax(p_teacher_logits / temperature, dim=-1)
    log_ps = F.log_softmax(p_student_logits / temperature, dim=-1)
    ps = log_ps.exp()
    return (ps * (log_ps - log_pt)).sum(dim=-1)  # (B, T)


def jsd(p_teacher_logits, p_student_logits, temperature: float = 1.0):
    """
    Jensen-Shannon Divergence: ½ KL(p_T||M) + ½ KL(p_S||M), M = ½(p_T + p_S)
    Symmetric and bounded in [0, log 2].
    """
    log_pt = F.log_softmax(p_teacher_logits / temperature, dim=-1)
    log_ps = F.log_softmax(p_student_logits / temperature, dim=-1)
    pt = log_pt.exp()
    ps = log_ps.exp()
    m = 0.5 * (pt + ps)
    log_m = m.log().clamp(min=-1e9)
    kl_tm = (pt * (log_pt - log_m)).sum(dim=-1)
    kl_sm = (ps * (log_ps - log_m)).sum(dim=-1)
    return 0.5 * (kl_tm + kl_sm)  # (B, T)


def tvd(p_teacher_logits, p_student_logits, temperature: float = 1.0):
    """
    Total Variation Distance: ½ Σ |p_T - p_S|
    """
    pt = F.softmax(p_teacher_logits / temperature, dim=-1)
    ps = F.softmax(p_student_logits / temperature, dim=-1)
    return 0.5 * (pt - ps).abs().sum(dim=-1)  # (B, T)


DIVERGENCE_FNS = {
    "fkl": forward_kl,
    "rkl": reverse_kl,
    "jsd": jsd,
    "tvd": tvd,
}


# ---------------------------------------------------------------------------
# GKD training step
# ---------------------------------------------------------------------------

def gkd_loss(
    teacher,
    student,
    input_ids,
    attention_mask,
    divergence_fn,
    temperature: float = 1.0,
):
    """
    Compute GKD loss at each token position:
      L = (1/T) * Σ_t D(p_teacher(·|x,y_{<t}) || p_student(·|x,y_{<t}))

    Both teacher and student forward-pass on the same sequence.
    Teacher runs in no-grad mode.
    """
    with torch.no_grad():
        teacher_out = teacher(input_ids=input_ids, attention_mask=attention_mask)
    student_out = student(input_ids=input_ids, attention_mask=attention_mask)

    teacher_logits = teacher_out.logits[:, :-1, :]  # (B, T-1, V)
    student_logits = student_out.logits[:, :-1, :]  # (B, T-1, V)

    # Compute token-level divergence
    div = divergence_fn(teacher_logits, student_logits, temperature)  # (B, T-1)

    # Mask padding
    pad_id = student.config.pad_token_id or 0
    mask = (input_ids[:, 1:] != pad_id).float()  # (B, T-1)
    div = div * mask

    # Mean over non-padding tokens
    loss = div.sum() / mask.sum().clamp(min=1)
    return loss


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(cfg: dict):
    set_seed(42)
    accelerator = Accelerator(
        mixed_precision="bf16" if cfg.get("bf16", True) else "no",
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 4),
    )

    # --- Tokenizer ---
    # NOTE: padding_side is set dynamically:
    #   "left"  during generation (model.generate requires left-padding for batched decoding)
    #   "right" during loss computation (attention masks exclude right-side padding correctly)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["student_model"], trust_remote_code=True, padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

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
        device_map="auto" if quant_mode is not None else None,
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

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

    # --- Divergence function ---
    div_name = cfg.get("divergence", "jsd").lower()
    if div_name not in DIVERGENCE_FNS:
        raise ValueError(f"Unknown divergence '{div_name}'. Choose from: {list(DIVERGENCE_FNS)}")
    divergence_fn = DIVERGENCE_FNS[div_name]
    print(f"Using divergence: {div_name.upper()}")

    # --- Dataset ---
    train_path = cfg.get("train_data")
    if not train_path or not Path(train_path).exists():
        raise ValueError(f"train_data not found: {train_path}")

    records = load_jsonl(train_path)
    dataset = GKDDataset(
        records, tokenizer,
        max_prompt_length=cfg.get("max_prompt_length", 512),
        max_seq_length=cfg.get("max_seq_length", 1024),
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.get("per_device_train_batch_size", 2),
        shuffle=True,
        num_workers=2,
    )
    print(f"Training samples: {len(dataset)}")

    # --- Optimizer ---
    lam = cfg.get("lam", 0.5)  # mixing ratio: fraction of student-generated sequences
    temperature = cfg.get("temperature", 1.0)
    max_new_tokens = cfg.get("max_new_tokens", 256)
    top_p = cfg.get("top_p", 0.9)

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

    student, optimizer, loader, scheduler = accelerator.prepare(
        student, optimizer, loader, scheduler
    )
    if quant_mode is None:
        teacher = teacher.to(accelerator.device)

    output_dir = cfg.get("output_dir", "./output")
    os.makedirs(output_dir, exist_ok=True)
    global_step = 0

    # --- Training loop ---
    for epoch in range(cfg.get("num_epochs", 3)):
        student.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}", disable=not accelerator.is_main_process)

        for batch in pbar:
            with accelerator.accumulate(student):
                # GKD mixing: with prob lam use student-generated, else use reference
                if random.random() < lam:
                    # On-policy: generate from student
                    # Switch to left-padding for batched generation
                    tokenizer.padding_side = "left"
                    student.eval()
                    with torch.no_grad():
                        gen_ids = accelerator.unwrap_model(student).generate(
                            input_ids=batch["prompt_input_ids"],
                            attention_mask=batch["prompt_attention_mask"],
                            max_new_tokens=max_new_tokens,
                            do_sample=True,
                            temperature=temperature,
                            top_p=top_p,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                        )
                    student.train()
                    tokenizer.padding_side = "right"  # restore for loss computation
                    pad_id = tokenizer.pad_token_id or 0
                    gen_mask = (gen_ids != pad_id).long()
                    seq_ids = gen_ids
                    seq_mask = gen_mask
                else:
                    # Off-policy: use reference sequence from data
                    seq_ids = batch["ref_input_ids"]
                    seq_mask = batch["ref_attention_mask"]

                loss = gkd_loss(
                    teacher, student,
                    seq_ids, seq_mask,
                    divergence_fn, temperature,
                )

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        student.parameters(), cfg.get("max_grad_norm", 1.0)
                    )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            if accelerator.is_main_process:
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    div=div_name,
                    lam=lam,
                )

            if global_step % cfg.get("save_steps", 500) == 0 and accelerator.is_main_process:
                ckpt_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
                unwrapped = accelerator.unwrap_model(student)
                unwrapped.save_pretrained(ckpt_dir)
                tokenizer.save_pretrained(ckpt_dir)
                print(f"Saved checkpoint: {ckpt_dir}")

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
        description="DIA-GUARD GKD — On-Policy Generalized Knowledge Distillation"
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--teacher_model", type=str, required=True)
    parser.add_argument("--student_model", type=str, required=True)
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--lam", type=float, default=0.5,
                        help="Mixing ratio: fraction of student-generated sequences (0=off-policy, 1=on-policy)")
    parser.add_argument("--divergence", type=str, default="jsd",
                        choices=["fkl", "rkl", "jsd", "tvd"],
                        help="Divergence function for distillation loss")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2,
                        dest="per_device_train_batch_size")
    parser.add_argument("--lr", type=float, default=1e-5, dest="learning_rate")
    parser.add_argument("--teacher_load_in_8bit", action="store_true",
                        help="Load teacher in 8-bit (bnb)")
    parser.add_argument("--teacher_load_in_4bit", action="store_true")
    parser.add_argument("--bf16", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--save_steps", type=int, default=500)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    overrides = {k: v for k, v in vars(args).items() if k != "config"}
    cfg = load_config(args.config, overrides)
    train(cfg)
