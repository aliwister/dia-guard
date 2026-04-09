"""
DIA-GUARD PEFT Fine-Tuning — LoRA / QLoRA + Contrastive Loss
=============================================================
Fine-tunes a decoder-only LLM using LoRA adapters with a combined
Cross-Entropy + Triplet Contrastive loss. The LoRA-adapted model learns to
separate safe and unsafe response representations via its hidden states.

Loss:
  L_total = alpha * L_CE + (1 - alpha) * L_triplet

  L_CE      = next-token prediction on safe responses
  L_triplet = max(0, margin - cos(h_anc, h_pos) + cos(h_anc, h_neg))

References:
  - LoRA: https://arxiv.org/abs/2106.09685
  - QLoRA: https://arxiv.org/abs/2305.14314
  - SimCSE: https://arxiv.org/abs/2104.08821

Data format (JSONL triplets):
  {"prompt": "...", "response": "safe ...", "neg_response": "unsafe ...", "label": 1}

Usage:
  python train_contrastive_lora.py --config configs/qwen3_4b_lora.yaml \\
      --train_data /data/triplets.jsonl --alpha 0.7 --margin 0.3
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
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

# Add parent dir to path for data_utils
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_utils import load_jsonl_records


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are DIA-Guard, a multilingual safety assistant. "
    "Analyze the following input and provide a safe, informed response."
)


def build_text(prompt: str, response: str, tokenizer) -> str:
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
    skipped = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            # Contrastive training requires a paired neg_text (dia_llm records only).
            if "neg_text" in rec and not rec["neg_text"]:
                skipped += 1
                continue
            records.append(rec)
    if skipped:
        print(f"[contrastive] Skipped {skipped:,} records with no neg_text (multi_value).")
    return records


class TripletLoRADataset(torch.utils.data.Dataset):
    def __init__(self, records: List[Dict], tokenizer, max_length: int = 2048):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        # Support both dia_splits format {"text": ..., "label": ..., "neg_text": ...}
        # and legacy format {"prompt": ..., "response": ..., "neg_response": ...}
        prompt = rec.get("prompt", rec.get("text", ""))
        label  = rec.get("label", 0)
        if "response" in rec:
            safe_resp   = rec["response"]
            unsafe_resp = rec.get("neg_response", "safe" if label == 1 else "unsafe")
        else:
            safe_resp   = "safe"   if label == 0 else "unsafe"
            unsafe_resp = "unsafe" if label == 0 else "safe"

        anchor_text = f"<|system|>{SYSTEM_PROMPT}</s>\n<|user|>{prompt}</s>\n<|assistant|>"
        pos_text = build_text(prompt, safe_resp, self.tokenizer)
        neg_text = build_text(prompt, unsafe_resp, self.tokenizer)

        def tok(text):
            enc = self.tokenizer(
                text, max_length=self.max_length, truncation=True,
                padding="max_length", return_tensors="pt",
            )
            return enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0)

        a_ids, a_mask = tok(anchor_text)
        p_ids, p_mask = tok(pos_text)
        n_ids, n_mask = tok(neg_text)

        return {
            "anchor_input_ids": a_ids, "anchor_attention_mask": a_mask,
            "pos_input_ids": p_ids, "pos_attention_mask": p_mask,
            "neg_input_ids": n_ids, "neg_attention_mask": n_mask,
            "ce_input_ids": p_ids, "ce_labels": p_ids.clone(),
        }


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

def get_last_hidden(model, input_ids, attention_mask):
    out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    hs = out.hidden_states[-1]               # (B, T, H)
    lengths = attention_mask.sum(dim=1) - 1  # last non-padding position
    return hs[torch.arange(hs.size(0), device=hs.device), lengths]  # (B, H)


def triplet_loss(h_a, h_p, h_n, margin: float = 0.3):
    sim_p = F.cosine_similarity(h_a, h_p, dim=-1)
    sim_n = F.cosine_similarity(h_a, h_n, dim=-1)
    return F.relu(margin - sim_p + sim_n).mean()


def ce_loss(model, input_ids, labels):
    logits = model(input_ids=input_ids).logits
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_labels[shift_labels == model.config.pad_token_id] = -100
    return nn.CrossEntropyLoss(ignore_index=-100)(
        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(cfg: dict):
    set_seed(42)
    use_qlora = cfg.get("use_qlora", False)
    accelerator = Accelerator(
        mixed_precision="bf16" if cfg.get("bf16", True) and not use_qlora else "no",
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 4),
    )

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model_name"], trust_remote_code=True, padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Model
    bnb_config = None
    if use_qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=cfg.get("bnb_4bit_use_double_quant", True),
        )

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16 if not use_qlora else None,
        attn_implementation=cfg.get("attn_implementation", "eager"),
        trust_remote_code=True,
        device_map="auto" if use_qlora else None,
    )
    model.config.use_cache = False

    if use_qlora:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=cfg.get("gradient_checkpointing", True)
        )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.get("lora_r", 64),
        lora_alpha=cfg.get("lora_alpha", 128),
        lora_dropout=cfg.get("lora_dropout", 0.05),
        bias=cfg.get("lora_bias", "none"),
        target_modules=cfg.get("lora_target_modules", [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]),
        use_rslora=cfg.get("use_rslora", True),
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    if cfg.get("gradient_checkpointing", True) and not use_qlora:
        model.gradient_checkpointing_enable()

    # Apply Liger Kernel for fused lm_head + cross-entropy (helps the CE forward pass)
    if cfg.get("use_liger_kernel", False):
        try:
            from liger_kernel.transformers import _apply_liger_kernel_to_instance
            _apply_liger_kernel_to_instance(model=model)
            print("Liger Kernel applied to model (fused lm_head+CE for CE forward pass)")
        except Exception as e:
            print(f"Warning: failed to apply Liger Kernel: {e}")

    # Dataset
    records = load_jsonl_records(cfg["train_data"], cfg["model_name"], split="train")
    dataset = TripletLoRADataset(records, tokenizer, cfg.get("max_seq_length", 2048))
    loader = DataLoader(dataset, batch_size=cfg.get("per_device_train_batch_size", 4),
                        shuffle=True, num_workers=2)

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(cfg.get("learning_rate", 2e-4)),
        weight_decay=cfg.get("weight_decay", 0.01),
    )
    total_steps = len(loader) * cfg.get("num_epochs", 3) // cfg.get("gradient_accumulation_steps", 4)
    warmup_steps = int(total_steps * cfg.get("warmup_ratio", 0.03))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    if not use_qlora:
        model, optimizer, loader, scheduler = accelerator.prepare(
            model, optimizer, loader, scheduler
        )

    alpha = cfg.get("alpha", 0.7)
    margin = cfg.get("margin", 0.3)
    output_dir = cfg.get("output_dir", "./output")
    os.makedirs(output_dir, exist_ok=True)
    global_step = 0
    best_eval_init = float("inf")
    no_improve_init = 0
    resume_epoch = 0

    # Resume from checkpoint if requested
    resume_ckpt = cfg.get("resume_from_checkpoint")
    if resume_ckpt == "auto":
        ckpts = sorted(
            [p for p in Path(output_dir).glob("checkpoint-*") if p.name != "checkpoint-best"],
            key=lambda p: int(p.name.split("-")[1]),
        )
        resume_ckpt = str(ckpts[-1]) if ckpts else None
    if resume_ckpt and Path(resume_ckpt).exists():
        state_path = Path(resume_ckpt) / "trainer_state.pt"
        if state_path.exists():
            print(f"Resuming contrastive run from {resume_ckpt}")
            # Load LoRA adapter weights
            from peft import PeftModel
            # The model passed to get_peft_model is already a PeftModel; load adapter into it
            adapter_state = torch.load(
                Path(resume_ckpt) / "adapter_model.safetensors"
                if (Path(resume_ckpt) / "adapter_model.safetensors").exists()
                else Path(resume_ckpt) / "adapter_model.bin",
                map_location="cpu",
            ) if (Path(resume_ckpt) / "adapter_model.bin").exists() else None
            # Easier: use model.load_adapter
            try:
                model.load_adapter(str(resume_ckpt), adapter_name="default", is_trainable=True)
            except Exception as e:
                print(f"  load_adapter failed ({e}), trying set_peft_model_state_dict")
                from peft import set_peft_model_state_dict
                from safetensors.torch import load_file
                sf = Path(resume_ckpt) / "adapter_model.safetensors"
                if sf.exists():
                    state = load_file(str(sf))
                    set_peft_model_state_dict(model, state)
            # Load trainer state (optimizer, scheduler, step, best_eval)
            saved = torch.load(state_path, map_location="cpu", weights_only=False)
            optimizer.load_state_dict(saved["optimizer"])
            scheduler.load_state_dict(saved["scheduler"])
            global_step = saved.get("global_step", 0)
            best_eval_init = saved.get("best_eval", float("inf"))
            no_improve_init = saved.get("no_improve", 0)
            resume_epoch = saved.get("epoch", 0)
            print(
                f"  Resumed: global_step={global_step}, epoch={resume_epoch}, "
                f"best_eval={best_eval_init:.4f}, no_improve={no_improve_init}"
            )
        else:
            print(f"  WARNING: {state_path} not found, starting fresh from step 0")

    # Optional eval loader (for early stopping)
    eval_loader = None
    eval_path = cfg.get("eval_data")
    if eval_path and Path(eval_path).exists():
        eval_records = load_jsonl_records(eval_path, cfg["model_name"], split="val")
        eval_dataset = TripletLoRADataset(eval_records, tokenizer, cfg.get("max_seq_length", 2048))
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=cfg.get("per_device_eval_batch_size", cfg.get("per_device_train_batch_size", 4)),
            shuffle=False, num_workers=2,
        )
        if not use_qlora:
            eval_loader = accelerator.prepare(eval_loader)

    early_stopping = cfg.get("early_stopping", True) and eval_loader is not None
    patience = int(cfg.get("early_stopping_patience", 3))
    threshold = float(cfg.get("early_stopping_threshold", 0.0))
    eval_steps = int(cfg.get("eval_steps", 200))
    best_eval = best_eval_init
    no_improve = no_improve_init
    stop_training = False
    if early_stopping:
        print(
            f"Early stopping enabled: patience={patience}, threshold={threshold}, "
            f"eval_steps={eval_steps}"
        )

    @torch.no_grad()
    def run_eval():
        model.eval()
        total = 0.0
        n = 0
        for batch in eval_loader:
            l_ce = ce_loss(model, batch["ce_input_ids"], batch["ce_labels"])
            h_a = get_last_hidden(model, batch["anchor_input_ids"], batch["anchor_attention_mask"])
            h_p = get_last_hidden(model, batch["pos_input_ids"], batch["pos_attention_mask"])
            h_n = get_last_hidden(model, batch["neg_input_ids"], batch["neg_attention_mask"])
            l_t = triplet_loss(h_a, h_p, h_n, margin)
            total += (alpha * l_ce + (1.0 - alpha) * l_t).item()
            n += 1
        model.train()
        return total / max(n, 1)

    num_epochs = cfg.get("num_epochs", 3)
    for epoch in range(resume_epoch, num_epochs):
        if stop_training:
            break
        model.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            ctx = accelerator.accumulate(model) if not use_qlora else __import__("contextlib").nullcontext()
            with ctx:
                l_ce = ce_loss(model, batch["ce_input_ids"], batch["ce_labels"])
                h_a = get_last_hidden(model, batch["anchor_input_ids"], batch["anchor_attention_mask"])
                h_p = get_last_hidden(model, batch["pos_input_ids"], batch["pos_attention_mask"])
                h_n = get_last_hidden(model, batch["neg_input_ids"], batch["neg_attention_mask"])
                l_t = triplet_loss(h_a, h_p, h_n, margin)
                loss = alpha * l_ce + (1.0 - alpha) * l_t

                if use_qlora:
                    loss.backward()
                else:
                    accelerator.backward(loss)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            pbar.set_postfix(loss=f"{loss.item():.4f}", ce=f"{l_ce.item():.4f}", contra=f"{l_t.item():.4f}")

            if global_step % cfg.get("save_steps", 500) == 0:
                ckpt = os.path.join(output_dir, f"checkpoint-{global_step}")
                model.save_pretrained(ckpt)
                tokenizer.save_pretrained(ckpt)
                # Save full training state for resume
                torch.save(
                    {
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "global_step": global_step,
                        "epoch": epoch,
                        "best_eval": best_eval,
                        "no_improve": no_improve,
                    },
                    os.path.join(ckpt, "trainer_state.pt"),
                )
                print(f"Saved checkpoint: {ckpt}")
                # Prune old checkpoints (keep last save_total_limit)
                save_total_limit = int(cfg.get("save_total_limit", 3))
                if save_total_limit > 0:
                    all_ckpts = sorted(
                        [p for p in Path(output_dir).glob("checkpoint-*") if p.name != "checkpoint-best"],
                        key=lambda p: int(p.name.split("-")[1]),
                    )
                    for old in all_ckpts[:-save_total_limit]:
                        import shutil
                        shutil.rmtree(old, ignore_errors=True)

            if early_stopping and global_step % eval_steps == 0:
                eval_loss = run_eval()
                print(f"[step {global_step}] eval_loss={eval_loss:.4f} best={best_eval:.4f} no_improve={no_improve}")
                if eval_loss < best_eval - threshold:
                    best_eval = eval_loss
                    no_improve = 0
                    # Save best checkpoint
                    best_dir = os.path.join(output_dir, "checkpoint-best")
                    model.save_pretrained(best_dir)
                    tokenizer.save_pretrained(best_dir)
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        print(f"Early stopping at step {global_step}: no improvement for {patience} evals")
                        stop_training = True
                        break

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    with open(os.path.join(output_dir, "training_config.yaml"), "w") as f:
        yaml.dump(cfg, f)
    print(f"Done. Adapter saved to: {output_dir}")


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--train_data", type=str, default=None)
    parser.add_argument("--eval_data", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="'auto' to find latest checkpoint, or path to a specific checkpoint")
    parser.add_argument("--use_qlora", type=lambda x: x.lower() == "true", default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--margin", type=float, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    overrides = {k: v for k, v in vars(args).items() if k != "config"}
    cfg = load_config(args.config, overrides)
    train(cfg)
