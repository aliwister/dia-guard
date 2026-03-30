#!/usr/bin/env python3
"""
DIA-GUARD — Upload All Trained Models to HuggingFace Hub
=========================================================

Scans the local models/ directory for completed training runs across all
three experiment groups and uploads the best checkpoint for each to
HuggingFace, preserving the group/method/model folder hierarchy.

HuggingFace repo naming:
  Group 1 (Teacher FT):        {org}/DIA-Guard-Teacher-{model}-{method}
  Group 2 (KD Students):       {org}/DIA-Guard-KD-{model}-{kd_method}
  Group 3 (Student FT):        {org}/DIA-Guard-Student-{model}-{method}

Usage:
  # Dry run — see what would be uploaded
  python upload_all_models.py --dry_run

  # Upload everything that is complete
  python upload_all_models.py --hf_token hf_xxxxx

  # Upload only Group 3
  python upload_all_models.py --hf_token hf_xxxxx --group 3

  # Upload a specific model
  python upload_all_models.py --hf_token hf_xxxxx --only gemma_3_270m_it
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

from huggingface_hub import HfApi, create_repo, upload_folder


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODELS_ROOT = Path(__file__).resolve().parent  # dia-guard/models/

# Slug → HuggingFace base model ID
SLUG_TO_BASE_MODEL = {
    "gemma_3_270m_it":        "google/gemma-3-270m-it",
    "gemma_3_1b_it":          "google/gemma-3-1b-it",
    "llama_3_2_1b_instruct":  "meta-llama/Llama-3.2-1B-Instruct",
    "qwen3guard_gen_0_6b":    "Qwen/Qwen3Guard-Gen-0.6B",
    "qwen3_5_0_8b":           "Qwen/Qwen3.5-0.8B",
    "smollm2_1_7b_instruct":  "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "qwen3_1_7b":             "Qwen/Qwen3-1.7B",
    "qwen3_4b_saferl":        "Qwen/Qwen3-4B-SafeRL",
    "tiny_aya_global":        "CohereForAI/aya-expanse-8b",
}

# Slug → short display name for HF repo
SLUG_TO_SHORT = {
    "gemma_3_270m_it":        "Gemma-270M",
    "gemma_3_1b_it":          "Gemma-1B",
    "llama_3_2_1b_instruct":  "Llama-1B",
    "qwen3guard_gen_0_6b":    "QwenGuard-0.6B",
    "qwen3_5_0_8b":           "Qwen-0.8B",
    "smollm2_1_7b_instruct":  "SmolLM-1.7B",
    "qwen3_1_7b":             "Qwen-1.7B",
    "qwen3_4b_saferl":        "Qwen-4B",
    "tiny_aya_global":        "Aya-3B",
}

METHOD_DISPLAY = {
    "peft":        "LoRA",
    "full_ft":     "FullFT",
    "minillm":     "MiniLLM",
    "gkd":         "GKD",
    "ted":         "TED",
}

GROUP_DIRS = {
    1: "FT",                        # Group 1: Teacher FT
    2: "KD",                        # Group 2: KD Students
    3: "group3_student_ft_baseline", # Group 3: Student FT Baseline
}

GROUP_LABELS = {
    1: "Teacher",
    2: "KD",
    3: "Student",
}


# ---------------------------------------------------------------------------
# Completion check
# ---------------------------------------------------------------------------

def is_model_complete(model_dir: Path) -> bool:
    """Check if a training run produced a usable model."""
    # LoRA adapters
    if (model_dir / "adapter_model.safetensors").exists():
        return True
    if (model_dir / "adapter_model.bin").exists():
        return True
    # Full fine-tune
    if (model_dir / "model.safetensors").exists():
        return True
    if (model_dir / "pytorch_model.bin").exists():
        return True
    # Sharded
    if (model_dir / "model.safetensors.index.json").exists():
        return True
    # config.json as minimum signal (some trainers save this)
    if (model_dir / "config.json").exists():
        return True
    return False


def count_files(model_dir: Path) -> tuple:
    """Return (num_files, total_size_mb)."""
    files = [f for f in model_dir.rglob("*") if f.is_file()]
    total = sum(f.stat().st_size for f in files)
    return len(files), total / 1e6


# ---------------------------------------------------------------------------
# Model card
# ---------------------------------------------------------------------------

def generate_model_card(
    repo_id: str,
    group: int,
    method: str,
    model_slug: str,
    base_model: str,
) -> str:
    group_label = GROUP_LABELS.get(group, "Unknown")
    method_label = METHOD_DISPLAY.get(method, method)

    return f"""---
language:
  - en
  - multilingual
license: apache-2.0
tags:
  - safety
  - content-moderation
  - dialectal-safety
  - dia-guard
  - group{group}-{group_label.lower()}
  - {method}
pipeline_tag: text-generation
base_model: {base_model}
datasets:
  - jsl5710/Shield
---

# {repo_id.split("/")[-1]}

Part of the **DIA-GUARD** (Dialect-Informed Adversarial Guard) framework.

## Overview

| Field | Value |
|-------|-------|
| **Experiment Group** | Group {group} — {group_label} |
| **Method** | {method_label} |
| **Base Model** | [{base_model}](https://huggingface.co/{base_model}) |
| **Task** | Binary safety classification (safe / unsafe) |
| **Training Data** | [jsl5710/Shield](https://huggingface.co/datasets/jsl5710/Shield) — 749,864 train / 158,887 val / 163,174 test |

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

base = AutoModelForCausalLM.from_pretrained("{base_model}", torch_dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(base, "{repo_id}")
tokenizer = AutoTokenizer.from_pretrained("{repo_id}")

messages = [
    {{"role": "system", "content": "You are DIA-Guard, a multilingual safety assistant. Classify the following text as safe or unsafe."}},
    {{"role": "user", "content": "Your text here"}},
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=8, do_sample=False)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

## Citation

```bibtex
@misc{{dia-guard,
  title={{DIA-GUARD: Dialect-Informed Adversarial Guard for LLM Safety}},
  author={{Lucas, Jason}},
  year={{2025}},
  url={{https://github.com/jsl5710/dia-guard}}
}}
```
"""


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_models(groups: list[int] = None) -> list[dict]:
    """Walk models/ and return list of dicts describing each completed model."""
    if groups is None:
        groups = [1, 2, 3]

    found = []

    for group_num in groups:
        group_dir_name = GROUP_DIRS[group_num]
        group_path = MODELS_ROOT / group_dir_name
        if not group_path.exists():
            continue

        # Group 1 & 3: group_path / {method} / {model_slug}
        # Group 2 (KD): group_path / {kd_method} / {model_slug}
        for method_dir in sorted(group_path.iterdir()):
            if not method_dir.is_dir() or method_dir.name.startswith("."):
                continue
            method = method_dir.name  # peft, full_ft, minillm, gkd, ted

            for model_dir in sorted(method_dir.iterdir()):
                if not model_dir.is_dir() or model_dir.name.startswith("."):
                    continue
                model_slug = model_dir.name
                complete = is_model_complete(model_dir)
                base_model = SLUG_TO_BASE_MODEL.get(model_slug, "unknown")
                short_name = SLUG_TO_SHORT.get(model_slug, model_slug)
                method_label = METHOD_DISPLAY.get(method, method)
                group_label = GROUP_LABELS[group_num]

                repo_name = f"DIA-Guard-{group_label}-{short_name}-{method_label}"
                n_files, size_mb = count_files(model_dir) if complete else (0, 0)

                found.append({
                    "group": group_num,
                    "method": method,
                    "model_slug": model_slug,
                    "model_dir": model_dir,
                    "base_model": base_model,
                    "short_name": short_name,
                    "repo_name": repo_name,
                    "complete": complete,
                    "n_files": n_files,
                    "size_mb": size_mb,
                })

    return found


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

def upload_model(entry: dict, org: str, token: str, dry_run: bool = False):
    """Upload a single model to HuggingFace."""
    repo_id = f"{org}/{entry['repo_name']}"
    model_dir = entry["model_dir"]

    print(f"\n{'[DRY RUN] ' if dry_run else ''}Uploading: {repo_id}")
    print(f"  Local:  {model_dir}")
    print(f"  Files:  {entry['n_files']}  ({entry['size_mb']:.1f} MB)")
    print(f"  Base:   {entry['base_model']}")

    if dry_run:
        return True

    # Generate model card if not present
    card_path = model_dir / "README.md"
    if not card_path.exists():
        card = generate_model_card(
            repo_id=repo_id,
            group=entry["group"],
            method=entry["method"],
            model_slug=entry["model_slug"],
            base_model=entry["base_model"],
        )
        card_path.write_text(card)
        print(f"  Generated model card")

    try:
        create_repo(
            repo_id=repo_id,
            repo_type="model",
            token=token,
            exist_ok=True,
        )
        upload_folder(
            folder_path=str(model_dir),
            repo_id=repo_id,
            repo_type="model",
            token=token,
            commit_message=f"Upload DIA-GUARD Group {entry['group']} — {entry['short_name']} ({entry['method']})",
            ignore_patterns=[
                "*.log", "__pycache__", "*.pyc", "wandb", "runs",
                "checkpoint-*",   # skip intermediate checkpoints, keep only best
                "global_step*",
            ],
        )
        print(f"  Done: https://huggingface.co/{repo_id}")
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Upload all completed DIA-GUARD models to HuggingFace"
    )
    parser.add_argument("--hf_token", type=str, default=None,
                        help="HuggingFace API token (or set HF_TOKEN env var)")
    parser.add_argument("--org", type=str, default="jsl5710",
                        help="HuggingFace org/user (default: jsl5710)")
    parser.add_argument("--group", type=int, default=None, choices=[1, 2, 3],
                        help="Upload only this group (default: all)")
    parser.add_argument("--only", type=str, default=None,
                        help="Upload only model matching this slug (e.g., gemma_3_270m_it)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Show status without uploading")
    parser.add_argument("--include_incomplete", action="store_true",
                        help="Also list models that haven't finished training")
    args = parser.parse_args()

    token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token and not args.dry_run:
        print("ERROR: Provide --hf_token or set HF_TOKEN env var")
        sys.exit(1)

    groups = [args.group] if args.group else [1, 2, 3]
    models = discover_models(groups)

    if args.only:
        models = [m for m in models if args.only in m["model_slug"]]

    # --- Status table ---
    complete = [m for m in models if m["complete"]]
    incomplete = [m for m in models if not m["complete"]]

    print("=" * 75)
    print("DIA-GUARD Model Upload Status")
    print("=" * 75)

    if complete:
        print(f"\nReady to upload ({len(complete)} models):")
        print(f"  {'Group':<8} {'Method':<10} {'Model':<25} {'Files':<8} {'Size':<10}")
        print(f"  {'-'*7:<8} {'-'*9:<10} {'-'*24:<25} {'-'*7:<8} {'-'*9:<10}")
        for m in complete:
            print(f"  G{m['group']:<7} {m['method']:<10} {m['short_name']:<25} {m['n_files']:<8} {m['size_mb']:.1f} MB")

    if incomplete:
        print(f"\nNot yet complete ({len(incomplete)} models):")
        for m in incomplete:
            print(f"  G{m['group']}  {m['method']:<10} {m['model_slug']}")

    if not complete:
        print("\nNo completed models found. Training may still be in progress.")
        print(f"Checked: {MODELS_ROOT}")
        sys.exit(0)

    # --- Upload ---
    if args.dry_run:
        print(f"\n[DRY RUN] Would upload {len(complete)} models to {args.org}/")
        for m in complete:
            print(f"  {args.org}/{m['repo_name']}")
        return

    print(f"\nUploading {len(complete)} models to https://huggingface.co/{args.org}/...")
    success, fail = 0, 0
    for m in complete:
        ok = upload_model(m, org=args.org, token=token)
        if ok:
            success += 1
        else:
            fail += 1

    print(f"\n{'=' * 75}")
    print(f"Upload complete: {success} succeeded, {fail} failed")
    print(f"{'=' * 75}")


if __name__ == "__main__":
    main()
