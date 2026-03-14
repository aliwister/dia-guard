"""
DIA-GUARD — Upload Model to HuggingFace Hub
==========================================
Uploads any DIA-GUARD model checkpoint to HuggingFace Hub.

Supports:
  - Single model upload
  - Batch upload of all KD models
  - Auto-generated model card (README.md) for HuggingFace
  - Private or public repositories

Prerequisites:
  pip install huggingface_hub
  huggingface-cli login   OR   set HUGGINGFACE_TOKEN env var

Usage:
  # Upload a single model
  python upload_to_hub.py \\
      --model_dir KD/minillm/qwen3-guard-0.6b \\
      --repo_id jsl5710/Dia-Guard-0.6B-MINILLM

  # Upload with model card metadata
  python upload_to_hub.py \\
      --model_dir KD/ted/gemma-270m \\
      --repo_id jsl5710/Dia-Guard-270M-TED \\
      --method ted \\
      --base_model google/gemma-3-270m-it \\
      --teacher_model Qwen/Qwen3-4B-SafeRL \\
      --private

  # Upload all KD models
  python upload_to_hub.py --upload_all_kd --org jsl5710

  # Dry run (list files, no upload)
  python upload_to_hub.py --model_dir KD/minillm/qwen3-guard-0.6b --dry_run
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

from huggingface_hub import HfApi, create_repo, upload_folder, login


# ---------------------------------------------------------------------------
# Model card generation
# ---------------------------------------------------------------------------

KD_METHOD_DESCRIPTIONS = {
    "minillm": (
        "MINILLM (Gu et al., ICLR 2024)",
        "Reverse KL divergence with policy gradient optimization.",
        "https://arxiv.org/abs/2306.08543",
    ),
    "gkd": (
        "GKD (Agarwal et al., ICLR 2024)",
        "On-policy distillation with JSD/KL loss and student-generated sequences.",
        "https://arxiv.org/abs/2306.13649",
    ),
    "ted": (
        "TED (Liang et al., ICML 2023)",
        "Task-aware layer-wise distillation with learned hidden state alignment filters.",
        "https://arxiv.org/abs/2210.01351",
    ),
    "full_ft": (
        "Full Fine-Tuning",
        "All model parameters updated with cross-entropy or contrastive loss.",
        "",
    ),
    "peft": (
        "LoRA Fine-Tuning",
        "Parameter-efficient fine-tuning using low-rank adapters (LoRA/QLoRA).",
        "https://arxiv.org/abs/2106.09685",
    ),
}


def generate_model_card(
    repo_id: str,
    model_dir: str,
    method: Optional[str] = None,
    base_model: Optional[str] = None,
    teacher_model: Optional[str] = None,
    tags: Optional[list] = None,
) -> str:
    """Generate a HuggingFace model card (README.md) for the DIA-GUARD model."""

    method_lower = (method or "").lower()
    method_info = KD_METHOD_DESCRIPTIONS.get(method_lower, ("Unknown", "", ""))

    # Load training config if available
    config_path = Path(model_dir) / "training_config.yaml"
    config_str = ""
    if config_path.exists():
        with open(config_path) as f:
            config_str = f.read()

    # Determine pipeline tag
    pipeline_tag = "text-generation"

    card = f"""---
language:
  - en
  - multilingual
license: apache-2.0
tags:
  - safety
  - content-moderation
  - dialectal-safety
  - dia-guard
  - knowledge-distillation
  - decoder
{chr(10).join("  - " + t for t in (tags or []))}
pipeline_tag: {pipeline_tag}
base_model: {base_model or "unknown"}
---

# {repo_id.split("/")[-1]}

Part of the **DIA-GUARD** framework — a multilingual, dialectally-aware safety system for LLMs.

## Model Description

This model was produced by the DIA-GUARD evaluation pipeline:

| Field | Value |
|-------|-------|
| **Method** | {method_info[0]} |
| **Description** | {method_info[1]} |
| **Base Model** | `{base_model or "unknown"}` |
| **Teacher Model** | `{teacher_model or "N/A"}` |
| **Task** | Multilingual Safety / Harm Detection |

## DIA-GUARD Pipeline

```
DIA-GUARD
└── Dia-LLM
    ├── Harm-SHIELD       ← dialect harm transformation
    ├── CounterHarm-SHIELD ← safe counterexample generation
    └── Evaluation
        ├── FineTune      ← teacher model training
        └── KD            ← this model was produced here
```

## Training Details

{"See `training_config.yaml` in this repository for full hyperparameters." if config_path.exists() else "Training config not available."}

{"### Hyperparameters" if config_str else ""}
{"```yaml" if config_str else ""}
{config_str}
{"```" if config_str else ""}

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    "{repo_id}",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("{repo_id}")

messages = [
    {{"role": "system", "content": "You are DIA-Guard, a multilingual safety assistant."}},
    {{"role": "user", "content": "Is it okay to make fun of someone's dialect?"}},
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Citation

If you use this model, please cite:

```bibtex
@misc{{dia-guard,
  title={{DIA-GUARD: Dialectally-Aware Safety for Large Language Models}},
  author={{jsl5710}},
  year={{2024}},
  url={{https://github.com/jsl5710/dia-guard}}
}}
```
{"" if not method_info[2] else f"""
### Distillation Method

```bibtex
@article{{method-paper,
  note={{See: {method_info[2]}}}
}}
```
"""}

## License

See base model license. DIA-GUARD training code is released under Apache 2.0.
"""
    return card


# ---------------------------------------------------------------------------
# Upload functions
# ---------------------------------------------------------------------------

def upload_model(
    model_dir: str,
    repo_id: str,
    private: bool = False,
    method: Optional[str] = None,
    base_model: Optional[str] = None,
    teacher_model: Optional[str] = None,
    token: Optional[str] = None,
    dry_run: bool = False,
    generate_card: bool = True,
):
    """Upload a model directory to HuggingFace Hub."""
    model_path = Path(model_dir)
    if not model_path.exists():
        print(f"ERROR: model_dir does not exist: {model_dir}")
        sys.exit(1)

    # List files to upload
    files = list(model_path.rglob("*"))
    files = [f for f in files if f.is_file()]
    print(f"\nModel: {model_dir}")
    print(f"Repo:  {repo_id}")
    print(f"Files: {len(files)}")
    for f in files[:10]:
        size_mb = f.stat().st_size / 1e6
        print(f"  {f.relative_to(model_path)} ({size_mb:.1f} MB)")
    if len(files) > 10:
        print(f"  ... and {len(files) - 10} more")

    if dry_run:
        print("\n[DRY RUN] No files uploaded.")
        return

    # Create repo
    api = HfApi(token=token)
    try:
        create_repo(
            repo_id=repo_id,
            private=private,
            repo_type="model",
            token=token,
            exist_ok=True,
        )
        print(f"Repository ready: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"ERROR creating repo: {e}")
        sys.exit(1)

    # Generate and write model card
    if generate_card:
        card_content = generate_model_card(
            repo_id=repo_id,
            model_dir=model_dir,
            method=method,
            base_model=base_model,
            teacher_model=teacher_model,
        )
        card_path = model_path / "README.md"
        # Don't overwrite if user has a custom README
        if not card_path.exists():
            with open(card_path, "w") as f:
                f.write(card_content)
            print(f"Generated model card: {card_path}")

    # Upload
    print(f"\nUploading to {repo_id}...")
    try:
        upload_folder(
            folder_path=str(model_path),
            repo_id=repo_id,
            repo_type="model",
            token=token,
            commit_message=f"Upload DIA-GUARD model: {model_path.name}",
            ignore_patterns=["*.log", "__pycache__", "*.pyc"],
        )
        print(f"Upload complete: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"ERROR uploading: {e}")
        sys.exit(1)


def detect_method_from_path(model_dir: str) -> Optional[str]:
    """Infer KD method from directory path."""
    path_lower = model_dir.lower()
    for method in ["minillm", "gkd", "ted", "full_ft", "peft"]:
        if method in path_lower:
            return method
    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Upload DIA-GUARD model to HuggingFace Hub"
    )

    # Single model upload
    parser.add_argument("--model_dir", type=str, default=None,
                        help="Path to model directory (HuggingFace format)")
    parser.add_argument("--repo_id", type=str, default=None,
                        help="HuggingFace repo ID (e.g., jsl5710/Dia-Guard-0.6B-MINILLM)")

    # Metadata for model card
    parser.add_argument("--method", type=str, default=None,
                        choices=["minillm", "gkd", "ted", "full_ft", "peft"],
                        help="Training method (auto-detected from path if not set)")
    parser.add_argument("--base_model", type=str, default=None,
                        help="Base model HF ID (e.g., Qwen/Qwen3Guard-Gen-0.6B)")
    parser.add_argument("--teacher_model", type=str, default=None,
                        help="Teacher model HF ID or path")

    # Upload options
    parser.add_argument("--private", action="store_true",
                        help="Make the HF repository private")
    parser.add_argument("--token", type=str, default=None,
                        help="HuggingFace API token (or set HUGGINGFACE_TOKEN env var)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Show what would be uploaded without actually uploading")
    parser.add_argument("--no_model_card", action="store_true",
                        help="Skip generating model card")

    # Batch upload
    parser.add_argument("--upload_all_kd", action="store_true",
                        help="Upload all KD model subdirectories")
    parser.add_argument("--org", type=str, default=None,
                        help="HuggingFace org/user for batch upload (e.g., jsl5710)")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Get token
    token = args.token or os.environ.get("HUGGINGFACE_TOKEN")
    if not token and not args.dry_run:
        print("No HuggingFace token found. Set --token or HUGGINGFACE_TOKEN env var.")
        print("Or run: huggingface-cli login")

    if args.upload_all_kd:
        # Batch upload all KD models
        if not args.org:
            print("ERROR: --org required for batch upload (e.g., --org jsl5710)")
            sys.exit(1)

        kd_base = Path(__file__).parent / "KD"
        if not kd_base.exists():
            print(f"ERROR: KD directory not found: {kd_base}")
            sys.exit(1)

        uploaded = 0
        for method_dir in sorted(kd_base.iterdir()):
            if not method_dir.is_dir() or method_dir.name.startswith("."):
                continue
            for model_dir in sorted(method_dir.iterdir()):
                if not model_dir.is_dir() or model_dir.name.startswith("."):
                    continue
                # Check if there's a model to upload
                has_model = (model_dir / "config.json").exists()
                if not has_model:
                    print(f"SKIP (no config.json): {model_dir}")
                    continue

                repo_id = f"{args.org}/Dia-Guard-{model_dir.name}-{method_dir.name.upper()}"
                method = detect_method_from_path(str(method_dir))
                upload_model(
                    model_dir=str(model_dir),
                    repo_id=repo_id,
                    private=args.private,
                    method=method,
                    token=token,
                    dry_run=args.dry_run,
                    generate_card=not args.no_model_card,
                )
                uploaded += 1

        print(f"\nTotal models uploaded: {uploaded}")

    elif args.model_dir and args.repo_id:
        method = args.method or detect_method_from_path(args.model_dir)
        upload_model(
            model_dir=args.model_dir,
            repo_id=args.repo_id,
            private=args.private,
            method=method,
            base_model=args.base_model,
            teacher_model=args.teacher_model,
            token=token,
            dry_run=args.dry_run,
            generate_card=not args.no_model_card,
        )

    else:
        print("ERROR: Provide --model_dir + --repo_id, or use --upload_all_kd --org <org>")
        print("Run with --help for usage.")
        sys.exit(1)
