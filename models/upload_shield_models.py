#!/usr/bin/env python3
"""
Shield — Upload All Trained Models to HuggingFace Hub
======================================================

Uploads all completed DIA-GUARD models under the `Shield-` naming prefix
in the user's HF account, then creates an HF Collection grouping them.

Each model is published as a public repo with:
  - Model weights (LoRA adapter or full model)
  - Tokenizer files
  - Auto-generated model card with metrics, base model, training config
  - License inherited from the base model

Usage:
  # Dry run — see what would be uploaded
  python upload_shield_models.py --dry-run

  # Upload everything
  python upload_shield_models.py

  # Upload only one model
  python upload_shield_models.py --only qwen3guard_gen_0_6b

  # Skip the collection creation
  python upload_shield_models.py --no-collection
"""

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo, upload_folder

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODELS_ROOT = Path("/data/vibe_exp/dia-guard/models")

# Cached username — populated in main() once to avoid /whoami-v2 rate limits
HF_USER = "jsl5710"

# Slug → (HF base model ID, license tag for HF, license name)
# License inherited from base model per HF Hub policy for derivatives
MODEL_INFO = {
    "gemma_3_270m_it":        ("google/gemma-3-270m-it",          "gemma",       "Gemma Terms of Use"),
    "gemma_3_1b_it":          ("google/gemma-3-1b-it",            "gemma",       "Gemma Terms of Use"),
    "llama_3_2_1b_instruct":  ("meta-llama/Llama-3.2-1B-Instruct","llama3.2",    "Llama 3.2 Community License"),
    "qwen3guard_gen_0_6b":    ("Qwen/Qwen3Guard-Gen-0.6B",        "apache-2.0",  "Apache 2.0"),
    "qwen3_5_0_8b":           ("Qwen/Qwen3.5-0.8B",               "apache-2.0",  "Apache 2.0"),
    "qwen3_1_7b":             ("Qwen/Qwen3-1.7B",                 "apache-2.0",  "Apache 2.0"),
    "smollm2_1_7b_instruct":  ("HuggingFaceTB/SmolLM2-1.7B-Instruct","apache-2.0","Apache 2.0"),
    "tiny_aya_global":        ("CohereLabs/tiny-aya-global",      "apache-2.0",  "Apache 2.0"),
    "qwen3_4b_saferl":        ("Qwen/Qwen3-4B-SafeRL",            "apache-2.0",  "Apache 2.0"),
    "qwen3guard_gen_8b":      ("Qwen/Qwen3Guard-Gen-8B",          "apache-2.0",  "Apache 2.0"),
}

# Pretty model names for the collection
PRETTY_NAMES = {
    "gemma_3_270m_it":        "Gemma-3-270m",
    "gemma_3_1b_it":          "Gemma-3-1B",
    "llama_3_2_1b_instruct":  "Llama-3.2-1B",
    "qwen3guard_gen_0_6b":    "Qwen3Guard-Gen-0.6B",
    "qwen3_5_0_8b":           "Qwen3.5-0.8B",
    "qwen3_1_7b":             "Qwen3-1.7B",
    "smollm2_1_7b_instruct":  "SmolLM2-1.7B",
    "tiny_aya_global":        "Aya-3B",
    "qwen3_4b_saferl":        "Qwen3-4B-SafeRL",
    "qwen3guard_gen_8b":      "Qwen3Guard-Gen-8B",
}


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_models():
    """
    Walk the models tree and yield (slug, method, loss, model_dir) for each
    completed run that has either adapter_model.safetensors or model*.safetensors
    at the root.
    """
    runs = []

    # Group 3 students (PEFT and Full FT)
    for method in ["peft", "full_ft"]:
        method_dir = MODELS_ROOT / "group3_student_ft_baseline" / method
        if not method_dir.is_dir():
            continue
        for model_dir in method_dir.iterdir():
            if not model_dir.is_dir():
                continue
            slug = model_dir.name
            if has_completed_weights(model_dir):
                # Loss type — for now everything completed is CE; contrastive completes later
                loss = "ce"
                runs.append((slug, method, loss, model_dir, "student"))

    # Group 1 teachers
    for method in ["peft", "full_ft"]:
        method_dir = MODELS_ROOT / "FT" / method
        if not method_dir.is_dir():
            continue
        for model_dir in method_dir.iterdir():
            if not model_dir.is_dir():
                continue
            slug = model_dir.name
            if has_completed_weights(model_dir):
                runs.append((slug, method, "ce", model_dir, "teacher"))

    return runs


def has_completed_weights(d: Path) -> bool:
    """A model is 'completed' if it has either an adapter or model safetensors at root."""
    if (d / "adapter_model.safetensors").exists():
        return True
    if (d / "adapter_model.bin").exists():
        return True
    # Full model — multiple sharded files possible
    if list(d.glob("model*.safetensors")):
        return True
    return False


# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------

def make_repo_name(user: str, slug: str, method: str, loss: str, role: str) -> str:
    """
    Build the Shield-prefixed repo name.

    Pattern: jsl5710/Shield-<PrettyName>-<Method>-<Loss>
    Examples:
      jsl5710/Shield-Qwen3Guard-Gen-0.6B-Full-FT-CE
      jsl5710/Shield-Llama-3.2-1B-PEFT-CE
    """
    pretty = PRETTY_NAMES.get(slug, slug)
    method_label = "Full-FT" if method == "full_ft" else "PEFT"
    loss_label = loss.upper()
    if role == "teacher":
        return f"{user}/Shield-Teacher-{pretty}-{method_label}-{loss_label}"
    return f"{user}/Shield-{pretty}-{method_label}-{loss_label}"


# ---------------------------------------------------------------------------
# Model card
# ---------------------------------------------------------------------------

CARD_TEMPLATE = """\
---
license: {license_tag}
base_model: {base_model}
tags:
  - dia-guard
  - shield
  - safety
  - dialect
  - {method_tag}
  - {loss_tag}
language:
  - en
library_name: {library}
pipeline_tag: text-generation
---

# {pretty_name} — {method_label}/{loss_label} (Shield Project)

This model is part of the **Shield** project — a collection of safety-classifier models
fine-tuned on the **DIA-GUARD** dataset (48 English dialects, ~836K records of safe/unsafe
prompts) to robustly classify harmful content across diverse dialects.

## Model Summary

| Field | Value |
|-------|-------|
| **Base model** | [`{base_model}`](https://huggingface.co/{base_model}) |
| **Training method** | {method_label} ({loss_label} loss) |
| **Training data** | DIA-GUARD splits (~836K train, 178K val) |
| **Domain** | LLM safety classification across 48 English dialects |
| **Role** | {role_label} |
| **License** | {license_name} (inherited from base model) |

## Intended Use

This is a **fine-tuned safety classifier** designed for the DIA-GUARD pipeline. It is intended
for use as:

1. **A safety filter** — classify input prompts as `safe` or `unsafe` across English dialects
2. **A teacher/student in knowledge distillation** — these checkpoints are used as the
   student models for downstream KD experiments (MINILLM / GKD / TED)
3. **A research baseline** — for studies on dialect-aware safety in LLMs

### How to use

{usage_example}

## Training Setup

- **Training objective:** {loss_label_full}
- **Optimizer:** AdamW with cosine LR schedule
- **Precision:** bf16 mixed precision
- **Frameworks:** transformers, peft, trl, accelerate
- **Hardware:** A100 40GB
{liger_line}

## Dataset

**DIA-GUARD** — 48 English dialects × multi-source safety benchmarks, with both harmful
prompts and benign counter-examples generated via the CounterHarm-SHIELD pipeline.

- ~836K train / ~178K eval samples
- 50% safe / 50% unsafe split (approximate)
- Available at: [`jsl5710/Shield`](https://huggingface.co/datasets/jsl5710/Shield)

## Citation

```bibtex
@misc{{diaguard2026,
  title         = {{DIA-GUARD: Dialect-Informed Adversarial Guard for LLM Safety}},
  author        = {{Jason Lucas et al.}},
  year          = {{2026}},
  howpublished  = {{\\url{{https://github.com/jsl5710/dia-guard}}}}
}}
```

## Limitations

- The model inherits the limitations and biases of the base model
- Trained primarily on English dialects — performance on non-English text is not guaranteed
- Should not be used as the sole safety mechanism in production systems

## License

This model is released under the **{license_name}**, inherited from the base model.
Please review the base model's license at the link above before use.
"""

PEFT_USAGE = """\
```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("{base_model}", torch_dtype="bfloat16")
tokenizer = AutoTokenizer.from_pretrained("{base_model}")
model = PeftModel.from_pretrained(base, "{repo_id}")

prompt = "<your prompt here>"
inputs = tokenizer.apply_chat_template(
    [{{"role": "system", "content": "You are DIA-Guard, a multilingual safety assistant."}},
     {{"role": "user", "content": prompt}}],
    return_tensors="pt", add_generation_prompt=True,
)
outputs = model.generate(inputs, max_new_tokens=4)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
# Expected: 'safe' or 'unsafe'
```
"""

FULL_USAGE = """\
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{repo_id}", torch_dtype="bfloat16")
tokenizer = AutoTokenizer.from_pretrained("{repo_id}")

prompt = "<your prompt here>"
inputs = tokenizer.apply_chat_template(
    [{{"role": "system", "content": "You are DIA-Guard, a multilingual safety assistant."}},
     {{"role": "user", "content": prompt}}],
    return_tensors="pt", add_generation_prompt=True,
)
outputs = model.generate(inputs, max_new_tokens=4)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
# Expected: 'safe' or 'unsafe'
```
"""


def build_model_card(slug, method, loss, role, repo_id, used_liger=False):
    base_model, license_tag, license_name = MODEL_INFO[slug]
    pretty_name = PRETTY_NAMES[slug]
    method_label = "Full-FT" if method == "full_ft" else "PEFT (LoRA)"
    method_tag = "full-ft" if method == "full_ft" else "peft-lora"
    loss_label = loss.upper()
    loss_tag = loss
    loss_label_full = "Cross-Entropy (next-token prediction)" if loss == "ce" else "Triplet Contrastive + CE"
    library = "transformers" if method == "full_ft" else "peft"
    role_label = "Student model (used as KD student in DIA-GUARD pipeline)" if role == "student" else "Teacher model"
    usage_template = FULL_USAGE if method == "full_ft" else PEFT_USAGE
    usage_example = usage_template.format(base_model=base_model, repo_id=repo_id)
    liger_line = "- **Optimization:** Liger Kernel (fused lm_head + cross-entropy)" if used_liger else ""

    return CARD_TEMPLATE.format(
        license_tag=license_tag,
        license_name=license_name,
        base_model=base_model,
        pretty_name=pretty_name,
        method_label=method_label,
        method_tag=method_tag,
        loss_label=loss_label,
        loss_tag=loss_tag,
        loss_label_full=loss_label_full,
        library=library,
        role_label=role_label,
        liger_line=liger_line,
        usage_example=usage_example,
    )


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

def upload_one(api: HfApi, slug, method, loss, model_dir: Path, role, dry_run=False):
    user = HF_USER
    repo_id = make_repo_name(user, slug, method, loss, role)

    print(f"\n[{slug}] {method}/{loss}")
    print(f"  source: {model_dir}")
    print(f"  target: https://huggingface.co/{repo_id}")

    if dry_run:
        print("  (dry-run, skipping upload)")
        return repo_id

    # Detect if Liger was used (check training_config.yaml or training_args.bin if present)
    used_liger = False
    cfg_path = model_dir / "training_config.yaml"
    if cfg_path.exists():
        used_liger = "use_liger_kernel: true" in cfg_path.read_text()

    # Create the repo
    create_repo(repo_id=repo_id, exist_ok=True, private=False, token=api.token)

    # Write the model card
    card = build_model_card(slug, method, loss, role, repo_id, used_liger=used_liger)
    (model_dir / "README.md").write_text(card)

    # Upload everything except checkpoints (only the final adapter/model files)
    upload_folder(
        repo_id=repo_id,
        folder_path=str(model_dir),
        token=api.token,
        ignore_patterns=["checkpoint-*", "wandb/", "*.pyc", "__pycache__/"],
        commit_message=f"Upload Shield model: {slug} ({method}/{loss})",
    )
    print(f"  ✓ uploaded")
    return repo_id


def create_collection(api: HfApi, repo_ids):
    """Create the Shield collection grouping all uploaded models."""
    user = HF_USER
    title = "Shield — Dialect-Aware LLM Safety Classifiers"
    description = (
        "**Shield** is a collection of fine-tuned safety classifier models from the "
        "**DIA-GUARD** project, trained on prompts across 48 English dialects. "
        "Each model classifies input as `safe` or `unsafe` and is intended for use "
        "as either a standalone safety filter or as a student model in knowledge "
        "distillation experiments.\n\n"
        "Models include both **PEFT (LoRA)** and **Full Fine-Tuning** variants across "
        "7 student model sizes (270M to 1.7B parameters)."
    )

    try:
        from huggingface_hub import create_collection as create_coll
        from huggingface_hub import add_collection_item
        coll = create_coll(
            title=title,
            description=description,
            namespace=user,
            private=False,
            token=api.token,
        )
        print(f"\n✓ Created collection: {coll.url}")

        for rid in repo_ids:
            try:
                add_collection_item(
                    collection_slug=coll.slug,
                    item_id=rid,
                    item_type="model",
                    token=api.token,
                )
                print(f"  + added {rid}")
            except Exception as e:
                print(f"  ! failed to add {rid}: {e}")
        return coll.url
    except Exception as e:
        print(f"  ! failed to create collection: {e}")
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Don't actually upload")
    parser.add_argument("--only", type=str, default=None, help="Upload only this model slug")
    parser.add_argument("--no-collection", action="store_true", help="Skip collection creation")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        token_path = Path("/data/huggingface/token")
        if token_path.exists():
            token = token_path.read_text().strip()
    if not token:
        print("ERROR: no HF token found in env HF_TOKEN or /data/huggingface/token")
        sys.exit(1)

    api = HfApi(token=token)
    user = HF_USER  # cached to avoid hitting /whoami-v2 rate limit
    print(f"User: {user} (cached)")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'UPLOAD'}")

    runs = discover_models()
    if args.only:
        runs = [r for r in runs if r[0] == args.only]
    if not runs:
        print("No completed models found")
        return

    print(f"\nFound {len(runs)} completed model(s):")
    for slug, method, loss, mdir, role in runs:
        print(f"  - {slug} ({method}/{loss}, {role})")

    uploaded = []
    for slug, method, loss, mdir, role in runs:
        try:
            rid = upload_one(api, slug, method, loss, mdir, role, dry_run=args.dry_run)
            if rid:
                uploaded.append(rid)
        except Exception as e:
            print(f"  ! upload failed: {e}")

    print(f"\n=== Summary ===")
    print(f"Uploaded: {len(uploaded)}/{len(runs)}")

    if uploaded and not args.no_collection and not args.dry_run:
        print("\nCreating Shield collection...")
        create_collection(api, uploaded)


if __name__ == "__main__":
    main()
