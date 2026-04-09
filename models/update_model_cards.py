#!/usr/bin/env python3
"""
Update Shield model cards on HuggingFace with final training metrics.

Adds a "Performance" section to each model's README with:
  - Final epoch (early-stopped)
  - Train loss / accuracy
  - Eval loss / accuracy
  - Training method, batch size, Liger usage

Only the README.md file is re-uploaded (not the model weights), so this
is fast and bandwidth-cheap.
"""

import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, upload_file

USER = "jsl5710"

# Per-model final metrics (slug → method → metrics dict)
METRICS = {
    "qwen3guard_gen_0_6b": {
        "peft": {
            "epoch": "0.0095/3 (early-stopped)",
            "train_loss": "0.2716",
            "train_acc": "—",
            "eval_loss": "0.33",
            "eval_acc": "95.2%",
            "bs": "8 × 1 = 8",
            "liger": False,
        },
        "full_ft": {
            "epoch": "0.52/3 (early-stopped)",
            "train_loss": "0.0977",
            "train_acc": "97.9%",
            "eval_loss": "0.17",
            "eval_acc": "96.8%",
            "bs": "128 × 1 = 128",
            "liger": True,
        },
    },
    "llama_3_2_1b_instruct": {
        "peft": {
            "epoch": "0.68/3 (early-stopped)",
            "train_loss": "0.5274",
            "train_acc": "—",
            "eval_loss": "1.02",
            "eval_acc": "88.6%",
            "bs": "8 × 2 = 16",
            "liger": False,
        },
        "full_ft": {
            "epoch": "0.71/3 (early-stopped)",
            "train_loss": "0.5147",
            "train_acc": "—",
            "eval_loss": "0.6634",
            "eval_acc": "85.67%",
            "bs": "96 × 1 = 96",
            "liger": True,
        },
    },
    "smollm2_1_7b_instruct": {
        "peft": {
            "epoch": "0.04/3 (early-stopped)",
            "train_loss": "0.5301",
            "train_acc": "—",
            "eval_loss": "0.92",
            "eval_acc": "85.3%",
            "bs": "16 × 1 = 16",
            "liger": False,
        },
        "full_ft": {
            "epoch": "0.60/3 (early-stopped)",
            "train_loss": "0.6234",
            "train_acc": "82.67%",
            "eval_loss": "0.7843",
            "eval_acc": "77.93%",
            "bs": "64 × 1 = 64",
            "liger": True,
        },
    },
    "qwen3_1_7b": {
        "peft": {
            "epoch": "0.03/3 (early-stopped)",
            "train_loss": "0.7572",
            "train_acc": "—",
            "eval_loss": "1.16",
            "eval_acc": "85.0%",
            "bs": "8 × 2 = 16",
            "liger": False,
        },
        "full_ft": {
            "epoch": "0.26/3 (early-stopped)",
            "train_loss": "0.5209",
            "train_acc": "89.37%",
            "eval_loss": "0.9483",
            "eval_acc": "81.85%",
            "bs": "64 × 1 = 64",
            "liger": True,
        },
    },
    "gemma_3_1b_it": {
        "peft": {
            "epoch": "0.02/3 (early-stopped)",
            "train_loss": "0.8187",
            "train_acc": "—",
            "eval_loss": "1.47",
            "eval_acc": "80.0%",
            "bs": "4 × 4 = 16",
            "liger": False,
        },
        "full_ft": {
            "epoch": "0.48/3 (early-stopped)",
            "train_loss": "0.4769",
            "train_acc": "88.65%",
            "eval_loss": "0.9158",
            "eval_acc": "80.91%",
            "bs": "96 × 1 = 96",
            "liger": True,
        },
    },
    "gemma_3_270m_it": {
        "peft": {
            "epoch": "0.51/3 (early-stopped)",
            "train_loss": "1.21",
            "train_acc": "—",
            "eval_loss": "1.61",
            "eval_acc": "78.7%",
            "bs": "6 × 2 = 12",
            "liger": False,
        },
        "full_ft": {
            "epoch": "0.73/3 (early-stopped)",
            "train_loss": "0.5839",
            "train_acc": "87.29%",
            "eval_loss": "1.078",
            "eval_acc": "79.68%",
            "bs": "256 × 1 = 256",
            "liger": True,
        },
    },
    "qwen3_5_0_8b": {
        "peft": {
            "epoch": "0.01/3 (early-stopped)",
            "train_loss": "1.16",
            "train_acc": "—",
            "eval_loss": "1.55",
            "eval_acc": "77.7%",
            "bs": "6 × 2 = 12",
            "liger": False,
        },
    },
}

# Slug → pretty name (matches the upload script)
PRETTY = {
    "gemma_3_270m_it":        "Gemma-3-270m",
    "gemma_3_1b_it":          "Gemma-3-1B",
    "llama_3_2_1b_instruct":  "Llama-3.2-1B",
    "qwen3guard_gen_0_6b":    "Qwen3Guard-Gen-0.6B",
    "qwen3_5_0_8b":           "Qwen3.5-0.8B",
    "qwen3_1_7b":             "Qwen3-1.7B",
    "smollm2_1_7b_instruct":  "SmolLM2-1.7B",
}


def make_repo_id(slug, method):
    pretty = PRETTY[slug]
    method_label = "Full-FT" if method == "full_ft" else "PEFT"
    return f"{USER}/Shield-{pretty}-{method_label}-CE"


def make_local_dir(slug, method):
    base = Path("/data/vibe_exp/dia-guard/models/group3_student_ft_baseline")
    return base / method / slug


PERFORMANCE_BLOCK = """\
## Performance

| Metric | Value |
|--------|-------|
| **Final epoch** | {epoch} |
| **Train loss** | {train_loss} |
| **Train accuracy** | {train_acc} |
| **Eval loss** | {eval_loss} |
| **Eval accuracy** | **{eval_acc}** |
| **Batch size (per_device × grad_accum)** | {bs} |
| **Liger Kernel** | {liger} |
| **Stopped via** | EarlyStoppingCallback (patience=3, metric=eval_loss) |

> Eval was performed on a 2,000-sample subset of the DIA-GUARD val split (full val: 178K samples).
> Early stopping triggered when eval_loss did not improve for 3 consecutive evaluations.

"""


def update_card(api: HfApi, slug, method, metrics):
    repo_id = make_repo_id(slug, method)
    local_dir = make_local_dir(slug, method)
    readme_path = local_dir / "README.md"

    if not readme_path.exists():
        print(f"  ! {repo_id}: README.md not found at {readme_path}, skipping")
        return False

    original = readme_path.read_text()

    # Build the performance section
    perf = PERFORMANCE_BLOCK.format(
        epoch=metrics["epoch"],
        train_loss=metrics["train_loss"],
        train_acc=metrics["train_acc"],
        eval_loss=metrics["eval_loss"],
        eval_acc=metrics["eval_acc"],
        bs=metrics["bs"],
        liger="✅ enabled" if metrics["liger"] else "❌ disabled",
    )

    # Insert performance block after "## Intended Use" section's "How to use" subsection
    # Strategy: insert before "## Training Setup"
    if "## Performance" in original:
        # Replace existing performance block
        import re
        updated = re.sub(
            r"## Performance\n.*?(?=\n## )",
            perf,
            original,
            count=1,
            flags=re.DOTALL,
        )
    elif "## Training Setup" in original:
        updated = original.replace("## Training Setup", perf + "## Training Setup")
    else:
        # Just append at end
        updated = original + "\n" + perf

    readme_path.write_text(updated)

    try:
        upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            token=api.token,
            commit_message="Add training performance metrics to model card",
        )
        print(f"  ✓ {repo_id} — eval_acc {metrics['eval_acc']}")
        return True
    except Exception as e:
        print(f"  ✗ {repo_id} — {e}")
        return False


def main():
    token = Path("/data/huggingface/token").read_text().strip()
    api = HfApi(token=token)
    print(f"Updating model cards for user: {USER}")

    success = 0
    total = 0
    for slug, methods in METRICS.items():
        for method, metrics in methods.items():
            total += 1
            if update_card(api, slug, method, metrics):
                success += 1

    print(f"\n=== Updated {success}/{total} model cards ===")


if __name__ == "__main__":
    main()
