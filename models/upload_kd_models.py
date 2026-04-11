#!/usr/bin/env python3
"""
Upload KD (Knowledge Distillation) models from Scenario 1 to HuggingFace.

Naming scheme:
    jsl5710/Shield-<PrettyStudent>-KD-<METHOD>-<TEACHER_SHORT>-<SCENARIO>

Example:
    jsl5710/Shield-Qwen3Guard-Gen-0.6B-KD-GKD-Qwen3-4B-SafeRL-OOB
"""
import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, create_repo, upload_folder

HF_USER = "jsl5710"
MODELS_ROOT = Path("/data/vibe_exp/dia-guard/models/KD")
RESULTS_ROOT = Path("/data/vibe_exp/dia-guard/codes/evaluation/results/Shield")

METHODS = {"minillm": "MINILLM", "gkd": "GKD", "ted": "TED"}
TEACHERS = {
    "qwen3-4b-saferl": ("Qwen3-4B-SafeRL", "Qwen/Qwen3-4B-SafeRL"),
    "qwen3guard-gen-8b": ("Qwen3Guard-Gen-8B", "Qwen/Qwen3Guard-Gen-8B"),
}
STUDENTS = {
    "qwen3guard_gen_0_6b": ("Qwen3Guard-Gen-0.6B", "Qwen/Qwen3Guard-Gen-0.6B"),
    "qwen3_1_7b": ("Qwen3-1.7B", "Qwen/Qwen3-1.7B"),
}
SCENARIOS = {"oob": "OOB"}


def make_repo_id(student_pretty, method, teacher_pretty, scenario):
    # Shield-<student>-KD-<METHOD>-<teacher>-<scenario>
    return f"{HF_USER}/Shield-{student_pretty}-KD-{method.upper()}-{teacher_pretty}-{scenario.upper()}"


def build_card(student_pretty, student_hf, method, teacher_pretty, teacher_hf,
               scenario, repo_id, metrics):
    o = metrics["overall"]
    cm = metrics["confusion_matrix"]
    pc = metrics["per_class"]
    return f"""---
license: apache-2.0
base_model: {student_hf}
tags:
- safety
- classifier
- knowledge-distillation
- {method.lower()}
- dia-guard
language:
- en
---

# Shield-{student_pretty}-KD-{method.upper()}-{teacher_pretty}-{scenario.upper()}

**Student:** `{student_hf}`
**Teacher:** `{teacher_hf}`
**KD method:** {method.upper()}
**Scenario:** {scenario.upper()} (out-of-box — neither teacher nor student was fine-tuned on DIA-GUARD before KD)

Part of the **DIA-GUARD** dialect-aware safety classifier suite. This checkpoint
is the result of distilling an off-the-shelf 4B/8B safety teacher into a smaller
student using 50K dialect-stratified samples from the DIA-GUARD train split, and
evaluating on the full 181,874-sample dialect holdout test.

## Test Set Results

| Metric | Value |
|--------|-------|
| **Accuracy** | **{o['accuracy']:.4f}** |
| **Precision** | {o['precision']:.4f} |
| **Recall** | {o['recall']:.4f} |
| **F1** | {o['f1']:.4f} |
| Test samples | {o['support']:,} |

### Per-class breakdown

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| safe | {pc['safe']['precision']:.4f} | {pc['safe']['recall']:.4f} | {pc['safe']['f1']:.4f} | {pc['safe']['support']:,} |
| unsafe | {pc['unsafe']['precision']:.4f} | {pc['unsafe']['recall']:.4f} | {pc['unsafe']['f1']:.4f} | {pc['unsafe']['support']:,} |

### Confusion matrix

|               | Predicted safe | Predicted unsafe |
|---------------|---------------:|-----------------:|
| **Actual safe**   | TN={cm['tn']:,} | FP={cm['fp']:,} |
| **Actual unsafe** | FN={cm['fn']:,} | TP={cm['tp']:,} |

## Training Setup

| | |
|---|---|
| Method | {method.upper()} |
| Teacher | {teacher_hf} |
| Student base | {student_hf} |
| Train data | 50,000 dialect-stratified DIA-GUARD samples |
| Epochs | 1 |
| Framework | HuggingFace transformers + accelerate |

## How to use

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("{repo_id}")
model = AutoModelForCausalLM.from_pretrained("{repo_id}", torch_dtype="bfloat16")

prompt = "Classify the following text as safe or unsafe.\\n\\nText: Hello\\n\\nAnswer:"
inputs = tokenizer(prompt, return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=8)
print(tokenizer.decode(out[0], skip_special_tokens=True))
```
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", type=str, default=None, help="Substring filter on repo_id")
    args = ap.parse_args()

    token = Path("/data/huggingface/token").read_text().strip()
    api = HfApi(token=token)

    uploaded = 0
    skipped = 0
    failed = 0

    for method in METHODS:
        for t_slug, (t_pretty, t_hf) in TEACHERS.items():
            for s_slug, (s_pretty, s_hf) in STUDENTS.items():
                for sc in SCENARIOS:
                    local_dir = MODELS_ROOT / method / t_slug / sc / s_slug
                    if not (local_dir / "model.safetensors").exists():
                        skipped += 1
                        continue

                    repo_id = make_repo_id(s_pretty, method, t_pretty, sc)
                    if args.only and args.only not in repo_id:
                        continue

                    # Load metrics
                    result_dir = RESULTS_ROOT / f"KD-{method.upper()}-{t_pretty}-{sc.upper()}" / s_pretty
                    metrics_file = result_dir / "metrics.json"
                    if not metrics_file.exists():
                        print(f"  ✗ {repo_id}: no metrics.json, skipping")
                        skipped += 1
                        continue
                    metrics = json.load(open(metrics_file))

                    print(f"→ {repo_id}")
                    print(f"    local: {local_dir}")
                    print(f"    acc={metrics['overall']['accuracy']:.4f}")

                    if args.dry_run:
                        continue

                    try:
                        create_repo(repo_id=repo_id, exist_ok=True, private=False, token=token)
                        # Write model card locally before upload
                        card = build_card(s_pretty, s_hf, method, t_pretty, t_hf, sc, repo_id, metrics)
                        (local_dir / "README.md").write_text(card)
                        upload_folder(
                            folder_path=str(local_dir),
                            repo_id=repo_id,
                            token=token,
                            ignore_patterns=["checkpoint-*", "*.log", "training_config.yaml"],
                            commit_message=f"Upload {method.upper()} KD model ({sc})",
                        )
                        print(f"  ✓ uploaded")
                        uploaded += 1
                    except Exception as e:
                        print(f"  ✗ {type(e).__name__}: {e}")
                        failed += 1

    print(f"\n=== Uploaded {uploaded}, skipped {skipped}, failed {failed} ===")


if __name__ == "__main__":
    main()
