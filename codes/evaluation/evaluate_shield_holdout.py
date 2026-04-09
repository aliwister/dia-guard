#!/usr/bin/env python3
"""
Evaluate all Shield models on the DIA-GUARD holdout test set.

For each completed Shield model (PEFT or Full FT):
  1. Load model (base + LoRA adapter for PEFT, direct for Full FT)
  2. Run inference on the full test split (181,874 samples)
  3. Compute overall + per-dialect metrics
  4. Write results to results/Shield/<Method>-CE/<ModelName>/
  5. Append a "Test Set Results" section to the model card on HF Hub

Usage:
  CUDA_VISIBLE_DEVICES=1 python evaluate_shield_holdout.py
  CUDA_VISIBLE_DEVICES=1 python evaluate_shield_holdout.py --only Qwen3Guard-Gen-0.6B
  CUDA_VISIBLE_DEVICES=1 python evaluate_shield_holdout.py --no-upload
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

import torch
from huggingface_hub import HfApi, upload_file
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Reuse logic from the existing evaluate.py in this directory
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import (
    load_jsonl,
    run_inference,
    compute_metrics,
    compute_per_dialect,
    write_predictions,
    write_confusion_matrix,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

USER = "jsl5710"
TEST_DATA = "/data/vibe_exp/dia-guard/dataset/dia_splits/test.jsonl"
RESULTS_ROOT = Path("/data/vibe_exp/dia-guard/codes/evaluation/results/Shield")
MODELS_ROOT = Path("/data/vibe_exp/dia-guard/models/group3_student_ft_baseline")
HF_TOKEN_PATH = Path("/data/huggingface/token")

# Slug → HF base model id (needed for PEFT loading)
BASE_MODEL = {
    "gemma_3_270m_it":        "google/gemma-3-270m-it",
    "gemma_3_1b_it":          "google/gemma-3-1b-it",
    "llama_3_2_1b_instruct":  "meta-llama/Llama-3.2-1B-Instruct",
    "qwen3guard_gen_0_6b":    "Qwen/Qwen3Guard-Gen-0.6B",
    "qwen3_5_0_8b":           "Qwen/Qwen3.5-0.8B",
    "qwen3_1_7b":             "Qwen/Qwen3-1.7B",
    "smollm2_1_7b_instruct":  "HuggingFaceTB/SmolLM2-1.7B-Instruct",
}

# Slug → HF Shield repo pretty name (matches upload_shield_models.py)
PRETTY = {
    "gemma_3_270m_it":        "Gemma-3-270m",
    "gemma_3_1b_it":          "Gemma-3-1B",
    "llama_3_2_1b_instruct":  "Llama-3.2-1B",
    "qwen3guard_gen_0_6b":    "Qwen3Guard-Gen-0.6B",
    "qwen3_5_0_8b":           "Qwen3.5-0.8B",
    "qwen3_1_7b":             "Qwen3-1.7B",
    "smollm2_1_7b_instruct":  "SmolLM2-1.7B",
}

# Per-model inference batch size on A100-40GB (no gradients → much bigger than training)
# Tuned for ~32-36 GB target VRAM. Full FT uses smaller bs because the full model
# is loaded (vs LoRA where base is shared). hidden_size matters too — Llama-1B has
# hidden_size=2048 vs Gemma-1B's 1152, so attention activations are much bigger.
BATCH_SIZE_PEFT = {
    # After merge_and_unload(), PEFT inference uses same VRAM as Full FT.
    # These match BATCH_SIZE_FULL for safety.
    "gemma_3_270m_it":        1024,
    "qwen3guard_gen_0_6b":    256,
    "qwen3_5_0_8b":           256,
    "gemma_3_1b_it":          384,
    "llama_3_2_1b_instruct":  256,
    "qwen3_1_7b":             192,
    "smollm2_1_7b_instruct":  96,
}
BATCH_SIZE_FULL = {
    "gemma_3_270m_it":        1024,
    "qwen3guard_gen_0_6b":    256,
    "qwen3_5_0_8b":           256,
    "gemma_3_1b_it":          384,
    "llama_3_2_1b_instruct":  256,  # hidden_size=2048 → big attention activations
    "qwen3_1_7b":             192,
    "smollm2_1_7b_instruct":  96,
}
BATCH_SIZE = BATCH_SIZE_PEFT  # legacy alias


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def has_completed_weights(d: Path) -> bool:
    if (d / "adapter_model.safetensors").exists() or (d / "adapter_model.bin").exists():
        return True
    if list(d.glob("model*.safetensors")):
        return True
    return False


def discover_models():
    """Yield (slug, method, model_dir) for each completed Shield model."""
    runs = []
    for method in ["peft", "full_ft"]:
        method_dir = MODELS_ROOT / method
        if not method_dir.is_dir():
            continue
        for model_dir in sorted(method_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            slug = model_dir.name
            if slug not in PRETTY:
                continue
            if has_completed_weights(model_dir):
                runs.append((slug, method, model_dir))
    return runs


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(slug, method, model_dir):
    """Returns (model, tokenizer) ready for inference on cuda."""
    base_id = BASE_MODEL[slug]

    # Tokenizer always loaded from base model (works for both methods)
    tokenizer = AutoTokenizer.from_pretrained(
        base_id,
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if method == "full_ft":
        # Load full model directly from local dir
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
            trust_remote_code=True,
            device_map="cuda",
        )
    else:  # peft
        # Load base model from HF, then apply adapter from local dir
        base = AutoModelForCausalLM.from_pretrained(
            base_id,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
            trust_remote_code=True,
            device_map="cuda",
            token=os.environ.get("HF_TOKEN"),
        )
        model = PeftModel.from_pretrained(base, str(model_dir))
        model = model.merge_and_unload()  # merge LoRA into base for fast inference
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Output paths and HF model card update
# ---------------------------------------------------------------------------

METHOD_SUFFIX = ""  # set via CLI to e.g. "-SAE"

def make_output_dir(slug, method):
    method_label = ("Full-FT-CE" if method == "full_ft" else "PEFT-CE") + METHOD_SUFFIX
    out = RESULTS_ROOT / method_label / PRETTY[slug]
    out.mkdir(parents=True, exist_ok=True)
    return out


def make_repo_id(slug, method):
    method_label = "Full-FT" if method == "full_ft" else "PEFT"
    return f"{USER}/Shield-{PRETTY[slug]}-{method_label}-CE"


def write_metrics_json(metrics, per_dialect, output_dir, slug, method):
    summary = {
        "model": PRETTY[slug],
        "slug": slug,
        "method": method,
        "loss": "ce",
        "test_split": TEST_DATA,
        "test_n": metrics.get("support", 0),
        "evaluated_at": datetime.utcnow().isoformat() + "Z",
        "overall": {
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "support": metrics["support"],
        },
        "confusion_matrix": metrics["confusion"],
        "per_class": metrics["per_class"],
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(output_dir / "per_dialect.json", "w") as f:
        json.dump(per_dialect, f, indent=2)
    return summary


TEST_RESULTS_BLOCK = """\
## Test Set Results

Evaluated on the **DIA-GUARD holdout test split** ({n_test:,} samples across 48 English dialects).

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **{accuracy:.4f}** |
| **Macro Precision** | {precision:.4f} |
| **Macro Recall** | {recall:.4f} |
| **Macro F1** | **{f1:.4f}** |
| **Support** | {support:,} |

### Per-class

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| **safe** | {safe_p:.4f} | {safe_r:.4f} | {safe_f:.4f} | {safe_n:,} |
| **unsafe** | {unsafe_p:.4f} | {unsafe_r:.4f} | {unsafe_f:.4f} | {unsafe_n:,} |

### Confusion Matrix

|             | Pred safe | Pred unsafe |
|-------------|-----------|-------------|
| **True safe** | {tn:,} | {fp:,} |
| **True unsafe** | {fn:,} | {tp:,} |

> Per-dialect breakdown available in `per_dialect.json` in the corresponding results folder.

"""


def update_model_card_on_hf(api, repo_id, summary, output_dir):
    """Append test results to the model card on HF."""
    try:
        # Read current README from HF
        from huggingface_hub import hf_hub_download
        try:
            local_readme = hf_hub_download(
                repo_id=repo_id,
                filename="README.md",
                token=api.token,
            )
            current = Path(local_readme).read_text()
        except Exception:
            current = ""

        block = TEST_RESULTS_BLOCK.format(
            n_test=summary["test_n"],
            accuracy=summary["overall"]["accuracy"],
            precision=summary["overall"]["precision"],
            recall=summary["overall"]["recall"],
            f1=summary["overall"]["f1"],
            support=summary["overall"]["support"],
            safe_p=summary["per_class"]["safe"]["precision"],
            safe_r=summary["per_class"]["safe"]["recall"],
            safe_f=summary["per_class"]["safe"]["f1"],
            safe_n=summary["per_class"]["safe"]["support"],
            unsafe_p=summary["per_class"]["unsafe"]["precision"],
            unsafe_r=summary["per_class"]["unsafe"]["recall"],
            unsafe_f=summary["per_class"]["unsafe"]["f1"],
            unsafe_n=summary["per_class"]["unsafe"]["support"],
            tp=summary["confusion_matrix"]["tp"],
            tn=summary["confusion_matrix"]["tn"],
            fp=summary["confusion_matrix"]["fp"],
            fn=summary["confusion_matrix"]["fn"],
        )

        # Replace existing block or insert before "## Training Setup"
        if "## Test Set Results" in current:
            import re
            updated = re.sub(
                r"## Test Set Results\n.*?(?=\n## |\Z)",
                block,
                current,
                count=1,
                flags=re.DOTALL,
            )
        elif "## Training Setup" in current:
            updated = current.replace("## Training Setup", block + "## Training Setup")
        else:
            updated = current + "\n" + block

        # Write locally and upload
        local_path = output_dir / "README_test_block.md"
        local_path.write_text(updated)
        upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            token=api.token,
            commit_message="Add holdout test set results to model card",
        )
        print(f"  ✓ Updated model card on HF: {repo_id}")
    except Exception as e:
        print(f"  ! Failed to update model card for {repo_id}: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def evaluate_one(slug, method, model_dir, records, api, no_upload=False, force=False):
    pretty = PRETTY[slug]
    bs_table = BATCH_SIZE_FULL if method == "full_ft" else BATCH_SIZE_PEFT
    bs = bs_table.get(slug, 64)
    print(f"\n{'='*70}")
    print(f"  Evaluating: {pretty} ({method}/CE)  bs={bs}")
    print(f"  Source: {model_dir}")
    print(f"{'='*70}")

    output_dir = make_output_dir(slug, method)
    print(f"  Output: {output_dir}")

    # Skip if metrics.json already exists (unless --force)
    if not force and (output_dir / "metrics.json").exists():
        existing = json.load(open(output_dir / "metrics.json"))
        print(f"  ⏭  Already evaluated (acc={existing['overall']['accuracy']:.4f}), skipping. Use --force to re-run.")
        return existing

    print(f"  Loading model...")
    model, tokenizer = load_model_and_tokenizer(slug, method, model_dir)

    print(f"  Running inference on {len(records):,} samples...")
    predictions = run_inference(
        model, tokenizer, records,
        batch_size=bs,
        max_new_tokens=8,
        device="cuda",
    )

    print(f"  Computing metrics...")
    metrics = compute_metrics(predictions)
    per_dialect = compute_per_dialect(predictions)

    print(f"  Overall accuracy: {metrics['accuracy']:.4f}  F1: {metrics['f1']:.4f}")

    # Write outputs
    write_predictions(predictions, str(output_dir / "predictions.jsonl"))
    write_confusion_matrix(metrics["confusion"], str(output_dir / "confusion_matrix.csv"))
    summary = write_metrics_json(metrics, per_dialect, output_dir, slug, method)
    print(f"  ✓ Wrote 4 files to {output_dir}")

    # Free GPU memory before next model
    del model, tokenizer
    torch.cuda.empty_cache()

    # Push to HF
    if not no_upload and api is not None:
        repo_id = make_repo_id(slug, method)
        update_model_card_on_hf(api, repo_id, summary, output_dir)

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=str, default=None,
                        help="Pretty name (e.g. 'Qwen3Guard-Gen-0.6B') to evaluate only that model")
    parser.add_argument("--method", type=str, choices=["peft", "full_ft", None], default=None,
                        help="Filter by method")
    parser.add_argument("--no-upload", action="store_true",
                        help="Don't push results to HF model cards")
    parser.add_argument("--force", action="store_true",
                        help="Re-evaluate even if metrics.json already exists")
    parser.add_argument("--test_data", type=str, default=None,
                        help="Override path to the test JSONL")
    parser.add_argument("--method_suffix", type=str, default="",
                        help="Suffix to append to method dir (e.g. '-SAE')")
    args = parser.parse_args()

    global TEST_DATA, METHOD_SUFFIX
    if args.test_data:
        TEST_DATA = args.test_data
    METHOD_SUFFIX = args.method_suffix

    # Token
    if "HF_TOKEN" not in os.environ and HF_TOKEN_PATH.exists():
        os.environ["HF_TOKEN"] = HF_TOKEN_PATH.read_text().strip()

    api = None
    if not args.no_upload:
        api = HfApi(token=os.environ.get("HF_TOKEN"))

    # Discover
    runs = discover_models()
    # Only those with "Done." in their training log (true completions)
    truly_done = []
    log_dir = Path("/data/vibe_exp/dia-guard/logs")
    log_map = {
        "gemma_3_270m_it": "gemma270m",
        "gemma_3_1b_it": "gemma1b",
        "llama_3_2_1b_instruct": "llama1b",
        "qwen3_1_7b": "qwen17b",
        "qwen3_5_0_8b": "qwen35",
        "qwen3guard_gen_0_6b": "qwen_guard",
        "smollm2_1_7b_instruct": "smollm",
    }
    for slug, method, mdir in runs:
        logbase = log_map.get(slug, slug)
        log = log_dir / (f"{logbase}_peft_ce.log" if method == "peft" else f"{logbase}_full_ce.log")
        if log.exists():
            tail = log.read_bytes()[-2000:].decode("utf-8", errors="ignore")
            if "Done." in tail:
                truly_done.append((slug, method, mdir))
        elif method == "peft":
            # Some PEFT logs may use a different name; trust file presence
            truly_done.append((slug, method, mdir))

    if args.only:
        truly_done = [r for r in truly_done if PRETTY[r[0]] == args.only]
    if args.method:
        truly_done = [r for r in truly_done if r[1] == args.method]

    if not truly_done:
        print("No completed Shield models found")
        return

    print(f"Found {len(truly_done)} completed models to evaluate:")
    for slug, method, _ in truly_done:
        print(f"  - {PRETTY[slug]} ({method}/CE)")

    # Load test data ONCE
    print(f"\nLoading test data: {TEST_DATA}")
    records = load_jsonl(TEST_DATA)
    print(f"  {len(records):,} test records")

    all_summaries = []
    for slug, method, mdir in truly_done:
        try:
            summary = evaluate_one(slug, method, mdir, records, api, no_upload=args.no_upload, force=args.force)
            all_summaries.append(summary)
        except Exception as e:
            import traceback
            print(f"  ! FAILED: {e}")
            traceback.print_exc()

    # Final leaderboard
    print(f"\n{'='*70}")
    print("  HOLDOUT TEST RESULTS — LEADERBOARD")
    print(f"{'='*70}")
    print(f"  {'Model':<35} {'Method':<10} {'Accuracy':>10} {'F1':>10}")
    print(f"  {'-'*35} {'-'*10} {'-'*10} {'-'*10}")
    for s in sorted(all_summaries, key=lambda x: -x["overall"]["accuracy"]):
        m = "Full FT" if s["method"] == "full_ft" else "PEFT"
        print(f"  {s['model']:<35} {m:<10} {s['overall']['accuracy']:>10.4f} {s['overall']['f1']:>10.4f}")

    # Write combined leaderboard
    leaderboard_path = RESULTS_ROOT / "leaderboard.json"
    leaderboard_path.parent.mkdir(parents=True, exist_ok=True)
    leaderboard_path.write_text(json.dumps(all_summaries, indent=2))
    print(f"\nLeaderboard written to: {leaderboard_path}")


if __name__ == "__main__":
    main()
