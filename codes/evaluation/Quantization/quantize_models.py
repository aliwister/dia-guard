#!/usr/bin/env python3
"""
DIA-GUARD Group 4 — Post-Training Quantization
================================================

Quantizes all distilled student models (Group 2 KD) and optionally
Group 3 (Student FT Baseline) models at 16-bit, 8-bit, and 4-bit
precision using bitsandbytes.

Quantization levels:
  - fp16:  torch.float16 (baseline, 2 bytes/param)
  - int8:  LLM.int8() (1 byte/param, ~0% accuracy loss)
  - int4:  NF4 with double quantization (0.5 bytes/param)

Usage:
  # Quantize all completed KD models
  python quantize_models.py

  # Quantize a specific model
  python quantize_models.py --model_dir ../../models/KD/minillm/qwen3guard_gen_0_6b

  # Quantize only at 4-bit
  python quantize_models.py --bits 4

  # Quantize Group 3 models too
  python quantize_models.py --include_group3

  # Dry run
  python quantize_models.py --dry_run

  # Upload quantized models to HuggingFace
  python quantize_models.py --push_to_hub --hf_org jsl5710 --hf_token YOUR_TOKEN
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent  # dia-guard/
MODELS_DIR = REPO_ROOT / "models"
SPLITS_DIR = REPO_ROOT / "dataset" / "dia_splits"

QUANT_CONFIGS = {
    16: {
        "label": "fp16",
        "suffix": "fp16",
        "description": "Float16 baseline (2 bytes/param)",
        "load_kwargs": {
            "torch_dtype": torch.float16,
        },
    },
    8: {
        "label": "int8",
        "suffix": "int8",
        "description": "LLM.int8() quantization (1 byte/param)",
        "load_kwargs": {
            "quantization_config": BitsAndBytesConfig(load_in_8bit=True),
        },
    },
    4: {
        "label": "nf4",
        "suffix": "nf4",
        "description": "NF4 4-bit with double quantization (0.5 bytes/param)",
        "load_kwargs": {
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            ),
        },
    },
}

# Slug → base model HF ID (for PEFT adapter loading)
SLUG_TO_BASE_MODEL = {
    "gemma_3_270m_it":        "google/gemma-3-270m-it",
    "gemma_3_1b_it":          "google/gemma-3-1b-it",
    "llama_3_2_1b_instruct":  "meta-llama/Llama-3.2-1B-Instruct",
    "qwen3guard_gen_0_6b":    "Qwen/Qwen3Guard-Gen-0.6B",
    "qwen3_5_0_8b":           "Qwen/Qwen3.5-0.8B",
    "smollm2_1_7b_instruct":  "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "qwen3_1_7b":             "Qwen/Qwen3-1.7B",
}

SLUG_TO_SHORT = {
    "gemma_3_270m_it":        "Gemma-270M",
    "gemma_3_1b_it":          "Gemma-1B",
    "llama_3_2_1b_instruct":  "Llama-1B",
    "qwen3guard_gen_0_6b":    "QwenGuard-0.6B",
    "qwen3_5_0_8b":           "Qwen-0.8B",
    "smollm2_1_7b_instruct":  "SmolLM-1.7B",
    "qwen3_1_7b":             "Qwen-1.7B",
}


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def is_model_complete(model_dir: Path) -> bool:
    """Check if a model directory has a trained model."""
    markers = [
        "adapter_model.safetensors", "adapter_model.bin",
        "model.safetensors", "pytorch_model.bin",
        "model.safetensors.index.json", "config.json",
    ]
    return any((model_dir / m).exists() for m in markers)


def is_peft_model(model_dir: Path) -> bool:
    """Check if model is a PEFT adapter (not full weights)."""
    return (model_dir / "adapter_config.json").exists()


def discover_models(include_group3: bool = False) -> list[dict]:
    """Find all completed KD (and optionally G3) models."""
    found = []

    # Group 2: KD models
    kd_dir = MODELS_DIR / "KD"
    if kd_dir.exists():
        for method_dir in sorted(kd_dir.iterdir()):
            if not method_dir.is_dir() or method_dir.name.startswith("."):
                continue
            for model_dir in sorted(method_dir.iterdir()):
                if not model_dir.is_dir() or model_dir.name.startswith("."):
                    continue
                if is_model_complete(model_dir):
                    found.append({
                        "group": 2,
                        "method": method_dir.name,
                        "model_slug": model_dir.name,
                        "model_dir": model_dir,
                        "is_peft": is_peft_model(model_dir),
                    })

    # Group 3: Student FT Baseline
    if include_group3:
        g3_dir = MODELS_DIR / "group3_student_ft_baseline"
        if g3_dir.exists():
            for method_dir in sorted(g3_dir.iterdir()):
                if not method_dir.is_dir() or method_dir.name.startswith("."):
                    continue
                for model_dir in sorted(method_dir.iterdir()):
                    if not model_dir.is_dir() or model_dir.name.startswith("."):
                        continue
                    if is_model_complete(model_dir):
                        found.append({
                            "group": 3,
                            "method": method_dir.name,
                            "model_slug": model_dir.name,
                            "model_dir": model_dir,
                            "is_peft": is_peft_model(model_dir),
                        })

    return found


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------

def get_model_size_mb(model_dir: Path) -> float:
    """Get total size of model files in MB."""
    total = 0
    for f in model_dir.rglob("*"):
        if f.is_file() and f.suffix in (".safetensors", ".bin", ".pt"):
            total += f.stat().st_size
    return total / 1e6


def load_model_for_quantization(
    model_dir: Path,
    model_slug: str,
    is_peft: bool,
    quant_kwargs: dict,
) -> tuple:
    """Load a model with quantization config applied."""
    trust_remote_code = True

    if is_peft:
        # PEFT model: load base model with quantization, then apply adapter
        base_model_id = SLUG_TO_BASE_MODEL.get(model_slug)
        if not base_model_id:
            raise ValueError(f"Unknown base model for PEFT slug: {model_slug}")

        from peft import PeftModel

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            device_map="auto",
            trust_remote_code=trust_remote_code,
            **quant_kwargs,
        )
        model = PeftModel.from_pretrained(base_model, str(model_dir))
        # Merge adapter into base for quantized saving
        model = model.merge_and_unload()
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    else:
        # Full model: load with quantization directly
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            device_map="auto",
            trust_remote_code=trust_remote_code,
            **quant_kwargs,
        )
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

    return model, tokenizer


def quantize_model(
    model_entry: dict,
    bits: int,
    output_base: Path,
    push_to_hub: bool = False,
    hf_org: Optional[str] = None,
    hf_token: Optional[str] = None,
) -> dict:
    """Quantize a single model at the specified bit width."""
    qcfg = QUANT_CONFIGS[bits]
    model_dir = model_entry["model_dir"]
    model_slug = model_entry["model_slug"]
    method = model_entry["method"]
    group = model_entry["group"]
    is_peft = model_entry["is_peft"]

    # Output directory
    group_label = "KD" if group == 2 else "group3_student_ft_baseline"
    output_dir = output_base / group_label / method / model_slug / qcfg["suffix"]

    result = {
        "model_slug": model_slug,
        "method": method,
        "bits": bits,
        "label": qcfg["label"],
        "output_dir": str(output_dir),
        "status": "pending",
    }

    # Skip if already quantized
    if (output_dir / "config.json").exists() or (output_dir / "model.safetensors").exists():
        print(f"  SKIP {qcfg['label']:>4} — already exists: {output_dir.name}")
        result["status"] = "skipped"
        return result

    print(f"  {qcfg['label']:>4} — loading with {qcfg['description']}...")
    t0 = time.time()

    try:
        model, tokenizer = load_model_for_quantization(
            model_dir=model_dir,
            model_slug=model_slug,
            is_peft=is_peft,
            quant_kwargs=qcfg["load_kwargs"],
        )

        # Save quantized model
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        # Save quantization metadata
        meta = {
            "source_model": str(model_dir),
            "quantization": qcfg["label"],
            "bits": bits,
            "description": qcfg["description"],
            "group": group,
            "method": method,
            "model_slug": model_slug,
        }
        with open(output_dir / "quantization_config.json", "w") as f:
            json.dump(meta, f, indent=2)

        elapsed = time.time() - t0
        size_mb = get_model_size_mb(output_dir)
        print(f"       Saved to {output_dir} ({size_mb:.1f} MB, {elapsed:.1f}s)")

        result["status"] = "done"
        result["size_mb"] = size_mb
        result["elapsed_s"] = elapsed

        # Push to HuggingFace
        if push_to_hub and hf_org and hf_token:
            short = SLUG_TO_SHORT.get(model_slug, model_slug)
            method_upper = method.upper()
            repo_id = f"{hf_org}/DIA-Guard-{short}-{method_upper}-{qcfg['suffix']}"
            print(f"       Pushing to {repo_id}...")
            model.push_to_hub(repo_id, token=hf_token)
            tokenizer.push_to_hub(repo_id, token=hf_token)
            result["hf_repo"] = repo_id

        # Free GPU memory
        del model
        del tokenizer
        torch.cuda.empty_cache()

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        print(f"       ERROR: {e}")

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="DIA-GUARD Group 4 — Quantize distilled models at 16/8/4-bit"
    )
    parser.add_argument("--model_dir", type=str, default=None,
                        help="Quantize a specific model directory")
    parser.add_argument("--bits", type=int, nargs="+", default=[16, 8, 4],
                        choices=[16, 8, 4],
                        help="Bit widths to quantize (default: 16 8 4)")
    parser.add_argument("--include_group3", action="store_true",
                        help="Also quantize Group 3 (Student FT Baseline) models")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output base dir (default: models/Quantized/)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Show what would be quantized without doing it")
    parser.add_argument("--push_to_hub", action="store_true",
                        help="Push quantized models to HuggingFace")
    parser.add_argument("--hf_org", type=str, default="jsl5710",
                        help="HuggingFace org for push (default: jsl5710)")
    parser.add_argument("--hf_token", type=str, default=None,
                        help="HuggingFace API token")
    args = parser.parse_args()

    output_base = Path(args.output_dir) if args.output_dir else MODELS_DIR / "Quantized"

    # Discover models
    if args.model_dir:
        model_path = Path(args.model_dir)
        if not model_path.exists():
            print(f"ERROR: {args.model_dir} does not exist")
            sys.exit(1)
        models = [{
            "group": 2,
            "method": model_path.parent.name,
            "model_slug": model_path.name,
            "model_dir": model_path,
            "is_peft": is_peft_model(model_path),
        }]
    else:
        models = discover_models(include_group3=args.include_group3)

    if not models:
        print("No completed models found to quantize.")
        print(f"Checked: {MODELS_DIR / 'KD'}")
        if args.include_group3:
            print(f"Checked: {MODELS_DIR / 'group3_student_ft_baseline'}")
        sys.exit(0)

    # Status table
    print("=" * 70)
    print("DIA-GUARD Group 4 — Post-Training Quantization")
    print("=" * 70)
    print(f"\nModels found: {len(models)}")
    print(f"Bit widths:   {args.bits}")
    print(f"Output:       {output_base}")
    print()

    for m in models:
        g = f"G{m['group']}"
        peft_tag = " (PEFT)" if m["is_peft"] else ""
        print(f"  {g}  {m['method']:<10} {m['model_slug']}{peft_tag}")

    if args.dry_run:
        print(f"\n[DRY RUN] Would quantize {len(models)} models × {len(args.bits)} bit widths = {len(models) * len(args.bits)} outputs")
        return

    # Quantize
    print(f"\nStarting quantization...")
    results = []

    for m in models:
        print(f"\n{'─' * 60}")
        print(f"G{m['group']} | {m['method']} | {m['model_slug']}")
        print(f"  Source: {m['model_dir']}")

        for bits in sorted(args.bits, reverse=True):  # 16 → 8 → 4
            r = quantize_model(
                model_entry=m,
                bits=bits,
                output_base=output_base,
                push_to_hub=args.push_to_hub,
                hf_org=args.hf_org,
                hf_token=args.hf_token,
            )
            results.append(r)

    # Summary
    done = [r for r in results if r["status"] == "done"]
    skipped = [r for r in results if r["status"] == "skipped"]
    errors = [r for r in results if r["status"] == "error"]

    print(f"\n{'=' * 70}")
    print(f"Quantization complete: {len(done)} done, {len(skipped)} skipped, {len(errors)} errors")

    if done:
        print(f"\nQuantized models saved to: {output_base}")
        total_mb = sum(r.get("size_mb", 0) for r in done)
        print(f"Total size: {total_mb:.1f} MB")

    if errors:
        print("\nErrors:")
        for r in errors:
            print(f"  {r['model_slug']} ({r['label']}): {r.get('error', 'unknown')}")

    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
