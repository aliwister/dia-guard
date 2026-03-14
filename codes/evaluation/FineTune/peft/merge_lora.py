"""
DIA-GUARD — Merge LoRA Adapters into Base Model
================================================
After PEFT fine-tuning, merges the LoRA adapter weights back into the base model
to produce a standalone full model suitable for use as a teacher in KD.

Usage:
  python merge_lora.py \\
      --adapter_dir ../../../models/FT/peft/qwen3-4b-lora \\
      --base_model Qwen/Qwen3-4B-SafeRL \\
      --output_dir ../../../models/FT/peft/qwen3-4b-lora-merged

  # With explicit dtype
  python merge_lora.py \\
      --adapter_dir ../../../models/FT/peft/aya-3b-lora \\
      --base_model CohereLabs/tiny-aya-global \\
      --output_dir ../../../models/FT/peft/aya-3b-lora-merged \\
      --dtype bfloat16
"""

import argparse
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def merge(adapter_dir: str, base_model: str, output_dir: str, dtype_str: str = "bfloat16"):
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(dtype_str, torch.bfloat16)

    print(f"Loading base model: {base_model}")
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map="cpu",  # merge on CPU to avoid OOM
    )

    print(f"Loading LoRA adapter: {adapter_dir}")
    model = PeftModel.from_pretrained(base, adapter_dir)

    print("Merging LoRA weights into base model...")
    merged = model.merge_and_unload()

    print(f"Saving merged model to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    merged.save_pretrained(output_dir, safe_serialization=True)

    # Copy tokenizer
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)

    print(f"Done. Merged model saved to: {output_dir}")
    print("This model can now be used as a teacher for knowledge distillation.")


def parse_args():
    parser = argparse.ArgumentParser(description="Merge LoRA adapters into base model")
    parser.add_argument("--adapter_dir", type=str, required=True,
                        help="Path to LoRA adapter directory")
    parser.add_argument("--base_model", type=str, required=True,
                        help="Base model HF ID or local path")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for merged model")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    merge(args.adapter_dir, args.base_model, args.output_dir, args.dtype)
