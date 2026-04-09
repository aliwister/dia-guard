#!/usr/bin/env python3
"""
Evaluate BASE (un-fine-tuned) models on the DIA-GUARD holdout test set.

These serve as baselines to compare against our fine-tuned Shield models.
For each of the 7 base student models, we load it directly from HF and run
the same safety classification prompt.

Usage:
  CUDA_VISIBLE_DEVICES=3 python evaluate_baseline_holdout.py --gpu_split first
  CUDA_VISIBLE_DEVICES=7 python evaluate_baseline_holdout.py --gpu_split second
  CUDA_VISIBLE_DEVICES=0 python evaluate_baseline_holdout.py --only Llama-3.2-1B
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Reuse logic from evaluate.py
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import (
    load_jsonl,
    run_inference,
    compute_metrics,
    compute_per_dialect,
    write_predictions,
    write_confusion_matrix,
)

TEST_DATA = "/data/vibe_exp/dia-guard/dataset/dia_splits/test.jsonl"
RESULTS_ROOT = Path("/data/vibe_exp/dia-guard/codes/evaluation/results/Shield/Baseline")
HF_TOKEN_PATH = Path("/data/huggingface/token")

# 7 student base models (no fine-tuning)
BASE_MODELS = {
    "Gemma-3-270m":          "google/gemma-3-270m-it",
    "Gemma-3-1B":            "google/gemma-3-1b-it",
    "Llama-3.2-1B":          "meta-llama/Llama-3.2-1B-Instruct",
    "Qwen3Guard-Gen-0.6B":   "Qwen/Qwen3Guard-Gen-0.6B",
    "Qwen3.5-0.8B":          "Qwen/Qwen3.5-0.8B",
    "Qwen3-1.7B":            "Qwen/Qwen3-1.7B",
    "SmolLM2-1.7B":          "HuggingFaceTB/SmolLM2-1.7B-Instruct",
}

# Per-model batch size for inference (no gradients) - same as Full FT
BATCH_SIZE = {
    "Gemma-3-270m":          1024,
    "Qwen3Guard-Gen-0.6B":   256,
    "Qwen3.5-0.8B":          256,
    "Gemma-3-1B":            384,
    "Llama-3.2-1B":          256,
    "Qwen3-1.7B":            192,
    "SmolLM2-1.7B":          96,
}

# GPU split: assign small models to one GPU, big to the other
SPLIT_FIRST = [
    "Gemma-3-270m", "Qwen3Guard-Gen-0.6B", "Qwen3.5-0.8B", "Gemma-3-1B",
]
SPLIT_SECOND = [
    "Llama-3.2-1B", "Qwen3-1.7B", "SmolLM2-1.7B",
]


def make_output_dir(pretty):
    out = RESULTS_ROOT / pretty
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_metrics_json(metrics, per_dialect, output_dir, pretty, base_id):
    summary = {
        "model": pretty,
        "base_model": base_id,
        "method": "baseline",
        "loss": "none",
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


def evaluate_one(pretty, base_id, records, force=False):
    bs = BATCH_SIZE.get(pretty, 64)
    print(f"\n{'='*70}")
    print(f"  BASELINE: {pretty}  bs={bs}")
    print(f"  Base model: {base_id}")
    print(f"{'='*70}")

    output_dir = make_output_dir(pretty)
    print(f"  Output: {output_dir}")

    if not force and (output_dir / "metrics.json").exists():
        existing = json.load(open(output_dir / "metrics.json"))
        print(f"  ⏭  Already evaluated (acc={existing['overall']['accuracy']:.4f}), skipping")
        return existing

    print(f"  Loading base model from HF...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_id, trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        trust_remote_code=True,
        device_map="cuda",
        token=os.environ.get("HF_TOKEN"),
    )
    model.eval()

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

    write_predictions(predictions, str(output_dir / "predictions.jsonl"))
    write_confusion_matrix(metrics["confusion"], str(output_dir / "confusion_matrix.csv"))
    summary = write_metrics_json(metrics, per_dialect, output_dir, pretty, base_id)

    del model, tokenizer
    torch.cuda.empty_cache()

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=str, default=None,
                        help="Run only this pretty name (e.g. 'Llama-3.2-1B')")
    parser.add_argument("--gpu_split", type=str, choices=["first", "second", "all"],
                        default="all", help="Which subset of models to run")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--test_data", type=str, default=None,
                        help="Override path to the test JSONL")
    parser.add_argument("--results_root", type=str, default=None,
                        help="Override results dir (default: results/Shield/Baseline)")
    args = parser.parse_args()

    global TEST_DATA, RESULTS_ROOT
    if args.test_data:
        TEST_DATA = args.test_data
    if args.results_root:
        RESULTS_ROOT = Path(args.results_root)

    if "HF_TOKEN" not in os.environ and HF_TOKEN_PATH.exists():
        os.environ["HF_TOKEN"] = HF_TOKEN_PATH.read_text().strip()

    if args.only:
        pretty_names = [args.only]
    elif args.gpu_split == "first":
        pretty_names = SPLIT_FIRST
    elif args.gpu_split == "second":
        pretty_names = SPLIT_SECOND
    else:
        pretty_names = list(BASE_MODELS.keys())

    print(f"Will evaluate {len(pretty_names)} base models:")
    for n in pretty_names:
        print(f"  - {n} ({BASE_MODELS[n]})")

    print(f"\nLoading test data: {TEST_DATA}")
    records = load_jsonl(TEST_DATA)
    print(f"  {len(records):,} test records")

    summaries = []
    for pretty in pretty_names:
        base_id = BASE_MODELS[pretty]
        try:
            s = evaluate_one(pretty, base_id, records, force=args.force)
            summaries.append(s)
        except Exception as e:
            import traceback
            print(f"  ! FAILED: {e}")
            traceback.print_exc()

    print(f"\n{'='*70}")
    print("  BASELINE LEADERBOARD")
    print(f"{'='*70}")
    print(f"  {'Model':<25} {'Acc':>10} {'Prec':>10} {'Rec':>10} {'F1':>10}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for s in sorted(summaries, key=lambda x: -x["overall"]["accuracy"]):
        o = s["overall"]
        print(f"  {s['model']:<25} {o['accuracy']:>10.4f} {o['precision']:>10.4f} {o['recall']:>10.4f} {o['f1']:>10.4f}")

    leaderboard_path = RESULTS_ROOT / "leaderboard.json"
    leaderboard_path.parent.mkdir(parents=True, exist_ok=True)
    leaderboard_path.write_text(json.dumps(summaries, indent=2))
    print(f"\nLeaderboard written to: {leaderboard_path}")


if __name__ == "__main__":
    main()
