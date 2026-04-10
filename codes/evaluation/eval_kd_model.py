#!/usr/bin/env python3
"""Evaluate a KD-distilled model on the DIA-GUARD holdout test set."""
import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import (
    load_jsonl, run_inference, compute_metrics,
    compute_per_dialect, write_predictions, write_confusion_matrix,
)

TEST_DATA = "/data/vibe_exp/dia-guard/dataset/dia_splits/test.jsonl"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--method", required=True, help="e.g. MINILLM, GKD, TED")
    p.add_argument("--teacher", required=True, help="e.g. Qwen3-4B-SafeRL")
    p.add_argument("--student_name", required=True, help="e.g. Qwen3Guard-Gen-0.6B")
    p.add_argument("--scenario", default="oob")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--test_data", default=TEST_DATA)
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if (out_dir / "metrics.json").exists():
        existing = json.load(open(out_dir / "metrics.json"))
        print(f"Already evaluated (acc={existing['overall']['accuracy']:.4f}), skipping.")
        return

    print(f"Loading KD model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16,
        attn_implementation="eager", trust_remote_code=True, device_map="cuda",
    )
    model.eval()

    print(f"Loading test data: {args.test_data}")
    records = load_jsonl(args.test_data)
    print(f"  {len(records):,} test records")

    print("Running inference...")
    predictions = run_inference(
        model, tokenizer, records,
        batch_size=args.batch_size, max_new_tokens=8, device="cuda",
    )

    print("Computing metrics...")
    metrics = compute_metrics(predictions)
    per_dialect = compute_per_dialect(predictions)
    print(f"  Accuracy: {metrics['accuracy']:.4f}  F1: {metrics['f1']:.4f}")

    write_predictions(predictions, str(out_dir / "predictions.jsonl"))
    write_confusion_matrix(metrics["confusion"], str(out_dir / "confusion_matrix.csv"))

    summary = {
        "model": args.student_name,
        "method": f"KD-{args.method.upper()}",
        "teacher": args.teacher,
        "scenario": args.scenario,
        "test_n": metrics.get("support", 0),
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
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "per_dialect.json", "w") as f:
        json.dump(per_dialect, f, indent=2)
    print(f"Results saved to {out_dir}")
    print("DONE")


if __name__ == "__main__":
    main()
