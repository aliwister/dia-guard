#!/usr/bin/env python3
"""
D-PURiFY: Dataset Purity Evaluation Framework
DIA-GUARD -> Dia-LLM -> D-PURiFY

Main evaluation runner that computes all automatic metrics:
  1. Text Similarity  (CPU)  : BLEU, METEOR, ROUGE-L, DiffLib
  2. Neural Metrics   (GPU)  : BERTScore, BARTScore, AlignScore
  3. Dialect Validation       : eWAVE accuracy, valid features, errors
  4. CounterHarm Metrics (GPU): BERTScore, Semantic Similarity, NLI

Usage:
    # Evaluate all dialects (all metrics)
    python evaluate.py --data_dir ../LLM_Data

    # Evaluate specific dialect
    python evaluate.py --data_dir ../LLM_Data --dialect aboriginal_english

    # Evaluate only counterharm metrics
    python evaluate.py --data_dir ../LLM_Data --counterharm-only

    # Evaluate only dialect transform metrics
    python evaluate.py --data_dir ../LLM_Data --transform-only

    # Test on first 2 files
    python evaluate.py --data_dir ../LLM_Data --test
"""

import argparse
import json
import sys
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from metrics.text_similarity import TextSimilarityMetrics
from metrics.neural_metrics import NeuralMetrics
from metrics.counterharm_metrics import CounterHarmMetrics

POLICY_VIOLATION = "guardrail policy violation"
PROGRESS_FILE_NAME = ".d_purify_progress.json"


def load_progress(data_dir: Path) -> dict:
    progress_file = data_dir / PROGRESS_FILE_NAME
    if progress_file.exists():
        with open(progress_file) as f:
            return json.load(f)
    return {"transform_completed": [], "counterharm_completed": [], "last_updated": None}


def save_progress(data_dir: Path, progress: dict):
    progress["last_updated"] = datetime.now().isoformat()
    with open(data_dir / PROGRESS_FILE_NAME, "w") as f:
        json.dump(progress, f, indent=2)


def evaluate_transforms(csv_path: Path, text_metrics, neural_metrics, progress: dict) -> bool:
    """Evaluate dialect transformation quality (basic_transform, coi_transform vs original)."""
    file_key = str(csv_path)
    if file_key in progress.get("transform_completed", []):
        return True

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"    Error reading {csv_path.name}: {e}")
        return False

    required = ["original_input", "basic_transform", "coi_transform"]
    if not all(c in df.columns for c in required):
        print(f"    Skipping {csv_path.name} (missing columns)")
        return False

    valid = df["basic_transform"].notna() & df["coi_transform"].notna()
    df_valid = df[valid].copy()
    if len(df_valid) == 0:
        print(f"    Skipping {csv_path.name} (no valid transforms)")
        return False

    refs = df_valid["original_input"].fillna("").astype(str).tolist()
    basic = df_valid["basic_transform"].fillna("").astype(str).tolist()
    coi = df_valid["coi_transform"].fillna("").astype(str).tolist()

    changed = False

    # --- Text similarity (CPU) ---
    if "basic_bleu" not in df.columns:
        print(f"    Computing text similarity metrics...")
        for prefix, candidates in [("basic", basic), ("coi", coi)]:
            results = text_metrics.compute_all(refs, candidates, prefix=prefix)
            for col, scores in results.items():
                df.loc[valid, col] = scores
        changed = True

    # --- Neural metrics (GPU) ---
    if "basic_bertscore" not in df.columns:
        print(f"    Computing neural metrics...")
        for prefix, candidates in [("basic", basic), ("coi", coi)]:
            results = neural_metrics.compute_all(refs, candidates, prefix=prefix)
            for col, scores in results.items():
                df.loc[valid, col] = scores
        changed = True

    if changed:
        df.to_csv(csv_path, index=False)

    progress.setdefault("transform_completed", []).append(file_key)
    return True


def evaluate_counterharm(csv_path: Path, ch_metrics, progress: dict) -> bool:
    """Evaluate counterharm benign samples (BERTScore, Semantic Sim, NLI)."""
    file_key = str(csv_path)
    if file_key in progress.get("counterharm_completed", []):
        return True

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"    Error reading {csv_path.name}: {e}")
        return False

    # Check if any counterharm columns exist
    ch_cols = ["counterharm_original", "counterharm_transformed", "counterharm_basic", "counterharm_coi"]
    if not any(c in df.columns for c in ch_cols):
        return False

    # Check if already evaluated
    if "ch_original_bertscore" in df.columns:
        progress.setdefault("counterharm_completed", []).append(file_key)
        return True

    print(f"    Computing counterharm metrics...")
    df = ch_metrics.evaluate_csv(df)
    df.to_csv(csv_path, index=False)

    progress.setdefault("counterharm_completed", []).append(file_key)
    return True


def main():
    parser = argparse.ArgumentParser(description="D-PURiFY: Dataset Purity Evaluation")
    parser.add_argument("--data_dir", type=str, default="../LLM_Data", help="Path to LLM_Data directory")
    parser.add_argument("--dialect", type=str, default=None, help="Specific dialect to evaluate")
    parser.add_argument("--transform-only", action="store_true", help="Only evaluate transform metrics")
    parser.add_argument("--counterharm-only", action="store_true", help="Only evaluate counterharm metrics")
    parser.add_argument("--test", action="store_true", help="Test on first 2 files only")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: {data_dir} does not exist")
        sys.exit(1)

    progress = load_progress(data_dir)

    # Find CSV files
    if args.dialect:
        dialects = [args.dialect]
    else:
        dialects = sorted([
            d.name for d in data_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ])

    csv_files = []
    for dialect in dialects:
        dialect_dir = data_dir / dialect
        if dialect_dir.exists():
            csv_files.extend(sorted(dialect_dir.glob("*_with_transforms.csv")))

    if args.test:
        csv_files = csv_files[:2]

    # Header
    print(f"\n{'=' * 80}")
    print(f"D-PURiFY: Dataset Purity Evaluation")
    print(f"DIA-GUARD -> Dia-LLM -> D-PURiFY")
    print(f"{'=' * 80}")
    print(f"Data Directory : {data_dir}")
    print(f"Dialects       : {len(dialects)}")
    print(f"CSV Files      : {len(csv_files)}")
    if not args.counterharm_only:
        print(f"Transform Metrics : BLEU, METEOR, ROUGE-L, DiffLib, BERTScore, BARTScore, AlignScore")
    if not args.transform_only:
        print(f"CounterHarm Metrics : BERTScore, Semantic Similarity, NLI (Contradiction/Entailment/Neutral)")
    print(f"{'=' * 80}\n")

    # Initialize metrics
    text_metrics = None
    neural_metrics = None
    ch_metrics = None

    if not args.counterharm_only:
        text_metrics = TextSimilarityMetrics()
        neural_metrics = NeuralMetrics()

    if not args.transform_only:
        ch_metrics = CounterHarmMetrics()

    # Process files
    transform_ok = 0
    counterharm_ok = 0
    failed = 0

    for i, csv_file in enumerate(csv_files):
        dialect_name = csv_file.parent.name
        print(f"\n[{i+1}/{len(csv_files)}] {dialect_name}/{csv_file.name}")

        try:
            if not args.counterharm_only:
                if evaluate_transforms(csv_file, text_metrics, neural_metrics, progress):
                    transform_ok += 1

            if not args.transform_only:
                if evaluate_counterharm(csv_file, ch_metrics, progress):
                    counterharm_ok += 1
        except Exception as e:
            print(f"    FAILED: {e}")
            failed += 1

        # Save progress every 5 files
        if (i + 1) % 5 == 0:
            save_progress(data_dir, progress)

    # Final save
    save_progress(data_dir, progress)

    # Cleanup
    if neural_metrics:
        neural_metrics.cleanup()
    if ch_metrics:
        ch_metrics.cleanup()

    # Summary
    print(f"\n{'=' * 80}")
    print(f"D-PURiFY EVALUATION COMPLETE")
    print(f"{'=' * 80}")
    if not args.counterharm_only:
        print(f"Transform evaluations : {transform_ok}")
    if not args.transform_only:
        print(f"CounterHarm evaluations: {counterharm_ok}")
    print(f"Failed               : {failed}")
    print(f"Total files          : {len(csv_files)}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
