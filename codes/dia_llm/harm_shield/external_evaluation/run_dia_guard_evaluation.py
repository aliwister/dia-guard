#!/usr/bin/env python3
"""
DIA-Guard Dataset Evaluation Runner

Processes all dialect folders in the DIA-Guard dataset with:
- GPU metrics: BERTScore, BARTScore, AlignScore, METEOR
- CPU metrics: ROUGE-L, DiffLib
- Feature Accuracy (eWAVE validation)
- LLM-as-a-Judge (optional, expensive)

Optimized for T4 16GB GPU with resume capability.

Usage:
    python run_dia_guard_evaluation.py --gpu-only
    python run_dia_guard_evaluation.py --cpu-only
    python run_dia_guard_evaluation.py --all
    python run_dia_guard_evaluation.py --bertscore-only
    python run_dia_guard_evaluation.py --llm-only --llm-backend azure
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
from tqdm import tqdm
import gc
import json

# Add parent directory for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)


# Default paths
DIA_GUARD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "dataset", "dia_guard")

# Batch sizes optimized for T4 16GB - maximized for faster processing
BATCH_SIZES = {
    "BERTScore": 256,     # BERTScore using sentence-transformers - can go high
    "BARTScore": 64,      # BART-large - increased batch size
    "AlignScore": 128,    # Sentence-transformer fallback - can go higher
    "METEOR": 500,        # CPU-based, no GPU memory concern
}


def find_all_csv_files(dataset_dir: str) -> list:
    """Find all CSV files in the dataset directory."""
    dataset_path = Path(dataset_dir)
    csv_files = []

    # Find CSVs directly in the root directory
    for csv_file in dataset_path.glob("*.csv"):
        csv_files.append({
            "path": str(csv_file),
            "dialect": "root",
            "relative": csv_file.name
        })

    for dialect_folder in dataset_path.iterdir():
        if not dialect_folder.is_dir() or dialect_folder.name.startswith('.'):
            continue

        # Find CSVs directly in dialect folders
        for csv_file in dialect_folder.glob("*.csv"):
            csv_files.append({
                "path": str(csv_file),
                "dialect": dialect_folder.name,
                "relative": str(csv_file.relative_to(dataset_path))
            })

    return sorted(csv_files, key=lambda x: x["relative"])


def get_progress_file(dataset_dir: str) -> str:
    """Get path to progress tracking file."""
    return os.path.join(dataset_dir, ".evaluation_progress_guard.json")


def load_progress(dataset_dir: str) -> dict:
    """Load evaluation progress."""
    progress_file = get_progress_file(dataset_dir)
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            return json.load(f)
    return {"completed": {}, "last_updated": None}


def save_progress(dataset_dir: str, progress: dict):
    """Save evaluation progress."""
    progress_file = get_progress_file(dataset_dir)
    progress["last_updated"] = datetime.now().isoformat()
    with open(progress_file, 'w') as f:
        json.dump(progress, f, indent=2)


def detect_column_pairs(df: pd.DataFrame) -> list:
    """Detect the correct column pairs based on dataset format."""
    if "human_content" in df.columns:
        return [
            ("human_content", "human_content_transformed", "human_"),
            ("ai_content", "ai_content_transformed", "ai_")
        ]
    elif "prompt" in df.columns and "prompt_transformed" in df.columns:
        return [("prompt", "prompt_transformed", "")]
    elif "Behavior" in df.columns and "Behavior_transformed" in df.columns:
        return [("Behavior", "Behavior_transformed", "")]
    else:
        # Try to auto-detect any *_transformed column
        for col in df.columns:
            if col.endswith("_transformed"):
                orig_col = col.replace("_transformed", "")
                if orig_col in df.columns:
                    return [(orig_col, col, "")]
        return []


def run_single_gpu_metric(
    csv_path: str,
    dialect: str,
    metric: str,
    verbose: bool = True
) -> pd.DataFrame:
    """Run a single GPU-based metric evaluation."""
    import torch

    if verbose:
        print(f"\n{'='*60}")
        print(f"{metric}: {csv_path}")
        print(f"Dialect: {dialect}")
        if torch.cuda.is_available():
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        print(f"{'='*60}")

    df = pd.read_csv(csv_path)
    pairs = detect_column_pairs(df)

    if not pairs:
        print(f"Warning: No valid column pairs found in {csv_path}")
        return df

    for orig_col, trans_col, prefix in pairs:
        if orig_col not in df.columns or trans_col not in df.columns:
            continue

        originals = df[orig_col].fillna("").astype(str).tolist()
        transformed = df[trans_col].fillna("").astype(str).tolist()

        if metric == "bertscore":
            col = f"{prefix}bertscore_f1"
            if col not in df.columns or df[col].isna().all():
                if verbose:
                    print(f"\nComputing BERTScore ({prefix or 'main'})...")
                try:
                    # Use sentence-transformers approach which is more stable
                    from sentence_transformers import SentenceTransformer, util

                    model = SentenceTransformer('all-mpnet-base-v2')

                    # Compute embeddings in batches
                    batch_size = BATCH_SIZES["BERTScore"]
                    f1_scores = []
                    precision_scores = []
                    recall_scores = []

                    for i in tqdm(range(0, len(originals), batch_size), desc="BERTScore"):
                        batch_orig = originals[i:i+batch_size]
                        batch_trans = transformed[i:i+batch_size]

                        orig_embeddings = model.encode(batch_orig, convert_to_tensor=True)
                        trans_embeddings = model.encode(batch_trans, convert_to_tensor=True)

                        # Compute cosine similarity as F1 approximation
                        similarities = util.cos_sim(orig_embeddings, trans_embeddings)
                        for j in range(len(batch_orig)):
                            sim = float(similarities[j][j])
                            f1_scores.append(sim)
                            precision_scores.append(sim)
                            recall_scores.append(sim)

                    df[col] = f1_scores
                    df[f"{prefix}bertscore_precision"] = precision_scores
                    df[f"{prefix}bertscore_recall"] = recall_scores

                    del model
                    gc.collect()
                    torch.cuda.empty_cache()
                    df.to_csv(csv_path, index=False)
                except Exception as e:
                    print(f"BERTScore error: {e}")

        elif metric == "bartscore":
            col = f"{prefix}bartscore"
            if col not in df.columns or df[col].isna().all():
                if verbose:
                    print(f"\nComputing BARTScore ({prefix or 'main'})...")
                # Dynamically adjust batch size for large files
                n_samples = len(originals)
                if n_samples > 5000:
                    batch_size = 16  # Small batch for very large files
                elif n_samples > 2000:
                    batch_size = 32  # Medium batch for large files
                else:
                    batch_size = BATCH_SIZES["BARTScore"]
                if verbose:
                    print(f"  Using batch size: {batch_size} for {n_samples} samples")

                scores = None
                for attempt, bs in enumerate([batch_size, batch_size // 2, 4, 2], 1):
                    try:
                        from gpu_metrics import BARTScoreEvaluator
                        gc.collect()
                        torch.cuda.empty_cache()
                        evaluator = BARTScoreEvaluator(
                            batch_size=bs,
                            verbose=verbose
                        )
                        scores = evaluator.score_batch(originals, transformed)
                        evaluator.cleanup()
                        break
                    except RuntimeError as e:
                        if "CUDA out of memory" in str(e) and attempt < 4:
                            print(f"  OOM with batch_size={bs}, retrying with smaller batch...")
                            gc.collect()
                            torch.cuda.empty_cache()
                            continue
                        else:
                            raise e

                if scores is not None:
                    df[col] = scores
                    gc.collect()
                    torch.cuda.empty_cache()
                    df.to_csv(csv_path, index=False)
                else:
                    print(f"BARTScore failed for {csv_path}")

        elif metric == "alignscore":
            col = f"{prefix}alignscore"
            if col not in df.columns or df[col].isna().all():
                if verbose:
                    print(f"\nComputing AlignScore ({prefix or 'main'})...")
                try:
                    from gpu_metrics import AlignScoreEvaluator
                    evaluator = AlignScoreEvaluator(
                        batch_size=BATCH_SIZES["AlignScore"],
                        verbose=verbose
                    )
                    scores = evaluator.score_batch(originals, transformed)
                    df[col] = scores
                    evaluator.cleanup()
                    gc.collect()
                    torch.cuda.empty_cache()
                    df.to_csv(csv_path, index=False)
                except Exception as e:
                    print(f"AlignScore error: {e}")

        elif metric == "meteor":
            col = f"{prefix}meteor"
            if col not in df.columns or df[col].isna().all():
                if verbose:
                    print(f"\nComputing METEOR ({prefix or 'main'})...")
                try:
                    from gpu_metrics import METEORScoreEvaluator
                    evaluator = METEORScoreEvaluator(verbose=verbose)
                    scores = evaluator.score_batch(originals, transformed)
                    df[col] = scores
                    df.to_csv(csv_path, index=False)
                except Exception as e:
                    print(f"METEOR error: {e}")

    return df


def run_gpu_metrics(
    csv_path: str,
    dialect: str,
    batch_size: int = 8,
    verbose: bool = True
) -> pd.DataFrame:
    """Run all GPU-based metrics evaluation."""
    for metric in ["bertscore", "bartscore", "alignscore", "meteor"]:
        run_single_gpu_metric(csv_path, dialect, metric, verbose)

    return pd.read_csv(csv_path)


def run_cpu_metrics(
    csv_path: str,
    dialect: str,
    verbose: bool = True
) -> pd.DataFrame:
    """Run CPU-based metrics evaluation."""
    from cpu_metrics import ROUGELEvaluator, DiffLibEvaluator, BLEUEvaluator

    if verbose:
        print(f"\n{'='*60}")
        print(f"CPU Metrics: {csv_path}")
        print(f"{'='*60}")

    df = pd.read_csv(csv_path)
    pairs = detect_column_pairs(df)

    if not pairs:
        print(f"Warning: No valid column pairs found in {csv_path}")
        return df

    for orig_col, trans_col, prefix in pairs:
        if orig_col not in df.columns or trans_col not in df.columns:
            continue

        originals = df[orig_col].fillna("").astype(str).tolist()
        transformed = df[trans_col].fillna("").astype(str).tolist()

        # ROUGE-L
        col = f"{prefix}rouge_l"
        if col not in df.columns or df[col].isna().all():
            if verbose:
                print(f"Computing ROUGE-L ({prefix or 'main'})...")
            try:
                evaluator = ROUGELEvaluator(verbose=verbose)
                detailed = evaluator.score_detailed(originals, transformed)
                df[col] = detailed['f1']
                df[f"{prefix}rouge_l_precision"] = detailed['precision']
                df[f"{prefix}rouge_l_recall"] = detailed['recall']
                df.to_csv(csv_path, index=False)
            except Exception as e:
                print(f"ROUGE-L error: {e}")

        # DiffLib
        col = f"{prefix}difflib_ratio"
        if col not in df.columns or df[col].isna().all():
            if verbose:
                print(f"Computing DiffLib ({prefix or 'main'})...")
            try:
                evaluator = DiffLibEvaluator(verbose=verbose)
                detailed = evaluator.score_detailed(originals, transformed)
                df[col] = detailed['ratio']
                df.to_csv(csv_path, index=False)
            except Exception as e:
                print(f"DiffLib error: {e}")

        # BLEU
        col = f"{prefix}bleu"
        if col not in df.columns or df[col].isna().all():
            if verbose:
                print(f"Computing BLEU ({prefix or 'main'})...")
            try:
                evaluator = BLEUEvaluator(verbose=verbose)
                scores = evaluator.score_batch(originals, transformed)
                df[col] = scores
                df.to_csv(csv_path, index=False)
            except Exception as e:
                print(f"BLEU error: {e}")

    return df


def run_feature_accuracy(
    csv_path: str,
    dialect: str,
    verbose: bool = True
) -> pd.DataFrame:
    """Run Feature Accuracy evaluation."""
    from llm_evaluation import FeatureAccuracyEvaluator, convert_dialect_folder_to_key

    if verbose:
        print(f"\n{'='*60}")
        print(f"Feature Accuracy: {csv_path}")
        print(f"{'='*60}")

    df = pd.read_csv(csv_path)
    dialect_key = convert_dialect_folder_to_key(dialect)
    pairs = detect_column_pairs(df)

    if not pairs:
        print(f"Warning: No valid column pairs found in {csv_path}")
        return df

    evaluator = FeatureAccuracyEvaluator(verbose=verbose)

    for orig_col, trans_col, prefix in pairs:
        if orig_col not in df.columns or trans_col not in df.columns:
            continue

        col = f"{prefix}feature_accuracy"
        if col in df.columns and df[col].notna().any():
            if verbose:
                print(f"Skipping already computed: {col}")
            continue

        originals = df[orig_col].fillna("").astype(str).tolist()
        transformed = df[trans_col].fillna("").astype(str).tolist()

        try:
            results = evaluator.evaluate_batch(originals, transformed, dialect_key, show_progress=verbose)
            scores = evaluator.get_scores_dict(results)

            for key, values in scores.items():
                df[f"{prefix}{key}"] = values

            df.to_csv(csv_path, index=False)
        except Exception as e:
            print(f"Feature Accuracy error: {e}")

    return df


def run_llm_evaluation(
    csv_path: str,
    dialect: str,
    backend: str = "azure",
    model: str = None,
    verbose: bool = True
) -> pd.DataFrame:
    """Run LLM-as-a-Judge evaluation."""
    from llm_evaluation import LLMJudgeEvaluator

    if verbose:
        print(f"\n{'='*60}")
        print(f"LLM-as-a-Judge: {csv_path}")
        print(f"Backend: {backend}")
        print(f"{'='*60}")

    df = pd.read_csv(csv_path)
    dialect_name = dialect.replace("_", " ").title()
    pairs = detect_column_pairs(df)

    if not pairs:
        print(f"Warning: No valid column pairs found in {csv_path}")
        return df

    evaluator = LLMJudgeEvaluator(
        backend=backend,
        model=model,
        verbose=verbose
    )

    for orig_col, trans_col, prefix in pairs:
        if orig_col not in df.columns or trans_col not in df.columns:
            continue

        col = f"{prefix}llm_fluency"
        if col in df.columns and df[col].notna().any():
            if verbose:
                print(f"Skipping already computed LLM evaluation for {prefix}")
            continue

        originals = df[orig_col].fillna("").astype(str).tolist()
        transformed = df[trans_col].fillna("").astype(str).tolist()

        try:
            results = evaluator.evaluate_batch(originals, transformed, dialect_name, show_progress=verbose)
            scores = evaluator.get_scores_dict(results)

            for key, values in scores.items():
                df[f"{prefix}{key}"] = values

            df.to_csv(csv_path, index=False)
        except Exception as e:
            print(f"LLM evaluation error: {e}")

    return df


def main():
    parser = argparse.ArgumentParser(description="DIA-Guard Dataset Evaluation")

    parser.add_argument(
        "--dataset-dir",
        default=DIA_GUARD_PATH,
        help="Path to DIA-Guard Dataset"
    )

    parser.add_argument(
        "--gpu-only",
        action="store_true",
        help="Run only GPU metrics"
    )

    parser.add_argument(
        "--bertscore-only",
        action="store_true",
        help="Run only BERTScore"
    )

    parser.add_argument(
        "--bartscore-only",
        action="store_true",
        help="Run only BARTScore"
    )

    parser.add_argument(
        "--alignscore-only",
        action="store_true",
        help="Run only AlignScore"
    )

    parser.add_argument(
        "--meteor-only",
        action="store_true",
        help="Run only METEOR"
    )

    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Run only CPU metrics"
    )

    parser.add_argument(
        "--features-only",
        action="store_true",
        help="Run only Feature Accuracy"
    )

    parser.add_argument(
        "--llm-only",
        action="store_true",
        help="Run only LLM-as-a-Judge"
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all metrics (default: GPU + CPU + Features, no LLM)"
    )

    parser.add_argument(
        "--include-llm",
        action="store_true",
        help="Include LLM-as-a-Judge in evaluation"
    )

    parser.add_argument(
        "--llm-backend",
        default="azure",
        choices=["azure", "openai", "anthropic", "ollama"],
        help="LLM backend"
    )

    parser.add_argument(
        "--llm-model",
        help="LLM model name"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Base batch size for GPU metrics"
    )

    parser.add_argument(
        "--dialect-filter",
        nargs="+",
        help="Only process these dialects"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=True,
        help="Print progress"
    )

    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress output"
    )

    args = parser.parse_args()
    verbose = args.verbose and not args.quiet

    # Check for individual metric flags
    single_metric = None
    if args.bertscore_only:
        single_metric = "bertscore"
    elif args.bartscore_only:
        single_metric = "bartscore"
    elif args.alignscore_only:
        single_metric = "alignscore"
    elif args.meteor_only:
        single_metric = "meteor"

    # Determine what to run
    if single_metric:
        run_gpu = False
        run_cpu = False
        run_features = False
        run_llm = False
    elif args.gpu_only:
        run_gpu = True
        run_cpu = False
        run_features = False
        run_llm = False
    elif args.cpu_only:
        run_gpu = False
        run_cpu = True
        run_features = False
        run_llm = False
    elif args.features_only:
        run_gpu = False
        run_cpu = False
        run_features = True
        run_llm = False
    elif args.llm_only:
        run_gpu = False
        run_cpu = False
        run_features = False
        run_llm = True
    else:
        # Default: GPU + CPU + Features
        run_gpu = True
        run_cpu = True
        run_features = True
        run_llm = args.include_llm

    # Find all CSV files
    csv_files = find_all_csv_files(args.dataset_dir)

    if args.dialect_filter:
        csv_files = [f for f in csv_files if f["dialect"] in args.dialect_filter]

    if verbose:
        print(f"\n{'#'*70}")
        print(f"DIA-Guard Dataset Evaluation")
        print(f"{'#'*70}")
        print(f"Dataset: {args.dataset_dir}")
        print(f"CSV files: {len(csv_files)}")
        if single_metric:
            print(f"Running single metric: {single_metric.upper()}")
        else:
            print(f"Run GPU: {run_gpu}")
            print(f"Run CPU: {run_cpu}")
            print(f"Run Features: {run_features}")
            print(f"Run LLM: {run_llm}")
        print(f"{'#'*70}\n")

    # Load progress
    progress = load_progress(args.dataset_dir)

    # Process each file
    for file_info in tqdm(csv_files, desc="Processing files"):
        csv_path = file_info["path"]
        dialect = file_info["dialect"]
        relative = file_info["relative"]

        if verbose:
            print(f"\n>>> Processing: {relative}")

        try:
            # Handle single metric mode
            if single_metric:
                run_single_gpu_metric(csv_path, dialect, single_metric, verbose)
                progress["completed"][f"{relative}_{single_metric}"] = datetime.now().isoformat()
                save_progress(args.dataset_dir, progress)
            else:
                if run_gpu:
                    run_gpu_metrics(csv_path, dialect, args.batch_size, verbose)
                    progress["completed"][f"{relative}_gpu"] = datetime.now().isoformat()
                    save_progress(args.dataset_dir, progress)

                if run_cpu:
                    run_cpu_metrics(csv_path, dialect, verbose)
                    progress["completed"][f"{relative}_cpu"] = datetime.now().isoformat()
                    save_progress(args.dataset_dir, progress)

                if run_features:
                    run_feature_accuracy(csv_path, dialect, verbose)
                    progress["completed"][f"{relative}_features"] = datetime.now().isoformat()
                    save_progress(args.dataset_dir, progress)

                if run_llm:
                    run_llm_evaluation(
                        csv_path, dialect,
                        backend=args.llm_backend,
                        model=args.llm_model,
                        verbose=verbose
                    )
                    progress["completed"][f"{relative}_llm"] = datetime.now().isoformat()
                    save_progress(args.dataset_dir, progress)

        except Exception as e:
            print(f"Error processing {relative}: {e}")
            continue

    if verbose:
        print(f"\n{'#'*70}")
        print(f"Evaluation Complete!")
        print(f"{'#'*70}\n")


if __name__ == "__main__":
    main()
