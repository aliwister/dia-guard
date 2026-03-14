"""
Main evaluation runner for dialect transformation datasets.

Supports:
- Resume capability (continues from where it left off)
- GPU metrics: BERTScore, BARTScore, AlignScore, METEOR
- CPU metrics: ROUGE-L
- LLM-as-a-Judge: fluency, faithfulness, authenticity, coherence, readability
- Feature Accuracy: eWAVE validation

Optimized for T4 16GB GPU with configurable batch sizes.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from tqdm import tqdm
import gc
import json

# Add parent directory for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


# Column mappings for different datasets
COLUMN_MAPPINGS = {
    "dia_harm": {
        "original_cols": ["human_content", "ai_content"],
        "transformed_cols": ["human_content_transformed", "ai_content_transformed"]
    },
    "dia_guard": {
        "original_cols": ["Behavior"],
        "transformed_cols": ["Behavior_transformed"]
    }
}


def detect_dataset_type(df: pd.DataFrame) -> str:
    """Detect dataset type based on columns."""
    if "human_content" in df.columns and "human_content_transformed" in df.columns:
        return "dia_harm"
    elif "Behavior" in df.columns and "Behavior_transformed" in df.columns:
        return "dia_guard"
    else:
        raise ValueError(f"Unknown dataset format. Columns: {df.columns.tolist()}")


def get_text_pairs(
    df: pd.DataFrame,
    dataset_type: str,
    content_type: str = "human"
) -> Tuple[List[str], List[str]]:
    """
    Extract original and transformed text pairs from dataframe.

    Args:
        df: DataFrame
        dataset_type: "dia_harm" or "dia_guard"
        content_type: "human" or "ai" (for dia_harm)

    Returns:
        Tuple of (originals, transformed)
    """
    mapping = COLUMN_MAPPINGS[dataset_type]

    if dataset_type == "dia_harm":
        if content_type == "human":
            orig_col = "human_content"
            trans_col = "human_content_transformed"
        else:
            orig_col = "ai_content"
            trans_col = "ai_content_transformed"
    else:
        orig_col = mapping["original_cols"][0]
        trans_col = mapping["transformed_cols"][0]

    # Handle missing values
    originals = df[orig_col].fillna("").astype(str).tolist()
    transformed = df[trans_col].fillna("").astype(str).tolist()

    return originals, transformed


def check_columns_exist(df: pd.DataFrame, columns: List[str]) -> List[str]:
    """Check which columns already exist (for resume)."""
    return [col for col in columns if col in df.columns and df[col].notna().any()]


def run_gpu_evaluation(
    csv_path: str,
    output_path: Optional[str] = None,
    metrics: List[str] = None,
    batch_size: int = 8,
    content_type: str = "human",
    verbose: bool = True
) -> pd.DataFrame:
    """
    Run GPU-based metrics evaluation.

    Args:
        csv_path: Path to input CSV
        output_path: Path to save output (modifies in-place if None)
        metrics: List of metrics to run (default: all)
        batch_size: Batch size for GPU processing
        content_type: "human" or "ai" content to evaluate
        verbose: Print progress

    Returns:
        DataFrame with added metric columns
    """
    from gpu_metrics import (
        BERTScoreEvaluator,
        BARTScoreEvaluator,
        AlignScoreEvaluator,
        METEORScoreEvaluator
    )

    metrics = metrics or ["BERTScore", "BARTScore", "AlignScore", "METEOR"]
    output_path = output_path or csv_path

    if verbose:
        print(f"\n{'='*60}")
        print(f"GPU Evaluation: {csv_path}")
        print(f"Metrics: {metrics}")
        print(f"Batch size: {batch_size}")
        print(f"{'='*60}\n")

    # Load data
    df = pd.read_csv(csv_path)
    dataset_type = detect_dataset_type(df)

    # Get text pairs
    originals, transformed = get_text_pairs(df, dataset_type, content_type)

    # Define column names based on content type
    prefix = f"{content_type}_" if dataset_type == "dia_harm" else ""

    # Check for existing columns (resume capability)
    new_columns = {
        "BERTScore": f"{prefix}bertscore_f1",
        "BARTScore": f"{prefix}bartscore",
        "AlignScore": f"{prefix}alignscore",
        "METEOR": f"{prefix}meteor"
    }

    existing = check_columns_exist(df, list(new_columns.values()))
    if existing and verbose:
        print(f"Skipping already computed metrics: {existing}")

    # Run each metric
    for metric in metrics:
        col_name = new_columns[metric]
        if col_name in existing:
            continue

        if verbose:
            print(f"\nComputing {metric}...")

        try:
            if metric == "BERTScore":
                evaluator = BERTScoreEvaluator(
                    batch_size=batch_size,
                    verbose=verbose
                )
                _, _, scores = evaluator.score_batch(originals, transformed)
                df[col_name] = scores
                # Also save precision and recall
                df[f"{prefix}bertscore_precision"] = _
                df[f"{prefix}bertscore_recall"] = _

            elif metric == "BARTScore":
                evaluator = BARTScoreEvaluator(
                    batch_size=batch_size,
                    verbose=verbose
                )
                scores = evaluator.score_batch(originals, transformed)
                df[col_name] = scores

            elif metric == "AlignScore":
                evaluator = AlignScoreEvaluator(
                    batch_size=batch_size,
                    verbose=verbose
                )
                scores = evaluator.score_batch(originals, transformed)
                df[col_name] = scores

            elif metric == "METEOR":
                evaluator = METEORScoreEvaluator(verbose=verbose)
                scores = evaluator.score_batch(originals, transformed)
                df[col_name] = scores

            # Cleanup after each metric
            evaluator.cleanup()
            gc.collect()

            # Save progress after each metric
            df.to_csv(output_path, index=False)
            if verbose:
                print(f"Saved progress: {col_name}")

        except Exception as e:
            print(f"Error computing {metric}: {e}")
            df[col_name] = np.nan

    return df


def run_cpu_evaluation(
    csv_path: str,
    output_path: Optional[str] = None,
    content_type: str = "human",
    verbose: bool = True
) -> pd.DataFrame:
    """
    Run CPU-based ROUGE-L and DiffLib evaluation.

    Args:
        csv_path: Path to input CSV
        output_path: Path to save output
        content_type: "human" or "ai" content
        verbose: Print progress

    Returns:
        DataFrame with ROUGE-L and DiffLib columns
    """
    from cpu_metrics import ROUGELEvaluator, DiffLibEvaluator

    output_path = output_path or csv_path

    if verbose:
        print(f"\n{'='*60}")
        print(f"CPU Evaluation (ROUGE-L, DiffLib): {csv_path}")
        print(f"{'='*60}\n")

    # Load data
    df = pd.read_csv(csv_path)
    dataset_type = detect_dataset_type(df)

    # Get text pairs
    originals, transformed = get_text_pairs(df, dataset_type, content_type)

    # Column names
    prefix = f"{content_type}_" if dataset_type == "dia_harm" else ""
    rouge_col = f"{prefix}rouge_l"
    difflib_col = f"{prefix}difflib_ratio"

    # Compute ROUGE-L if not already done
    if rouge_col not in df.columns or not df[rouge_col].notna().any():
        if verbose:
            print("Computing ROUGE-L...")
        evaluator = ROUGELEvaluator(verbose=verbose)
        detailed = evaluator.score_detailed(originals, transformed)

        df[rouge_col] = detailed['f1']
        df[f"{prefix}rouge_l_precision"] = detailed['precision']
        df[f"{prefix}rouge_l_recall"] = detailed['recall']
    else:
        if verbose:
            print(f"Skipping already computed: {rouge_col}")

    # Compute DiffLib if not already done
    if difflib_col not in df.columns or not df[difflib_col].notna().any():
        if verbose:
            print("Computing DiffLib similarity...")
        evaluator = DiffLibEvaluator(verbose=verbose)
        detailed = evaluator.score_detailed(originals, transformed)

        df[difflib_col] = detailed['ratio']
        df[f"{prefix}difflib_quick_ratio"] = detailed['quick_ratio']
    else:
        if verbose:
            print(f"Skipping already computed: {difflib_col}")

    # Save
    df.to_csv(output_path, index=False)
    if verbose:
        print(f"Saved: {output_path}")

    return df


def run_llm_evaluation(
    csv_path: str,
    dialect_name: str,
    output_path: Optional[str] = None,
    backend: str = "azure",
    model: Optional[str] = None,
    content_type: str = "human",
    verbose: bool = True
) -> pd.DataFrame:
    """
    Run LLM-as-a-Judge evaluation.

    Args:
        csv_path: Path to input CSV
        dialect_name: Full dialect name for evaluation
        output_path: Path to save output
        backend: LLM backend to use
        model: Model name/deployment
        content_type: "human" or "ai" content
        verbose: Print progress

    Returns:
        DataFrame with LLM evaluation columns
    """
    from llm_evaluation import LLMJudgeEvaluator, FeatureAccuracyEvaluator, convert_dialect_folder_to_key

    output_path = output_path or csv_path

    if verbose:
        print(f"\n{'='*60}")
        print(f"LLM-as-a-Judge Evaluation: {csv_path}")
        print(f"Dialect: {dialect_name}")
        print(f"Backend: {backend}")
        print(f"{'='*60}\n")

    # Load data
    df = pd.read_csv(csv_path)
    dataset_type = detect_dataset_type(df)

    # Get text pairs
    originals, transformed = get_text_pairs(df, dataset_type, content_type)

    # Column names
    prefix = f"{content_type}_" if dataset_type == "dia_harm" else ""

    llm_columns = [
        f"{prefix}llm_fluency",
        f"{prefix}llm_faithfulness",
        f"{prefix}llm_authenticity",
        f"{prefix}llm_coherence",
        f"{prefix}llm_readability",
        f"{prefix}llm_overall"
    ]

    # Check for existing
    existing = check_columns_exist(df, llm_columns)
    if len(existing) == len(llm_columns):
        if verbose:
            print("LLM evaluation already complete, skipping...")
        return df

    # Initialize evaluator
    evaluator = LLMJudgeEvaluator(
        backend=backend,
        model=model,
        verbose=verbose
    )

    # Find rows that need evaluation
    if existing:
        # Partial resume: find rows with missing values
        needs_eval = df[llm_columns[0]].isna()
    else:
        needs_eval = pd.Series([True] * len(df))

    # Evaluate only missing rows
    results = []
    for i, (orig, trans) in enumerate(tqdm(
        zip(originals, transformed),
        total=len(originals),
        desc="LLM Judge"
    )):
        if not needs_eval.iloc[i]:
            # Keep existing values
            results.append(None)
            continue

        result = evaluator.evaluate(orig, trans, dialect_name)
        results.append(result)

        # Save progress every 10 rows
        if (i + 1) % 10 == 0:
            for j, res in enumerate(results):
                if res is not None:
                    df.loc[j, f"{prefix}llm_fluency"] = res.fluency
                    df.loc[j, f"{prefix}llm_faithfulness"] = res.faithfulness
                    df.loc[j, f"{prefix}llm_authenticity"] = res.dialect_authenticity
                    df.loc[j, f"{prefix}llm_coherence"] = res.coherence
                    df.loc[j, f"{prefix}llm_readability"] = res.readability
                    df.loc[j, f"{prefix}llm_overall"] = res.overall
            df.to_csv(output_path, index=False)

    # Final save
    for i, res in enumerate(results):
        if res is not None:
            df.loc[i, f"{prefix}llm_fluency"] = res.fluency
            df.loc[i, f"{prefix}llm_faithfulness"] = res.faithfulness
            df.loc[i, f"{prefix}llm_authenticity"] = res.dialect_authenticity
            df.loc[i, f"{prefix}llm_coherence"] = res.coherence
            df.loc[i, f"{prefix}llm_readability"] = res.readability
            df.loc[i, f"{prefix}llm_overall"] = res.overall

    df.to_csv(output_path, index=False)
    if verbose:
        print(f"Saved: {output_path}")

    return df


def run_feature_accuracy(
    csv_path: str,
    dialect_key: str,
    output_path: Optional[str] = None,
    content_type: str = "human",
    verbose: bool = True
) -> pd.DataFrame:
    """
    Run eWAVE Feature Accuracy evaluation.

    Args:
        csv_path: Path to input CSV
        dialect_key: Dialect key (e.g., "aboriginal", "urban_aave")
        output_path: Path to save output
        content_type: "human" or "ai" content
        verbose: Print progress

    Returns:
        DataFrame with feature accuracy columns
    """
    from llm_evaluation import FeatureAccuracyEvaluator

    output_path = output_path or csv_path

    if verbose:
        print(f"\n{'='*60}")
        print(f"Feature Accuracy Evaluation: {csv_path}")
        print(f"Dialect key: {dialect_key}")
        print(f"{'='*60}\n")

    # Load data
    df = pd.read_csv(csv_path)
    dataset_type = detect_dataset_type(df)

    # Get text pairs
    originals, transformed = get_text_pairs(df, dataset_type, content_type)

    # Column names
    prefix = f"{content_type}_" if dataset_type == "dia_harm" else ""
    col_name = f"{prefix}feature_accuracy"

    # Check for existing
    if col_name in df.columns and df[col_name].notna().any():
        if verbose:
            print(f"Skipping already computed: {col_name}")
        return df

    # Initialize evaluator
    evaluator = FeatureAccuracyEvaluator(verbose=verbose)

    # Evaluate
    results = evaluator.evaluate_batch(originals, transformed, dialect_key)
    scores = evaluator.get_scores_dict(results)

    # Add columns
    for key, values in scores.items():
        df[f"{prefix}{key}"] = values

    # Save
    df.to_csv(output_path, index=False)
    if verbose:
        print(f"Saved: {output_path}")

    return df


def run_full_evaluation(
    csv_path: str,
    dialect_folder: str,
    output_path: Optional[str] = None,
    gpu_batch_size: int = 8,
    run_gpu: bool = True,
    run_cpu: bool = True,
    run_llm: bool = True,
    run_features: bool = True,
    llm_backend: str = "azure",
    llm_model: Optional[str] = None,
    content_types: List[str] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Run complete evaluation pipeline.

    Args:
        csv_path: Path to input CSV
        dialect_folder: Dialect folder name (e.g., "aboriginal_english")
        output_path: Path to save output
        gpu_batch_size: Batch size for GPU metrics
        run_gpu: Run GPU metrics
        run_cpu: Run CPU metrics
        run_llm: Run LLM-as-a-Judge
        run_features: Run Feature Accuracy
        llm_backend: LLM backend for judge
        llm_model: LLM model name
        content_types: Content types to evaluate (for DIA-HARM)
        verbose: Print progress

    Returns:
        DataFrame with all evaluation columns
    """
    from llm_evaluation import convert_dialect_folder_to_key

    output_path = output_path or csv_path

    # Detect dataset type
    df = pd.read_csv(csv_path)
    dataset_type = detect_dataset_type(df)

    # Determine content types
    if content_types is None:
        if dataset_type == "dia_harm":
            content_types = ["human", "ai"]
        else:
            content_types = ["human"]  # Dummy for dia_guard

    # Convert folder name to dialect key
    dialect_key = convert_dialect_folder_to_key(dialect_folder)

    # Human-readable dialect name
    dialect_name = dialect_folder.replace("_", " ").title()

    if verbose:
        print(f"\n{'='*70}")
        print(f"FULL EVALUATION PIPELINE")
        print(f"{'='*70}")
        print(f"Input: {csv_path}")
        print(f"Output: {output_path}")
        print(f"Dataset type: {dataset_type}")
        print(f"Dialect: {dialect_name} (key: {dialect_key})")
        print(f"Content types: {content_types}")
        print(f"{'='*70}\n")

    for content_type in content_types:
        if verbose:
            print(f"\n>>> Evaluating {content_type} content...")

        # GPU metrics
        if run_gpu:
            df = run_gpu_evaluation(
                csv_path=output_path if os.path.exists(output_path) else csv_path,
                output_path=output_path,
                batch_size=gpu_batch_size,
                content_type=content_type,
                verbose=verbose
            )

        # CPU metrics
        if run_cpu:
            df = run_cpu_evaluation(
                csv_path=output_path,
                output_path=output_path,
                content_type=content_type,
                verbose=verbose
            )

        # Feature Accuracy
        if run_features:
            df = run_feature_accuracy(
                csv_path=output_path,
                dialect_key=dialect_key,
                output_path=output_path,
                content_type=content_type,
                verbose=verbose
            )

        # LLM-as-a-Judge (last, most expensive)
        if run_llm:
            df = run_llm_evaluation(
                csv_path=output_path,
                dialect_name=dialect_name,
                output_path=output_path,
                backend=llm_backend,
                model=llm_model,
                content_type=content_type,
                verbose=verbose
            )

    if verbose:
        print(f"\n{'='*70}")
        print(f"EVALUATION COMPLETE")
        print(f"Output saved to: {output_path}")
        print(f"{'='*70}\n")

    return df


def process_dataset_folder(
    dataset_dir: str,
    output_dir: Optional[str] = None,
    gpu_batch_size: int = 8,
    run_gpu: bool = True,
    run_cpu: bool = True,
    run_llm: bool = False,  # Off by default (expensive)
    run_features: bool = True,
    llm_backend: str = "azure",
    llm_model: Optional[str] = None,
    dialect_filter: Optional[List[str]] = None,
    verbose: bool = True
):
    """
    Process all CSV files in a dataset directory.

    Args:
        dataset_dir: Path to dataset directory (e.g., DIA-HARM Dataset)
        output_dir: Output directory (uses input dir if None)
        gpu_batch_size: Batch size for GPU metrics
        run_gpu: Run GPU metrics
        run_cpu: Run CPU metrics
        run_llm: Run LLM-as-a-Judge
        run_features: Run Feature Accuracy
        llm_backend: LLM backend
        llm_model: LLM model
        dialect_filter: List of dialects to process (None = all)
        verbose: Print progress
    """
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir) if output_dir else dataset_dir

    # Find all dialect folders
    dialect_folders = [
        d for d in dataset_dir.iterdir()
        if d.is_dir() and not d.name.startswith('.')
    ]

    if dialect_filter:
        dialect_folders = [d for d in dialect_folders if d.name in dialect_filter]

    if verbose:
        print(f"Found {len(dialect_folders)} dialect folders to process")

    for dialect_folder in tqdm(dialect_folders, desc="Dialects"):
        if verbose:
            print(f"\n\n{'#'*70}")
            print(f"Processing: {dialect_folder.name}")
            print(f"{'#'*70}")

        # Find all CSV files in dialect folder (including subfolders)
        csv_files = list(dialect_folder.rglob("*.csv"))

        for csv_file in csv_files:
            # Create output path maintaining folder structure
            relative_path = csv_file.relative_to(dataset_dir)
            output_path = output_dir / relative_path

            # Create output directory if needed
            output_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                run_full_evaluation(
                    csv_path=str(csv_file),
                    dialect_folder=dialect_folder.name,
                    output_path=str(output_path),
                    gpu_batch_size=gpu_batch_size,
                    run_gpu=run_gpu,
                    run_cpu=run_cpu,
                    run_llm=run_llm,
                    run_features=run_features,
                    llm_backend=llm_backend,
                    llm_model=llm_model,
                    verbose=verbose
                )
            except Exception as e:
                print(f"Error processing {csv_file}: {e}")
                continue


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate dialect transformations with multiple metrics"
    )

    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input CSV file or dataset directory"
    )

    parser.add_argument(
        "--output", "-o",
        help="Output path (default: modify in place)"
    )

    parser.add_argument(
        "--dialect", "-d",
        help="Dialect folder name (required for single file)"
    )

    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=8,
        help="Batch size for GPU metrics (default: 8 for T4 16GB)"
    )

    parser.add_argument(
        "--gpu-metrics",
        action="store_true",
        default=True,
        help="Run GPU metrics (BERTScore, BARTScore, AlignScore, METEOR)"
    )

    parser.add_argument(
        "--no-gpu-metrics",
        action="store_true",
        help="Skip GPU metrics"
    )

    parser.add_argument(
        "--cpu-metrics",
        action="store_true",
        default=True,
        help="Run CPU metrics (ROUGE-L)"
    )

    parser.add_argument(
        "--no-cpu-metrics",
        action="store_true",
        help="Skip CPU metrics"
    )

    parser.add_argument(
        "--llm-judge",
        action="store_true",
        help="Run LLM-as-a-Judge evaluation"
    )

    parser.add_argument(
        "--feature-accuracy",
        action="store_true",
        default=True,
        help="Run Feature Accuracy evaluation"
    )

    parser.add_argument(
        "--no-feature-accuracy",
        action="store_true",
        help="Skip Feature Accuracy"
    )

    parser.add_argument(
        "--llm-backend",
        default="azure",
        choices=["azure", "openai", "anthropic", "ollama"],
        help="LLM backend for judge"
    )

    parser.add_argument(
        "--llm-model",
        help="LLM model name/deployment"
    )

    parser.add_argument(
        "--content-type",
        default="human",
        choices=["human", "ai", "both"],
        help="Content type to evaluate (for DIA-HARM)"
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

    # Determine what to run
    run_gpu = args.gpu_metrics and not args.no_gpu_metrics
    run_cpu = args.cpu_metrics and not args.no_cpu_metrics
    run_llm = args.llm_judge
    run_features = args.feature_accuracy and not args.no_feature_accuracy
    verbose = args.verbose and not args.quiet

    # Content types
    if args.content_type == "both":
        content_types = ["human", "ai"]
    else:
        content_types = [args.content_type]

    input_path = Path(args.input)

    if input_path.is_file():
        # Single file
        if not args.dialect:
            # Try to infer from path
            args.dialect = input_path.parent.name

        run_full_evaluation(
            csv_path=str(input_path),
            dialect_folder=args.dialect,
            output_path=args.output,
            gpu_batch_size=args.batch_size,
            run_gpu=run_gpu,
            run_cpu=run_cpu,
            run_llm=run_llm,
            run_features=run_features,
            llm_backend=args.llm_backend,
            llm_model=args.llm_model,
            content_types=content_types,
            verbose=verbose
        )
    else:
        # Directory
        process_dataset_folder(
            dataset_dir=str(input_path),
            output_dir=args.output,
            gpu_batch_size=args.batch_size,
            run_gpu=run_gpu,
            run_cpu=run_cpu,
            run_llm=run_llm,
            run_features=run_features,
            llm_backend=args.llm_backend,
            llm_model=args.llm_model,
            verbose=verbose
        )


if __name__ == "__main__":
    main()
