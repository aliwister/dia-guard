#!/usr/bin/env python3
"""
Main pipeline: Convert safety/security benchmark datasets into 50 English dialect
varieties using the Multi-Value framework.

Usage:
    python dialect_transform_pipeline.py --output-dir ./output
    python dialect_transform_pipeline.py --output-dir ./output --resume
    python dialect_transform_pipeline.py --datasets securityeval --dialects AfricanAmericanVernacular
"""

import argparse
import json
import logging
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from dataset_configs import DATASET_CONFIGS, PROCESSING_ORDER
from dialect_utils import (
    BROKEN_DIALECTS,
    DialectWorker,
    discover_dialects,
    setup_nlp_resources,
)
from progress_tracker import ProgressTracker

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("pipeline.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def to_serializable(value):
    """Convert numpy/pandas types to JSON-serializable Python types."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (list, tuple)):
        return [to_serializable(v) for v in value]
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------
def load_dataset_by_config(ds_name, config, hf_token=None):
    """Load a dataset based on its configuration. Returns a pandas DataFrame."""
    loader = config["loader"]

    if loader == "huggingface":
        from datasets import load_dataset

        hf_path = config["hf_path"]
        hf_configs = config.get("hf_config")
        hf_split = config.get("hf_split")

        if isinstance(hf_configs, list):
            # Multiple configs (e.g. CyberSecEval: autocomplete + instruct)
            frames = []
            for cfg in hf_configs:
                try:
                    ds = load_dataset(hf_path, cfg, token=hf_token)
                except Exception:
                    ds = load_dataset(hf_path, cfg)
                for split_name in ds:
                    df = ds[split_name].to_pandas()
                    df["_config"] = cfg
                    df["_split"] = split_name
                    frames.append(df)
            return pd.concat(frames, ignore_index=True)
        else:
            try:
                ds = load_dataset(hf_path, hf_configs, token=hf_token)
            except Exception:
                ds = load_dataset(hf_path, hf_configs)
            if hf_split and hf_split != "all" and hf_split in ds:
                return ds[hf_split].to_pandas()
            else:
                # Merge all splits
                frames = []
                for split_name in ds:
                    df = ds[split_name].to_pandas()
                    df["_split"] = split_name
                    frames.append(df)
                return pd.concat(frames, ignore_index=True)

    elif loader == "github_json":
        frames = []
        for file_key, url in config["github_files"].items():
            logger.info(f"  Downloading {file_key} from GitHub...")
            resp = requests.get(url, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            df = pd.DataFrame(data)
            df["_source_file"] = file_key
            frames.append(df)
        return pd.concat(frames, ignore_index=True)

    elif loader == "github_csv":
        url = config["github_url"]
        logger.info(f"  Downloading CSV from GitHub...")
        df = pd.read_csv(url)
        return df

    else:
        raise ValueError(f"Unknown loader: {loader}")


def maybe_sample(df, config, seed=42):
    """Apply stratified sampling if configured."""
    sample_size = config.get("sample_size")
    if not sample_size or len(df) <= sample_size:
        return df

    stratify_col = config.get("sample_stratify_col")
    if stratify_col and stratify_col in df.columns:
        # Stratified sample
        sampled = df.groupby(stratify_col, group_keys=False).apply(
            lambda x: x.sample(
                n=min(len(x), int(sample_size * len(x) / len(df))),
                random_state=seed,
            )
        )
        # Top up if rounding left us short
        if len(sampled) < sample_size:
            remaining = df.drop(sampled.index)
            extra = remaining.sample(n=sample_size - len(sampled), random_state=seed)
            sampled = pd.concat([sampled, extra])
        return sampled.reset_index(drop=True)
    else:
        return df.sample(n=sample_size, random_state=seed).reset_index(drop=True)


def detect_text_columns(df, ds_name, config):
    """Auto-detect text columns for datasets like LLMSecEval."""
    text_cols = config["text_columns"]
    if text_cols:
        # Verify they exist
        missing = [c for c in text_cols if c not in df.columns]
        if missing:
            logger.warning(f"Columns {missing} not found in {ds_name}. Available: {list(df.columns)}")
            text_cols = [c for c in text_cols if c in df.columns]
        return text_cols

    # Auto-detect for LLMSecEval and similar
    candidates = [c for c in df.columns if "prompt" in c.lower() or "nl" in c.lower()]
    if candidates:
        logger.info(f"  Auto-detected text columns: {candidates}")
        return candidates

    # Fallback: first string column
    for c in df.columns:
        if df[c].dtype == object:
            logger.info(f"  Fallback text column: {c}")
            return [c]

    return []


# ---------------------------------------------------------------------------
# Transformation
# ---------------------------------------------------------------------------
def transform_dataset_for_dialect(
    df, worker, dialect_name, ds_name, config, tracker, timeout_sec=30
):
    """Transform all text columns in a DataFrame for a single dialect.

    Args:
        worker: DialectWorker instance (subprocess-based, killable on hang)
    """
    text_cols = detect_text_columns(df, ds_name, config)
    if not text_cols:
        logger.error(f"No text columns found for {ds_name}")
        return [], 0

    col_types = config.get("text_column_types", {})
    contains_code = config.get("contains_code_mixed", False)
    done_rows = tracker.get_partial_rows(ds_name, dialect_name)

    records = []
    errors = 0
    consecutive_timeouts = 0
    MAX_CONSECUTIVE_TIMEOUTS = 20

    for idx in tqdm(range(len(df)), desc=f"  {dialect_name}", leave=False):
        if idx in done_rows:
            continue

        # If worker died (was killed due to hang), skip remaining rows
        if not worker.alive:
            logger.warning(f"  {dialect_name}: worker dead — skipping remaining rows")
            break

        row = df.iloc[idx]
        record = {"original_index": idx}

        # Copy all non-text columns
        for col in df.columns:
            if col not in text_cols:
                record[col] = to_serializable(row[col])

        # Transform each text column
        row_ok = True
        for col in text_cols:
            original = row[col]
            record[f"{col}_original"] = to_serializable(original)

            col_type = col_types.get(col, "string")

            try:
                if col_type == "list_of_strings" and isinstance(original, (list, tuple)):
                    transformed_list = []
                    for item in original:
                        if item and str(item).strip():
                            t, ok = worker.transform(
                                str(item),
                                contains_code=contains_code,
                                timeout_sec=timeout_sec,
                            )
                            transformed_list.append(t)
                            if not ok:
                                row_ok = False
                        else:
                            transformed_list.append(item)
                    record[f"{col}_dialect"] = transformed_list
                else:
                    text_str = str(original) if original is not None else ""
                    if text_str.strip():
                        t, ok = worker.transform(
                            text_str,
                            contains_code=contains_code,
                            timeout_sec=timeout_sec,
                        )
                        record[f"{col}_dialect"] = t
                        if not ok:
                            row_ok = False
                    else:
                        record[f"{col}_dialect"] = text_str
            except Exception as e:
                logger.warning(f"Error at {ds_name}[{idx}].{col}/{dialect_name}: {e}")
                record[f"{col}_dialect"] = to_serializable(original)
                row_ok = False

        record["transform_success"] = row_ok
        if not row_ok:
            errors += 1
            consecutive_timeouts += 1
            if consecutive_timeouts >= MAX_CONSECUTIVE_TIMEOUTS:
                logger.warning(
                    f"  {dialect_name}: {MAX_CONSECUTIVE_TIMEOUTS} consecutive "
                    f"failures — skipping remaining rows"
                )
                break
        else:
            consecutive_timeouts = 0
        records.append(record)
        tracker.mark_row_done(ds_name, dialect_name, idx)

    tracker.flush()
    return records, errors


def save_output(records, ds_name, dialect_name, dialect_idx, config, output_root, errors=0):
    """Save transformed records to a JSON file."""
    dir_name = f"{dialect_idx:02d}_{dialect_name}"
    out_dir = output_root / ds_name / dir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    text_cols = config.get("text_columns", [])
    out_file = out_dir / f"{ds_name}.json"

    output = {
        "metadata": {
            "source_dataset": ds_name,
            "dialect_name": dialect_name,
            "num_records": len(records),
            "num_transform_errors": errors,
            "transformed_columns": text_cols,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "data": records,
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    logger.info(f"  Saved {len(records)} records -> {out_file}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Transform safety datasets into 50 English dialects")
    p.add_argument("--output-dir", type=str, default="./output")
    p.add_argument("--datasets", nargs="*", default=None,
                   help="Specific dataset keys to process (default: all)")
    p.add_argument("--dialects", nargs="*", default=None,
                   help="Specific dialect class names (default: all 50)")
    p.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    p.add_argument("--hf-token", type=str, default=None,
                   help="HuggingFace API token for gated datasets")
    p.add_argument("--timeout", type=int, default=30,
                   help="Per-transform timeout in seconds")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Multi-Value Dialect Transform Pipeline")
    logger.info("=" * 60)

    # Step 1: Setup NLP resources
    logger.info("Setting up NLP resources...")
    setup_nlp_resources()

    # Step 2: Discover dialects
    logger.info("Discovering dialects...")
    all_dialects = discover_dialects()
    dialect_names = sorted(all_dialects.keys())
    logger.info(f"Found {len(dialect_names)} dialects")

    if args.dialects:
        dialect_names = [d for d in dialect_names if d in args.dialects]
        logger.info(f"Filtered to {len(dialect_names)} dialects: {dialect_names}")

    # Step 3: Determine datasets
    dataset_keys = [k for k in PROCESSING_ORDER if k in DATASET_CONFIGS]
    if args.datasets:
        dataset_keys = [k for k in dataset_keys if k in args.datasets]
    logger.info(f"Will process {len(dataset_keys)} datasets: {dataset_keys}")

    # Step 4: Init progress tracker
    tracker = ProgressTracker(output_root / "progress.db", resume=args.resume)

    total = len(dataset_keys) * len(dialect_names)
    logger.info(f"Total combinations: {total}")

    # Step 5: Main loop
    for ds_name in dataset_keys:
        config = DATASET_CONFIGS[ds_name]
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"DATASET: {ds_name} (est. {config.get('estimated_size', '?')} rows)")
        logger.info("=" * 60)

        # Load dataset
        try:
            df = load_dataset_by_config(ds_name, config, hf_token=args.hf_token)
            logger.info(f"Loaded {len(df)} rows, columns: {list(df.columns)}")
        except Exception as e:
            logger.error(f"Failed to load {ds_name}: {e}")
            logger.error(traceback.format_exc())
            continue

        # Auto-detect and update text columns
        text_cols = detect_text_columns(df, ds_name, config)
        config["text_columns"] = text_cols
        logger.info(f"Text columns to transform: {text_cols}")

        # Apply sampling if needed
        df = maybe_sample(df, config, seed=args.seed)
        logger.info(f"Processing {len(df)} rows")

        # Each dialect gets a fresh subprocess worker (killable on hang)
        for dial_idx, dial_name in enumerate(dialect_names, start=1):
            if dial_name in BROKEN_DIALECTS:
                logger.info(f"  [{dial_idx}/{len(dialect_names)}] {dial_name} - SKIP (known broken)")
                continue

            if tracker.is_combination_done(ds_name, dial_name):
                logger.info(f"  [{dial_idx}/{len(dialect_names)}] {dial_name} - SKIP (done)")
                continue

            logger.info(f"  [{dial_idx}/{len(dialect_names)}] {dial_name}")

            # Spawn a subprocess worker for this dialect
            worker = DialectWorker(dial_name, seed=args.seed, init_timeout=120)
            if not worker.alive:
                logger.error(f"  Failed to start worker for {dial_name}")
                worker.close()
                continue

            try:
                records, errors = transform_dataset_for_dialect(
                    df, worker, dial_name, ds_name,
                    config, tracker, timeout_sec=args.timeout,
                )
                save_output(records, ds_name, dial_name, dial_idx, config, output_root, errors)
                tracker.mark_combination_done(ds_name, dial_name, len(records), errors)
            except Exception as e:
                logger.error(f"  Error processing {ds_name}/{dial_name}: {e}")
                logger.error(traceback.format_exc())
            finally:
                worker.close()

    logger.info("")
    logger.info("=" * 60)
    logger.info("Pipeline complete!")
    tracker.print_summary()
    tracker.close()


if __name__ == "__main__":
    main()
