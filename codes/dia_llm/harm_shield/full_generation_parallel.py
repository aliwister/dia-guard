#!/usr/bin/env python3
"""
Parallel generation script for DIA_LLM dataset.

Processes multiple rows concurrently to maximize API utilization.
Target: ~100 RPM with 8 calls/row = ~12 rows/minute with 6 concurrent workers.
"""

import sys
import os
import csv
import time
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import AzureOpenAIBackend, BedrockBackend
from transformer import DialectTransformer
from coi_transformation import CoIDialectTransformer
from feature_validator import LLMComprehensiveValidator

# Dataset folder names to eWAVE dialect keys mapping
DIALECT_MAPPING = {
    # North America
    "appalachian_english": "appalachian",
    "chicano_english": "chicano",
    "colloquial_american_english": "colloquial_american",
    "earlier_african_american_vernacular_english": "earlier_aave",
    "newfoundland_english": "newfoundland",
    "ozark_english": "ozark",
    "rural_african_american_vernacular_english": "rural_aave",
    "southeast_american_enclave_dialects": "southeast_enclave",
    "urban_african_american_vernacular_english": "urban_aave",

    # British Isles
    "channel_islands_english": "channel_islands",
    "east_anglian_english": "east_anglian",
    "english_dialects_in_the_north_of_england": "northern_england",
    "english_dialects_in_the_southeast_of_england": "southeast_england",
    "english_dialects_in_the_southwest_of_england": "southwest_england",
    "irish_english": "irish",
    "maltese_english": "maltese_english",
    "manx_english": "manx",
    "orkney_and_shetland_english": "orkney_shetland",
    "scottish_english": "scottish",
    "welsh_english": "welsh",

    # Caribbean
    "bahamian_english": "bahamian",
    "jamaican_english": "jamaican",

    # Africa
    "black_south_african_english": "south_african_black",
    "cameroon_english": "cameroon",
    "cape_flats_english": "cape_flats",
    "ghanaian_english": "ghanaian",
    "indian_south_african_english": "south_african_indian",
    "kenyan_english": "kenyan",
    "liberian_settler_english": "liberian_settler",
    "nigerian_english": "nigerian",
    "tanzanian_english": "tanzanian",
    "ugandan_english": "ugandan",
    "white_south_african_english": "south_african_white",

    # South/Southeast Asia
    "colloquial_singapore_english_singlish": "singapore",
    "hong_kong_english": "hong_kong",
    "indian_english": "indian",
    "malaysian_english": "malaysian",
    "pakistani_english": "pakistani",
    "philippine_english": "philippine",
    "sri_lankan_english": "sri_lankan",

    # Australia & Pacific
    "aboriginal_english": "aboriginal",
    "acrolectal_fiji_english": "fiji_acrolectal",
    "australian_english": "australian",
    "australian_vernacular_english": "australian_vernacular",
    "new_zealand_english": "new_zealand",
    "pure_fiji_english_basilectal_fijie": "fiji_basilectal",

    # Atlantic Islands
    "falkland_islands_english": "falkland",
    "st_helena_english": "st_helena",
    "tristan_da_cunha_english": "tristan",
}

POLICY_VIOLATION = "guardrail policy violation"

# New columns to add
NEW_COLUMNS = [
    "basic_transform",
    "coi_transform",
    "mv_ewave_accuracy",
    "mv_errors",
    "mv_valid_features",
    "basic_ewave_accuracy",
    "basic_errors",
    "basic_valid_features",
    "coi_ewave_accuracy",
    "coi_errors",
    "coi_valid_features",
    "generation_model",
]


MODEL_CONFIGS = {
    "gpt4.1": {
        "deployment_name": "gpt-4.1",
        "base_url": "https://jsl-diaguard.openai.azure.com/openai/v1",
        "api_key": os.getenv("AZURE_OPENAI_API_KEY", ""),
    },
    "deepseek": {
        "deployment_name": "DeepSeek-V3.2",
        "base_url": "https://ai-jsl57102192ai462716477695.services.ai.azure.com/openai/v1/",
        "api_key": os.getenv("AZURE_AI_API_KEY", ""),
    },
    "kimi": {
        "deployment_name": "Kimi-K2.5",
        "base_url": "https://ai-jsl57102192ai462716477695.services.ai.azure.com/openai/v1/",
        "api_key": os.getenv("AZURE_AI_API_KEY", ""),
    },
    "gemini": {
        "deployment_name": "gemini-2.5-flash-lite",
        "project_id": "diaguard-new-project",
        "location": "us-central1",
    },
    "gemini2.5": {
        "deployment_name": "gemini-2.5-flash-lite",
        "project_id": "diaguard-new-project",
        "location": "us-central1",
    },
    "gemini2.5flash": {
        "deployment_name": "gemini-2.5-flash",
        "project_id": "diaguard-new-project",
        "location": "us-central1",
    },
    # ── AWS Bedrock models ──
    "bedrock-deepseek": {
        "model_id": "deepseek.v3.2",
        "region": "us-east-1",
    },
    "bedrock-llama3-8b": {
        "model_id": "meta.llama3-8b-instruct-v1:0",
        "region": "us-east-1",
    },
    "bedrock-llama4-maverick": {
        "model_id": "meta.llama4-maverick-17b-instruct-v1:0",
        "region": "us-east-1",
    },
    "bedrock-llama4-scout": {
        "model_id": "meta.llama4-scout-17b-instruct-v1:0",
        "region": "us-east-1",
    },
    "bedrock-mistral-large3": {
        "model_id": "mistral.mistral-large-3-675b-instruct",
        "region": "us-east-1",
    },
    "bedrock-safeguard-120b": {
        "model_id": "openai.gpt-oss-safeguard-120b",
        "region": "us-east-1",
    },
    "bedrock-safeguard-20b": {
        "model_id": "openai.gpt-oss-safeguard-20b",
        "region": "us-east-1",
    },
    "bedrock-gpt-oss-120b": {
        "model_id": "openai.gpt-oss-120b-1:0",
        "region": "us-east-1",
    },
    "bedrock-gpt-oss-20b": {
        "model_id": "openai.gpt-oss-20b-1:0",
        "region": "us-east-1",
    },
    "bedrock-qwen3-32b": {
        "model_id": "qwen.qwen3-32b-v1:0",
        "region": "us-east-1",
    },
}


def create_llm(model_key: str = "gpt4.1"):
    """Create LLM backend for the specified model."""
    if model_key not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_key}. Available: {list(MODEL_CONFIGS.keys())}")
    config = MODEL_CONFIGS[model_key]
    if model_key in ("gemini", "gemini2.5", "gemini2.5flash"):
        from models import GeminiBackend
        return GeminiBackend(
            model=config["deployment_name"],
            use_vertex=True,
            project_id=config["project_id"],
            location=config["location"],
        )
    if model_key.startswith("bedrock-"):
        return BedrockBackend(
            model_id=config["model_id"],
            region=config["region"],
        )
    return AzureOpenAIBackend(
        deployment_name=config["deployment_name"],
        base_url=config["base_url"],
        api_key=config["api_key"],
    )


def is_policy_violation(error_msg: str) -> bool:
    """Check if error is a content policy violation."""
    return "content_filter" in error_msg or "content management policy" in error_msg


def row_already_processed(row: Dict[str, Any]) -> bool:
    """Check if row has already been processed (has both transforms).
    Rows with 'guardrail policy violation' are considered NOT processed
    so they get retried with research framing.
    """
    basic = row.get("basic_transform", "")
    coi = row.get("coi_transform", "")
    # Retry rows where either transform is empty or a policy violation
    if not basic or not coi:
        return False
    if basic == POLICY_VIOLATION or coi == POLICY_VIOLATION:
        return False
    return True


def process_row(
    row: Dict[str, Any],
    row_idx: int,
    dialect_key: str,
    llm,
    basic_transformer: DialectTransformer,
    coi_transformer: CoIDialectTransformer,
    validator: LLMComprehensiveValidator
) -> Dict[str, Any]:
    """Process a single row, adding transforms and validation metrics."""
    original = row.get("original_input", "")
    multivalue = row.get("transformed_input", "")

    if not original:
        return row

    result = dict(row)

    # Check existing values -- only regenerate if empty or policy violation
    existing_basic = row.get("basic_transform", "").strip()
    existing_coi = row.get("coi_transform", "").strip()

    # Generate Basic Transform (skip if already has valid result)
    if existing_basic and existing_basic != POLICY_VIOLATION:
        basic_output = existing_basic
    else:
        basic_output = POLICY_VIOLATION
        try:
            basic_result = basic_transformer.transform(original, dialect_key)
            basic_output = basic_result.transformed
        except Exception as e:
            error_msg = str(e)
            if is_policy_violation(error_msg):
                basic_output = POLICY_VIOLATION
            else:
                basic_output = original  # Fallback to original

    # Generate CoI Transform (skip if already has valid result)
    if existing_coi and existing_coi != POLICY_VIOLATION:
        coi_output = existing_coi
    else:
        coi_output = POLICY_VIOLATION
        try:
            coi_result = coi_transformer.transform(original, dialect_key, skip_validation=True)
            coi_output = coi_result.final_output
        except Exception as e:
            error_msg = str(e)
            if is_policy_violation(error_msg):
                coi_output = POLICY_VIOLATION
            else:
                coi_output = original  # Fallback to original

    result["basic_transform"] = basic_output
    result["coi_transform"] = coi_output
    result["generation_model"] = llm.name

    # Validate Multi-VALUE (if available and not policy violation)
    if multivalue and multivalue != POLICY_VIOLATION:
        try:
            mv_validation = validator.validate(original, multivalue)
            result["mv_ewave_accuracy"] = f"{mv_validation.ewave_accuracy:.4f}"
            result["mv_errors"] = len(mv_validation.non_ewave_errors)
            result["mv_valid_features"] = len(mv_validation.valid_changes)
        except Exception:
            result["mv_ewave_accuracy"] = ""
            result["mv_errors"] = ""
            result["mv_valid_features"] = ""
    else:
        result["mv_ewave_accuracy"] = ""
        result["mv_errors"] = ""
        result["mv_valid_features"] = ""

    # Validate Basic Transform
    if basic_output and basic_output != POLICY_VIOLATION:
        try:
            basic_validation = validator.validate(original, basic_output)
            result["basic_ewave_accuracy"] = f"{basic_validation.ewave_accuracy:.4f}"
            result["basic_errors"] = len(basic_validation.non_ewave_errors)
            result["basic_valid_features"] = len(basic_validation.valid_changes)
        except Exception:
            result["basic_ewave_accuracy"] = ""
            result["basic_errors"] = ""
            result["basic_valid_features"] = ""
    else:
        result["basic_ewave_accuracy"] = ""
        result["basic_errors"] = ""
        result["basic_valid_features"] = ""

    # Validate CoI Transform
    if coi_output and coi_output != POLICY_VIOLATION:
        try:
            coi_validation = validator.validate(original, coi_output)
            result["coi_ewave_accuracy"] = f"{coi_validation.ewave_accuracy:.4f}"
            result["coi_errors"] = len(coi_validation.non_ewave_errors)
            result["coi_valid_features"] = len(coi_validation.valid_changes)
        except Exception:
            result["coi_ewave_accuracy"] = ""
            result["coi_errors"] = ""
            result["coi_valid_features"] = ""
    else:
        result["coi_ewave_accuracy"] = ""
        result["coi_errors"] = ""
        result["coi_valid_features"] = ""

    return result


class ParallelProcessor:
    """Process rows in parallel with thread-safe saving."""

    def __init__(self, output_path: str, fieldnames: list, max_workers: int = 6):
        self.output_path = output_path
        self.fieldnames = fieldnames
        self.max_workers = max_workers
        self.results = {}  # row_idx -> result
        self.lock = threading.Lock()
        self.processed_count = 0
        self.start_time = time.time()

    def save_results(self, all_rows: list):
        """Save all results in order."""
        with self.lock:
            output_rows = []
            for i, row in enumerate(all_rows):
                if i in self.results:
                    output_rows.append(self.results[i])
                else:
                    # Row not processed yet, add original with empty new columns
                    row_copy = dict(row)
                    for col in NEW_COLUMNS:
                        if col not in row_copy:
                            row_copy[col] = ""
                    output_rows.append(row_copy)

            with open(self.output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()
                writer.writerows(output_rows)

    def process_batch(
        self,
        rows_to_process: List[tuple],  # List of (idx, row)
        all_rows: list,
        dialect_key: str,
        llm,
        rating_level: str = "AB"
    ):
        """Process a batch of rows in parallel."""
        # Create transformers for each thread (thread-local)
        def worker(idx_row_tuple):
            idx, row = idx_row_tuple

            # Create thread-local transformers and validator
            basic_transformer = DialectTransformer(
                llm=llm,
                rating_level=rating_level,
                use_ewave=True,
                verbose=False
            )
            coi_transformer = CoIDialectTransformer(
                llm=llm,
                rating_level=rating_level,
                verbose=False
            )
            validator = LLMComprehensiveValidator(
                llm, dialect_key, verbose=False,
                use_explicit_ratings=True,
                use_post_correction=True
            )

            try:
                result = process_row(
                    row, idx, dialect_key, llm,
                    basic_transformer, coi_transformer, validator
                )
                return (idx, result, None)
            except Exception as e:
                # Add row with empty new columns on error
                row_copy = dict(row)
                for col in NEW_COLUMNS:
                    row_copy[col] = ""
                return (idx, row_copy, str(e))

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(worker, item): item for item in rows_to_process}

            for future in as_completed(futures):
                idx, result, error = future.result()

                with self.lock:
                    self.results[idx] = result
                    self.processed_count += 1

                    # Calculate rate
                    elapsed = time.time() - self.start_time
                    rate = self.processed_count / (elapsed / 60) if elapsed > 0 else 0

                    sample_id = result.get("sample_id", "N/A")[:15]
                    if error:
                        print(f"    [{self.processed_count}] idx={idx} sample={sample_id} ERROR: {error[:30]}")
                    else:
                        print(f"    [{self.processed_count}] idx={idx} sample={sample_id} ({rate:.1f} rows/min)")

                # Save after each completion
                self.save_results(all_rows)


def process_csv(
    input_path: str,
    output_path: str,
    dialect_key: str,
    llm,
    rating_level: str = "AB",
    max_workers: int = 6,
    test_mode: bool = False,
    test_rows: int = 1
):
    """Process a single CSV file with parallel workers."""
    # Load existing output if resuming
    existing_rows = {}
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row.get("sample_id", ""), row.get("original_input", ""))
                existing_rows[key] = row

    # Load input CSV
    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        input_fieldnames = reader.fieldnames
        input_rows = list(reader)

    # Determine output fieldnames
    output_fieldnames = list(input_fieldnames)
    for col in NEW_COLUMNS:
        if col not in output_fieldnames:
            output_fieldnames.append(col)

    # Determine rows to process
    # In test mode, only process up to test_rows NEW rows, but always save ALL rows
    rows_to_process = []
    skipped_count = 0
    process_limit = test_rows if test_mode else len(input_rows)

    for i, row in enumerate(input_rows):
        key = (row.get("sample_id", ""), row.get("original_input", ""))

        # Check if already processed
        if key in existing_rows and row_already_processed(existing_rows[key]):
            skipped_count += 1
            continue

        # In test mode, limit how many new rows we process
        if test_mode and len(rows_to_process) >= process_limit:
            continue

        rows_to_process.append((i, row))

    if not rows_to_process:
        print(f"    All {skipped_count} rows already processed")
        return 0, skipped_count

    print(f"    Processing {len(rows_to_process)} rows ({skipped_count} skipped) with {max_workers} workers")

    # Create processor and run — always use ALL rows for saving
    processor = ParallelProcessor(output_path, output_fieldnames, max_workers)

    # Pre-populate with existing results for all rows
    for i, row in enumerate(input_rows):
        key = (row.get("sample_id", ""), row.get("original_input", ""))
        if key in existing_rows and row_already_processed(existing_rows[key]):
            processor.results[i] = existing_rows[key]

    processor.process_batch(
        rows_to_process,
        input_rows,  # Always pass ALL rows so save_results preserves them
        dialect_key,
        llm,
        rating_level
    )

    return len(rows_to_process), skipped_count


def main(
    data_dir: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "dataset", "dia_llm"),
    max_workers: int = 6,
    test_mode: bool = False,
    test_rows: int = 1,
    specific_dialect: Optional[str] = None,
    specific_dataset: Optional[str] = None,
    model_key: str = "gpt4.1"
):
    """Main function to process all dialects and datasets in-place.

    Reads and writes the same _with_transforms.csv files in data_dir.
    Rows that already have basic_transform and coi_transform are skipped.
    """
    print("=" * 100)
    print("PARALLEL GENERATION: Multi-VALUE + Basic Transform + CoI Transform")
    print(f"Data dir (in-place): {data_dir}")
    print(f"Model: {model_key}")
    print(f"Workers: {max_workers}")
    print(f"Test mode: {test_mode} (rows: {test_rows})")
    print("=" * 100)

    # Create LLM (shared across threads - client is thread-safe)
    llm = create_llm(model_key)

    # Get list of dialects to process
    dialect_folders = sorted(os.listdir(data_dir))
    dialect_folders = [d for d in dialect_folders if os.path.isdir(os.path.join(data_dir, d))]

    if specific_dialect:
        dialect_folders = [d for d in dialect_folders if d == specific_dialect]

    total_processed = 0
    total_skipped = 0
    start_time = time.time()

    for dialect_folder in dialect_folders:
        # Check if dialect is mapped
        if dialect_folder not in DIALECT_MAPPING:
            print(f"\n[SKIP] {dialect_folder}: No dialect mapping")
            continue

        dialect_key = DIALECT_MAPPING[dialect_folder]
        dialect_path = os.path.join(data_dir, dialect_folder)

        print(f"\n{'=' * 80}")
        print(f"DIALECT: {dialect_folder} -> {dialect_key}")
        print(f"{'=' * 80}")

        # Get CSV files in dialect folder
        csv_files = [f for f in os.listdir(dialect_path) if f.endswith('.csv')]

        if specific_dataset:
            csv_files = [f for f in csv_files if specific_dataset in f]

        for csv_file in sorted(csv_files):
            csv_path = os.path.join(dialect_path, csv_file)

            print(f"\n  Dataset: {csv_file}")

            try:
                # In-place: same file is both input and output
                processed, skipped = process_csv(
                    csv_path, csv_path, dialect_key, llm,
                    max_workers=max_workers,
                    test_mode=test_mode, test_rows=test_rows
                )
                total_processed += processed
                total_skipped += skipped

            except Exception as e:
                print(f"    ERROR: {str(e)[:60]}...")

            if test_mode:
                break  # Only process one dataset in test mode

        if test_mode:
            break  # Only process one dialect in test mode

    elapsed = time.time() - start_time
    rate = total_processed / (elapsed / 60) if elapsed > 0 else 0

    print("\n" + "=" * 100)
    print("GENERATION COMPLETE")
    print(f"Total processed: {total_processed}")
    print(f"Total skipped (already done): {total_skipped}")
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Average rate: {rate:.1f} rows/minute")
    print("=" * 100)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parallel generation for DIA_LLM dataset (in-place)")
    parser.add_argument("--test", action="store_true", help="Test mode (limited rows)")
    parser.add_argument("--test-rows", type=int, default=5, help="Number of rows in test mode")
    parser.add_argument("--workers", type=int, default=6, help="Number of parallel workers")
    parser.add_argument("--dialect", type=str, help="Specific dialect folder to process")
    parser.add_argument("--dataset", type=str, help="Specific dataset name pattern to process")
    parser.add_argument("--data-dir", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "dataset", "dia_llm"))
    parser.add_argument("--model", type=str, default="gpt4.1", choices=list(MODEL_CONFIGS.keys()),
                        help="Model to use for generation (default: gpt4.1)")

    args = parser.parse_args()

    main(
        data_dir=args.data_dir,
        max_workers=args.workers,
        test_mode=args.test,
        test_rows=args.test_rows,
        specific_dialect=args.dialect,
        specific_dataset=args.dataset,
        model_key=args.model
    )
