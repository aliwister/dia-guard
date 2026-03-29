#!/usr/bin/env python3
"""
Parallel generation script for CounterHarm-SHIELD benign sample generation.

Processes multiple rows concurrently, running the 6-chain CoI pipeline on
4 text columns per row: original_input, transformed_input, basic_transform, coi_transform.

Part of DIA-GUARD → Dia-LLM → CounterHarm-SHIELD.
"""

import sys
import os
import csv
import time
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent dir so we can import from Harm_Shield
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "harm_shield"))

from models import AzureOpenAIBackend, BedrockBackend
from counterharm_pipeline import CounterHarmSHIELD

# Same dialect mapping as Harm_Shield
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

# The 4 source text columns to generate benign counterparts for
SOURCE_COLUMNS = ["original_input", "transformed_input", "basic_transform", "coi_transform"]

# New columns added by CounterHarm-SHIELD
COUNTERHARM_COLUMNS = [
    "counterharm_original",
    "counterharm_transformed",
    "counterharm_basic",
    "counterharm_coi",
    "counterharm_score",
    "counterharm_validated",
    "counterharm_model",
]

# Mapping from source column to output column
SOURCE_TO_OUTPUT = {
    "original_input": "counterharm_original",
    "transformed_input": "counterharm_transformed",
    "basic_transform": "counterharm_basic",
    "coi_transform": "counterharm_coi",
}

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
    "bedrock-llama3.3-70b": {
        "model_id": "us.meta.llama3-3-70b-instruct-v1:0",
        "region": "us-east-1",
    },
    "bedrock-llama4-maverick": {
        "model_id": "us.meta.llama4-maverick-17b-instruct-v1:0",
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
    "bedrock-claude-opus-4.6": {
        "model_id": "us.anthropic.claude-opus-4-6-v1",
        "region": "us-east-1",
    },
    "bedrock-claude-sonnet-4.6": {
        "model_id": "us.anthropic.claude-sonnet-4-6",
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


def row_already_processed(row: Dict[str, Any]) -> bool:
    """Check if row already has all CounterHarm outputs filled."""
    src_cols = {"counterharm_original": "original_input",
                "counterharm_transformed": "transformed_input",
                "counterharm_basic": "basic_transform",
                "counterharm_coi": "coi_transform"}
    for out_col, src_col in src_cols.items():
        src_val = row.get(src_col, "").strip()
        out_val = row.get(out_col, "").strip()
        # Skip check if source is empty or a policy violation
        if not src_val or src_val == POLICY_VIOLATION:
            continue
        # Treat [FAILED] as done (permanently failed — don't retry)
        if out_val == "[FAILED]":
            continue
        if not out_val:
            return False
    return True


def process_row(
    row: Dict[str, Any],
    row_idx: int,
    dialect_key: str,
    pipeline: CounterHarmSHIELD,
) -> Dict[str, Any]:
    """Process a single row: run 6-chain pipeline on each of 4 text columns."""
    result = dict(row)

    scores = []
    all_validated = True

    for src_col, out_col in SOURCE_TO_OUTPUT.items():
        # Skip columns that already have results or permanently failed (partial resume)
        existing_val = row.get(out_col, "").strip()
        if existing_val:
            continue  # Includes [FAILED] — don't retry permanently failed cells

        text = row.get(src_col, "").strip()

        # Skip empty or policy-violation texts
        if not text or text == POLICY_VIOLATION:
            result[out_col] = ""
            continue

        try:
            state = pipeline.run(harmful_seed=text, dialect=dialect_key)
            output = state.refined_text or ""
            if not output:
                # Pipeline exhausted retries without producing output
                result[out_col] = "[FAILED]"
                all_validated = False
            else:
                result[out_col] = output
                scores.append(state.harmlessness_score)
                if not state.validated:
                    all_validated = False
        except Exception as e:
            result[out_col] = "[FAILED]"
            all_validated = False

    # Aggregate scores: average of all non-zero scores
    avg_score = sum(scores) / len(scores) if scores else 0.0
    result["counterharm_score"] = f"{avg_score:.4f}"
    result["counterharm_validated"] = str(all_validated)
    result["counterharm_model"] = pipeline.llm.name

    return result


class ParallelProcessor:
    """Process rows in parallel with thread-safe saving."""

    def __init__(self, output_path: str, fieldnames: list, max_workers: int = 4):
        self.output_path = output_path
        self.fieldnames = fieldnames
        self.max_workers = max_workers
        self.results = {}  # row_idx -> result
        self.lock = threading.Lock()
        self.processed_count = 0
        self.start_time = time.time()

    def save_results(self, all_rows: list):
        """Save all results in order, preserving unprocessed rows."""
        with self.lock:
            output_rows = []
            for i, row in enumerate(all_rows):
                if i in self.results:
                    output_rows.append(self.results[i])
                else:
                    row_copy = dict(row)
                    for col in COUNTERHARM_COLUMNS:
                        if col not in row_copy:
                            row_copy[col] = ""
                    output_rows.append(row_copy)

            with open(self.output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()
                writer.writerows(output_rows)

    def process_batch(
        self,
        rows_to_process: List[tuple],
        all_rows: list,
        dialect_key: str,
        llm,
    ):
        """Process a batch of rows in parallel."""
        def worker(idx_row_tuple):
            idx, row = idx_row_tuple

            # Create thread-local pipeline instance
            pipeline = CounterHarmSHIELD(llm=llm, verbose=False)

            try:
                result = process_row(row, idx, dialect_key, pipeline)
                return (idx, result, None)
            except Exception as e:
                row_copy = dict(row)
                for col in COUNTERHARM_COLUMNS:
                    row_copy[col] = ""
                return (idx, row_copy, str(e))

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(worker, item): item for item in rows_to_process}

            for future in as_completed(futures):
                idx, result, error = future.result()

                with self.lock:
                    self.results[idx] = result
                    self.processed_count += 1

                    elapsed = time.time() - self.start_time
                    rate = self.processed_count / (elapsed / 60) if elapsed > 0 else 0

                    sample_id = result.get("sample_id", "N/A")[:15]
                    if error:
                        print(f"    [{self.processed_count}] idx={idx} sample={sample_id} ERROR: {error[:50]}")
                    else:
                        validated = result.get("counterharm_validated", "?")
                        score = result.get("counterharm_score", "?")
                        print(f"    [{self.processed_count}] idx={idx} sample={sample_id} "
                              f"score={score} valid={validated} ({rate:.1f} rows/min)")

                # Save after each completion
                self.save_results(all_rows)


def process_csv(
    input_path: str,
    output_path: str,
    dialect_key: str,
    llm,
    max_workers: int = 4,
    test_mode: bool = False,
    test_rows: int = 1,
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
    for col in COUNTERHARM_COLUMNS:
        if col not in output_fieldnames:
            output_fieldnames.append(col)

    # Determine rows to process (skip already-processed)
    rows_to_process = []
    skipped_count = 0
    process_limit = test_rows if test_mode else len(input_rows)

    for i, row in enumerate(input_rows):
        key = (row.get("sample_id", ""), row.get("original_input", ""))

        if key in existing_rows and row_already_processed(existing_rows[key]):
            skipped_count += 1
            continue

        if test_mode and len(rows_to_process) >= process_limit:
            continue

        rows_to_process.append((i, row))

    if not rows_to_process:
        print(f"    All {skipped_count} rows already processed")
        return 0, skipped_count

    print(f"    Processing {len(rows_to_process)} rows ({skipped_count} skipped) with {max_workers} workers")

    # Create processor and run
    processor = ParallelProcessor(output_path, output_fieldnames, max_workers)

    # Pre-populate with existing results
    for i, row in enumerate(input_rows):
        key = (row.get("sample_id", ""), row.get("original_input", ""))
        if key in existing_rows and row_already_processed(existing_rows[key]):
            processor.results[i] = existing_rows[key]

    processor.process_batch(rows_to_process, input_rows, dialect_key, llm)

    return len(rows_to_process), skipped_count


def main(
    data_dir: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "dataset", "dia_llm"),
    max_workers: int = 4,
    test_mode: bool = False,
    test_rows: int = 1,
    specific_dialect: Optional[str] = None,
    specific_dataset: Optional[str] = None,
    model_key: str = "gpt4.1",
):
    """Main function to run CounterHarm-SHIELD benign generation in-place.

    Reads and writes the same _with_transforms.csv files in data_dir.
    For each row, generates benign counterparts for 4 text columns using
    the 6-chain CoI pipeline (ToxiCraft + PromptSafe + FIZLE).

    ~24 API calls per row (6 chains x 4 columns).
    """
    print("=" * 100)
    print("CounterHarm-SHIELD: Dialect-Aware Benign Sample Generation")
    print("DIA-GUARD -> Dia-LLM -> CounterHarm-SHIELD")
    print(f"Data dir (in-place): {data_dir}")
    print(f"Model: {model_key}")
    print(f"Workers: {max_workers}")
    print(f"Test mode: {test_mode} (rows: {test_rows})")
    print(f"Columns: {SOURCE_COLUMNS}")
    print(f"Pipeline: 6-chain CoI (ToxiCraft + PromptSafe + FIZLE)")
    print("=" * 100)

    llm = create_llm(model_key)

    dialect_folders = sorted(os.listdir(data_dir))
    dialect_folders = [d for d in dialect_folders if os.path.isdir(os.path.join(data_dir, d))]

    if specific_dialect:
        dialect_folders = [d for d in dialect_folders if d == specific_dialect]

    total_processed = 0
    total_skipped = 0
    start_time = time.time()

    for dialect_folder in dialect_folders:
        if dialect_folder not in DIALECT_MAPPING:
            print(f"\n[SKIP] {dialect_folder}: No dialect mapping")
            continue

        dialect_key = DIALECT_MAPPING[dialect_folder]
        dialect_path = os.path.join(data_dir, dialect_folder)

        print(f"\n{'=' * 80}")
        print(f"DIALECT: {dialect_folder} -> {dialect_key}")
        print(f"{'=' * 80}")

        csv_files = [f for f in os.listdir(dialect_path) if f.endswith('.csv')]

        if specific_dataset:
            csv_files = [f for f in csv_files if specific_dataset in f]

        for csv_file in sorted(csv_files):
            csv_path = os.path.join(dialect_path, csv_file)

            print(f"\n  Dataset: {csv_file}")

            try:
                processed, skipped = process_csv(
                    csv_path, csv_path, dialect_key, llm,
                    max_workers=max_workers,
                    test_mode=test_mode, test_rows=test_rows,
                )
                total_processed += processed
                total_skipped += skipped
            except Exception as e:
                print(f"    ERROR: {str(e)[:60]}...")

            if test_mode:
                break

        if test_mode:
            break

    elapsed = time.time() - start_time
    rate = total_processed / (elapsed / 60) if elapsed > 0 else 0

    print("\n" + "=" * 100)
    print("CounterHarm-SHIELD GENERATION COMPLETE")
    print(f"Total processed: {total_processed}")
    print(f"Total skipped (already done): {total_skipped}")
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Average rate: {rate:.1f} rows/minute")
    print("=" * 100)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="CounterHarm-SHIELD: Parallel benign sample generation (in-place)"
    )
    parser.add_argument("--test", action="store_true", help="Test mode (limited rows)")
    parser.add_argument("--test-rows", type=int, default=1, help="Number of rows in test mode")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--dialect", type=str, help="Specific dialect folder to process")
    parser.add_argument("--dataset", type=str, help="Specific dataset name pattern to process")
    parser.add_argument("--data-dir", type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "dataset", "dia_llm"))
    parser.add_argument("--model", type=str, default="gpt4.1",
                        choices=list(MODEL_CONFIGS.keys()),
                        help="Model to use for generation (default: gpt4.1)")

    args = parser.parse_args()

    main(
        data_dir=args.data_dir,
        max_workers=args.workers,
        test_mode=args.test,
        test_rows=args.test_rows,
        specific_dialect=args.dialect,
        specific_dataset=args.dataset,
        model_key=args.model,
    )
