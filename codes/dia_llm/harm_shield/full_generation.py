#!/usr/bin/env python3
"""
Full generation script for DIA_LLM dataset.

Processes all dialects and datasets, adding:
- basic_transform: Basic Transform output (single-shot with eWAVE)
- coi_transform: CoI Transform output (4-chain agentic)
- Validation metrics for Multi-VALUE, Basic, and CoI transforms

Features:
- Incremental saving after each row
- Resume capability (skips already processed rows)
- Policy violation handling
"""

import sys
import os
import csv
import time
from pathlib import Path
from typing import Optional, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import AzureOpenAIBackend
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
]


def create_llm():
    """Create Azure OpenAI backend."""
    return AzureOpenAIBackend(
        deployment_name="gpt-4.1",
        endpoint="https://jsl-diaguard.openai.azure.com",
        api_version="2024-02-15-preview",
        api_key="YOUR_AZURE_OPENAI_API_KEY"
    )


def is_policy_violation(error_msg: str) -> bool:
    """Check if error is a content policy violation."""
    return "content_filter" in error_msg or "content management policy" in error_msg


def row_already_processed(row: Dict[str, Any]) -> bool:
    """Check if row has already been processed (has both transforms)."""
    basic = row.get("basic_transform", "")
    coi = row.get("coi_transform", "")

    # Consider processed if both have non-empty values
    if basic and coi:
        return True
    return False


def process_row(
    row: Dict[str, Any],
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

    # Generate Basic Transform
    basic_output = POLICY_VIOLATION
    basic_filtered = False
    try:
        basic_result = basic_transformer.transform(original, dialect_key)
        basic_output = basic_result.transformed
    except Exception as e:
        error_msg = str(e)
        if is_policy_violation(error_msg):
            basic_filtered = True
            basic_output = POLICY_VIOLATION
        else:
            print(f"      Basic Error: {error_msg[:50]}...")
            basic_output = original  # Fallback to original

    # Generate CoI Transform
    coi_output = POLICY_VIOLATION
    coi_filtered = False
    try:
        coi_result = coi_transformer.transform(original, dialect_key, skip_validation=True)
        coi_output = coi_result.final_output
    except Exception as e:
        error_msg = str(e)
        if is_policy_violation(error_msg):
            coi_filtered = True
            coi_output = POLICY_VIOLATION
        else:
            print(f"      CoI Error: {error_msg[:50]}...")
            coi_output = original  # Fallback to original

    result["basic_transform"] = basic_output
    result["coi_transform"] = coi_output

    # Validate Multi-VALUE (if available and not policy violation)
    if multivalue and multivalue != POLICY_VIOLATION:
        try:
            mv_validation = validator.validate(original, multivalue)
            result["mv_ewave_accuracy"] = f"{mv_validation.ewave_accuracy:.4f}"
            result["mv_errors"] = len(mv_validation.non_ewave_errors)
            result["mv_valid_features"] = len(mv_validation.valid_changes)
        except Exception as e:
            print(f"      MV Validation Error: {str(e)[:50]}...")
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
        except Exception as e:
            print(f"      Basic Validation Error: {str(e)[:50]}...")
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
        except Exception as e:
            print(f"      CoI Validation Error: {str(e)[:50]}...")
            result["coi_ewave_accuracy"] = ""
            result["coi_errors"] = ""
            result["coi_valid_features"] = ""
    else:
        result["coi_ewave_accuracy"] = ""
        result["coi_errors"] = ""
        result["coi_valid_features"] = ""

    return result


def save_rows(rows: list, output_path: str, fieldnames: list):
    """Save rows to CSV file."""
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def process_csv(
    input_path: str,
    output_path: str,
    dialect_key: str,
    llm,
    rating_level: str = "AB",
    test_mode: bool = False,
    test_rows: int = 1
):
    """Process a single CSV file."""
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

    # Create transformers and validator
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

    # Process rows
    output_rows = []
    processed_count = 0
    skipped_count = 0

    rows_to_process = input_rows[:test_rows] if test_mode else input_rows

    for i, row in enumerate(rows_to_process):
        key = (row.get("sample_id", ""), row.get("original_input", ""))

        # Check if already processed
        if key in existing_rows and row_already_processed(existing_rows[key]):
            output_rows.append(existing_rows[key])
            skipped_count += 1
            continue

        print(f"    [{i+1}/{len(rows_to_process)}] Processing sample_id={row.get('sample_id', 'N/A')[:20]}...")

        try:
            processed_row = process_row(
                row, dialect_key, llm,
                basic_transformer, coi_transformer, validator
            )
            output_rows.append(processed_row)
            processed_count += 1

            # Save after each successful row
            save_rows(output_rows, output_path, output_fieldnames)

        except Exception as e:
            print(f"      Error processing row: {str(e)[:50]}...")
            # Add row with empty new columns
            for col in NEW_COLUMNS:
                row[col] = ""
            output_rows.append(row)
            save_rows(output_rows, output_path, output_fieldnames)

    return processed_count, skipped_count


def main(
    input_dir: str = "/Users/jsl/Downloads/DIA_LLM/data_100_per_dataset",
    output_dir: str = "/Users/jsl/Downloads/DIA_LLM/outputs",
    test_mode: bool = False,
    test_rows: int = 1,
    specific_dialect: Optional[str] = None,
    specific_dataset: Optional[str] = None
):
    """Main function to process all dialects and datasets."""
    print("=" * 100)
    print("FULL GENERATION: Multi-VALUE + Basic Transform + CoI Transform")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Test mode: {test_mode} (rows: {test_rows})")
    print("=" * 100)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Create LLM
    llm = create_llm()

    # Get list of dialects to process
    dialect_folders = sorted(os.listdir(input_dir))
    dialect_folders = [d for d in dialect_folders if os.path.isdir(os.path.join(input_dir, d))]

    if specific_dialect:
        dialect_folders = [d for d in dialect_folders if d == specific_dialect]

    total_processed = 0
    total_skipped = 0

    for dialect_folder in dialect_folders:
        # Check if dialect is mapped
        if dialect_folder not in DIALECT_MAPPING:
            print(f"\n[SKIP] {dialect_folder}: No dialect mapping")
            continue

        dialect_key = DIALECT_MAPPING[dialect_folder]
        dialect_path = os.path.join(input_dir, dialect_folder)

        print(f"\n{'=' * 80}")
        print(f"DIALECT: {dialect_folder} -> {dialect_key}")
        print(f"{'=' * 80}")

        # Create output folder for dialect
        dialect_output_dir = os.path.join(output_dir, dialect_folder)
        os.makedirs(dialect_output_dir, exist_ok=True)

        # Get CSV files in dialect folder
        csv_files = [f for f in os.listdir(dialect_path) if f.endswith('.csv')]

        if specific_dataset:
            csv_files = [f for f in csv_files if specific_dataset in f]

        for csv_file in sorted(csv_files):
            input_csv = os.path.join(dialect_path, csv_file)
            output_csv = os.path.join(dialect_output_dir, csv_file.replace('.csv', '_with_transforms.csv'))

            print(f"\n  Dataset: {csv_file}")

            try:
                processed, skipped = process_csv(
                    input_csv, output_csv, dialect_key, llm,
                    test_mode=test_mode, test_rows=test_rows
                )
                total_processed += processed
                total_skipped += skipped
                print(f"    Processed: {processed}, Skipped: {skipped}")

            except Exception as e:
                print(f"    ERROR: {str(e)[:60]}...")

            if test_mode:
                break  # Only process one dataset in test mode

        if test_mode:
            break  # Only process one dialect in test mode

    print("\n" + "=" * 100)
    print("GENERATION COMPLETE")
    print(f"Total processed: {total_processed}")
    print(f"Total skipped (already done): {total_skipped}")
    print("=" * 100)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Full generation for DIA_LLM dataset")
    parser.add_argument("--test", action="store_true", help="Test mode (1 row only)")
    parser.add_argument("--test-rows", type=int, default=1, help="Number of rows in test mode")
    parser.add_argument("--dialect", type=str, help="Specific dialect folder to process")
    parser.add_argument("--dataset", type=str, help="Specific dataset name pattern to process")
    parser.add_argument("--input-dir", type=str, default="/Users/jsl/Downloads/DIA_LLM/data_100_per_dataset")
    parser.add_argument("--output-dir", type=str, default="/Users/jsl/Downloads/DIA_LLM/outputs")

    args = parser.parse_args()

    main(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        test_mode=args.test,
        test_rows=args.test_rows,
        specific_dialect=args.dialect,
        specific_dataset=args.dataset
    )
