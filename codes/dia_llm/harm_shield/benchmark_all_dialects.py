#!/usr/bin/env python3
"""
Comprehensive benchmark: Multi-VALUE vs CoI Transform across ALL dialects.

Runs 1 sample per dialect and compares transformation quality using eWAVE validation.
"""

import sys
import os
import csv
import random
from typing import List, Dict, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import AzureOpenAIBackend
from coi_transformation import CoIDialectTransformer
from feature_validator import LLMComprehensiveValidator


# Mapping from dataset folder names to eWAVE dialect keys
# Complete mapping for all 48 dialects in DIA_LLM dataset
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


def create_llm():
    """Create Azure OpenAI backend."""
    return AzureOpenAIBackend(
        deployment_name="gpt-4.1",
        endpoint="https://jsl-diaguard.openai.azure.com",
        api_version="2024-02-15-preview",
        api_key="YOUR_AZURE_OPENAI_API_KEY"
    )


def find_dialect_csv(base_path: str, dialect_folder: str) -> Optional[str]:
    """Find the harmBench CSV file for a dialect."""
    folder_path = os.path.join(base_path, dialect_folder)
    if not os.path.isdir(folder_path):
        return None

    # Look for harmBench file first
    for f in os.listdir(folder_path):
        if f.startswith("harmBench") and f.endswith(".csv"):
            return os.path.join(folder_path, f)

    # Fallback to any CSV
    for f in os.listdir(folder_path):
        if f.endswith(".csv") and not f.startswith("f3"):
            return os.path.join(folder_path, f)

    return None


def load_sample(csv_path: str, seed: int = 42) -> Optional[Dict]:
    """Load a single random sample from dataset."""
    random.seed(seed)

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return None

    row = random.choice(rows)
    return {
        "original": row.get("Behavior", ""),
        "multivalue": row.get("Behavior_transformed", "")
    }


def run_single_comparison(
    sample: Dict,
    dialect_key: str,
    dialect_name: str,
    llm,
    transformer: CoIDialectTransformer,
    rating_level: str = "AB"
) -> Optional[Dict]:
    """Run comparison for a single sample."""
    original = sample["original"]
    multivalue = sample["multivalue"]

    if not original or not multivalue:
        return None

    # Generate CoI transform
    coi_filtered = False  # Track if Azure content filter blocked
    try:
        coi_result = transformer.transform(original, dialect_key, skip_validation=True)
        coi_output = coi_result.final_output
    except Exception as e:
        error_msg = str(e)
        print(f"    CoI Error: {error_msg[:50]}...")
        # Check if it's Azure content filter
        if "content_filter" in error_msg or "content management policy" in error_msg:
            coi_filtered = True
        coi_output = original  # Fallback

    # Validate both
    try:
        validator = LLMComprehensiveValidator(
            llm, dialect_key, verbose=False,
            use_explicit_ratings=True,
            use_post_correction=True
        )

        mv_validation = validator.validate(original, multivalue)
        coi_validation = validator.validate(original, coi_output)

        return {
            "original": original,
            "multivalue": multivalue,
            "coi": coi_output,
            "mv_validation": mv_validation,
            "coi_validation": coi_validation,
            "coi_filtered": coi_filtered  # Track content filter
        }
    except Exception as e:
        print(f"    Validation Error: {str(e)[:50]}...")
        return None


def main():
    """Run benchmark across all dialects."""
    print("=" * 100)
    print("COMPREHENSIVE BENCHMARK: MULTI-VALUE vs COI TRANSFORM")
    print("1 Sample per Dialect | AB-rated eWAVE features")
    print("=" * 100)

    random.seed(42)
    llm = create_llm()
    transformer = CoIDialectTransformer(llm=llm, verbose=False, rating_level="AB")

    base_path = "/Users/jsl/Downloads/outputs"

    results = []
    skipped = []

    # Process each dialect
    for dialect_folder, dialect_key in sorted(DIALECT_MAPPING.items()):
        csv_path = find_dialect_csv(base_path, dialect_folder)

        if not csv_path:
            skipped.append((dialect_folder, "No CSV found"))
            continue

        sample = load_sample(csv_path)
        if not sample:
            skipped.append((dialect_folder, "No samples"))
            continue

        dialect_name = dialect_folder.replace("_", " ").title()
        print(f"\n[{len(results)+1:02d}] {dialect_name}")
        print(f"    Original: {sample['original'][:60]}...")

        result = run_single_comparison(
            sample=sample,
            dialect_key=dialect_key,
            dialect_name=dialect_name,
            llm=llm,
            transformer=transformer
        )

        if result:
            mv_v = result["mv_validation"]
            coi_v = result["coi_validation"]
            filtered_mark = " [FILTERED]" if result.get("coi_filtered") else ""

            print(f"    Multi-VALUE: eWAVE={mv_v.ewave_accuracy:.0%}, Valid={len(mv_v.valid_changes)}, Error={len(mv_v.non_ewave_errors)}")
            print(f"    CoI Transform: eWAVE={coi_v.ewave_accuracy:.0%}, Valid={len(coi_v.valid_changes)}, Error={len(coi_v.non_ewave_errors)}{filtered_mark}")

            results.append({
                "dialect": dialect_name,
                "dialect_key": dialect_key,
                **result
            })
        else:
            skipped.append((dialect_folder, "Validation failed"))

    # Separate filtered and non-filtered results
    filtered_results = [r for r in results if r.get("coi_filtered")]
    valid_results = [r for r in results if not r.get("coi_filtered")]

    # Print summary table
    print("\n" + "=" * 100)
    print("RESULTS SUMMARY (excluding Azure content-filtered samples)")
    print("=" * 100)

    print(f"\n{'Dialect':<45} {'Multi-VALUE':^20} {'CoI Transform':^20} {'Winner':^12}")
    print(f"{'':45} {'eWAVE':>8} {'Err':>5} {'Valid':>5} {'eWAVE':>8} {'Err':>5} {'Valid':>5} {'':<12}")
    print("-" * 100)

    mv_wins = 0
    coi_wins = 0
    ties = 0

    mv_total_ewave = 0
    coi_total_ewave = 0
    mv_total_errors = 0
    coi_total_errors = 0
    mv_total_valid = 0
    coi_total_valid = 0

    for r in valid_results:
        mv = r["mv_validation"]
        coi = r["coi_validation"]

        mv_ewave = mv.ewave_accuracy
        coi_ewave = coi.ewave_accuracy
        mv_err = len(mv.non_ewave_errors)
        coi_err = len(coi.non_ewave_errors)
        mv_valid = len(mv.valid_changes)
        coi_valid = len(coi.valid_changes)

        # Determine winner (higher eWAVE, fewer errors)
        if coi_ewave > mv_ewave or (coi_ewave == mv_ewave and coi_err < mv_err):
            winner = "CoI"
            coi_wins += 1
        elif mv_ewave > coi_ewave or (mv_ewave == coi_ewave and mv_err < coi_err):
            winner = "Multi-VALUE"
            mv_wins += 1
        else:
            winner = "Tie"
            ties += 1

        print(f"{r['dialect']:<45} {mv_ewave:>7.0%} {mv_err:>5} {mv_valid:>5} {coi_ewave:>7.0%} {coi_err:>5} {coi_valid:>5} {winner:^12}")

        mv_total_ewave += mv_ewave
        coi_total_ewave += coi_ewave
        mv_total_errors += mv_err
        coi_total_errors += coi_err
        mv_total_valid += mv_valid
        coi_total_valid += coi_valid

    n = len(valid_results)
    print("-" * 100)
    if n > 0:
        print(f"{'AVERAGE':<45} {mv_total_ewave/n:>7.0%} {mv_total_errors:>5} {mv_total_valid:>5} {coi_total_ewave/n:>7.0%} {coi_total_errors:>5} {coi_total_valid:>5}")

    # Final summary
    print("\n" + "=" * 100)
    print("FINAL COMPARISON")
    print("=" * 100)
    print(f"\nDialects tested (valid): {n}")
    print(f"Dialects filtered (Azure content policy): {len(filtered_results)}")
    print(f"Dialects skipped (other errors): {len(skipped)}")

    if n > 0:
        print(f"\nWINS (excluding filtered):")
        print(f"  Multi-VALUE: {mv_wins} ({mv_wins/n:.0%})")
        print(f"  CoI Transform: {coi_wins} ({coi_wins/n:.0%})")
        print(f"  Ties: {ties} ({ties/n:.0%})")

        print(f"\nAVERAGE METRICS (excluding filtered):")
        print(f"  Multi-VALUE:  eWAVE={mv_total_ewave/n:.0%}, Errors={mv_total_errors}, Valid Features={mv_total_valid}")
        print(f"  CoI Transform: eWAVE={coi_total_ewave/n:.0%}, Errors={coi_total_errors}, Valid Features={coi_total_valid}")

    if filtered_results:
        print(f"\nFILTERED DIALECTS (Azure content policy):")
        for r in filtered_results:
            print(f"  - {r['dialect']}")

    if skipped:
        print(f"\nSKIPPED DIALECTS:")
        for folder, reason in skipped:
            print(f"  - {folder}: {reason}")

    print("\n" + "=" * 100)
    print("BENCHMARK COMPLETED")
    print("=" * 100)

    return 0


if __name__ == "__main__":
    sys.exit(main())
