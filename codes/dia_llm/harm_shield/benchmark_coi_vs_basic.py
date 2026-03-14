#!/usr/bin/env python3
"""
Benchmark comparison: Multi-VALUE vs CoI Transform vs Basic Transform

Compares three approaches:
1. Multi-VALUE: Dataset transformations (pre-computed)
2. CoI Transform: 4-chain agentic approach
3. Basic Transform: Single-shot with eWAVE ratings
"""

import sys
import os
import csv
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import AzureOpenAIBackend
from transformer import DialectTransformer
from coi_transformation import CoIDialectTransformer
from feature_validator import LLMComprehensiveValidator


# Dataset paths for Multi-VALUE samples
DATASETS = {
    "urban_aave": {
        "path": "/Users/jsl/Downloads/outputs/urban_african_american_vernacular_english/harmBench_urban_african_american_vernacular_english.csv",
        "name": "Urban AAVE"
    },
    "aboriginal": {
        "path": "/Users/jsl/Downloads/outputs/aboriginal_english/harmBench_aboriginal_english.csv",
        "name": "Aboriginal English"
    },
    "jamaican": {
        "path": "/Users/jsl/Downloads/outputs/jamaican_english/harmBench_jamaican_english.csv",
        "name": "Jamaican English"
    },
    "indian": {
        "path": "/Users/jsl/Downloads/outputs/indian_english/harmBench_indian_english.csv",
        "name": "Indian English"
    },
}


def create_llm():
    """Create Azure OpenAI backend."""
    return AzureOpenAIBackend(
        deployment_name="gpt-4.1",
        endpoint="https://jsl-diaguard.openai.azure.com",
        api_version="2024-02-15-preview",
        api_key="YOUR_AZURE_OPENAI_API_KEY"
    )


def load_samples_from_csv(csv_path: str, n_samples: int = 3):
    """Load samples from Multi-VALUE dataset."""
    samples = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if len(rows) > n_samples:
        rows = random.sample(rows, n_samples)

    for row in rows:
        samples.append({
            "original": row.get("Behavior", ""),
            "multivalue": row.get("Behavior_transformed", "")
        })

    return samples


def run_comparison(
    sample: dict,
    dialect_key: str,
    dialect_name: str,
    llm,
    basic_transformer,
    coi_transformer,
    validator,
    rating_level: str = "AB"
):
    """Run comparison between Multi-VALUE, CoI, and Basic Transform."""
    text = sample["original"]
    multivalue_output = sample["multivalue"]

    if not text or not multivalue_output:
        return None

    print(f"\n  Original: {text[:70]}{'...' if len(text) > 70 else ''}")

    # Multi-VALUE (pre-computed)
    mv_output = multivalue_output

    # Basic Transform (single-shot with eWAVE)
    try:
        basic_result = basic_transformer.transform(text, dialect_key)
        basic_output = basic_result.transformed
    except Exception as e:
        print(f"    Basic Error: {str(e)[:40]}...")
        basic_output = text

    # CoI Transform (4-chain agentic)
    coi_filtered = False
    try:
        coi_result = coi_transformer.transform(text, dialect_key, skip_validation=True)
        coi_output = coi_result.final_output
    except Exception as e:
        error_msg = str(e)
        print(f"    CoI Error: {error_msg[:40]}...")
        if "content_filter" in error_msg or "content management policy" in error_msg:
            coi_filtered = True
        coi_output = text

    # Validate all three
    try:
        mv_validation = validator.validate(text, mv_output)
        basic_validation = validator.validate(text, basic_output)
        coi_validation = validator.validate(text, coi_output)

        print(f"    Multi-VALUE: eWAVE={mv_validation.ewave_accuracy:.0%}, Valid={len(mv_validation.valid_changes)}, Err={len(mv_validation.non_ewave_errors)}")
        print(f"    Basic:       eWAVE={basic_validation.ewave_accuracy:.0%}, Valid={len(basic_validation.valid_changes)}, Err={len(basic_validation.non_ewave_errors)}")
        print(f"    CoI:         eWAVE={coi_validation.ewave_accuracy:.0%}, Valid={len(coi_validation.valid_changes)}, Err={len(coi_validation.non_ewave_errors)}{' [FILTERED]' if coi_filtered else ''}")

        return {
            "original": text,
            "mv_output": mv_output,
            "basic_output": basic_output,
            "coi_output": coi_output,
            "mv_validation": mv_validation,
            "basic_validation": basic_validation,
            "coi_validation": coi_validation,
            "coi_filtered": coi_filtered
        }
    except Exception as e:
        print(f"    Validation Error: {str(e)[:40]}...")
        return None


def main():
    """Run benchmark comparison."""
    print("=" * 100)
    print("BENCHMARK: MULTI-VALUE vs BASIC TRANSFORM vs COI TRANSFORM")
    print("Using AB-rated eWAVE features (pervasive + common)")
    print("=" * 100)

    random.seed(42)
    llm = create_llm()

    n_samples = 3  # Samples per dialect
    rating_level = "AB"

    all_results = []

    for dialect_key, ds_info in DATASETS.items():
        dialect_name = ds_info["name"]
        csv_path = ds_info["path"]

        if not os.path.exists(csv_path):
            print(f"\n[SKIP] {dialect_name}: CSV not found")
            continue

        print(f"\n{'=' * 80}")
        print(f"DIALECT: {dialect_name}")
        print(f"{'=' * 80}")

        samples = load_samples_from_csv(csv_path, n_samples)

        # Create transformers once per dialect
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

        for i, sample in enumerate(samples, 1):
            print(f"\n[Sample {i}/{len(samples)}]")
            result = run_comparison(
                sample=sample,
                dialect_key=dialect_key,
                dialect_name=dialect_name,
                llm=llm,
                basic_transformer=basic_transformer,
                coi_transformer=coi_transformer,
                validator=validator,
                rating_level=rating_level
            )
            if result:
                result["dialect"] = dialect_name
                result["dialect_key"] = dialect_key
                all_results.append(result)

    # Exclude filtered results
    valid_results = [r for r in all_results if not r.get("coi_filtered")]
    filtered_results = [r for r in all_results if r.get("coi_filtered")]

    # Summary table
    print("\n" + "=" * 100)
    print("RESULTS SUMMARY")
    print("=" * 100)

    print(f"\n{'Dialect':<25} {'Multi-VALUE':^22} {'Basic Transform':^22} {'CoI Transform':^22}")
    print(f"{'':25} {'eWAVE':>7} {'Err':>5} {'Valid':>5} {'eWAVE':>7} {'Err':>5} {'Valid':>5} {'eWAVE':>7} {'Err':>5} {'Valid':>5}")
    print("-" * 100)

    # Aggregate by dialect
    dialect_stats = {}
    for r in valid_results:
        d = r["dialect"]
        if d not in dialect_stats:
            dialect_stats[d] = {
                "mv_ewave": 0, "mv_err": 0, "mv_valid": 0,
                "basic_ewave": 0, "basic_err": 0, "basic_valid": 0,
                "coi_ewave": 0, "coi_err": 0, "coi_valid": 0,
                "count": 0
            }

        s = dialect_stats[d]
        s["mv_ewave"] += r["mv_validation"].ewave_accuracy
        s["mv_err"] += len(r["mv_validation"].non_ewave_errors)
        s["mv_valid"] += len(r["mv_validation"].valid_changes)
        s["basic_ewave"] += r["basic_validation"].ewave_accuracy
        s["basic_err"] += len(r["basic_validation"].non_ewave_errors)
        s["basic_valid"] += len(r["basic_validation"].valid_changes)
        s["coi_ewave"] += r["coi_validation"].ewave_accuracy
        s["coi_err"] += len(r["coi_validation"].non_ewave_errors)
        s["coi_valid"] += len(r["coi_validation"].valid_changes)
        s["count"] += 1

    # Print per-dialect averages
    total_mv_ewave = 0
    total_basic_ewave = 0
    total_coi_ewave = 0
    total_mv_err = 0
    total_basic_err = 0
    total_coi_err = 0
    total_mv_valid = 0
    total_basic_valid = 0
    total_coi_valid = 0
    total_count = 0

    for dialect, s in dialect_stats.items():
        n = s["count"]
        if n == 0:
            continue

        print(f"{dialect:<25} {s['mv_ewave']/n:>6.0%} {s['mv_err']:>5} {s['mv_valid']:>5} "
              f"{s['basic_ewave']/n:>6.0%} {s['basic_err']:>5} {s['basic_valid']:>5} "
              f"{s['coi_ewave']/n:>6.0%} {s['coi_err']:>5} {s['coi_valid']:>5}")

        total_mv_ewave += s["mv_ewave"]
        total_basic_ewave += s["basic_ewave"]
        total_coi_ewave += s["coi_ewave"]
        total_mv_err += s["mv_err"]
        total_basic_err += s["basic_err"]
        total_coi_err += s["coi_err"]
        total_mv_valid += s["mv_valid"]
        total_basic_valid += s["basic_valid"]
        total_coi_valid += s["coi_valid"]
        total_count += n

    print("-" * 100)
    if total_count > 0:
        print(f"{'AVERAGE':<25} {total_mv_ewave/total_count:>6.0%} {total_mv_err:>5} {total_mv_valid:>5} "
              f"{total_basic_ewave/total_count:>6.0%} {total_basic_err:>5} {total_basic_valid:>5} "
              f"{total_coi_ewave/total_count:>6.0%} {total_coi_err:>5} {total_coi_valid:>5}")

    # Winner analysis
    mv_wins = 0
    basic_wins = 0
    coi_wins = 0
    ties = 0

    for r in valid_results:
        mv_score = (r["mv_validation"].ewave_accuracy, -len(r["mv_validation"].non_ewave_errors))
        basic_score = (r["basic_validation"].ewave_accuracy, -len(r["basic_validation"].non_ewave_errors))
        coi_score = (r["coi_validation"].ewave_accuracy, -len(r["coi_validation"].non_ewave_errors))

        scores = [("Multi-VALUE", mv_score), ("Basic", basic_score), ("CoI", coi_score)]
        scores.sort(key=lambda x: x[1], reverse=True)

        if scores[0][1] == scores[1][1] == scores[2][1]:
            ties += 1
        elif scores[0][0] == "Multi-VALUE":
            mv_wins += 1
        elif scores[0][0] == "Basic":
            basic_wins += 1
        else:
            coi_wins += 1

    # Final summary
    print("\n" + "=" * 100)
    print("FINAL COMPARISON")
    print("=" * 100)

    print(f"\nSamples tested: {total_count}")
    print(f"Samples filtered (Azure content policy): {len(filtered_results)}")

    if total_count > 0:
        print(f"\nWINS (by eWAVE accuracy, then fewer errors):")
        print(f"  Multi-VALUE:     {mv_wins:>3} ({mv_wins/total_count:.0%})")
        print(f"  Basic Transform: {basic_wins:>3} ({basic_wins/total_count:.0%})")
        print(f"  CoI Transform:   {coi_wins:>3} ({coi_wins/total_count:.0%})")
        print(f"  Ties:            {ties:>3} ({ties/total_count:.0%})")

        print(f"\nAVERAGE METRICS:")
        print(f"  Multi-VALUE:     eWAVE={total_mv_ewave/total_count:.0%}, Errors={total_mv_err}, Valid={total_mv_valid}")
        print(f"  Basic Transform: eWAVE={total_basic_ewave/total_count:.0%}, Errors={total_basic_err}, Valid={total_basic_valid}")
        print(f"  CoI Transform:   eWAVE={total_coi_ewave/total_count:.0%}, Errors={total_coi_err}, Valid={total_coi_valid}")

    print("\n" + "=" * 100)
    print("BENCHMARK COMPLETED")
    print("=" * 100)

    return 0


if __name__ == "__main__":
    sys.exit(main())
