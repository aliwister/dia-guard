#!/usr/bin/env python3
"""
Benchmark comparison: Multi-VALUE vs CoI Transform (DIA-LLM)

Compares transformation quality on real datasets using eWAVE validation.
"""

import sys
import os
import csv
import random
from typing import List, Dict, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import AzureOpenAIBackend
from coi_transformation import CoIDialectTransformer
from feature_validator import LLMComprehensiveValidator


def create_llm():
    """Create Azure OpenAI backend."""
    return AzureOpenAIBackend(
        deployment_name="gpt-4.1",
        endpoint="https://jsl-diaguard.openai.azure.com",
        api_version="2024-02-15-preview",
        api_key="YOUR_AZURE_OPENAI_API_KEY"
    )


def load_samples(csv_path: str, n_samples: int = 5) -> List[Dict]:
    """Load random samples from dataset."""
    samples = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Random sample
    if len(rows) > n_samples:
        rows = random.sample(rows, n_samples)

    for row in rows:
        samples.append({
            "original": row.get("Behavior", ""),
            "multivalue": row.get("Behavior_transformed", "")
        })

    return samples


def run_comparison(
    samples: List[Dict],
    dialect_key: str,
    dialect_name: str,
    llm,
    rating_level: str = "A"
):
    """Run comparison between Multi-VALUE and CoI Transform."""
    print("\n" + "=" * 80)
    print(f"BENCHMARK: {dialect_name}")
    print(f"Rating Level: {rating_level} | Samples: {len(samples)}")
    print("=" * 80)

    transformer = CoIDialectTransformer(llm=llm, verbose=False, rating_level=rating_level)
    validator = LLMComprehensiveValidator(
        llm, dialect_key, verbose=False,
        use_explicit_ratings=True,
        use_post_correction=True
    )

    results = []

    for i, sample in enumerate(samples, 1):
        original = sample["original"]
        multivalue = sample["multivalue"]

        print(f"\n{'─' * 60}")
        print(f"Sample {i}/{len(samples)}")
        print(f"{'─' * 60}")
        print(f"Original: {original[:80]}{'...' if len(original) > 80 else ''}")
        print(f"Multi-VALUE: {multivalue[:80]}{'...' if len(multivalue) > 80 else ''}")

        # Generate CoI transform
        try:
            coi_result = transformer.transform(original, dialect_key, skip_validation=True)
            coi_output = coi_result.final_output
        except Exception as e:
            print(f"  CoI Transform Error: {e}")
            coi_output = original  # Fallback

        print(f"CoI Transform: {coi_output[:80]}{'...' if len(coi_output) > 80 else ''}")

        # Validate both
        print("\n  Validating...")

        try:
            mv_validation = validator.validate(original, multivalue)
            coi_validation = validator.validate(original, coi_output)

            print(f"  Multi-VALUE: eWAVE={mv_validation.ewave_accuracy:.0%}, Dialect={mv_validation.dialect_accuracy:.0%}, "
                  f"Valid={len(mv_validation.valid_changes)}, Wrong={len(mv_validation.wrong_dialect_changes)}, Error={len(mv_validation.non_ewave_errors)}")
            print(f"  CoI Transform: eWAVE={coi_validation.ewave_accuracy:.0%}, Dialect={coi_validation.dialect_accuracy:.0%}, "
                  f"Valid={len(coi_validation.valid_changes)}, Wrong={len(coi_validation.wrong_dialect_changes)}, Error={len(coi_validation.non_ewave_errors)}")

            results.append({
                "original": original,
                "multivalue": multivalue,
                "coi": coi_output,
                "mv_validation": mv_validation,
                "coi_validation": coi_validation
            })
        except Exception as e:
            print(f"  Validation Error: {e}")

    return results


def print_summary(all_results: Dict[str, List]):
    """Print aggregate summary."""
    print("\n" + "=" * 80)
    print("AGGREGATE SUMMARY")
    print("=" * 80)

    print(f"\n{'Dialect':<30} {'Method':<15} {'eWAVE':>8} {'Dialect':>8} {'Valid':>6} {'Wrong':>6} {'Error':>6}")
    print("-" * 85)

    totals = {
        "Multi-VALUE": {"ewave": 0, "dialect": 0, "valid": 0, "wrong": 0, "error": 0, "count": 0},
        "CoI Transform": {"ewave": 0, "dialect": 0, "valid": 0, "wrong": 0, "error": 0, "count": 0}
    }

    for dialect_name, results in all_results.items():
        if not results:
            continue

        # Aggregate for this dialect
        mv_ewave = sum(r["mv_validation"].ewave_accuracy for r in results) / len(results)
        mv_dialect = sum(r["mv_validation"].dialect_accuracy for r in results) / len(results)
        mv_valid = sum(len(r["mv_validation"].valid_changes) for r in results)
        mv_wrong = sum(len(r["mv_validation"].wrong_dialect_changes) for r in results)
        mv_error = sum(len(r["mv_validation"].non_ewave_errors) for r in results)

        coi_ewave = sum(r["coi_validation"].ewave_accuracy for r in results) / len(results)
        coi_dialect = sum(r["coi_validation"].dialect_accuracy for r in results) / len(results)
        coi_valid = sum(len(r["coi_validation"].valid_changes) for r in results)
        coi_wrong = sum(len(r["coi_validation"].wrong_dialect_changes) for r in results)
        coi_error = sum(len(r["coi_validation"].non_ewave_errors) for r in results)

        print(f"{dialect_name:<30} {'Multi-VALUE':<15} {mv_ewave:>7.0%} {mv_dialect:>7.0%} {mv_valid:>6} {mv_wrong:>6} {mv_error:>6}")
        print(f"{'':<30} {'CoI Transform':<15} {coi_ewave:>7.0%} {coi_dialect:>7.0%} {coi_valid:>6} {coi_wrong:>6} {coi_error:>6}")
        print()

        # Add to totals
        totals["Multi-VALUE"]["ewave"] += mv_ewave * len(results)
        totals["Multi-VALUE"]["dialect"] += mv_dialect * len(results)
        totals["Multi-VALUE"]["valid"] += mv_valid
        totals["Multi-VALUE"]["wrong"] += mv_wrong
        totals["Multi-VALUE"]["error"] += mv_error
        totals["Multi-VALUE"]["count"] += len(results)

        totals["CoI Transform"]["ewave"] += coi_ewave * len(results)
        totals["CoI Transform"]["dialect"] += coi_dialect * len(results)
        totals["CoI Transform"]["valid"] += coi_valid
        totals["CoI Transform"]["wrong"] += coi_wrong
        totals["CoI Transform"]["error"] += coi_error
        totals["CoI Transform"]["count"] += len(results)

    print("-" * 85)
    print("OVERALL AVERAGE:")
    for method in ["Multi-VALUE", "CoI Transform"]:
        t = totals[method]
        if t["count"] > 0:
            print(f"  {method:<15}: eWAVE={t['ewave']/t['count']:.0%}, Dialect={t['dialect']/t['count']:.0%}, "
                  f"Valid={t['valid']}, Wrong={t['wrong']}, Error={t['error']}")


def main():
    """Run benchmark comparison."""
    print("=" * 80)
    print("MULTI-VALUE vs COI TRANSFORM BENCHMARK")
    print("Using AB-rated eWAVE features (pervasive + common)")
    print("=" * 80)

    # Set random seed for reproducibility
    random.seed(42)

    llm = create_llm()

    # Dataset paths
    datasets = [
        {
            "path": "/Users/jsl/Downloads/outputs/harm_aboriginal_english.csv",
            "dialect_key": "aboriginal",
            "dialect_name": "Aboriginal English"
        },
        {
            "path": "/Users/jsl/Downloads/outputs/urban_african_american_vernacular_english/harmBench_urban_african_american_vernacular_english.csv",
            "dialect_key": "urban_aave",
            "dialect_name": "Urban AAVE"
        }
    ]

    n_samples = 5  # Samples per dialect
    rating_level = "AB"  # Use A+B rated features (pervasive + common)

    all_results = {}

    for ds in datasets:
        print(f"\nLoading samples from: {ds['path']}")
        samples = load_samples(ds["path"], n_samples)

        results = run_comparison(
            samples=samples,
            dialect_key=ds["dialect_key"],
            dialect_name=ds["dialect_name"],
            llm=llm,
            rating_level=rating_level
        )

        all_results[ds["dialect_name"]] = results

    # Print summary
    print_summary(all_results)

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETED")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
