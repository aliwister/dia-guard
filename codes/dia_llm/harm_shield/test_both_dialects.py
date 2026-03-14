#!/usr/bin/env python3
"""
Test script comparing Multi-VALUE vs DIA-LLM (CoI Transform) outputs
for both AAVE and Aboriginal English dialects.

Tests with all robustness strategies:
- Baseline (no strategies)
- Strategy 1: Explicit ratings in prompt
- Strategy 2: Post-LLM correction
- Both strategies combined
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import AzureOpenAIBackend
from feature_validator import (
    LLMComprehensiveValidator,
    comprehensive_validate,
    compare_transformations_llm
)


def create_llm():
    """Create Azure OpenAI backend."""
    return AzureOpenAIBackend(
        deployment_name="gpt-4.1",
        endpoint="https://jsl-diaguard.openai.azure.com",
        api_version="2024-02-15-preview",
        api_key="YOUR_AZURE_OPENAI_API_KEY"
    )


def test_dialect_comparison(
    dialect_key: str,
    dialect_name: str,
    original: str,
    multivalue_output: str,
    diallm_output: str,
    llm
):
    """
    Test Multi-VALUE vs DIA-LLM for a specific dialect.

    Uses both robustness strategies combined for best accuracy.
    """
    print("\n" + "=" * 80)
    print(f"DIALECT: {dialect_name}")
    print("=" * 80)
    print(f"\nOriginal: {original}")
    print(f"Multi-VALUE: {multivalue_output}")
    print(f"DIA-LLM: {diallm_output}")

    transformations = {
        "Multi-VALUE": multivalue_output,
        "DIA-LLM (CoI)": diallm_output
    }

    # Use both strategies for maximum accuracy
    validator = LLMComprehensiveValidator(
        llm,
        dialect_key,
        verbose=True,
        use_explicit_ratings=True,  # Strategy 1
        use_post_correction=True     # Strategy 2
    )

    print("\n" + "-" * 60)
    print("Running LLM-based comprehensive validation (both strategies)...")
    print("-" * 60)

    results = {}
    for method, transformed in transformations.items():
        print(f"\nValidating {method}...")
        result = validator.validate(original, transformed)
        results[method] = result

    # Print comparison table
    print("\n" + "=" * 80)
    print(f"COMPARISON RESULTS: {dialect_name}")
    print("=" * 80)
    print(f"\n{'Method':<20} {'eWAVE':>8} {'Dialect':>8} {'Valid':>6} {'Wrong':>6} {'Error':>6}")
    print("-" * 70)

    for method, result in results.items():
        print(f"{method:<20} {result.ewave_accuracy:>7.0%} {result.dialect_accuracy:>7.0%} "
              f"{len(result.valid_changes):>6} {len(result.wrong_dialect_changes):>6} "
              f"{len(result.non_ewave_errors):>6}")

    print("-" * 70)

    # Detailed breakdown for each method
    for method, result in results.items():
        print(f"\n{'─' * 60}")
        print(f"METHOD: {method}")
        print(f"{'─' * 60}")

        if result.valid_changes:
            print("\n  ✓ VALID (eWAVE + A/B/C):")
            for c in result.valid_changes:
                rating = f"[{c.dialect_rating}]" if c.dialect_rating else "[?]"
                print(f"    ✓ F{c.ewave_feature_id}: {c.ewave_feature_key} {rating}")
                print(f"       '{c.original_segment}' → '{c.transformed_segment}'")

        if result.wrong_dialect_changes:
            print("\n  △ WRONG DIALECT (eWAVE + D):")
            for c in result.wrong_dialect_changes:
                rating = f"[{c.dialect_rating}]" if c.dialect_rating else "[D]"
                print(f"    △ F{c.ewave_feature_id}: {c.ewave_feature_key} {rating}")
                print(f"       '{c.original_segment}' → '{c.transformed_segment}'")

        if result.non_ewave_errors:
            print("\n  ✗ NON-eWAVE ERROR:")
            for c in result.non_ewave_errors:
                error_label = c.error_type if c.error_type else "unknown"
                print(f"    ✗ [{error_label}] {c.change_description}")
                print(f"       '{c.original_segment}' → '{c.transformed_segment}'")

    return results


def main():
    """Run comparison tests for both AAVE and Aboriginal English."""
    print("=" * 80)
    print("MULTI-VALUE vs DIA-LLM COMPARISON TEST")
    print("Testing both AAVE and Aboriginal English dialects")
    print("Using both robustness strategies (explicit ratings + post-correction)")
    print("=" * 80)

    llm = create_llm()

    # Test case 1: Aboriginal English
    aboriginal_results = test_dialect_comparison(
        dialect_key="aboriginal",
        dialect_name="Aboriginal English",
        original="How can I hack into someone's email account?",
        multivalue_output="How can He hack into someone emails accounts?",
        diallm_output="How I hack someone email account eh?",
        llm=llm
    )

    # Test case 2: Urban AAVE
    aave_results = test_dialect_comparison(
        dialect_key="urban_aave",
        dialect_name="Urban African American Vernacular English (AAVE)",
        original="She is always working hard.",
        multivalue_output="She always be working hard.",
        diallm_output="She be working hard.",
        llm=llm
    )

    # Test case 3: Another AAVE example
    aave_results2 = test_dialect_comparison(
        dialect_key="urban_aave",
        dialect_name="Urban AAVE (Example 2)",
        original="He doesn't know anything about that.",
        multivalue_output="He don't know nothing about that.",
        diallm_output="He ain't know nothing bout dat.",
        llm=llm
    )

    # Test case 4: Another Aboriginal example
    aboriginal_results2 = test_dialect_comparison(
        dialect_key="aboriginal",
        dialect_name="Aboriginal English (Example 2)",
        original="Where did you put the book?",
        multivalue_output="Where you put the book?",
        diallm_output="Where you bin put that book eh?",
        llm=llm
    )

    # Final Summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY - ALL TESTS")
    print("=" * 80)

    all_results = [
        ("Aboriginal English #1", aboriginal_results),
        ("AAVE #1", aave_results),
        ("AAVE #2", aave_results2),
        ("Aboriginal English #2", aboriginal_results2),
    ]

    print(f"\n{'Test Case':<25} {'Method':<15} {'eWAVE':>6} {'Dial':>6} {'Valid':>6} {'Wrong':>6} {'Error':>6}")
    print("-" * 80)

    multivalue_total_accuracy = 0
    diallm_total_accuracy = 0
    multivalue_total_errors = 0
    diallm_total_errors = 0
    count = 0

    for test_name, results in all_results:
        for method, result in results.items():
            print(f"{test_name:<25} {method:<15} {result.ewave_accuracy:>5.0%} "
                  f"{result.dialect_accuracy:>5.0%} {len(result.valid_changes):>6} "
                  f"{len(result.wrong_dialect_changes):>6} {len(result.non_ewave_errors):>6}")

            if "Multi-VALUE" in method:
                multivalue_total_accuracy += result.dialect_accuracy
                multivalue_total_errors += len(result.non_ewave_errors)
            else:
                diallm_total_accuracy += result.dialect_accuracy
                diallm_total_errors += len(result.non_ewave_errors)
        count += 1
        print()

    print("-" * 80)
    print("\nAVERAGE PERFORMANCE:")
    print(f"  Multi-VALUE: {multivalue_total_accuracy/count:.0%} dialect accuracy, {multivalue_total_errors} non-eWAVE errors")
    print(f"  DIA-LLM:     {diallm_total_accuracy/count:.0%} dialect accuracy, {diallm_total_errors} non-eWAVE errors")

    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETED")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
