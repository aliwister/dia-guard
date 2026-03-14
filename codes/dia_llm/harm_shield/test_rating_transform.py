#!/usr/bin/env python3
"""
Test CoI transformation with rating-based feature selection.

Tests dialect transformation using eWAVE A-rated (pervasive) features only.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import AzureOpenAIBackend
from coi_transformation import CoIDialectTransformer, coi_transform
from feature_validator import LLMComprehensiveValidator


def create_llm():
    """Create Azure OpenAI backend."""
    return AzureOpenAIBackend(
        deployment_name="gpt-4.1",
        endpoint="https://jsl-diaguard.openai.azure.com",
        api_version="2024-02-15-preview",
        api_key="YOUR_AZURE_OPENAI_API_KEY"
    )


def test_transform_with_rating(
    text: str,
    dialect_key: str,
    dialect_name: str,
    rating_level: str,
    llm
):
    """Transform text using rating-based features and validate."""
    print("\n" + "=" * 80)
    print(f"DIALECT: {dialect_name}")
    print(f"RATING LEVEL: {rating_level}")
    print("=" * 80)
    print(f"\nOriginal: {text}")

    # Transform using rating-based features
    transformer = CoIDialectTransformer(llm=llm, verbose=False, rating_level=rating_level)
    result = transformer.transform(text, dialect_key, skip_validation=True)

    print(f"Transformed: {result.final_output}")
    print(f"Features loaded: {len(result.features_applied)} applied")

    # Validate the transformation
    print("\n" + "-" * 60)
    print("Validating transformation...")
    print("-" * 60)

    validator = LLMComprehensiveValidator(
        llm,
        dialect_key,
        verbose=False,
        use_explicit_ratings=True,
        use_post_correction=True
    )

    validation = validator.validate(text, result.final_output)

    print(f"\neWAVE Accuracy: {validation.ewave_accuracy:.0%}")
    print(f"Dialect Accuracy: {validation.dialect_accuracy:.0%}")
    print(f"Valid: {len(validation.valid_changes)} | Wrong: {len(validation.wrong_dialect_changes)} | Error: {len(validation.non_ewave_errors)}")

    if validation.valid_changes:
        print("\n  ✓ VALID (eWAVE + A/B/C):")
        for c in validation.valid_changes:
            rating = f"[{c.dialect_rating}]" if c.dialect_rating else "[?]"
            print(f"    ✓ F{c.ewave_feature_id}: {c.ewave_feature_key} {rating}")

    if validation.wrong_dialect_changes:
        print("\n  △ WRONG DIALECT (eWAVE + D):")
        for c in validation.wrong_dialect_changes:
            rating = f"[{c.dialect_rating}]" if c.dialect_rating else "[D]"
            print(f"    △ F{c.ewave_feature_id}: {c.ewave_feature_key} {rating}")

    if validation.non_ewave_errors:
        print("\n  ✗ NON-eWAVE ERROR:")
        for c in validation.non_ewave_errors:
            error_label = c.error_type if c.error_type else "unknown"
            print(f"    ✗ [{error_label}] {c.change_description}")

    return {
        "original": text,
        "transformed": result.final_output,
        "validation": validation
    }


def main():
    """Test transformations with A-rated features."""
    print("=" * 80)
    print("COI TRANSFORMATION WITH RATING-BASED FEATURES")
    print("Using A-rated (pervasive/obligatory) features only")
    print("=" * 80)

    llm = create_llm()

    # Test cases
    test_cases = [
        {
            "text": "She is always working hard.",
            "dialect_key": "urban_aave",
            "dialect_name": "Urban AAVE"
        },
        {
            "text": "He doesn't know anything about that.",
            "dialect_key": "urban_aave",
            "dialect_name": "Urban AAVE"
        },
        {
            "text": "Where did you put the book?",
            "dialect_key": "aboriginal",
            "dialect_name": "Aboriginal English"
        },
    ]

    results = []
    for tc in test_cases:
        result = test_transform_with_rating(
            text=tc["text"],
            dialect_key=tc["dialect_key"],
            dialect_name=tc["dialect_name"],
            rating_level="A",
            llm=llm
        )
        results.append((tc["dialect_name"], result))

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\n{'Test Case':<40} {'eWAVE':>8} {'Dialect':>8} {'Valid':>6} {'Wrong':>6} {'Error':>6}")
    print("-" * 80)

    for name, result in results:
        v = result["validation"]
        print(f"{name:<40} {v.ewave_accuracy:>7.0%} {v.dialect_accuracy:>7.0%} "
              f"{len(v.valid_changes):>6} {len(v.wrong_dialect_changes):>6} "
              f"{len(v.non_ewave_errors):>6}")

    print("-" * 80)
    print("\n" + "=" * 80)
    print("TEST COMPLETED")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
