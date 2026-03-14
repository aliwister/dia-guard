#!/usr/bin/env python3
"""
Batch COI Dialect Transformation Script
Transforms text in CSV files to various English dialects using CoIDialectTransformer
"""

import os
import sys
import pandas as pd
from pathlib import Path
from typing import Optional, Dict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from models import AzureOpenAIBackend
from coi_transformation import CoIDialectTransformer
from dialects import DIALECT_REGISTRY

# Mapping from folder names to dialect registry keys
FOLDER_TO_DIALECT_KEY = {
    "aboriginal_english": "aboriginal",
    "acrolectal_fiji_english": "fiji",  # closest match
    "appalachian_english": "appalachian",
    "australian_english": "australian",
    "australian_vernacular_english": "australian_vernacular",
    "bahamian_english": "bahamian",
    "black_south_african_english": "south_african_black",
    "cameroon_english": "cameroon",
    "cape_flats_english": "cape_flats",
    "channel_islands_english": "channel_islands",
    "chicano_english": "chicano",
    "colloquial_american_english": "colloquial_american",
    "colloquial_singapore_english_singlish": "singapore",
    "earlier_african_american_vernacular_english": "earlier_aave",
    "east_anglian_english": "east_anglian",
    "english_dialects_in_the_north_of_england": "northern_england",
    "english_dialects_in_the_southeast_of_england": "southeast_england",
    "english_dialects_in_the_southwest_of_england": "southwest_england",
    "falkland_islands_english": "falkland",
    "ghanaian_english": "ghanaian",
    "hong_kong_english": "hong_kong",
    "indian_english": "indian",
    "indian_south_african_english": "south_african_indian",
    "irish_english": "irish",
    "jamaican_english": "jamaican",
    "kenyan_english": "kenyan",
    "liberian_settler_english": "liberian_settler",
    "malaysian_english": "malaysian",
    "maltese_english": "maltese_english",
    "manx_english": "manx",
    "new_zealand_english": "new_zealand",
    "newfoundland_english": "newfoundland",
    "nigerian_english": "nigerian",
    "orkney_and_shetland_english": "orkney_shetland",
    "ozark_english": "ozark",
    "pakistani_english": "pakistani",
    "philippine_english": "philippine",
    "pure_fiji_english_basilectal_fijie": "fiji",
    "rural_african_american_vernacular_english": "rural_aave",
    "scottish_english": "scottish",
    "southeast_american_enclave_dialects": "southeast_enclave",
    "sri_lankan_english": "sri_lankan",
    "st_helena_english": "st_helena",
    "tanzanian_english": "tanzanian",
    "tristan_da_cunha_english": "tristan",
    "ugandan_english": "ugandan",
    "urban_african_american_vernacular_english": "urban_aave",
    "welsh_english": "welsh",
}


def get_dialect_key(folder_name: str) -> Optional[str]:
    """Convert folder name to dialect registry key."""
    return FOLDER_TO_DIALECT_KEY.get(folder_name)


def create_azure_llm():
    """Create Azure OpenAI backend."""
    return AzureOpenAIBackend(
        deployment_name="gpt-4.1",
        endpoint="https://jsl-diaguard.openai.azure.com",
        api_version="2024-02-15-preview",
        api_key="YOUR_AZURE_OPENAI_API_KEY"
    )


def transform_csv(
    csv_path: str,
    dialect_key: str,
    source_column: str = "prompt",
    output_column: str = "coi_transformed",
    sample_limit: Optional[int] = None,
    save_output: bool = True
) -> pd.DataFrame:
    """
    Transform a CSV file using CoI dialect transformation.

    Args:
        csv_path: Path to the CSV file
        dialect_key: Dialect key from DIALECT_REGISTRY
        source_column: Column containing text to transform
        output_column: Column name for transformed output
        sample_limit: Limit number of rows to transform (None for all)
        save_output: Whether to save the transformed CSV

    Returns:
        DataFrame with transformed column
    """
    # Validate dialect key
    if dialect_key not in DIALECT_REGISTRY:
        raise ValueError(f"Unknown dialect: {dialect_key}")

    # Read CSV
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")

    if sample_limit:
        df = df.head(sample_limit)
        print(f"Processing first {sample_limit} sample(s)")

    # Initialize transformer
    llm = create_azure_llm()
    transformer = CoIDialectTransformer(llm=llm)

    # Transform each row
    transformed_texts = []
    for idx, row in df.iterrows():
        text = row[source_column]
        print(f"\n[{idx + 1}/{len(df)}] Transforming: {text[:50]}...")

        try:
            result = transformer.transform(text, dialect=dialect_key)
            # Extract just the transformed text (before verification info)
            raw_output = result.final_output
            # The output may contain verification text after double newlines
            if "\n\nDUAL ATTENTION" in raw_output:
                transformed = raw_output.split("\n\nDUAL ATTENTION")[0].strip()
            elif "\n\nFINAL FEATURES" in raw_output:
                transformed = raw_output.split("\n\nFINAL FEATURES")[0].strip()
            else:
                transformed = raw_output.strip()
            print(f"  -> {transformed[:80]}...")
            transformed_texts.append(transformed)
        except Exception as e:
            print(f"  ERROR: {e}")
            transformed_texts.append(f"ERROR: {e}")

    # Add to dataframe
    df[output_column] = transformed_texts

    # Save if requested
    if save_output:
        df.to_csv(csv_path, index=False)
        print(f"\nSaved to {csv_path}")

    return df


def process_folder(
    base_path: str,
    source_column: str = "prompt",
    output_column: str = "coi_transformed",
    sample_limit: Optional[int] = None
):
    """
    Process all CSV files in a folder structure.

    Args:
        base_path: Base path containing dialect folders
        source_column: Column containing text to transform
        output_column: Column name for transformed output
        sample_limit: Limit number of rows per file (None for all)
    """
    base = Path(base_path)

    for dialect_folder in base.iterdir():
        if not dialect_folder.is_dir() or dialect_folder.name.startswith('.'):
            continue

        dialect_key = get_dialect_key(dialect_folder.name)
        if not dialect_key:
            print(f"WARNING: No dialect mapping for folder: {dialect_folder.name}")
            continue

        print(f"\n{'='*60}")
        print(f"Processing dialect: {dialect_folder.name} -> {dialect_key}")
        print(f"{'='*60}")

        for csv_file in dialect_folder.glob("*.csv"):
            print(f"\n--- {csv_file.name} ---")
            try:
                transform_csv(
                    str(csv_file),
                    dialect_key,
                    source_column=source_column,
                    output_column=output_column,
                    sample_limit=sample_limit
                )
            except Exception as e:
                print(f"ERROR processing {csv_file}: {e}")


if __name__ == "__main__":
    # Default path
    BASE_PATH = "/Users/jsl/Library/CloudStorage/OneDrive-ThePennsylvaniaStateUniversity/JasonL Research Projects/DIA-Guard/data/outputs"

    # Test with 1 sample first
    print("=" * 60)
    print("TESTING: Running on 1 sample only")
    print("=" * 60)

    # Test on a single file
    test_file = f"{BASE_PATH}/aboriginal_english/forbiddent_questions_aboriginal_english.csv"

    df = transform_csv(
        test_file,
        dialect_key="aboriginal",
        sample_limit=1,
        save_output=False  # Don't save during test
    )

    print("\n" + "=" * 60)
    print("TEST RESULT:")
    print("=" * 60)
    print(f"Original: {df['prompt'].iloc[0]}")
    print(f"Existing transform: {df['prompt_transformed'].iloc[0]}")
    print(f"COI Transform: {df['coi_transformed'].iloc[0]}")
