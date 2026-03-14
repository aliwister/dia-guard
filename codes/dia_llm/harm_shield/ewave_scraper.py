#!/usr/bin/env python3
"""
eWAVE Feature Scraper

Scrapes feature ratings from the eWAVE (Electronic World Atlas of Varieties of English) database.
Uses the DataTables AJAX endpoint to get JSON data.

Usage:
    python ewave_scraper.py --language 61  # Aboriginal English
    python ewave_scraper.py --language 61 --output aboriginal_features.json
    python ewave_scraper.py --list-languages
    python ewave_scraper.py --all --output-dir ewave_data/  # Scrape all dialects
"""

import argparse
import json
import re
import requests
import time
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime


# Complete eWAVE language IDs (77 varieties)
# Source: https://ewave-atlas.org/languages.geojson
EWAVE_LANGUAGES = {
    # British Isles
    1: "Orkney and Shetland English",
    2: "Scottish English",
    3: "Irish English",
    4: "Manx English",
    5: "Welsh English",
    6: "English dialects in the North of England",
    7: "English dialects in the Southwest of England",
    8: "East Anglian English",
    9: "English dialects in the Southeast of England",
    10: "Channel Islands English",
    11: "British Creole",
    12: "Maltese English",

    # North America
    13: "Newfoundland English",
    14: "Colloquial American English",
    15: "Urban African American Vernacular English",
    16: "Rural African American Vernacular English",
    17: "Earlier African American Vernacular English",
    18: "Ozark English",
    19: "Appalachian English",
    20: "Southeast American enclave dialects",
    21: "Gullah",
    22: "Chicano English",

    # Caribbean
    23: "Bahamian English",
    24: "Bahamian Creole",
    25: "Barbadian Creole (Bajan)",
    26: "Jamaican English",
    27: "Jamaican Creole",
    28: "San Andrés Creole",
    29: "Belizean Creole",
    30: "Guyanese Creole (Creolese)",
    31: "Eastern Maroon Creole",
    32: "Saramaccan",
    33: "Sranan",
    34: "Trinidadian Creole",
    35: "Vincentian Creole",

    # West Africa
    36: "Krio (Sierra Leone Creole)",
    37: "Liberian Settler English",
    38: "Vernacular Liberian English",
    39: "Ghanaian English",
    40: "Ghanaian Pidgin",
    41: "Nigerian English",
    42: "Nigerian Pidgin",
    43: "Cameroon English",
    44: "Cameroon Pidgin",

    # East/South Africa
    45: "Tanzanian English",
    46: "Kenyan English",
    47: "Ugandan English",
    48: "White Zimbabwean English",
    49: "Black South African English",
    50: "Indian South African English",
    51: "White South African English",

    # South/Southeast Asia
    52: "Indian English",
    53: "Pakistani English",
    54: "Butler English",
    55: "Sri Lankan English",
    56: "Hong Kong English",
    57: "Colloquial Singapore English (Singlish)",
    58: "Malaysian English",

    # Australia/New Zealand
    59: "Australian English",
    60: "Australian Vernacular English",
    61: "Aboriginal English",
    62: "Roper River Creole (Kriol)",
    63: "Torres Strait Creole",
    64: "New Zealand English",

    # Pacific
    65: "Norfolk Island/Pitcairn English",
    66: "Palmerston English",
    67: "Acrolectal Fiji English",
    68: "Pure Fiji English (basilectal FijiE)",
    69: "Bislama",
    70: "Tok Pisin",
    71: "Hawai'i Creole",

    # Atlantic Islands
    72: "Falkland Islands English",
    73: "St. Helena English",
    74: "Tristan da Cunha English",

    # Additional
    75: "Philippine English",
    76: "Cape Flats English",
    77: "Croker Island English",
}

# Mapping from our dialect keys to eWAVE IDs
DIALECT_TO_EWAVE_ID = {
    # British Isles
    "orkney_shetland": 1,
    "scottish": 2,
    "irish": 3,
    "manx": 4,
    "welsh": 5,
    "northern_england": 6,
    "southwest_england": 7,
    "east_anglian": 8,
    "southeast_england": 9,
    "channel_islands": 10,
    "british_creole": 11,
    "maltese_english": 12,

    # North America
    "newfoundland": 13,
    "colloquial_american": 14,
    "urban_aave": 15,
    "rural_aave": 16,
    "earlier_aave": 17,
    "ozark": 18,
    "appalachian": 19,
    "southeast_enclave": 20,
    "gullah": 21,
    "chicano": 22,

    # Caribbean
    "bahamian": 23,
    "bahamian_creole": 24,
    "barbadian": 25,
    "jamaican_english": 26,
    "jamaican": 27,  # Jamaican Creole
    "san_andres": 28,
    "belizean": 29,
    "guyanese": 30,
    "eastern_maroon": 31,
    "saramaccan": 32,
    "sranan": 33,
    "trinidadian": 34,
    "vincentian": 35,

    # West Africa
    "krio": 36,
    "liberian_settler": 37,
    "liberian_vernacular": 38,
    "ghanaian": 39,
    "ghanaian_pidgin": 40,
    "nigerian": 41,
    "nigerian_pidgin": 42,
    "cameroon": 43,
    "cameroon_pidgin": 44,

    # East/South Africa
    "tanzanian": 45,
    "kenyan": 46,
    "ugandan": 47,
    "zimbabwean_white": 48,
    "south_african_black": 49,
    "south_african_indian": 50,
    "south_african_white": 51,

    # South/Southeast Asia
    "indian": 52,
    "pakistani": 53,
    "butler_english": 54,
    "sri_lankan": 55,
    "hong_kong": 56,
    "singapore": 57,
    "malaysian": 58,

    # Australia/New Zealand
    "australian": 59,
    "australian_vernacular": 60,
    "aboriginal": 61,
    "kriol": 62,
    "torres_strait": 63,
    "new_zealand": 64,

    # Pacific
    "pitcairn_norfolk": 65,
    "palmerston": 66,
    "fiji_acrolectal": 67,
    "fiji": 68,
    "bislama": 69,
    "tok_pisin": 70,
    "hawaii_creole": 71,

    # Atlantic Islands
    "falkland": 72,
    "st_helena": 73,
    "tristan": 74,

    # Additional
    "philippine": 75,
    "cape_flats": 76,
    "croker_island": 77,
}


def fetch_ewave_features(language_id: int, verbose: bool = False) -> List[Dict]:
    """
    Fetch all 235 features for a given language from eWAVE.

    Args:
        language_id: eWAVE language ID (e.g., 61 for Aboriginal English)
        verbose: Print progress

    Returns:
        List of feature dictionaries with id, name, and rating
    """
    features = []

    # eWAVE uses DataTables with server-side processing
    # We need to fetch in batches of 100
    for start in [0, 100, 200]:
        url = f"https://ewave-atlas.org/values?language={language_id}&sEcho=1&iDisplayStart={start}&iDisplayLength=100"

        if verbose:
            print(f"Fetching features {start+1}-{start+100}...")

        headers = {
            "Accept": "application/json",
            "X-Requested-With": "XMLHttpRequest"
        }

        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()

            # Parse the aaData array
            for row in data.get("aaData", []):
                if len(row) >= 3:
                    # Extract feature ID from HTML: <a ... title="1">1</a>
                    id_match = re.search(r'title="(\d+)">\d+</a>', row[0])
                    feature_id = int(id_match.group(1)) if id_match else None

                    # Extract feature name from HTML
                    name_match = re.search(r'title="([^"]+)">[^<]+</a>', row[1])
                    feature_name = name_match.group(1) if name_match else ""

                    # Extract rating from HTML: "A - feature is pervasive..."
                    rating_match = re.search(r'>([ABCDX\?])\s*-\s*([^<]+)</a>', row[2])
                    if rating_match:
                        rating = rating_match.group(1)
                        rating_desc = rating_match.group(2).strip()
                    else:
                        rating = "?"
                        rating_desc = "unknown"

                    features.append({
                        "id": feature_id,
                        "name": feature_name,
                        "rating": rating,
                        "rating_description": rating_desc
                    })

        except requests.RequestException as e:
            print(f"Error fetching data: {e}")
            break

    # Sort by feature ID
    features.sort(key=lambda x: x["id"] if x["id"] else 999)

    return features


def analyze_features(features: List[Dict]) -> Dict:
    """
    Analyze feature ratings and generate summary.

    Returns:
        Dictionary with rating counts and feature lists by rating
    """
    rating_counts = {"A": 0, "B": 0, "C": 0, "D": 0, "X": 0, "?": 0}
    by_rating = {"A": [], "B": [], "C": [], "D": [], "X": [], "?": []}

    for f in features:
        rating = f["rating"]
        rating_counts[rating] = rating_counts.get(rating, 0) + 1
        by_rating[rating].append(f)

    return {
        "total": len(features),
        "counts": rating_counts,
        "by_rating": by_rating
    }


def print_feature_summary(features: List[Dict], analysis: Dict):
    """Print a summary of features by rating."""
    print("\n" + "=" * 60)
    print("eWAVE FEATURE SUMMARY")
    print("=" * 60)
    print(f"Total features: {analysis['total']}")
    print()
    print("Rating Counts:")
    print(f"  A (pervasive):     {analysis['counts']['A']}")
    print(f"  B (neither rare):  {analysis['counts']['B']}")
    print(f"  C (rare):          {analysis['counts']['C']}")
    print(f"  D (absent):        {analysis['counts']['D']}")
    print(f"  X (not applicable):{analysis['counts']['X']}")
    print(f"  ? (no info):       {analysis['counts']['?']}")
    print()

    # Print A-rated features
    print("=" * 60)
    print("A-RATED FEATURES (Pervasive)")
    print("=" * 60)
    for f in analysis["by_rating"]["A"]:
        print(f"  F{f['id']}: {f['name']}")

    # Print B-rated features
    print()
    print("=" * 60)
    print("B-RATED FEATURES (Neither Rare nor Pervasive)")
    print("=" * 60)
    for f in analysis["by_rating"]["B"]:
        print(f"  F{f['id']}: {f['name']}")


def generate_feature_keys(features: List[Dict], rating_filter: str = "AB") -> List[str]:
    """
    Generate feature keys for dialects.py based on feature names.

    Args:
        features: List of feature dictionaries
        rating_filter: Which ratings to include (e.g., "A", "AB", "ABC")

    Returns:
        List of feature key strings
    """
    # Mapping from eWAVE feature IDs to our FEATURE_LIBRARY keys
    # This would need to be expanded based on actual mappings
    ID_TO_KEY = {
        1: "she_inanimate",
        2: "he_inanimate",
        3: "alternative_it_referential",
        4: "alternative_it_dummy",
        5: "generalized_3sg_subject",
        6: "generalized_3sg_object",
        7: "me_coordinate_subjects",
        10: "no_gender_distinction",
        11: "regularized_reflexives",
        26: "object_possessive_1sg",
        34: "second_person_plural",
        50: "plural_preposed",
        52: "associative_plural_them",
        57: "zero_plural_human",
        58: "zero_plural_nonhuman",
        62: "zero_for_definite",
        63: "zero_for_indefinite",
        68: "them_for_those",
        77: "zero_genitive",
        88: "progressive_stative",
        90: "invariant_be_habitual",
        104: "completive_done",
        111: "been_past_anterior",
        114: "go_future",
        128: "regularized_past",
        129: "unmarked_past",
        131: "participle_for_past",
        132: "zero_past_regular",
        148: "serial_verb_give",
        149: "serial_verb_go",
        150: "serial_verb_come",
        154: "negative_concord",
        158: "invariant_dont",
        159: "never_past_negator",
        165: "invariant_tag",
        170: "zero_3sg",
        173: "variant_existential",
        174: "delete_aux_progressive",
        175: "delete_aux_gonna",
        176: "delete_copula_np",
        177: "delete_copula_adjp",
        178: "delete_copula_locative",
        180: "was_were_generalization",
        193: "zero_relative_subject",
        208: "deletion_to_infinitive",
        216: "omit_prepositions",
        228: "no_inversion_wh",
        229: "no_inversion_yn",
    }

    keys = []
    for f in features:
        if f["rating"] in rating_filter:
            fid = f["id"]
            if fid in ID_TO_KEY:
                keys.append(ID_TO_KEY[fid])

    return keys


def save_features(features: List[Dict], output_path: str):
    """Save features to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(features, f, indent=2)
    print(f"Saved {len(features)} features to {output_path}")


def fetch_all_dialects(output_dir: str, delay: float = 1.0, verbose: bool = False) -> Dict:
    """
    Fetch features for all 77 eWAVE dialects and save to individual JSON files.

    Args:
        output_dir: Directory to save individual dialect JSON files
        delay: Delay between requests (seconds) to avoid rate limiting
        verbose: Print progress

    Returns:
        Dictionary with all dialect data and summary statistics
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_data = {
        "metadata": {
            "source": "eWAVE (Electronic World Atlas of Varieties of English)",
            "url": "https://ewave-atlas.org",
            "scraped_at": datetime.now().isoformat(),
            "total_features": 235,
            "total_dialects": len(EWAVE_LANGUAGES)
        },
        "dialects": {}
    }

    failed = []
    success_count = 0

    print(f"Fetching features for {len(EWAVE_LANGUAGES)} dialects...")
    print(f"Output directory: {output_path}")
    print()

    for lang_id, lang_name in EWAVE_LANGUAGES.items():
        print(f"[{lang_id:2d}/{len(EWAVE_LANGUAGES)}] {lang_name}...", end=" ", flush=True)

        try:
            features = fetch_ewave_features(lang_id, verbose=False)

            if features:
                analysis = analyze_features(features)

                # Create dialect key from name (sanitize for filesystem)
                dialect_key = lang_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("'", "").replace("/", "_")

                dialect_data = {
                    "ewave_id": lang_id,
                    "name": lang_name,
                    "features": features,
                    "summary": {
                        "total": analysis["total"],
                        "counts": analysis["counts"]
                    }
                }

                all_data["dialects"][dialect_key] = dialect_data

                # Save individual dialect file
                dialect_file = output_path / f"ewave_{lang_id:02d}_{dialect_key[:30]}.json"
                with open(dialect_file, 'w') as f:
                    json.dump(dialect_data, f, indent=2)

                print(f"OK (A:{analysis['counts']['A']}, B:{analysis['counts']['B']}, C:{analysis['counts']['C']}, D:{analysis['counts']['D']})")
                success_count += 1
            else:
                print("FAILED (no features)")
                failed.append((lang_id, lang_name, "No features returned"))

        except Exception as e:
            print(f"ERROR: {e}")
            failed.append((lang_id, lang_name, str(e)))

        # Rate limiting
        time.sleep(delay)

    # Save combined data
    combined_file = output_path / "ewave_all_dialects.json"
    with open(combined_file, 'w') as f:
        json.dump(all_data, f, indent=2)

    # Print summary
    print()
    print("=" * 60)
    print("SCRAPING COMPLETE")
    print("=" * 60)
    print(f"Successful: {success_count}/{len(EWAVE_LANGUAGES)}")
    print(f"Failed: {len(failed)}")
    if failed:
        print("\nFailed dialects:")
        for lang_id, lang_name, error in failed:
            print(f"  {lang_id}: {lang_name} - {error}")

    print(f"\nCombined data saved to: {combined_file}")

    return all_data


def generate_rating_summary(all_data: Dict, output_file: str = None) -> str:
    """
    Generate a summary report of all dialect feature ratings.

    Args:
        all_data: Combined dialect data from fetch_all_dialects
        output_file: Optional file to save report

    Returns:
        Summary report as string
    """
    lines = []
    lines.append("=" * 80)
    lines.append("eWAVE FEATURE RATINGS SUMMARY")
    lines.append("=" * 80)
    lines.append(f"Generated: {datetime.now().isoformat()}")
    lines.append(f"Total dialects: {len(all_data.get('dialects', {}))}")
    lines.append("")
    lines.append("Rating Legend:")
    lines.append("  A = Pervasive or obligatory")
    lines.append("  B = Neither pervasive nor rare")
    lines.append("  C = Exists but is rare")
    lines.append("  D = Attested absence (feature not used)")
    lines.append("  X = Not applicable")
    lines.append("  ? = No information")
    lines.append("")
    lines.append("-" * 80)
    lines.append(f"{'Dialect':<40} {'A':>5} {'B':>5} {'C':>5} {'D':>5} {'X':>5} {'?':>5}")
    lines.append("-" * 80)

    for key, data in sorted(all_data.get("dialects", {}).items(), key=lambda x: x[1].get("ewave_id", 0)):
        counts = data.get("summary", {}).get("counts", {})
        name = data.get("name", key)[:38]
        lines.append(f"{name:<40} {counts.get('A', 0):>5} {counts.get('B', 0):>5} {counts.get('C', 0):>5} {counts.get('D', 0):>5} {counts.get('X', 0):>5} {counts.get('?', 0):>5}")

    lines.append("-" * 80)

    report = "\n".join(lines)

    if output_file:
        with open(output_file, 'w') as f:
            f.write(report)
        print(f"Report saved to: {output_file}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Scrape eWAVE feature ratings")
    parser.add_argument("--language", "-l", type=int, default=None,
                       help="eWAVE language ID (e.g., 61 for Aboriginal English)")
    parser.add_argument("--output", "-o", type=str, default=None,
                       help="Output JSON file path")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")
    parser.add_argument("--summary", "-s", action="store_true", default=True,
                       help="Print summary (default: True)")
    parser.add_argument("--all", "-a", action="store_true",
                       help="Fetch all 77 dialects")
    parser.add_argument("--output-dir", type=str, default="ewave_data",
                       help="Output directory for batch processing (default: ewave_data)")
    parser.add_argument("--list-languages", action="store_true",
                       help="List all available eWAVE language IDs")
    parser.add_argument("--delay", type=float, default=1.0,
                       help="Delay between requests in seconds (default: 1.0)")

    args = parser.parse_args()

    # List languages mode
    if args.list_languages:
        print("eWAVE Language IDs:")
        print("-" * 60)
        for lang_id, lang_name in sorted(EWAVE_LANGUAGES.items()):
            print(f"  {lang_id:2d}: {lang_name}")
        return

    # Batch mode: fetch all dialects
    if args.all:
        all_data = fetch_all_dialects(
            output_dir=args.output_dir,
            delay=args.delay,
            verbose=args.verbose
        )
        # Generate summary report
        report = generate_rating_summary(
            all_data,
            output_file=Path(args.output_dir) / "ewave_summary_report.txt"
        )
        print()
        print(report)
        return

    # Single language mode
    if args.language is None:
        args.language = 61  # Default to Aboriginal English

    lang_name = EWAVE_LANGUAGES.get(args.language, f"Unknown (ID: {args.language})")
    print(f"Fetching eWAVE features for {lang_name} (ID: {args.language})...")
    features = fetch_ewave_features(args.language, verbose=args.verbose)

    if not features:
        print("No features found!")
        return

    print(f"Retrieved {len(features)} features")

    # Analyze
    analysis = analyze_features(features)

    if args.summary:
        print_feature_summary(features, analysis)

    # Save if output specified
    if args.output:
        save_features(features, args.output)

    # Generate feature keys for A+B rated features
    print()
    print("=" * 60)
    print("FEATURE KEYS FOR dialects.py (A+B rated)")
    print("=" * 60)
    keys = generate_feature_keys(features, "AB")
    for key in keys:
        print(f'    "{key}",')


if __name__ == "__main__":
    main()
