#!/usr/bin/env python3
"""
eWAVE Ratings Integration Module

Provides rating-based feature selection for dialect transformations.
Allows users to select comprehensiveness levels (A, AB, ABC, etc.) to control
which features are used in dialect transformations.

Rating Legend (from eWAVE):
    A = Pervasive or obligatory (feature is pervasive/obligatory in this variety)
    B = Neither pervasive nor rare (feature exists but is not pervasive)
    C = Exists but is rare (feature exists but is very rarely used)
    D = Attested absence (feature is documented as NOT used in this variety)
    X = Not applicable (feature not applicable to this variety type)
    ? = No information available

Usage:
    from ewave_ratings import get_dialect_features_by_rating, RatingLevel

    # Get only pervasive features (A-rated)
    features = get_dialect_features_by_rating("aboriginal", RatingLevel.A)

    # Get common features (A and B rated)
    features = get_dialect_features_by_rating("aboriginal", RatingLevel.AB)

    # Get all documented features (A, B, C rated)
    features = get_dialect_features_by_rating("aboriginal", RatingLevel.ABC)
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Set
from enum import Enum


class RatingLevel(Enum):
    """Feature rating levels for controlling transformation comprehensiveness."""
    A = "A"           # Only pervasive/obligatory features
    AB = "AB"         # Pervasive + common features
    ABC = "ABC"       # All documented features (including rare)
    ABCD = "ABCD"     # Include attested absences (for analysis)
    ALL = "ABCDX?"    # Everything including unknown


# Human-readable descriptions for each level
RATING_LEVEL_DESCRIPTIONS = {
    RatingLevel.A: "Pervasive Only - Only highly characteristic features (most conservative)",
    RatingLevel.AB: "Common Features - Pervasive and common features (recommended default)",
    RatingLevel.ABC: "Comprehensive - All documented features including rare ones",
    RatingLevel.ABCD: "Full Analysis - Include attested absences for linguistic analysis",
    RatingLevel.ALL: "Everything - All data including unknown features"
}


# eWAVE data directory
EWAVE_DATA_DIR = Path(__file__).parent / "ewave_data"


def load_ewave_dialect_data(ewave_id: int) -> Optional[Dict]:
    """
    Load eWAVE data for a specific dialect by its eWAVE ID.

    Args:
        ewave_id: eWAVE language ID (1-77)

    Returns:
        Dictionary with dialect data or None if not found
    """
    # Try to find the file
    for json_file in EWAVE_DATA_DIR.glob(f"ewave_{ewave_id:02d}_*.json"):
        with open(json_file, 'r') as f:
            return json.load(f)

    # Try the combined file
    combined_file = EWAVE_DATA_DIR / "ewave_all_dialects.json"
    if combined_file.exists():
        with open(combined_file, 'r') as f:
            data = json.load(f)
            for dialect_key, dialect_data in data.get("dialects", {}).items():
                if dialect_data.get("ewave_id") == ewave_id:
                    return dialect_data

    return None


def load_all_ewave_data() -> Dict:
    """Load all eWAVE dialect data from the combined file."""
    combined_file = EWAVE_DATA_DIR / "ewave_all_dialects.json"
    if combined_file.exists():
        with open(combined_file, 'r') as f:
            return json.load(f)
    return {"dialects": {}}


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
    "jamaican": 27,
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


# Mapping from eWAVE feature IDs to our FEATURE_LIBRARY keys
EWAVE_ID_TO_FEATURE_KEY = {
    1: "she_inanimate",
    2: "he_inanimate",
    3: "alternative_it_referential",
    4: "alternative_it_dummy",
    5: "generalized_3sg_subject",
    6: "generalized_3sg_object",
    7: "me_coordinate_subjects",
    8: "myself_coordinate_subjects",
    9: "benefactive_dative",
    10: "no_gender_distinction",
    11: "regularized_reflexives",
    12: "regularized_reflexives_object",
    13: "regularized_reflexives_double",
    14: "no_number_distinction_reflexives",
    15: "emphatic_reflexives_subject",
    16: "emphatic_reflexives_other",
    17: "fi_possessive",
    18: "me_possessive_singular",
    19: "possessive_pronoun_object",
    20: "possessive_pronoun_subject",
    21: "possessive_me_my_deletion",
    22: "emphatic_reflexives_own",
    23: "absolute_possessive_mines",
    24: "absolute_possessive_other",
    25: "nasal_possessive",
    26: "object_possessive_1sg",
    27: "object_possessive_2sg",
    28: "us_singular_object",
    29: "non_coordinated_subject_pronoun",
    30: "object_pronoun_subject",
    31: "non_coordinated_object_subject",
    32: "emphatic_pronoun_distinction",
    33: "nasal_independent_possessive",
    34: "second_person_plural",
    35: "second_person_singular",
    36: "inclusive_exclusive_1p",
    37: "more_number_distinctions",
    38: "pronoun_exchange_animate",
    39: "resumptive_subject",
    40: "resumptive_object",
    41: "singular_it_plural",
    42: "object_pronoun_drop",
    43: "subject_drop_referential",
    44: "subject_drop_dummy",
    45: "it_insertion",
    46: "it_deletion_referential",
    47: "it_deletion_non_referential",
    48: "plural_suffix",
    49: "plural_reduplication",
    50: "plural_preposed",
    51: "plural_postposed",
    52: "associative_plural_them",
    53: "associative_plural_suffix",
    54: "associative_plural_other",
    55: "group_plural",
    56: "mass_count",
    57: "zero_plural_human",
    58: "zero_plural_nonhuman",
    59: "double_determiners",
    60: "zero_indefinite_article",
    61: "indefinite_definite_contexts",
    62: "zero_for_definite",
    63: "zero_for_indefinite",
    64: "definite_indefinite_contexts",
    65: "some_indefinite",
    66: "indefinite_one",
    67: "definite_dem_personal",
    68: "them_for_those",
    69: "dis_here_demonstrative",
    70: "proximal_distal_conflation",
    71: "me_first_person_possessive",
    72: "simple_juxtaposition",
    73: "possession_for_phrase_preceding",
    74: "possession_for_phrase_following",
    75: "possession_own_following",
    76: "preposition_personal_pronoun",
    77: "zero_genitive",
    78: "double_genitives",
    79: "double_comparatives",
    80: "regularized_comparison_periphrastic",
    81: "regularized_comparison_synthetic",
    82: "adnominal_them",
    83: "wh_relative",
    84: "relative_what",
    85: "relative_as",
    86: "relative_at",
    87: "relative_nonstandard",
    88: "progressive_stative",
    89: "progressive_habitual",
    90: "invariant_be_habitual",
    91: "do_habitual",
    92: "other_habitual_synthetic",
    93: "other_habitual_analytic",
    94: "aspect_be_v_ing",
    95: "been_progressive",
    96: "been_perfect_progressive",
    97: "immediate_anteriority",
    98: "recency_just",
    99: "be_perfect_auxiliary",
    100: "have_perfect_auxiliary",
    101: "simple_present_for_perfect",
    102: "after_perfect",
    103: "do_tense_marker",
    104: "completive_done",
    105: "have_done_completive",
    106: "be_done_irrealis",
    107: "resultative_sit_stand_lie",
    108: "resultative_finish",
    109: "already_perfect",
    110: "past_continuous_counterfactual",
    111: "been_past_anterior",
    112: "other_past_anterior",
    113: "had_counterfactual",
    114: "go_future",
    115: "other_future",
    116: "would_hypothetical",
    117: "would_habitual",
    118: "could_would_interchangeable",
    119: "may_can_interchangeable",
    120: "was_subjunctive",
    121: "dont_have_to_prohibition",
    122: "aint_got_to_necessity",
    123: "double_modals",
    124: "other_modal_combinations",
    125: "quasi_modals_core",
    126: "quasi_modals_aspectual",
    127: "unsplit_want",
    128: "regularized_past",
    129: "unmarked_past",
    130: "unmarked_past_semantic",
    131: "participle_for_past",
    132: "zero_past_regular",
    133: "past_for_participle",
    134: "a_prefixing_ing",
    135: "a_prefixing_verbs",
    136: "present_participle_reduced",
    137: "special_do_forms",
    138: "been_as_be",
    139: "am_invariant",
    140: "other_copula_np",
    141: "other_copula_locative",
    142: "other_copula_adjp",
    143: "transitive_suffix",
    144: "di_transitive",
    145: "verb_transitivity",
    146: "deverbal_adjectives",
    147: "complex_predicates",
    148: "serial_verb_give",
    149: "serial_verb_go",
    150: "serial_verb_come",
    151: "serial_verb_other_motion",
    152: "serial_verb_other",
    153: "say_verb_introduce_speech",
    154: "negative_concord",
    155: "aint_be",
    156: "aint_have",
    157: "aint_did",
    158: "invariant_dont",
    159: "never_past_negator",
    160: "no_preverbal",
    161: "not_preverbal",
    162: "wasnt_werent_interchangeable",
    163: "negative_suffix",
    164: "never_without_auxiliary",
    165: "invariant_tag",
    166: "amnt_tag",
    167: "innit_tag",
    168: "can_or_not_tag",
    169: "other_negative",
    170: "zero_3sg",
    171: "regularized_3sg",
    172: "there_singular_plural",
    173: "variant_existential",
    174: "delete_aux_progressive",
    175: "delete_aux_gonna",
    176: "delete_copula_np",
    177: "delete_copula_adjp",
    178: "delete_copula_locative",
    179: "am_is_leveling",
    180: "was_were_generalization",
    181: "was_were_split",
    182: "done_as_auxiliary",
    183: "other_agreement",
    184: "singular_collective_nouns",
    185: "default_singulars",
    186: "northern_subject_rule",
    187: "relative_what_agreement",
    188: "wh_relative_human",
    189: "wh_relative_nonhuman",
    190: "relative_that_human",
    191: "relativizer_what",
    192: "relativizer_as",
    193: "zero_relative_subject",
    194: "resumptive_pronouns",
    195: "relative_pronoun_deletion",
    196: "one_relativizer",
    197: "double_object_order",
    198: "particle_placement",
    199: "preposition_stranding",
    200: "say_complementizer",
    201: "for_complementizer",
    202: "existential_get",
    203: "relative_infinitive",
    204: "for_to_infinitive",
    205: "existential_have",
    206: "existential_have_forms",
    207: "would_like_construction",
    208: "deletion_to_infinitive",
    209: "bare_root_complement",
    210: "non_finite_bare_root",
    211: "serial_like_construction",
    212: "clause_final_but_though",
    213: "conjunction_doubling",
    214: "sentence_initial_because",
    215: "fronting_prepositional",
    216: "omit_prepositions",
    217: "non_standard_preposition",
    218: "locative_adverb",
    219: "adverb_way_time",
    220: "flat_adverbs",
    221: "manner_adverb_degree",
    222: "comparative_adverb",
    223: "other_clefting",
    224: "other_fronting",
    225: "focus_marker",
    226: "negative_inversion",
    227: "inverted_indirect_questions",
    228: "no_inversion_wh",
    229: "no_inversion_yn",
    230: "habitual_inversion",
    231: "like_focus",
    232: "like_quotative",
    233: "subject_imperative",
    234: "singular_address_plural_verb",
    235: "pluralized_relative",
}


def get_features_by_rating(
    dialect_key: str,
    rating_level: RatingLevel = RatingLevel.AB,
    return_details: bool = False
) -> List[str]:
    """
    Get feature keys for a dialect filtered by rating level.

    Args:
        dialect_key: Dialect identifier (e.g., "aboriginal", "urban_aave")
        rating_level: Which ratings to include (A, AB, ABC, etc.)
        return_details: If True, return full feature info instead of just keys

    Returns:
        List of feature keys (or feature dictionaries if return_details=True)
    """
    ewave_id = DIALECT_TO_EWAVE_ID.get(dialect_key)
    if not ewave_id:
        raise ValueError(f"Unknown dialect: {dialect_key}")

    dialect_data = load_ewave_dialect_data(ewave_id)
    if not dialect_data:
        raise ValueError(f"No eWAVE data found for {dialect_key} (ID: {ewave_id})")

    features = dialect_data.get("features", [])
    allowed_ratings = set(rating_level.value)

    result = []
    for feature in features:
        if feature.get("rating") in allowed_ratings:
            feature_id = feature.get("id")
            feature_key = EWAVE_ID_TO_FEATURE_KEY.get(feature_id)

            if feature_key:
                if return_details:
                    result.append({
                        "key": feature_key,
                        "ewave_id": feature_id,
                        "name": feature.get("name"),
                        "rating": feature.get("rating"),
                        "rating_description": feature.get("rating_description")
                    })
                else:
                    result.append(feature_key)

    return result


def get_dialect_rating_summary(dialect_key: str) -> Dict:
    """
    Get a summary of feature ratings for a dialect.

    Args:
        dialect_key: Dialect identifier

    Returns:
        Dictionary with rating counts and percentages
    """
    ewave_id = DIALECT_TO_EWAVE_ID.get(dialect_key)
    if not ewave_id:
        raise ValueError(f"Unknown dialect: {dialect_key}")

    dialect_data = load_ewave_dialect_data(ewave_id)
    if not dialect_data:
        raise ValueError(f"No eWAVE data found for {dialect_key} (ID: {ewave_id})")

    summary = dialect_data.get("summary", {})
    counts = summary.get("counts", {})
    total = summary.get("total", 235)

    return {
        "dialect": dialect_key,
        "ewave_id": ewave_id,
        "name": dialect_data.get("name"),
        "total_features": total,
        "counts": counts,
        "percentages": {
            rating: round(count / total * 100, 1)
            for rating, count in counts.items()
        }
    }


def compare_dialects(
    dialect_keys: List[str],
    rating_level: RatingLevel = RatingLevel.AB
) -> Dict:
    """
    Compare features across multiple dialects at a given rating level.

    Args:
        dialect_keys: List of dialect identifiers to compare
        rating_level: Rating level for comparison

    Returns:
        Dictionary with shared and unique features
    """
    all_features = {}
    for dialect in dialect_keys:
        features = set(get_features_by_rating(dialect, rating_level))
        all_features[dialect] = features

    # Find shared features
    if all_features:
        shared = set.intersection(*all_features.values())
    else:
        shared = set()

    # Find unique features per dialect
    unique = {}
    for dialect, features in all_features.items():
        unique[dialect] = features - shared

    return {
        "rating_level": rating_level.value,
        "dialects": dialect_keys,
        "shared_features": list(shared),
        "shared_count": len(shared),
        "unique_features": {k: list(v) for k, v in unique.items()},
        "per_dialect_count": {k: len(v) for k, v in all_features.items()}
    }


def get_available_dialects() -> List[Dict]:
    """Get list of all available dialects with their eWAVE IDs."""
    return [
        {"key": key, "ewave_id": ewave_id}
        for key, ewave_id in sorted(DIALECT_TO_EWAVE_ID.items(), key=lambda x: x[1])
    ]


# Convenience function for CLI usage
def print_dialect_features(dialect_key: str, rating_level: RatingLevel = RatingLevel.AB):
    """Print features for a dialect at a given rating level."""
    try:
        features = get_features_by_rating(dialect_key, rating_level, return_details=True)
        summary = get_dialect_rating_summary(dialect_key)

        print(f"\n{'='*60}")
        print(f"DIALECT: {summary['name']}")
        print(f"eWAVE ID: {summary['ewave_id']}")
        print(f"Rating Level: {rating_level.value} - {RATING_LEVEL_DESCRIPTIONS[rating_level]}")
        print(f"{'='*60}")
        print(f"\nTotal features in eWAVE: {summary['total_features']}")
        print(f"Feature counts: A={summary['counts'].get('A', 0)}, B={summary['counts'].get('B', 0)}, "
              f"C={summary['counts'].get('C', 0)}, D={summary['counts'].get('D', 0)}")
        print(f"\nFeatures at level {rating_level.value} ({len(features)}):")
        print("-" * 60)

        for f in features:
            print(f"  F{f['ewave_id']:3d} [{f['rating']}]: {f['key']}")
            print(f"        {f['name']}")

    except ValueError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python ewave_ratings.py <dialect_key> [rating_level]")
        print("\nAvailable dialects:")
        for d in get_available_dialects()[:10]:
            print(f"  {d['key']} (ID: {d['ewave_id']})")
        print("  ...")
        print("\nRating levels: A, AB, ABC, ABCD, ALL")
        sys.exit(1)

    dialect = sys.argv[1]
    level = RatingLevel.AB

    if len(sys.argv) >= 3:
        level_str = sys.argv[2].upper()
        try:
            level = RatingLevel(level_str)
        except ValueError:
            print(f"Invalid rating level: {level_str}")
            print("Valid levels: A, AB, ABC, ABCD, ALL")
            sys.exit(1)

    print_dialect_features(dialect, level)
