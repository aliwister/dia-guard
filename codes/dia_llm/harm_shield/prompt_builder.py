# prompt_builder.py - Assembles dialect-specific prompts from features and dialects

from typing import List, Optional, Dict, Any
from features import FEATURE_LIBRARY, CATEGORIES
from dialects import DIALECT_REGISTRY, get_dialect, get_all_features_for_dialect


def build_feature_section(feature_name: str) -> str:
    """Build a formatted section for a single feature."""
    if feature_name not in FEATURE_LIBRARY:
        return f"### {feature_name.replace('_', ' ').title()}\n(Feature definition not found)\n"

    feat = FEATURE_LIBRARY[feature_name]

    # Format examples
    examples = ""
    if feat.get("examples"):
        examples = "\n".join([
            f'    "{ex[0]}" → "{ex[1]}"'
            for ex in feat["examples"]
        ])

    # Format constraints
    constraints = ""
    if feat.get("constraints"):
        constraints = f"\n    Note: {feat['constraints']}"

    # Format notes
    notes = ""
    if feat.get("notes"):
        notes = f"\n    Info: {feat['notes']}"

    section = f"""
### {feature_name.replace('_', ' ').title()}
{feat['description']}
Examples:
{examples}{constraints}{notes}
"""
    return section


def build_dialect_prompt(
    dialect_key: str,
    custom_features: Optional[List[str]] = None,
    strength: Optional[float] = None,
    include_all_examples: bool = True
) -> str:
    """
    Build a complete transformation prompt for a dialect.

    Args:
        dialect_key: Key from DIALECT_REGISTRY
        custom_features: Override the dialect's default features
        strength: Override the dialect's default strength
        include_all_examples: Include all examples for each feature

    Returns:
        Complete system prompt for the LLM
    """
    if dialect_key not in DIALECT_REGISTRY:
        available = ", ".join(list(DIALECT_REGISTRY.keys())[:10]) + "..."
        raise ValueError(f"Unknown dialect: {dialect_key}. Available: {available}")

    dialect = DIALECT_REGISTRY[dialect_key]
    features = custom_features or dialect.get("features", [])
    transform_strength = strength if strength is not None else dialect.get("strength", 1.0)

    # Build feature sections grouped by category
    feature_sections_by_category: Dict[str, List[str]] = {}

    for feat_name in features:
        if feat_name in FEATURE_LIBRARY:
            feat = FEATURE_LIBRARY[feat_name]
            category = feat.get("category", "other")
            if category not in feature_sections_by_category:
                feature_sections_by_category[category] = []
            feature_sections_by_category[category].append(build_feature_section(feat_name))

    # Format by category
    all_sections = ""
    for category, sections in feature_sections_by_category.items():
        category_info = CATEGORIES.get(category, {})
        if isinstance(category_info, dict):
            category_name = category_info.get("name", category.replace("_", " ").title())
        else:
            category_name = category_info if category_info else category.replace("_", " ").title()
        all_sections += f"\n## {category_name.upper()}\n"
        all_sections += "".join(sections)

    prompt = f"""# {dialect['name']} Transformation Guide

You are a linguistics expert specializing in {dialect['name']} from {dialect['region']}.
Your task is to transform Standard American English (SAE) text into authentic {dialect['name']}.

{dialect.get('description', '')}

## Transformation Strength: {transform_strength}
- 1.0 = Apply all applicable features consistently
- 0.5 = Apply features occasionally/subtly
- Current setting: {transform_strength}

{dialect.get('notes', '')}

# FEATURES TO APPLY:
{all_sections}

# CRITICAL RULES:

## 1. PRESERVE EXACTLY (Never Modify):
- URLs (http://, https://)
- Email addresses
- @mentions and #hashtags
- Emoji (all Unicode emoji)
- Code blocks and inline code
- File paths
- Phone numbers
- Markdown links [text](url)
- Any placeholder tokens (__PROTECTED_*__)

## 2. MAINTAIN MEANING:
- Never change the core meaning of sentences
- Preserve the intent and information content
- Keep proper nouns unchanged

## 3. NATURAL APPLICATION:
- Don't force every feature into every sentence
- Apply features where grammatically appropriate
- Some sentences may not need any changes
- Match the natural flow of the dialect

## 4. CONTEXT AWARENESS:
- Some features only apply in specific grammatical contexts
- Check subject-verb relationships before applying agreement changes
- Consider sentence type (statement, question, command)

## 5. REGISTER MATCHING:
- Maintain the formality level of the original text
- Formal text stays relatively formal
- Informal text can use more dialectal features

# OUTPUT FORMAT:
Return ONLY the transformed text.
No explanations, annotations, or meta-commentary.
Do not include quotation marks around your output unless they were in the input.
"""

    return prompt


def build_concise_prompt(
    dialect_key: str,
    max_examples_per_feature: int = 2
) -> str:
    """
    Build a more concise prompt for models with smaller context windows.

    Args:
        dialect_key: Key from DIALECT_REGISTRY
        max_examples_per_feature: Maximum examples to include per feature

    Returns:
        Concise system prompt
    """
    dialect = DIALECT_REGISTRY[dialect_key]
    features = dialect.get("features", [])

    # Build compact feature list
    feature_lines = []
    for feat_name in features:
        if feat_name in FEATURE_LIBRARY:
            feat = FEATURE_LIBRARY[feat_name]
            examples = feat.get("examples", [])[:max_examples_per_feature]
            example_str = "; ".join([f'"{e[0]}"→"{e[1]}"' for e in examples])
            feature_lines.append(f"- **{feat_name}**: {feat['description']} ({example_str})")

    prompt = f"""# Transform to {dialect['name']}

Region: {dialect['region']}
Strength: {dialect.get('strength', 1.0)}

## Features:
{chr(10).join(feature_lines)}

## Rules:
1. PRESERVE: URLs, emoji, @mentions, #hashtags, code
2. Keep meaning intact
3. Apply features naturally
4. Output transformed text only
"""
    return prompt


def build_multi_dialect_prompt() -> str:
    """Build a prompt that can transform to any dialect on demand."""

    # Organize dialects by region
    from dialects import REGIONS

    dialect_list = ""
    for region, dialect_keys in REGIONS.items():
        region_name = region.replace("_", " ").title()
        dialect_list += f"\n### {region_name}\n"
        for key in dialect_keys:
            if key in DIALECT_REGISTRY:
                name = DIALECT_REGISTRY[key]["name"]
                dialect_list += f"- `{key}`: {name}\n"

    # Key feature summaries
    feature_summary = """
## Feature Categories Summary:

### 1. PRONOUNS
- y'all, youse (2nd person plural)
- 'em for them
- regularized reflexives (hisself, theirselves)
- me in coordinate subjects (me and John)

### 2. COPULA/AUXILIARY
- Copula deletion: "He tall" (not "He is tall")
- Aux deletion: "She running" (not "She is running")
- Habitual be: "He be working" (habitual action)

### 3. TENSE/ASPECT
- done (completive): "I done finished"
- been (remote past): "I been knowing him"
- finna/gon (future): "I'm finna leave"
- a-prefixing: "He was a-running"

### 4. NEGATION
- ain't for isn't/aren't/haven't
- Negative concord: "I don't have nothing"
- Negative inversion: "Can't nobody do that"

### 5. AGREEMENT
- Uninflected verbs: "He walk" (not "He walks")
- was/were leveling: "We was happy"

### 6. ARTICLES
- Omission: "I went to hospital"
- Variation: "the" for "a" or vice versa

### 7. OTHER
- Double modals: "might could"
- Serial verbs: "take go give"
- Invariant tags: "isn't it" / "innit"
- Topic prominence
"""

    prompt = f"""# Universal English Dialect Transformer

You can transform Standard American English into any of 50 English dialects.

## Available Dialects:
{dialect_list}

{feature_summary}

## Usage Instructions:
When given input in the format:
```
[dialect_code] text to transform
```
Apply the appropriate morphosyntactic features for that dialect.

## CRITICAL RULES:
1. PRESERVE EXACTLY: URLs, @mentions, #hashtags, emoji, code blocks
2. Apply features naturally based on context
3. Maintain meaning and register/formality
4. Output only the transformed text

## Examples:

Input: [urban_aave] She is always working at the office.
Output: She be working at the office.

Input: [appalachian] He was running down the mountain.
Output: He was a-running down the mountain.

Input: [irish] I have just finished eating.
Output: I'm after eating.

Input: [singapore] I have already eaten the food.
Output: I eat the food already.
"""
    return prompt


def build_batch_prompt(dialect_key: str) -> str:
    """Build a prompt optimized for batch processing multiple texts."""

    base_prompt = build_concise_prompt(dialect_key)

    batch_instructions = """

## BATCH PROCESSING MODE

You will receive multiple texts to transform, separated by "---".
Each text is prefixed with a number in brackets like [0], [1], [2].

Transform each text independently and return them in the same format:
- Keep the [N] prefix for each
- Separate outputs with "---"
- Transform only the text after the [N] prefix

Example Input:
[0] She is always working.
---
[1] He doesn't have any money.
---
[2] They are at the store.

Example Output:
[0] She be working.
---
[1] He don't have no money.
---
[2] They at the store.
"""

    return base_prompt + batch_instructions


def get_quick_prompt(dialect_key: str) -> str:
    """
    Get a minimal prompt for quick transformations.
    Best for high-quality models that need less guidance.
    """
    dialect = DIALECT_REGISTRY.get(dialect_key, {})
    features = dialect.get("features", [])

    # Just list feature names
    feature_list = ", ".join(features[:15])  # Top 15 features
    if len(features) > 15:
        feature_list += f" (+{len(features) - 15} more)"

    return f"""Transform to {dialect.get('name', dialect_key)}.
Key features: {feature_list}
Preserve URLs/emoji/@mentions. Output transformed text only."""
