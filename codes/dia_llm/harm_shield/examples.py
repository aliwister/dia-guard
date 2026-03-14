#!/usr/bin/env python3
"""
Examples showing how to use the Dialect Transformer with different backends.
"""

# ═══════════════════════════════════════════════════════════════════════════
# EXAMPLE 1: Using OpenAI API
# ═══════════════════════════════════════════════════════════════════════════

def example_openai():
    """Transform text using OpenAI's GPT-4."""
    from transformer import DialectTransformer
    from models import OpenAIBackend

    # Create backend (uses OPENAI_API_KEY env var by default)
    llm = OpenAIBackend(model="gpt-4")

    # Or specify API key directly:
    # llm = OpenAIBackend(model="gpt-4", api_key="sk-...")

    # Create transformer
    transformer = DialectTransformer(llm=llm, verbose=True)

    # Transform some text
    text = """
    She is always working at the office. He doesn't have any money.
    I am going to the store to buy some groceries.
    Check out my website: https://example.com 🔥
    """

    # Transform to AAVE
    result = transformer.transform(text, dialect="urban_aave")
    print(f"\n{'='*60}")
    print("ORIGINAL:")
    print(result.original)
    print(f"\n{'='*60}")
    print(f"TRANSFORMED TO: {result.dialect_name}")
    print(result.transformed)
    print(f"\nModel used: {result.model_used}")


# ═══════════════════════════════════════════════════════════════════════════
# EXAMPLE 2: Using Ollama (Local Models)
# ═══════════════════════════════════════════════════════════════════════════

def example_ollama():
    """Transform text using Ollama with Llama 3.1."""
    from transformer import DialectTransformer
    from models import OllamaBackend

    # Make sure Ollama is running: ollama serve
    # And you have the model: ollama pull llama3.1

    llm = OllamaBackend(
        model="llama3.1",  # or "mistral", "mixtral", "phi3"
        host="http://localhost:11434"
    )

    transformer = DialectTransformer(llm=llm, verbose=True)

    text = "I am going to help them with their homework. She doesn't know anything about it."

    # Transform to Appalachian
    result = transformer.transform(text, dialect="appalachian")
    print(f"\nOriginal: {result.original}")
    print(f"Appalachian: {result.transformed}")

    # Transform to Jamaican
    result = transformer.transform(text, dialect="jamaican")
    print(f"Jamaican: {result.transformed}")


# ═══════════════════════════════════════════════════════════════════════════
# EXAMPLE 3: Using HuggingFace Transformers (Local Models)
# ═══════════════════════════════════════════════════════════════════════════

def example_huggingface():
    """Transform text using a local HuggingFace model."""
    from transformer import DialectTransformer
    from models import HuggingFaceBackend

    # Using Phi-3 (small, fast, ~4GB VRAM)
    llm = HuggingFaceBackend(
        model="microsoft/Phi-3-mini-4k-instruct",
        load_in_4bit=True  # Use 4-bit quantization to reduce memory
    )

    # Alternative models:
    # - "mistralai/Mistral-7B-Instruct-v0.3" (better quality, ~16GB)
    # - "meta-llama/Meta-Llama-3.1-8B-Instruct" (best quality, ~18GB)

    transformer = DialectTransformer(llm=llm, verbose=True)

    text = "She has been living there for many years. They are not going to help anyone."

    # Transform to Indian English
    result = transformer.transform(text, dialect="indian")
    print(f"\nOriginal: {result.original}")
    print(f"Indian English: {result.transformed}")


# ═══════════════════════════════════════════════════════════════════════════
# EXAMPLE 4: Auto-Detection (Uses Best Available Backend)
# ═══════════════════════════════════════════════════════════════════════════

def example_auto():
    """Let the system auto-detect the best available backend."""
    from transformer import DialectTransformer

    # Auto-detect: tries OpenAI > Anthropic > Ollama > HuggingFace
    transformer = DialectTransformer(backend="auto", verbose=True)

    text = "I have not seen him anywhere. She is always late to work."

    dialects_to_test = [
        "urban_aave",
        "appalachian",
        "irish",
        "singapore",
        "nigerian"
    ]

    for dialect in dialects_to_test:
        result = transformer.transform(text, dialect=dialect)
        print(f"\n{result.dialect_name}:")
        print(f"  {result.transformed}")


# ═══════════════════════════════════════════════════════════════════════════
# EXAMPLE 5: Batch Processing
# ═══════════════════════════════════════════════════════════════════════════

def example_batch():
    """Process multiple texts efficiently."""
    from transformer import DialectTransformer

    transformer = DialectTransformer(backend="auto", verbose=True)

    texts = [
        "She is working at home.",
        "He doesn't have any friends.",
        "They are going to the store.",
        "I have not finished my homework.",
        "We were happy to see you."
    ]

    results = transformer.transform_batch(
        texts,
        dialect="urban_aave",
        batch_size=3
    )

    print("\nBatch Results:")
    for result in results:
        print(f"  Original: {result.original}")
        print(f"  AAVE: {result.transformed}")
        print()


# ═══════════════════════════════════════════════════════════════════════════
# EXAMPLE 6: Custom Features
# ═══════════════════════════════════════════════════════════════════════════

def example_custom_features():
    """Use custom feature selection instead of dialect defaults."""
    from transformer import DialectTransformer

    transformer = DialectTransformer(backend="auto")

    text = "He is tall. She doesn't have anything. They are always working."

    # Use only specific features
    custom_features = [
        "drop_copula_be_AP",  # Copula deletion before adjectives
        "negative_concord",   # Double negatives
        "habitual_be"         # Habitual 'be'
    ]

    result = transformer.transform(
        text,
        dialect="urban_aave",
        custom_features=custom_features,
        strength=1.0
    )

    print(f"Original: {result.original}")
    print(f"Custom AAVE: {result.transformed}")


# ═══════════════════════════════════════════════════════════════════════════
# EXAMPLE 7: Handling Special Characters (URLs, Emoji, etc.)
# ═══════════════════════════════════════════════════════════════════════════

def example_special_chars():
    """Demonstrate that special characters are preserved."""
    from transformer import DialectTransformer

    transformer = DialectTransformer(backend="auto", verbose=True)

    text = """
    Hey @john, check out this link: https://example.com/path?query=1 🔥🎉
    She is always posting on #Twitter about her work.
    Email me at test@email.com if you are interested!
    The file is at /home/user/documents/file.txt
    """

    result = transformer.transform(text, dialect="urban_aave")

    print(f"\nOriginal:\n{result.original}")
    print(f"\nTransformed:\n{result.transformed}")
    print(f"\nProtected tokens: {len(result.protected_tokens)}")


# ═══════════════════════════════════════════════════════════════════════════
# EXAMPLE 8: Compare Multiple Dialects
# ═══════════════════════════════════════════════════════════════════════════

def example_compare_dialects():
    """Compare the same text across multiple dialects."""
    from transformer import DialectTransformer

    transformer = DialectTransformer(backend="auto")

    text = "I have been knowing him for years. She is always working hard. They don't have anything."

    dialects = {
        "urban_aave": "African American Vernacular",
        "appalachian": "Appalachian",
        "irish": "Irish English",
        "jamaican": "Jamaican Creole",
        "singapore": "Singlish",
        "indian": "Indian English",
        "nigerian": "Nigerian English"
    }

    print(f"\nOriginal: {text}\n")
    print("=" * 70)

    for dialect_key, dialect_name in dialects.items():
        result = transformer.transform(text, dialect=dialect_key)
        print(f"\n{dialect_name}:")
        print(f"  {result.transformed}")


# ═══════════════════════════════════════════════════════════════════════════
# EXAMPLE 9: Adjusting Transformation Strength
# ═══════════════════════════════════════════════════════════════════════════

def example_strength():
    """Show how strength affects transformation intensity."""
    from transformer import DialectTransformer

    transformer = DialectTransformer(backend="auto")

    text = "She is always working. He doesn't have any money. They are at the store."

    print(f"\nOriginal: {text}\n")

    for strength in [0.3, 0.6, 1.0]:
        result = transformer.transform(
            text,
            dialect="urban_aave",
            strength=strength
        )
        print(f"Strength {strength}: {result.transformed}")


# ═══════════════════════════════════════════════════════════════════════════
# EXAMPLE 10: Quick One-Liner
# ═══════════════════════════════════════════════════════════════════════════

def example_quick():
    """Quick one-liner for simple transformations."""
    from transformer import transform_to_dialect

    # Simple one-liner
    result = transform_to_dialect(
        "She is always working hard.",
        dialect="urban_aave"
    )
    print(f"Quick result: {result}")


# ═══════════════════════════════════════════════════════════════════════════
# EXAMPLE 11: CoI Token Reintegration (Rule-based)
# ═══════════════════════════════════════════════════════════════════════════

def example_coi_rulebased():
    """Demonstrate Chain of Interaction token reintegration without LLM."""
    from coi_reintegration import CoIDialectReintegrator, reintegrate_tokens

    print("CoI Token Reintegration (Rule-based)")
    print("=" * 50)

    # Original text with sensitive tokens
    original = "Check this out 🔥 https://example.com @john #trending — it's amazing!"

    # Multi-VALUE output (tokens removed to avoid OOV errors)
    multivalue = "Check dis out it amazing"  # Jamaican dialect, tokens stripped

    print(f"\nOriginal: {original}")
    print(f"Multi-VALUE: {multivalue}")

    # Quick function
    print("\n--- Quick Function ---")
    result = reintegrate_tokens(original, multivalue)
    print(f"Reintegrated: {result}")

    # Full class with verbose output
    print("\n--- Full CoI Process (Rule-based) ---")
    reintegrator = CoIDialectReintegrator(verbose=True)
    result = reintegrator.reintegrate(original, multivalue, use_llm=False)

    print(f"\nFinal Output: {result.final_output}")
    print(f"Tokens Reintegrated: {len(result.tokens_reintegrated)}")
    print(f"Validation Passed: {result.validation_passed}")

    print("\nChain Outputs:")
    for chain in result.chain_outputs:
        print(f"  Chain {chain.chain_id}: {chain.chain_name}")


# ═══════════════════════════════════════════════════════════════════════════
# EXAMPLE 12: CoI Token Reintegration (LLM-based)
# ═══════════════════════════════════════════════════════════════════════════

def example_coi_llm():
    """Demonstrate Chain of Interaction token reintegration with LLM."""
    from coi_reintegration import CoIDialectReintegrator
    from models import OpenAIBackend

    print("CoI Token Reintegration (LLM-based)")
    print("=" * 50)

    # Original text with sensitive tokens
    original = "She is always working 💪 https://example.com @jane #girlboss — amazing work ethic!"

    # Multi-VALUE AAVE output (tokens stripped)
    multivalue = "She be working amazing work ethic"

    print(f"\nOriginal: {original}")
    print(f"Multi-VALUE AAVE: {multivalue}")

    # Create LLM backend
    llm = OpenAIBackend(model="gpt-4")

    # Run CoI reintegration
    reintegrator = CoIDialectReintegrator(llm=llm, verbose=True)
    result = reintegrator.reintegrate(original, multivalue, use_llm=True)

    print(f"\n{'='*50}")
    print(f"FINAL OUTPUT: {result.final_output}")
    print(f"Tokens Reintegrated: {len(result.tokens_reintegrated)}")
    print(f"Validation Passed: {result.validation_passed}")
    if result.corrections_made:
        print(f"Corrections: {result.corrections_made}")


# ═══════════════════════════════════════════════════════════════════════════
# EXAMPLE 13: Full Pipeline - Transform + Reintegrate
# ═══════════════════════════════════════════════════════════════════════════

def example_full_pipeline():
    """Full pipeline: strip tokens, transform with Multi-VALUE sim, reintegrate."""
    from transformer import DialectTransformer
    from coi_reintegration import CoIDialectReintegrator
    import re

    print("Full Pipeline: Strip → Transform → Reintegrate")
    print("=" * 50)

    # Original text with sensitive tokens
    original = "Hey @mike check out https://news.com 🔥🎉 — breaking news! #viral"
    print(f"\nOriginal: {original}")

    # Step 1: Strip sensitive tokens (simulating Multi-VALUE preprocessing)
    stripped = original
    # Remove emoji
    stripped = re.sub(r'[🔥🎉💪👍❤️]+', '', stripped)
    # Remove URLs
    stripped = re.sub(r'https?://\S+', '', stripped)
    # Remove mentions/hashtags
    stripped = re.sub(r'[@#]\w+', '', stripped)
    # Remove special dashes
    stripped = re.sub(r'[—–]', '', stripped)
    # Clean up whitespace
    stripped = ' '.join(stripped.split())
    print(f"Stripped: {stripped}")

    # Step 2: Transform to dialect (auto backend)
    transformer = DialectTransformer(backend="auto")
    result = transformer.transform(stripped, dialect="urban_aave")
    dialect_text = result.transformed
    print(f"Dialect (AAVE): {dialect_text}")

    # Step 3: Reintegrate sensitive tokens
    reintegrator = CoIDialectReintegrator(llm=transformer.llm, verbose=False)
    final_result = reintegrator.reintegrate(original, dialect_text, use_llm=True)
    print(f"Reintegrated: {final_result.final_output}")

    print(f"\nTokens reintegrated: {[t.token for t in final_result.tokens_reintegrated]}")


# ═══════════════════════════════════════════════════════════════════════════
# EXAMPLE 14: LLM-as-a-Judge Evaluation (Single)
# ═══════════════════════════════════════════════════════════════════════════

def example_evaluation_single():
    """Evaluate a single dialect transformation using LLM-as-a-Judge."""
    from evaluation import DialectEvaluator, evaluate_transformation
    from models import OpenAIBackend

    print("LLM-as-a-Judge Evaluation (Single Text)")
    print("=" * 50)

    # Original and transformed texts
    original = "She is always working hard. He doesn't have any money to spend."
    dialect = "She be working hard. He don't got no money to spend."
    dialect_name = "Urban African American Vernacular English"

    print(f"\nOriginal (SAE): {original}")
    print(f"Dialect (AAVE): {dialect}")

    # Create evaluator with OpenAI
    llm = OpenAIBackend(model="gpt-4")
    evaluator = DialectEvaluator(llm=llm, verbose=True)

    # Run comprehensive evaluation
    result = evaluator.evaluate(
        original_text=original,
        dialect_text=dialect,
        dialect_name=dialect_name,
        comprehensive=True
    )

    # Display results
    print(f"\n{'='*50}")
    print("EVALUATION RESULTS")
    print(f"{'='*50}")
    print(f"Overall Score: {result.overall_score}/7")
    print(f"\nDimension Scores:")
    for dim, score in result.dimensions.items():
        print(f"  {dim.replace('_', ' ').title()}: {score.score}/7")
        if score.strengths:
            print(f"    Strengths: {', '.join(score.strengths[:2])}")
        if score.weaknesses:
            print(f"    Weaknesses: {', '.join(score.weaknesses[:2])}")

    print(f"\nSummary: {result.summary}")


# ═══════════════════════════════════════════════════════════════════════════
# EXAMPLE 15: LLM-as-a-Judge Evaluation (Ollama)
# ═══════════════════════════════════════════════════════════════════════════

def example_evaluation_ollama():
    """Evaluate using local Ollama model."""
    from evaluation import DialectEvaluator
    from models import OllamaBackend

    print("LLM-as-a-Judge Evaluation (Ollama Local)")
    print("=" * 50)

    original = "I am going to the store to buy some groceries."
    dialect = "Me a go a di store fi buy some groceries."
    dialect_name = "Jamaican Creole"

    print(f"\nOriginal (SAE): {original}")
    print(f"Dialect (Jamaican): {dialect}")

    # Use Ollama with llama3.1
    llm = OllamaBackend(model="llama3.1")
    evaluator = DialectEvaluator(llm=llm, verbose=True)

    result = evaluator.evaluate(original, dialect, dialect_name)

    print(f"\n{'='*50}")
    print(f"Overall Score: {result.overall_score}/7")
    for dim, score in result.dimensions.items():
        print(f"  {dim}: {score.score}/7")


# ═══════════════════════════════════════════════════════════════════════════
# EXAMPLE 16: Evaluate Individual Dimensions
# ═══════════════════════════════════════════════════════════════════════════

def example_evaluation_dimensions():
    """Evaluate individual dimensions separately."""
    from evaluation import DialectEvaluator
    from models import OpenAIBackend

    print("Individual Dimension Evaluation")
    print("=" * 50)

    original = "They are not going anywhere today."
    dialect = "Dey ain't goin' nowhere today."
    dialect_name = "Urban AAVE"

    print(f"\nOriginal: {original}")
    print(f"Dialect: {dialect}")

    llm = OpenAIBackend(model="gpt-4")
    evaluator = DialectEvaluator(llm=llm, verbose=False)

    # Evaluate each dimension separately
    dimensions = ["fluency", "faithfulness", "dialect_authenticity"]

    for dim in dimensions:
        print(f"\n--- Evaluating {dim.upper()} ---")
        score = evaluator.evaluate_dimension(original, dialect, dialect_name, dim)
        print(f"Score: {score.score}/7")
        print(f"Reasoning: {score.reasoning[:200]}...")
        if score.strengths:
            print(f"Strengths: {score.strengths}")
        if score.weaknesses:
            print(f"Weaknesses: {score.weaknesses}")


# ═══════════════════════════════════════════════════════════════════════════
# EXAMPLE 17: Batch Evaluation with Report
# ═══════════════════════════════════════════════════════════════════════════

def example_evaluation_batch():
    """Evaluate multiple transformations and generate a report."""
    from evaluation import DialectEvaluator, aggregate_results, format_results_report
    from models import OpenAIBackend

    print("Batch Evaluation with Report")
    print("=" * 50)

    # Multiple text pairs
    pairs = [
        {
            "original": "She is always working.",
            "dialect": "She be working."
        },
        {
            "original": "He doesn't have any friends.",
            "dialect": "He don't got no friends."
        },
        {
            "original": "They are going to the store.",
            "dialect": "They finna go to the store."
        },
        {
            "original": "I have not seen him anywhere.",
            "dialect": "I ain't seen him nowhere."
        }
    ]

    print(f"\nEvaluating {len(pairs)} text pairs...")

    llm = OpenAIBackend(model="gpt-4")
    evaluator = DialectEvaluator(llm=llm, verbose=False)

    results = evaluator.evaluate_batch(pairs, "Urban AAVE", comprehensive=True)

    # Generate and print report
    report = format_results_report(results)
    print(report)

    # Also show aggregated stats
    stats = aggregate_results(results)
    print("\nAggregated Statistics:")
    print(f"  Mean Overall Score: {stats['overall']['mean']}/7")
    for dim, dim_stats in stats['dimensions'].items():
        print(f"  {dim}: {dim_stats['mean']}/7")


# ═══════════════════════════════════════════════════════════════════════════
# EXAMPLE 18: Transform + Evaluate Pipeline
# ═══════════════════════════════════════════════════════════════════════════

def example_transform_evaluate():
    """Full pipeline: transform text and evaluate the result."""
    from transformer import DialectTransformer
    from evaluation import DialectEvaluator
    from dialects import get_dialect

    print("Transform + Evaluate Pipeline")
    print("=" * 50)

    original = "She is always working hard at the office. He doesn't have any idea what to do."

    # Transform
    transformer = DialectTransformer(backend="auto")
    transform_result = transformer.transform(original, dialect="urban_aave")

    dialect_info = get_dialect("urban_aave")
    print(f"\nOriginal: {original}")
    print(f"Transformed ({dialect_info['name']}): {transform_result.transformed}")

    # Evaluate
    evaluator = DialectEvaluator(llm=transformer.llm, verbose=False)
    eval_result = evaluator.evaluate(
        original_text=original,
        dialect_text=transform_result.transformed,
        dialect_name=dialect_info['name']
    )

    print(f"\n{'='*50}")
    print("QUALITY ASSESSMENT")
    print(f"{'='*50}")
    print(f"Overall Quality: {eval_result.overall_score}/7")
    print(f"\nBreakdown:")
    for dim, score in eval_result.dimensions.items():
        stars = "★" * score.score + "☆" * (7 - score.score)
        print(f"  {dim.replace('_', ' ').title():25} {stars} ({score.score}/7)")

    print(f"\n{eval_result.summary}")


# ═══════════════════════════════════════════════════════════════════════════
# EXAMPLE 19: Custom Evaluation Prompt
# ═══════════════════════════════════════════════════════════════════════════

def example_evaluation_custom_prompt():
    """Build custom evaluation prompts for external use."""
    from evaluation import build_evaluation_prompt

    print("Custom Evaluation Prompts")
    print("=" * 50)

    original = "She is working at home today."
    dialect = "She working at home today."
    dialect_name = "Singlish"

    # Build prompts for different dimensions
    print("\n--- Fluency Prompt (first 500 chars) ---")
    fluency_prompt = build_evaluation_prompt(original, dialect, dialect_name, "fluency")
    print(fluency_prompt[:500] + "...")

    print("\n--- Comprehensive Prompt (first 500 chars) ---")
    comprehensive_prompt = build_evaluation_prompt(original, dialect, dialect_name, "comprehensive")
    print(comprehensive_prompt[:500] + "...")

    print("\n(Use these prompts with any LLM of your choice)")


# ═══════════════════════════════════════════════════════════════════════════
# EXAMPLE 20: Feature Accuracy Evaluation (eWAVE Alignment)
# ═══════════════════════════════════════════════════════════════════════════

def example_evaluation_feature_accuracy():
    """Evaluate feature accuracy based on eWAVE linguistic specifications."""
    from evaluation import DialectEvaluator
    from models import OpenAIBackend

    print("Feature Accuracy Evaluation (eWAVE Alignment)")
    print("=" * 50)

    # Test cases with different feature applications
    test_cases = [
        {
            "original": "She is always working hard. He doesn't have any money.",
            "dialect": "She be working hard. He don't got no money.",
            "dialect_name": "Urban African American Vernacular English",
            "expected_features": ["habitual be", "negative concord", "don't for doesn't"]
        },
        {
            "original": "I am going to the store. We were not there.",
            "dialect": "I finna go to the store. We wasn't there.",
            "dialect_name": "Urban African American Vernacular English",
            "expected_features": ["finna (fixing to)", "was/were leveling"]
        }
    ]

    llm = OpenAIBackend(model="gpt-4")
    evaluator = DialectEvaluator(llm=llm, verbose=False)

    for i, case in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i} ---")
        print(f"Original: {case['original']}")
        print(f"Dialect: {case['dialect']}")
        print(f"Expected eWAVE features: {case['expected_features']}")

        # Evaluate feature accuracy specifically
        result = evaluator.evaluate_dimension(
            original_text=case["original"],
            dialect_text=case["dialect"],
            dialect_name=case["dialect_name"],
            dimension="feature_accuracy"
        )

        print(f"\nFeature Accuracy Score: {result.score}/7")

        if result.features_present:
            print(f"Features Present: {result.features_present[:5]}")

        if result.features_missing:
            print(f"Features Missing: {result.features_missing[:3]}")

        if result.strengths:
            print(f"Strengths: {result.strengths[:2]}")

        if result.weaknesses:
            print(f"Weaknesses: {result.weaknesses[:2]}")


# ═══════════════════════════════════════════════════════════════════════════
# RUN EXAMPLES
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    examples = {
        "1": ("OpenAI API", example_openai),
        "2": ("Ollama (Local)", example_ollama),
        "3": ("HuggingFace (Local)", example_huggingface),
        "4": ("Auto-Detection", example_auto),
        "5": ("Batch Processing", example_batch),
        "6": ("Custom Features", example_custom_features),
        "7": ("Special Characters", example_special_chars),
        "8": ("Compare Dialects", example_compare_dialects),
        "9": ("Transformation Strength", example_strength),
        "10": ("Quick One-Liner", example_quick),
        "11": ("CoI Reintegration (Rule-based)", example_coi_rulebased),
        "12": ("CoI Reintegration (LLM)", example_coi_llm),
        "13": ("Full Pipeline", example_full_pipeline),
        "14": ("LLM-as-a-Judge (Single)", example_evaluation_single),
        "15": ("LLM-as-a-Judge (Ollama)", example_evaluation_ollama),
        "16": ("Evaluate Dimensions", example_evaluation_dimensions),
        "17": ("Batch Evaluation", example_evaluation_batch),
        "18": ("Transform + Evaluate", example_transform_evaluate),
        "19": ("Custom Eval Prompts", example_evaluation_custom_prompt),
        "20": ("Feature Accuracy (eWAVE)", example_evaluation_feature_accuracy),
    }

    print("\nDialect Transformer Examples")
    print("=" * 50)

    if len(sys.argv) > 1:
        choice = sys.argv[1]
        if choice in examples:
            name, func = examples[choice]
            print(f"\nRunning Example {choice}: {name}\n")
            try:
                func()
            except Exception as e:
                print(f"Error: {e}")
                print("Make sure you have the required backend installed.")
        else:
            print(f"Unknown example: {choice}")
    else:
        print("\nAvailable examples:")
        for key, (name, _) in examples.items():
            print(f"  {key}: {name}")
        print("\nRun with: python examples.py <number>")
        print("Example: python examples.py 4  (for Auto-Detection)")
