#!/usr/bin/env python3
"""
Dialect Transformer - Transform Standard American English to 50 English dialects

Usage:
    # Using OpenAI API
    python main.py --backend openai --dialect urban_aave "She is always working"

    # Using Ollama (local)
    python main.py --backend ollama --model llama3.1 --dialect jamaican "I am going to the store"

    # Using HuggingFace (local)
    python main.py --backend huggingface --dialect indian "She doesn't have any money"

    # Interactive mode
    python main.py --interactive --dialect appalachian

    # List all dialects
    python main.py --list-dialects
"""

import argparse
import sys
import json
from typing import Optional

from transformer import DialectTransformer, list_available_dialects, get_dialect_info
from models import ModelFactory, OpenAIBackend, OllamaBackend, HuggingFaceBackend, AnthropicBackend
from dialects import REGIONS, DIALECT_REGISTRY


def parse_args():
    parser = argparse.ArgumentParser(
        description="Transform Standard American English to various English dialects",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Transform text using OpenAI
  python main.py --backend openai --dialect urban_aave "She is always working"

  # Transform using local Ollama
  python main.py --backend ollama --model llama3.1 --dialect jamaican "I have eaten"

  # Interactive mode
  python main.py --interactive --dialect appalachian

  # List all available dialects
  python main.py --list-dialects

  # Show dialect details
  python main.py --dialect-info urban_aave
        """
    )

    # Backend selection
    parser.add_argument(
        "--backend", "-b",
        choices=["auto", "openai", "anthropic", "ollama", "huggingface", "vllm"],
        default="auto",
        help="LLM backend to use (default: auto-detect)"
    )

    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Model name/path (e.g., gpt-4, llama3.1, mistral)"
    )

    # Dialect selection
    parser.add_argument(
        "--dialect", "-d",
        type=str,
        help="Target dialect (e.g., urban_aave, jamaican, indian)"
    )

    # Input text
    parser.add_argument(
        "text",
        nargs="*",
        help="Text to transform (or use --input-file)"
    )

    parser.add_argument(
        "--input-file", "-i",
        type=str,
        help="Read input from file"
    )

    parser.add_argument(
        "--output-file", "-o",
        type=str,
        help="Write output to file"
    )

    # Options
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )

    parser.add_argument(
        "--temperature", "-t",
        type=float,
        default=0.3,
        help="Sampling temperature (default: 0.3)"
    )

    parser.add_argument(
        "--strength", "-s",
        type=float,
        default=None,
        help="Transformation strength 0.0-1.0 (default: use dialect default)"
    )

    parser.add_argument(
        "--concise",
        action="store_true",
        help="Use concise prompts (for smaller models)"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print debug information"
    )

    # Info commands
    parser.add_argument(
        "--list-dialects",
        action="store_true",
        help="List all available dialects"
    )

    parser.add_argument(
        "--list-regions",
        action="store_true",
        help="List dialects by region"
    )

    parser.add_argument(
        "--dialect-info",
        type=str,
        metavar="DIALECT",
        help="Show detailed info about a dialect"
    )

    parser.add_argument(
        "--list-backends",
        action="store_true",
        help="List available LLM backends"
    )

    # Model-specific options
    parser.add_argument(
        "--api-key",
        type=str,
        help="API key for OpenAI/Anthropic"
    )

    parser.add_argument(
        "--ollama-host",
        type=str,
        default="http://localhost:11434",
        help="Ollama server URL"
    )

    parser.add_argument(
        "--quantize",
        choices=["4bit", "8bit", "none"],
        default="none",
        help="Quantization for HuggingFace models"
    )

    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process input as batch (one text per line)"
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON"
    )

    return parser.parse_args()


def list_dialects_command():
    """List all available dialects."""
    print("\n" + "=" * 70)
    print("AVAILABLE DIALECTS")
    print("=" * 70)

    dialects = list_available_dialects()
    for key, name in sorted(dialects.items()):
        strength = DIALECT_REGISTRY[key].get("strength", 1.0)
        print(f"  {key:30} {name} (strength: {strength})")

    print(f"\nTotal: {len(dialects)} dialects")
    print("=" * 70)


def list_regions_command():
    """List dialects organized by region."""
    print("\n" + "=" * 70)
    print("DIALECTS BY REGION")
    print("=" * 70)

    for region, dialect_keys in REGIONS.items():
        region_name = region.replace("_", " ").title()
        print(f"\n{region_name}:")
        print("-" * 40)
        for key in dialect_keys:
            if key in DIALECT_REGISTRY:
                d = DIALECT_REGISTRY[key]
                print(f"  {key:30} {d['name']}")

    print("\n" + "=" * 70)


def dialect_info_command(dialect: str):
    """Show detailed info about a dialect."""
    try:
        info = get_dialect_info(dialect)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print("\n" + "=" * 70)
    print(f"DIALECT: {info['name']}")
    print("=" * 70)
    print(f"Key: {dialect}")
    print(f"Region: {info['region']}")
    print(f"Strength: {info.get('strength', 1.0)}")

    if info.get("description"):
        print(f"\nDescription:")
        print(f"  {info['description']}")

    if info.get("notes"):
        print(f"\nNotes:")
        print(f"  {info['notes']}")

    print(f"\nFeatures ({len(info.get('features', []))}):")
    for feat in info.get("features", []):
        print(f"  - {feat}")

    print("=" * 70)


def list_backends_command():
    """List available LLM backends."""
    print("\n" + "=" * 70)
    print("AVAILABLE LLM BACKENDS")
    print("=" * 70)

    backends = ModelFactory.list_available()
    if backends:
        for backend in backends:
            print(f"  ✓ {backend}")
    else:
        print("  No backends currently available")

    print("\nBackend options:")
    print("  openai      - OpenAI API (GPT-4, GPT-3.5)")
    print("  anthropic   - Anthropic API (Claude)")
    print("  ollama      - Local models via Ollama")
    print("  huggingface - Local models via Transformers")
    print("  vllm        - High-performance local inference")
    print("=" * 70)


def create_llm_backend(args):
    """Create LLM backend from arguments."""
    if args.backend == "openai":
        model = args.model or "gpt-4"
        return OpenAIBackend(model=model, api_key=args.api_key)

    elif args.backend == "anthropic":
        model = args.model or "claude-3-5-sonnet-20241022"
        return AnthropicBackend(model=model, api_key=args.api_key)

    elif args.backend == "ollama":
        model = args.model or "llama3.1"
        return OllamaBackend(model=model, host=args.ollama_host)

    elif args.backend == "huggingface":
        model = args.model or "microsoft/Phi-3-mini-4k-instruct"
        load_in_4bit = args.quantize == "4bit"
        load_in_8bit = args.quantize == "8bit"
        return HuggingFaceBackend(
            model=model,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit
        )

    elif args.backend == "vllm":
        from models import VLLMBackend
        model = args.model or "meta-llama/Meta-Llama-3.1-8B-Instruct"
        return VLLMBackend(model=model)

    else:  # auto
        return ModelFactory.auto_detect()


def main():
    args = parse_args()

    # Handle info commands
    if args.list_dialects:
        list_dialects_command()
        return

    if args.list_regions:
        list_regions_command()
        return

    if args.dialect_info:
        dialect_info_command(args.dialect_info)
        return

    if args.list_backends:
        list_backends_command()
        return

    # Require dialect for transformation
    if not args.dialect:
        print("Error: --dialect is required for transformation")
        print("Use --list-dialects to see available options")
        sys.exit(1)

    # Validate dialect
    if args.dialect not in DIALECT_REGISTRY:
        print(f"Error: Unknown dialect '{args.dialect}'")
        print("Use --list-dialects to see available options")
        sys.exit(1)

    # Create LLM backend
    try:
        llm = create_llm_backend(args)
    except Exception as e:
        print(f"Error creating LLM backend: {e}")
        sys.exit(1)

    # Create transformer
    transformer = DialectTransformer(
        llm=llm,
        default_temperature=args.temperature,
        verbose=args.verbose
    )

    # Interactive mode
    if args.interactive:
        transformer.transform_interactive(
            args.dialect,
            prompt_mode="concise" if args.concise else "full"
        )
        return

    # Get input text
    if args.input_file:
        with open(args.input_file, "r") as f:
            if args.batch:
                texts = [line.strip() for line in f if line.strip()]
            else:
                texts = [f.read()]
    elif args.text:
        texts = [" ".join(args.text)]
    else:
        print("Error: No input text provided")
        print("Use positional argument, --input-file, or --interactive")
        sys.exit(1)

    # Transform
    results = []
    for text in texts:
        result = transformer.transform(
            text,
            args.dialect,
            strength=args.strength,
            use_concise_prompt=args.concise
        )
        results.append(result)

    # Output
    if args.json:
        output = [{
            "original": r.original,
            "transformed": r.transformed,
            "dialect": r.dialect,
            "dialect_name": r.dialect_name,
            "model": r.model_used
        } for r in results]
        output_text = json.dumps(output, indent=2)
    else:
        if len(results) == 1:
            output_text = results[0].transformed
        else:
            output_text = "\n".join([
                f"[{i}] {r.transformed}"
                for i, r in enumerate(results)
            ])

    if args.output_file:
        with open(args.output_file, "w") as f:
            f.write(output_text)
        if args.verbose:
            print(f"Output written to {args.output_file}")
    else:
        print(output_text)


if __name__ == "__main__":
    main()
