# Harm-SHIELD

**Harm-SHIELD** (Safety Harm Identification in English Language Dialects) transforms Standard American English (SAE) harmful content into 48 English dialect variants using LLMs. It is a sub-module of **Dia-LLM** within the **DIA-GUARD** framework.

```
DIA-GUARD
└── Dia-LLM (LLM-based transformation)
    ├── CounterHarm-SHIELD
    │     Generate validated benign counterexamples
    └── Harm-SHIELD  ← this module
          Transform SAE harmful content into dialectal variants
```

## Features

- **48 English Dialects**: Based on [eWAVE](https://ewave-atlas.org) (Electronic World Atlas of Varieties of English)
- **235 Linguistic Features**: Across 12 grammatical categories with AB-rated feature constraints
- **Basic Transform**: Direct SAE-to-dialect transformation with eWAVE feature guidance
- **Chain of Interaction (CoI) Transform**: 4-chain agentic dialect transformation with dual attention and strict eWAVE feature constraints
- **Three-Category eWAVE Validation**: Validates transformations against eWAVE database (valid features, wrong dialect, non-eWAVE errors)
- **Research Framing**: System prompts include computational linguistics research context to reduce content filter refusals
- **Resume Logic**: Automatically resumes from last completed row; retries rows with guardrail policy violations
- **OOV-Robust**: Preserves URLs, emoji, @mentions, #hashtags, code blocks

## Pipeline (8 API Calls per Row)

| Step | Description |
|------|-------------|
| Basic transform | Direct SAE-to-dialect transformation |
| CoI chain 1 | Dialect feature analysis |
| CoI chain 2 | Initial transformation |
| CoI chain 3 | Dual attention refinement |
| CoI chain 4 | Final transformation with eWAVE constraints |
| Validation 1 | eWAVE feature validation |
| Validation 2 | Dialect authenticity check |
| Validation 3 | Content preservation check |

## Output Columns

| Column | Description |
|--------|-------------|
| `basic_transform` | Direct dialect transformation |
| `coi_transform` | 4-chain CoI dialect transformation |
| `validation_*` | Validation scores and details |

## Supported Models

| Flag | Model |
|------|-------|
| `--model gpt4.1` | Azure GPT-4.1 |
| `--model deepseek` | DeepSeek-V3.2 |
| `--model kimi` | Kimi-K2.5 |
| `--model gemini` | Gemini 3.1 Flash Lite Preview |
| `--model gemini3.1` | Gemini 3.1 Flash Lite Preview |
| `--model gemini2.5` | Gemini 2.5 Flash Lite |

## Usage

```bash
# Run with DeepSeek and 100 workers on all dialects
python full_generation_parallel.py --model deepseek --workers 100

# Run with Gemini on a specific dialect
python full_generation_parallel.py --model gemini3.1 --workers 30 --dialect welsh_english

# Test mode (1 row only)
python full_generation_parallel.py --model gpt4.1 --workers 1 --test
```

## Resume

The pipeline automatically resumes from the last completed row. Rows where `basic_transform` or `coi_transform` contains "guardrail policy violation" are retried; successful regenerations replace the violation marker.

## Key Files

| File | Description |
|------|-------------|
| `full_generation_parallel.py` | Parallel generation script with resume logic |
| `transformer.py` | Basic SAE-to-dialect transformer |
| `coi_transformation.py` | 4-chain CoI dialect transformer |
| `models.py` | LLM backend implementations (Azure OpenAI, Gemini) |
| `feature_validator.py` | LLM-based eWAVE validation |
| `dialects.py` | Dialect definitions and eWAVE feature mappings |
| `ewave_data/` | eWAVE feature database |
