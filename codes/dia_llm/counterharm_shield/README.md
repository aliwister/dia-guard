# CounterHarm-SHIELD

**CounterHarm-SHIELD** (Safety Harm Identification in English Language Dialects) generates validated benign counterexamples to harmful content across 48 English dialects. It is a sub-module of **Dia-LLM** within the **DIA-GUARD** framework.

```
DIA-GUARD
└── Dia-LLM (LLM-based transformation)
    ├── CounterHarm-SHIELD  ← this module
    │     Generate validated benign counterexamples
    └── Harm-SHIELD
          Transform SAE harmful content into dialectal variants
```

## Pipeline (6-Chain CoI)

Each input text goes through a 6-chain Chain-of-Interactions pipeline:

| Chain | Name | Framework |
|-------|------|-----------|
| 1 | Harmful attribute extraction | ToxiCraft |
| 2 | Benign prompt construction | ToxiCraft |
| 3 | Contextual anchoring (CAE) | ToxiCraft |
| 4 | Thematic style refinement (TSR) | ToxiCraft |
| 5 | Gated harmlessness scoring | PromptSafe |
| 6 | Counterfactual label validation | FIZLE |

Chains 3-6 retry up to 2 times if validation fails (max 3 attempts per cell).

## Output Columns

For each row, the pipeline generates 4 benign counterexample columns:

| Output Column | Source Column |
|---------------|---------------|
| `counterharm_original` | `original_input` |
| `counterharm_transformed` | `transformed_input` |
| `counterharm_basic` | `basic_transform` |
| `counterharm_coi` | `coi_transform` |

If a source column contains a guardrail policy violation, the corresponding output is skipped (per-cell, not per-row).

## Length Mirroring

The pipeline constrains output length to match the source input, producing benign counterexamples with approximately the same word count and sentence structure as the original harmful text.

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
# Run with Gemini 3.1 and 10 workers on a specific dialect
python full_generation_parallel.py --model gemini3.1 --workers 10 --dialect welsh_english

# Run with DeepSeek and 2 workers
python full_generation_parallel.py --model deepseek --workers 2

# Run all dialects (default)
python full_generation_parallel.py --model gemini --workers 10
```

## Resume

The pipeline automatically resumes from the last completed row. Rows with all output columns filled (or source columns with policy violations) are skipped.

## Files

| File | Description |
|------|-------------|
| `counterharm_pipeline.py` | 6-chain CoI pipeline implementation |
| `full_generation_parallel.py` | Parallel generation script with resume logic |
