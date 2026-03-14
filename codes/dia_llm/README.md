# Dia-LLM

**Dia-LLM** is the LLM-based dialect transformation module of the **DIA-GUARD** framework. It generates dialect-transformed harmful content and validated benign counterexamples across 48 English dialects for the SHIELD dataset.

```
DIA-GUARD
└── Dia-LLM  ← this module
    ├── harm_shield/        Dialect transformation of harmful content
    └── counterharm_shield/ Benign counterexample generation
```

## Sub-modules

### [harm_shield](harm_shield/)

Transforms Standard American English (SAE) harmful text into 48 English dialect variants using a basic transformer and a 4-chain Chain-of-Interaction (CoI) pipeline. Validates transformations against the [eWAVE](https://ewave-atlas.org) linguistic feature database.

- **8 API calls per row** (basic transform + CoI 4-chain + 3 validations)
- Outputs: `basic_transform`, `coi_transform`, validation scores

### [counterharm_shield](counterharm_shield/)

Generates validated benign counterexamples for each harmful input using a 6-chain CoI pipeline (ToxiCraft + PromptSafe + FIZLE). Outputs mirror the length and structure of the source harmful text.

- **24 API calls per row** (6-chain pipeline x 4 text columns)
- Outputs: `counterharm_original`, `counterharm_transformed`, `counterharm_basic`, `counterharm_coi`

## Dataset

Processes 15 benchmark datasets across 48 English dialects (~2,500 rows per dialect):

| Dataset | Rows |
|---------|------|
| Salad_Bench | 200 |
| Simple_Safety_Tests | 100 |
| Toxic_Text | 200 |
| Toxicity_Jigsaw | 200 |
| advbench | 200 |
| bipia | 91 |
| cyberseceval | 200 |
| do_not_answer | 200 |
| forbiddent_questions | 200 |
| harmBench | 200 |
| injecagent | 200 |
| jailbreakbench | 200 |
| llmseceval | 150 |
| securityeval | 121 |
| sorry_bench | 200 |

## Supported Models

| Flag | Model |
|------|-------|
| `gpt4.1` | Azure GPT-4.1 |
| `deepseek` | DeepSeek-V3.2 |
| `kimi` | Kimi-K2.5 |
| `gemini` / `gemini3.1` | Gemini 3.1 Flash Lite Preview |
| `gemini2.5` | Gemini 2.5 Flash Lite |

## Quick Start

```bash
# Harm-SHIELD: transform harmful content into dialects
cd harm_shield
python full_generation_parallel.py --model deepseek --workers 100

# CounterHarm-SHIELD: generate benign counterexamples
cd counterharm_shield
python full_generation_parallel.py --model gemini3.1 --workers 10 --dialect welsh_english
```

Both pipelines support automatic resume from the last completed row.
