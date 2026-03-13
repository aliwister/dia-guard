# Multi-VALUE Dialect Generation Pipeline

This pipeline transforms safety and security benchmark datasets into 50 English dialect varieties using the [Multi-VALUE](https://github.com/SALT-NLP/multi-value) framework ([project page](https://value-nlp.github.io/multivalue/)).

## Differences from Original Multi-VALUE

The original [Multi-VALUE](https://github.com/SALT-NLP/multi-value) library (`value-nlp`) provides dialect transformation classes but is designed for interactive, single-threaded use. When applied at scale to large benchmark datasets, several failure modes emerge:

| Issue | Original Multi-VALUE | This Pipeline |
|-------|---------------------|---------------|
| **Hanging transforms** | Some dialect classes hang indefinitely on certain inputs due to blocking C extension calls (spaCy/stanza). No timeout mechanism exists. | Each dialect runs in a **dedicated subprocess** via `multiprocessing`. Hung transforms are killed after a configurable timeout (default: 30s) using `Process.terminate()`. Threading cannot kill C extensions blocked by the GIL — multiprocessing can. |
| **Network download failures** | spaCy, NLTK, and stanza attempt network downloads on every instantiation, causing hangs or crashes when GitHub/CDN is unreachable. | **Offline patches** monkey-patch all three libraries to skip network calls. Resources are verified once at startup, then `DownloadMethod.REUSE_RESOURCES` is enforced for all subsequent loads. |
| **PyTorch 2.6+ incompatibility** | stanza model loading fails with `weights_only=True` (the PyTorch 2.6+ default) because stanza checkpoints contain numpy globals not in the allowlist. | `torch.load()` is **monkey-patched** to default to `weights_only=False` in both the main process and subprocess. |
| **No resumability** | No checkpoint system. A crash after hours of processing means starting over. | **SQLite-based checkpoint** system with two-tier tracking (row-level + combination-level). Use `--resume` to continue from where you left off. |
| **Code corruption** | Transforming code-mixed prompts (e.g., SecurityEval, CyberSecEval) corrupts source code syntax. | **Code-aware transformation**: extracts NL text from docstrings/comments, transforms only those portions, and reconstructs the prompt with code blocks preserved verbatim. |
| **Consecutive failure cascades** | No detection of systematic failures — the pipeline grinds through thousands of doomed rows. | After **20 consecutive failures** for a dialect, the pipeline bails out and moves to the next dialect, saving time. |
| **Windows compatibility** | Some multiprocessing patterns (fork-based) don't work on Windows. | Explicit `spawn` context used throughout; worker functions are top-level and picklable. |

## Project Structure

```
multi_value_gen/
├── dialect_transform_pipeline.py   # Main entry point & orchestration
├── dialect_utils.py                # Subprocess-based dialect workers & NLP setup
├── dataset_configs.py              # Dataset registry (9 benchmarks)
├── progress_tracker.py             # SQLite checkpoint system
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Supported Datasets

The pipeline supports 9 safety/security benchmark datasets, processed smallest-first:

| Dataset | Source | Size | Text Columns |
|---------|--------|------|-------------|
| SecurityEval | HuggingFace | 121 | Prompt (code-mixed) |
| LLMSecEval | GitHub CSV | 150 | LLM-generated NL Prompt, Manually-fixed NL Prompt |
| JailbreakBench | HuggingFace | 200 | Goal, Target |
| SORRY-Bench | HuggingFace | ~450 | turns (list of strings) |
| AdvBench | HuggingFace | 520 | prompt |
| Do-Not-Answer | HuggingFace | 939 | question |
| InjecAgent | GitHub JSON | 1,054 | User Instruction |
| CyberSecEval | HuggingFace | ~3,840 | prompt (code-mixed) |
| BIPIA | HuggingFace | ~70K (sampled to 5K) | context, user_intent |

## 50 English Dialect Varieties

All 50 dialects from Multi-VALUE are supported, spanning varieties from across the English-speaking world including Aboriginal English, African American Vernacular English, Appalachian English, Australian English, Jamaican English, Scottish English, and 44 others. See the [Multi-VALUE documentation](https://value-nlp.github.io/multivalue/) for the full list.

## Installation

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
# or: venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Download required NLP models (done automatically on first run, but can be pre-cached)
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('cmudict'); nltk.download('wordnet')"
python -c "import stanza; stanza.download('en')"
```

## Usage

### Basic Usage

```bash
# Transform all 9 datasets across all 50 dialects
python dialect_transform_pipeline.py --output-dir ./output

# With a HuggingFace token (needed for some gated datasets)
python dialect_transform_pipeline.py --output-dir ./output --hf-token hf_YOUR_TOKEN
```

### Selective Processing

```bash
# Process only specific datasets
python dialect_transform_pipeline.py --datasets securityeval llmseceval

# Process only specific dialects
python dialect_transform_pipeline.py --dialects AppalachianDialect ScottishDialect

# Combine both
python dialect_transform_pipeline.py --datasets advbench --dialects JamaicanDialect
```

### Resuming After Interruption

```bash
# If the pipeline is interrupted, resume from the last checkpoint
python dialect_transform_pipeline.py --output-dir ./output --resume
```

The `--resume` flag reads from `output/progress.db` (SQLite) to skip completed dataset-dialect combinations and partially-completed rows.

### Tuning Timeouts

```bash
# Increase per-transform timeout for slow dialects (default: 30s)
python dialect_transform_pipeline.py --timeout 60

# Set random seed for reproducibility
python dialect_transform_pipeline.py --seed 42
```

## Output Format

Results are saved as JSON files organized by dataset and dialect:

```
output/
├── progress.db                              # SQLite checkpoint database
├── securityeval/
│   ├── 01_AboriginalDialect/
│   │   └── securityeval.json
│   ├── 02_AfricanAmericanVernacular/
│   │   └── securityeval.json
│   └── ...
├── advbench/
│   └── ...
└── ...
```

Each JSON file contains:

```json
{
  "metadata": {
    "source_dataset": "securityeval",
    "dialect_name": "AppalachianDialect",
    "num_records": 121,
    "num_transform_errors": 3,
    "transformed_columns": ["Prompt"],
    "timestamp": "2026-03-02T23:27:09.496469+00:00"
  },
  "data": [
    {
      "original_index": 0,
      "ID": "SEC-001",
      "Insecure_code": "[preserved verbatim]",
      "Prompt_original": "Write a function that...",
      "Prompt_dialect": "Write ye a function that...",
      "transform_success": true
    }
  ]
}
```

## Architecture

```
                    ┌──────────────────────────────────┐
                    │  dialect_transform_pipeline.py   │
                    │  (main orchestration)            │
                    └────────┬─────────────────────────┘
                             │
          ┌──────────────────┼──────────────────┐
          ▼                  ▼                  ▼
   dataset_configs.py  progress_tracker.py  dialect_utils.py
   (9 dataset specs)   (SQLite checkpoints) (subprocess workers)
                                                │
                                    ┌───────────┴───────────┐
                                    ▼                       ▼
                             Main Process            Subprocess
                            ┌───────────┐          ┌───────────┐
                            │ Send text  │──Queue──▶│ Dialect   │
                            │ via Queue  │          │ instance  │
                            │            │◀─Queue──│ transform │
                            │ Timeout    │          │           │
                            │ handling   │  kill ──▶│ (killable)│
                            └───────────┘          └───────────┘
```

Each dialect gets its own subprocess. If a transform hangs beyond the timeout, `Process.terminate()` kills the subprocess — something impossible with threads due to the GIL.

## Adding New Datasets

To add a new dataset, add an entry to `dataset_configs.py`:

```python
DATASET_CONFIGS["my_dataset"] = {
    "loader": "huggingface",           # or "github_json", "github_csv"
    "hf_path": "org/dataset-name",
    "hf_config": None,
    "hf_split": "test",
    "text_columns": ["prompt"],        # Columns to transform
    "code_columns": [],                # Columns with code (preserved)
    "preserve_columns": ["label"],     # Extra columns to keep
    "contains_code_mixed": False,      # Set True for code-mixed prompts
    "estimated_size": 500,
}
```

Then add the key to `PROCESSING_ORDER` in the same file.

## Citation

If you use this pipeline, please cite both Multi-VALUE and DIA-Guard:

```bibtex
@inproceedings{ziems2023multivalue,
    title={Multi-VALUE: A Framework for Cross-Dialectal English NLP},
    author={Ziems, Caleb and Held, William and Yang, Jingfeng and others},
    booktitle={ACL},
    year={2023}
}
```

## License

This pipeline builds on [Multi-VALUE](https://github.com/SALT-NLP/multi-value) (MIT License).
