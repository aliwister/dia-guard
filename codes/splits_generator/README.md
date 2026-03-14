# Splits Generator — DIA-GUARD

Generates stratified train/val/test splits from DIA-GUARD LLM data and writes them to `../DIA_Splits/`.

Uploaded to: `https://github.com/jsl5710/dia-guard/tree/main/codes/splits_generator`

---

## Data Sources

| Source | Description | Format |
|--------|-------------|--------|
| LLM-based (`--llm_data_dir`) | Per-dialect CSVs with dialectal transformations and counterharm | CSV |
| Rule-based (`--rule_data_dir`) | Multi-value EWAVE rule-based transformations | CSV |

**LLM data structure:** `{llm_data_dir}/{dialect}/{dataset}_zeroshot_harmfulness_results_with_transforms.csv`
- 48 dialects × 15 datasets

---

## Usage

```bash
conda env create -f environment.yml
conda activate dia_splits

# Full run (all dialects, all datasets)
python generate_splits.py \
    --llm_data_dir ../LLM_Data \
    --output_dir ../DIA_Splits \
    --train_ratio 0.70 \
    --val_ratio 0.15 \
    --test_ratio 0.15 \
    --seed 42

# With rule-based data
python generate_splits.py \
    --llm_data_dir ../LLM_Data \
    --rule_data_dir /path/to/multi_value_gen \
    --output_dir ../DIA_Splits \
    --seed 42

# Specific dialects only
python generate_splits.py \
    --llm_data_dir ../LLM_Data \
    --dialects aboriginal_english australian_english black_south_african_english \
    --output_dir ../DIA_Splits

# Dry run (print stats only, no files written)
python generate_splits.py --llm_data_dir ../LLM_Data --dry_run
```

---

## Output

```
DIA_Splits/
├── splits_metadata.json    ← statistics (sizes, label/dialect/dataset distributions)
├── train.jsonl
├── val.jsonl
├── test.jsonl
└── by_dialect/
    └── {dialect}/
        ├── train.jsonl
        ├── val.jsonl
        └── test.jsonl
```

---

## Text Variants per Source Row

Each CSV row is expanded into up to **8 training records** (rows with empty text are skipped):

| CSV Column | `text_type` | Label | Description |
|------------|-------------|-------|-------------|
| `original_input` | `original` | 1 (unsafe) | SAE original text |
| `transformed_input` | `transformed` | 1 (unsafe) | Full dialect transformation |
| `basic_transform` | `basic_transform` | 1 (unsafe) | Basic dialect features |
| `coi_transform` | `coi_transform` | 1 (unsafe) | Coarse-grained dialect features |
| `counterharm_original` | `counterharm_original` | 0 (safe) | Benign ↔ original_input |
| `counterharm_transformed` | `counterharm_transformed` | 0 (safe) | Benign ↔ transformed_input |
| `counterharm_basic` | `counterharm_basic` | 0 (safe) | Benign ↔ basic_transform |
| `counterharm_coi` | `counterharm_coi` | 0 (safe) | Benign ↔ coi_transform |

**NOT used** (metadata only): `counterharm_score`, `counterharm_validated`, `counterharm_model`

---

## Sample ID Format

Every record gets a **globally unique, traceable ID**:

```
{dataset}__{dialect}__{source_row_id}__{text_type}
```

Example: `advbench__aboriginal_english__0__basic_transform`

This lets you trace any model prediction back to its exact source row and column.

---

## Contrastive Training

Each harmful record's `neg_text` field is pre-populated with its paired benign counterpart text (e.g. `original_input` → `counterharm_original`), ready for triplet-loss training.

---

## Split Strategy

Stratified by **(dialect × label)** stratum so every dialect and both label classes are represented proportionally in train/val/test.
