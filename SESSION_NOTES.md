# DIA-GUARD Development Session Notes
**Last Updated:** 2026-03-14
**Repo:** https://github.com/jsl5710/dia-guard
**Local Base:** `/home/azureuser/cloudfiles/code/Users/jsl5710/DIA-LLM/`

---

## What Was Built and Pushed

### 1. `codes/evaluation/` — Training & Evaluation Pipeline

Full pipeline for fine-tuning teacher models (>2B) then distilling into student models (<2B).

#### Fine-Tuning (`codes/evaluation/FineTune/`)
| Script | Description |
|--------|-------------|
| `FineTune/full_ft/train_ce.py` | Full fine-tuning with CE loss via `SFTTrainer` (`processing_class=tokenizer`, `completion_only_loss=True`) |
| `FineTune/full_ft/train_contrastive.py` | Full FT with triplet contrastive loss + CE (custom Accelerator loop) |
| `FineTune/peft/train_ce_lora.py` | LoRA/QLoRA fine-tuning with CE loss; `use_rslora=True` |
| `FineTune/peft/train_contrastive_lora.py` | LoRA + contrastive loss |
| `FineTune/peft/merge_lora.py` | Merge LoRA adapter → base model before using as teacher |
| `FineTune/full_ft/configs/qwen3_4b.yaml` | Config for `Qwen/Qwen3-4B-SafeRL` |
| `FineTune/full_ft/configs/aya_3b.yaml` | Config for `CohereLabs/tiny-aya-global` |
| `FineTune/full_ft/environment.yml` | Conda env `dia_full_ft` |
| `FineTune/peft/environment.yml` | Conda env `dia_peft` |

**Key implementation details:**
- `trl==0.17.0`: use `SFTConfig(completion_only_loss=True)` + `SFTTrainer(processing_class=tokenizer, ...)`
- `use_rslora=True` in `LoraConfig` for rank-stabilized scaling (recommended for r≥16)
- Contrastive loss: triplet on last non-padding hidden state; `α·L_CE + (1-α)·L_contrastive`, default α=0.7

**Teacher models (>2B):**
- `Qwen/Qwen3-4B-SafeRL` (4B)
- `CohereLabs/tiny-aya-global` (3B)

#### Knowledge Distillation (`codes/evaluation/Knowledge_Distillation/`)
| Method | Script | Loss |
|--------|--------|------|
| MINILLM | `minillm/train_minillm.py` | Reverse KL via REINFORCE + EMA baseline + SFT regularization |
| GKD | `gkd/train_gkd.py` | On-policy mixing (FKL/RKL/JSD/TVD token-level divergence) |
| TED | `ted/train_ted.py` | `α·L_CE + β·L_KD + γ·L_layer` with task-aware projection filters |

**Student models (<2B):**
- `Qwen/Qwen3Guard-Gen-0.6B`, `google/gemma-3-270m-it`, `meta-llama/Llama-3.2-1B-Instruct`
- `google/gemma-3-1b-it`, `Qwen/Qwen3.5-0.8B`, `Qwen/Qwen3-1.7B`
- `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`, `HuggingFaceTB/SmolLM2-1.7B-Instruct`

**Critical padding side fix** (in both MINILLM and GKD): switch `tokenizer.padding_side = "left"` before `model.generate()`, restore `"right"` before loss computation.

#### Experiment Orchestrator (`codes/evaluation/run_experiment.py`)
```bash
# Full pipeline: FT then KD
python run_experiment.py --stage full --ft_method full_ft --kd_method minillm \
  --teacher_model Qwen/Qwen3-4B-SafeRL --student_model meta-llama/Llama-3.2-1B-Instruct \
  --loss ce --train_file /path/to/train.jsonl --val_file /path/to/val.jsonl

# FT only
python run_experiment.py --stage ft --ft_method peft --loss contrastive \
  --teacher_model CohereLabs/tiny-aya-global ...

# KD only (requires FT model to already exist in models/FT/)
python run_experiment.py --stage kd --kd_method ted \
  --teacher_model Qwen/Qwen3-4B-SafeRL --student_model Qwen/Qwen3-1.7B ...
```
- Resumeable: checks for existing checkpoints before starting
- KD auto-validates that FT teacher model exists in `models/FT/{method}/{model_name}/`
- Saves results to `Evaluation/results/{stage}/{method}/{model}-{timestamp}/`

#### Evaluator (`codes/evaluation/evaluate.py`)
- Reads predictions JSONL, computes Accuracy, Precision, Recall, F1, Confusion Matrix
- Per-dialect and per-dataset breakdowns
- Outputs `metrics.json` + confusion matrix CSV

#### Results Structure (`codes/evaluation/results/`)
```
results/
  FT/full_ft/{teacher_model}-{timestamp}/
    metrics.json          # Acc, P, R, F1, confusion matrix
    predictions.jsonl     # sample_id + prediction + label + dialect + dataset
    confusion_matrix.csv
  KD/minillm/{student_model}-{timestamp}/
    ...
```

---

### 2. `codes/splits_generator/` — Dataset Split Generator

**Script:** `generate_splits.py`
**Conda env:** `dia_splits` (pandas, scikit-learn, tqdm, pyyaml)

**Data sources:**
- LLM-generated data: `/home/azureuser/cloudfiles/code/Users/jsl5710/DIA-LLM/LLM_Data/` (48 dialects × 15 datasets)
- Rule-based data: GitHub `codes/multi_value_gen` → `dataset/multi_value/`

**Schema — what columns map to what:**
| Source Column | text_type in JSONL | label |
|---------------|--------------------|-------|
| `original_input` | `original` | 1 (harmful) |
| `transformed_input` | `transformed` | 1 (harmful) |
| `basic_transform` | `basic_transform` | 1 (harmful) |
| `coi_transform` | `coi_transform` | 1 (harmful) |
| `counterharm_original` | `counterharm_original` | 0 (safe) |
| `counterharm_transformed` | `counterharm_transformed` | 0 (safe) |
| `counterharm_basic` | `counterharm_basic` | 0 (safe) |
| `counterharm_coi` | `counterharm_coi` | 0 (safe) |

**NOT USED:** `coi_transform` (for safety training — see note below), `counterharm_score`, `counterharm_validated`, `counterharm_model`

**Refusal filter:** rows where text == `"guardrail policy violation"` (exact match, case-insensitive) are skipped. This is the only refusal pattern found in the data.

**Global sample ID format:** `{dataset}__{dialect}__{row_sample_id}__{text_type}`
Example: `advbench__aboriginal_english__42__original`

**Stratified split:** groups by `(dialect, label)` then proportionally splits each group.

```bash
# Run split generation (once LLM data is complete)
conda activate dia_splits
python generate_splits.py \
  --llm_data_dir /home/azureuser/cloudfiles/code/Users/jsl5710/DIA-LLM/LLM_Data \
  --rule_data_dir /tmp/dia-guard/dataset/multi_value \
  --output_dir /home/azureuser/cloudfiles/code/Users/jsl5710/DIA-LLM/DIA_Splits \
  --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1 --seed 42
```

---

### 3. `dataset/dia_splits/` — Split Storage (README only for now)

JSONL splits go here once `generate_splits.py` is run. Push `train.jsonl`, `val.jsonl`, `test.jsonl` to this folder.

---

### 4. `models/` — Model Checkpoint Directories

```
models/
  FT/full_ft/{model_name}/     # Saved after full FT
  FT/peft/{model_name}/        # Merged LoRA adapter (run merge_lora.py first)
  KD/minillm/{model_name}/
  KD/gkd/{model_name}/
  KD/ted/{model_name}/         # Also saves task_aware_filters.pt + alignment_metadata.json
  upload_to_hub.py             # Upload any checkpoint to HuggingFace Hub
```

HF upload:
```bash
python models/upload_to_hub.py \
  --model_dir models/KD/minillm/Llama-3.2-1B-Instruct \
  --repo_id jsl5710/dia-guard-minillm-llama3-1b \
  --token YOUR_HF_TOKEN
```

---

## What Needs to Be Done Next

### HIGH PRIORITY

#### 1. Update `generate_splits.py` with correct schema
The `generate_splits.py` currently uses an older column mapping. It needs to be updated to:
- Use exactly the 8 text variants listed in the table above
- Apply the `"guardrail policy violation"` refusal filter
- Generate global sample IDs in format `{dataset}__{dialect}__{row_sample_id}__{text_type}`
- For `neg_text` (contrastive training): pair each harmful variant with its `counterharm_*` counterpart
  - `original` ↔ `counterharm_original`
  - `transformed` ↔ `counterharm_transformed`
  - `basic_transform` ↔ `counterharm_basic`

> **Status:** The user confirmed the schema but the script was not yet rewritten before the session ended.

#### 2. Run `generate_splits.py` once LLM data finishes generating
- LLM data is at: `/home/azureuser/cloudfiles/code/Users/jsl5710/DIA-LLM/LLM_Data/`
- 48 dialects × 15 datasets = up to 719 CSV files
- Run the generator and push output JSONL to `dataset/dia_splits/`

#### 3. Create `Evaluation/results/` sub-structure + `evaluate.py`
- `evaluate.py` was mentioned in the plan but may not have been written before session end
- Check: `ls /home/azureuser/cloudfiles/code/Users/jsl5710/DIA-LLM/Evaluation/`
- If missing, implement: reads predictions JSONL → computes Acc/P/R/F1/confusion matrix with per-dialect breakdown

#### 4. Run actual training experiments
- First: generate splits (step 2 above)
- Then: run `run_experiment.py --stage ft` for teacher models
- Then: run `run_experiment.py --stage kd` for each student model
- Pipeline order: Full FT → PEFT → MINILLM → GKD → TED

### MEDIUM PRIORITY

#### 5. Create per-model training configs for all student models
Currently only `qwen3_4b.yaml` and `aya_3b.yaml` exist for teachers. Need configs for each student model in `Knowledge_Distillation/{method}/configs/`.

#### 6. Push trained model weights to HuggingFace
Use `models/upload_to_hub.py` once training completes.

#### 7. Upload LLM data to `codes/dia_llm` on GitHub
When LLM generation finishes, the LLM data code/results go to `codes/dia_llm` in the repo.

---

## Environment Setup Quick Reference

| Module | Conda Env | Create Command |
|--------|-----------|----------------|
| Full FT | `dia_full_ft` | `conda env create -f codes/evaluation/FineTune/full_ft/environment.yml` |
| PEFT | `dia_peft` | `conda env create -f codes/evaluation/FineTune/peft/environment.yml` |
| MINILLM | `dia_minillm` | `conda env create -f codes/evaluation/Knowledge_Distillation/minillm/environment.yml` |
| GKD | `dia_gkd` | `conda env create -f codes/evaluation/Knowledge_Distillation/gkd/environment.yml` |
| TED | `dia_ted` | `conda env create -f codes/evaluation/Knowledge_Distillation/ted/environment.yml` |
| Splits | `dia_splits` | `conda env create -f codes/splits_generator/environment.yml` |

---

## Key File Locations

| Item | Local Path |
|------|-----------|
| LLM-generated data (48 dialects) | `/home/azureuser/cloudfiles/code/Users/jsl5710/DIA-LLM/LLM_Data/` |
| Split generator | `/home/azureuser/cloudfiles/code/Users/jsl5710/DIA-LLM/Splits_Generator/generate_splits.py` |
| Generated splits (output) | `/home/azureuser/cloudfiles/code/Users/jsl5710/DIA-LLM/DIA_Splits/` |
| Evaluation pipeline | `/home/azureuser/cloudfiles/code/Users/jsl5710/DIA-LLM/Evaluation/` |
| Model checkpoints | `/home/azureuser/cloudfiles/code/Users/jsl5710/DIA-LLM/models/` |
| Cloned repo (temp) | `/tmp/dia-guard/` *(re-clone if instance restarted)* |

---

## Repo Structure (as of this session)

```
dia-guard/
  codes/
    evaluation/           ← NEW: full training pipeline
      FineTune/
        full_ft/          (train_ce.py, train_contrastive.py, configs/, env)
        peft/             (train_ce_lora.py, train_contrastive_lora.py, merge_lora.py)
      Knowledge_Distillation/
        minillm/          (train_minillm.py, env)
        gkd/              (train_gkd.py, env)
        ted/              (train_ted.py, env)
      run_experiment.py   ← orchestrator (resumeable, parametric)
      evaluate.py         ← metrics computation
      results/            ← organized by stage/method/model-timestamp/
    splits_generator/     ← NEW: split generation from raw LLM_Data CSVs
    dia_llm/              ← PENDING: LLM data generation code (not yet pushed)
    multi_value_gen/      ← EXISTS: rule-based data pipeline
    d_purify/             ← EXISTS: evaluation utilities
  dataset/
    dia_splits/           ← NEW (README only): JSONL splits go here when generated
    multi_value/          ← EXISTS: rule-based dialect data (48 dialects)
    dia_llm/              ← EXISTS (empty): LLM-based data placeholder
  models/                 ← NEW: checkpoint directories + HF upload utility
  README.md
```

---

## Data Schema (LLM_Data CSV columns used)

```
sample_id, dataset, dialect           ← metadata
original_input                        → text_type=original,     label=1 (harmful)
transformed_input                     → text_type=transformed,  label=1 (harmful, dialect)
basic_transform                       → text_type=basic,        label=1 (harmful, dialect)
coi_transform                         → text_type=coi,          label=1 (harmful, dialect)
counterharm_original                  → text_type=ch_original,  label=0 (safe)
counterharm_transformed               → text_type=ch_transformed, label=0 (safe)
counterharm_basic                     → text_type=ch_basic,     label=0 (safe)
counterharm_coi                       → text_type=ch_coi,       label=0 (safe)

IGNORED: original_harmfulness, transformed_harmfulness (model predictions, not ground truth)
IGNORED: counterharm_score, counterharm_validated, counterharm_model
FILTER:  skip any row where text == "guardrail policy violation" (LLM refusal artifact)
```

Global sample ID: `{dataset}__{dialect}__{sample_id}__{text_type}`
