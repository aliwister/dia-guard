# DIA-GUARD: Dialect-Informed Adversarial Guard for LLM Safety

DIA-GUARD is a comprehensive framework for evaluating and improving LLM safety across **48 English dialects**. It addresses the gap in dialect-aware safety evaluation by generating dialect-transformed harmful content, validated benign counterexamples, and training compact safety classifiers via knowledge distillation.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        DIA-GUARD                            │
├──────────────┬──────────────┬───────────────┬───────────────┤
│  Multi-VALUE │  Harm-SHIELD │  CounterHarm  │  D-PURiFY     │
│  (Rule-based │  (LLM-based  │  -SHIELD      │  (Quality     │
│   dialect    │   dialect    │  (Benign      │   evaluation) │
│   transform) │   transform) │   generation) │               │
├──────────────┴──────────────┴───────────────┴───────────────┤
│              Splits Generator → Train/Val/Test              │
├─────────────────────────────────────────────────────────────┤
│  Fine-Tuning (Teacher >2B) → Knowledge Distillation (<2B)  │
└─────────────────────────────────────────────────────────────┘
```

## Repository Structure

```
dia-guard/
├── codes/
│   ├── dia_llm/                    # LLM-based generation pipeline
│   │   ├── harm_shield/            # Dialect transformation (48 dialects, eWAVE-validated)
│   │   └── counterharm_shield/     # Benign counterexample generation (6-chain CoI)
│   ├── multi_value_gen/            # Rule-based dialect transformation pipeline
│   ├── d_purify/                   # Quality evaluation (similarity, neural, eWAVE, LLM-judge)
│   ├── splits_generator/           # Stratified train/val/test split generation
│   └── evaluation/                 # Training pipeline
│       ├── FineTune/               # Fine-tuning (Full FT + LoRA/QLoRA)
│       │   ├── launch_ft.sh        # Training launcher (GPU assignment, tmux)
│       │   ├── setup_gpu.sh        # Auto-configure for A100/H100/T4
│       │   ├── peft/               # LoRA training scripts + configs
│       │   ├── full_ft/            # Full FT training scripts + configs
│       │   └── configs/            # Accelerate multi-GPU configs
│       ├── Knowledge_Distillation/ # Student distillation (MINILLM, GKD, TED)
│       ├── Quantization/          # Post-training quantization (fp16, int8, nf4)
│       └── run_experiment.py       # Experiment orchestrator + status tracking
├── dataset/
│   ├── dia_llm/                    # LLM-generated data (48 dialects × 15 datasets)
│   ├── multi_value/                # Rule-based dialect data
│   └── dia_splits/                 # Train/val/test splits (hosted on HF: jsl5710/Shield)
└── models/                         # Trained checkpoints
    ├── upload_all_models.py        # Batch upload to HuggingFace Hub
    ├── FT/                         # Group 1: Teacher FT
    ├── KD/                         # Group 2: KD Students
    ├── group3_student_ft_baseline/ # Group 3: Student FT Baseline
    └── Quantized/                  # Group 4: Quantized models (fp16/int8/nf4)
```

---

## Modules

### Harm-SHIELD — Dialect Transformation

Transforms Standard American English harmful text into 48 English dialect variants using LLMs with [eWAVE](https://ewave-atlas.org/) linguistic feature validation. Applies both basic and 4-chain Chain-of-Interaction (CoI) transformations.

```bash
cd codes/dia_llm/harm_shield
python full_generation_parallel.py --model gemini --workers 4
```

### CounterHarm-SHIELD — Benign Counterexample Generation

Generates validated benign counterexamples using a 6-chain pipeline (ToxiCraft + PromptSafe + FIZLE) that mirrors the length and structure of source harmful text. ~24 API calls per row.

```bash
cd codes/dia_llm/counterharm_shield
python full_generation_parallel.py --model gemini --workers 4
```

**Available models:** `gpt4.1`, `deepseek`, `kimi`, `gemini`, `gemini2.5`, `gemini2.5flash`

**Useful flags:**
| Flag | Description |
|------|-------------|
| `--model MODEL` | LLM backend to use |
| `--workers N` | Parallel workers (default: 4) |
| `--dialect FOLDER` | Process only one dialect |
| `--dataset PATTERN` | Process only matching CSV files |
| `--test` | Test mode (limited rows) |
| `--test-rows N` | Number of rows in test mode |
| `--data-dir PATH` | Override data directory (auto-resolves to `dataset/dia_llm/`) |

### Multi-VALUE — Rule-Based Dialect Transformation

Scales the Multi-VALUE framework to process 9 safety/security benchmark datasets across 50 English dialects with resumable checkpoints and code-aware preservation.

### D-PURiFY — Quality Evaluation

Automatic evaluation framework assessing dialect transformation quality via text similarity, neural/semantic metrics, eWAVE feature validation, and LLM-as-a-Judge scoring.

### Evaluation Pipeline — Training & Distillation

Three experiment groups across 9 models:

| Group | Name | Models | Description |
|-------|------|--------|-------------|
| **G1** | Teacher FT | 2 teachers (3B, 4B) | Fine-tune large models on safety data |
| **G2** | KD Students | 7 students (<2B) | Distill from G1 teachers (MINILLM, GKD, TED) |
| **G3** | Student FT Baseline | 7 students (<2B) | Direct fine-tuning without KD |
| **G4** | Quantization | G2 + G3 models | Post-training quantization at fp16/int8/nf4 |

| Method | Type | Description |
|--------|------|-------------|
| Full FT | Fine-tune | Full fine-tuning with CE or contrastive loss |
| PEFT | Fine-tune | LoRA/QLoRA with rank-stabilized scaling (r=64, alpha=128) |
| MINILLM | Distillation | Reverse KL via REINFORCE + EMA baseline |
| GKD | Distillation | On-policy mixing (FKL/RKL/JSD/TVD) |
| TED | Distillation | Task-aware embedding distillation |
| Quantization | Compression | fp16 / LLM.int8() / NF4 4-bit via bitsandbytes |

```bash
# Configure for your GPU (A100/H100/T4)
bash codes/evaluation/FineTune/setup_gpu.sh a100

# Launch Group 3 student training with tmux
bash codes/evaluation/FineTune/launch_ft.sh peft ce google/gemma-3-270m-it 0

# Full pipeline: fine-tune teacher then distill
python codes/evaluation/run_experiment.py --stage full \
  --ft_method full_ft --kd_method minillm \
  --teacher_model Qwen/Qwen3-4B-SafeRL \
  --student_model meta-llama/Llama-3.2-1B-Instruct \
  --train_file dataset/dia_splits/train.jsonl \
  --val_file dataset/dia_splits/val.jsonl

# Check experiment status
python codes/evaluation/run_experiment.py --status

# Upload completed models to HuggingFace
python models/upload_all_models.py --hf_token YOUR_TOKEN
```

---

## Data Schema

Each CSV in `dataset/dia_llm/` contains these columns:

| Column | Text Type | Label | Description |
|--------|-----------|-------|-------------|
| `original_input` | original | 1 (harmful) | Original benchmark text |
| `transformed_input` | transformed | 1 (harmful) | Dialect-transformed text |
| `basic_transform` | basic | 1 (harmful) | Basic dialect transformation |
| `coi_transform` | coi | 1 (harmful) | CoI dialect transformation |
| `counterharm_original` | counterharm | 0 (safe) | Benign counterpart of original |
| `counterharm_transformed` | counterharm | 0 (safe) | Benign counterpart of transformed |
| `counterharm_basic` | counterharm | 0 (safe) | Benign counterpart of basic |
| `counterharm_coi` | counterharm | 0 (safe) | Benign counterpart of CoI |

**Coverage:** 48 dialects × 15 benchmark datasets × ~178 rows each = ~127,500 rows total

**Benchmark datasets:** Salad Bench, Simple Safety Tests, Toxic Text, Toxicity Jigsaw, AdvBench, BIPIA, CyberSecEval, Do Not Answer, Forbidden Questions, HarmBench, InjecAgent, JailbreakBench, LLMSecEval, SecurityEval, Sorry Bench

---

## Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/jsl5710/dia-guard.git
cd dia-guard
```

### 2. Set up environment variables

```bash
# For Azure OpenAI models (gpt4.1)
export AZURE_OPENAI_API_KEY="your-key"

# For Azure AI models (deepseek, kimi)
export AZURE_AI_API_KEY="your-key"

# For Gemini via Vertex AI
gcloud auth application-default login
```

### 3. Install dependencies

```bash
# For data generation
pip install openai google-genai

# For training
pip install trl peft transformers datasets accelerate huggingface_hub wandb
```

### 4. Continue CounterHarm-SHIELD generation

```bash
cd codes/dia_llm/counterharm_shield
python full_generation_parallel.py --model gemini --workers 4
```

Data is read from and written to `dataset/dia_llm/` automatically.

### 5. Generate splits (after data generation is complete)

```bash
conda env create -f codes/splits_generator/environment.yml
conda activate dia_splits
python codes/splits_generator/generate_splits.py \
  --llm_data_dir dataset/dia_llm \
  --rule_data_dir dataset/multi_value \
  --output_dir dataset/dia_splits
```

Or download pre-built splits from HuggingFace:
```bash
huggingface-cli download jsl5710/Shield --repo-type dataset --local-dir dataset/dia_splits
```

### 6. Train models

```bash
# Login (required for gated models: Gemma, Llama)
huggingface-cli login
wandb login

# Configure for your GPU
bash codes/evaluation/FineTune/setup_gpu.sh a100   # or h100, t4

# Launch Group 3 student FT (PEFT with CE loss)
bash codes/evaluation/FineTune/launch_ft.sh peft ce google/gemma-3-270m-it 0

# Check experiment status
python codes/evaluation/run_experiment.py --status
```

### 7. Upload trained models to HuggingFace

```bash
# See which models are complete
python models/upload_all_models.py --dry_run

# Upload all completed models
python models/upload_all_models.py --hf_token YOUR_HF_TOKEN
```

---

## Dialect Coverage (48 Dialects)

| Region | Dialects |
|--------|----------|
| **North America** | Appalachian, Chicano, Colloquial American, Earlier AAVE, Newfoundland, Ozark, Rural AAVE, Southeast Enclave, Urban AAVE |
| **British Isles** | Channel Islands, East Anglian, Northern England, Southeast England, Southwest England, Irish, Maltese, Manx, Orkney & Shetland, Scottish, Welsh |
| **Caribbean** | Bahamian, Jamaican |
| **Africa** | Black South African, Cameroon, Cape Flats, Ghanaian, Indian South African, Kenyan, Liberian Settler, Nigerian, Tanzanian, Ugandan, White South African |
| **South/Southeast Asia** | Singapore (Singlish), Hong Kong, Indian, Malaysian, Pakistani, Philippine, Sri Lankan |
| **Australia & Pacific** | Aboriginal, Acrolectal Fiji, Australian, Australian Vernacular, New Zealand, Basilectal Fiji |
| **Atlantic Islands** | Falkland Islands, St Helena, Tristan da Cunha |

---

## Generation Progress

| Status | Count | Details |
|--------|-------|---------|
| Completed | 6 dialects | Aboriginal, Acrolectal Fiji, Appalachian, Australian, Scottish, St Helena |
| Nearly Done (>99%) | 5 dialects | Tanzanian, Tristan da Cunha, Ugandan, Welsh, Urban AAVE |
| In Progress | 2 dialects | Southeast Enclave (52%), Australian Vernacular (34%) |
| Not Started | 35 dialects | Remaining dialects |

*Overall: ~31,360 / 127,561 rows completed (24.6%)*

---

## License

This project is for research purposes.
