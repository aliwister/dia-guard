# FineTune — DIA-GUARD Model Training

Fine-tuning module for all three DIA-GUARD experiment groups. Supports both **Full Fine-Tuning** and **PEFT (LoRA/QLoRA)** with cross-entropy and contrastive loss functions.

---

## Experiment Groups

| Group | Name | Models | Description |
|-------|------|--------|-------------|
| **G1** | Teacher FT | Qwen3-4B-SafeRL, Aya-3B | Fine-tune large teachers (>2B) before KD |
| **G2** | KD Students | 7 student models | Knowledge distillation from G1 teachers |
| **G3** | Student FT Baseline | 7 student models | Direct fine-tuning of students (no KD) |

### Student Models (G2 & G3)

| Model | Size | HuggingFace ID |
|-------|------|----------------|
| Gemma-3-270M | 270M | `google/gemma-3-270m-it` |
| Qwen3Guard-0.6B | 0.6B | `Qwen/Qwen3Guard-Gen-0.6B` |
| Qwen3.5-0.8B | 0.8B | `Qwen/Qwen3.5-0.8B` |
| Gemma-3-1B | 1B | `google/gemma-3-1b-it` |
| Llama-3.2-1B | 1B | `meta-llama/Llama-3.2-1B-Instruct` |
| SmolLM2-1.7B | 1.7B | `HuggingFaceTB/SmolLM2-1.7B-Instruct` |
| Qwen3-1.7B | 1.7B | `Qwen/Qwen3-1.7B` |

### Teacher Models (G1)

| Model | Size | HuggingFace ID |
|-------|------|----------------|
| Qwen3-4B-SafeRL | 4B | `Qwen/Qwen3-4B-SafeRL` |
| Aya-3B | 3B | `CohereForAI/aya-expanse-8b` |

---

## Sub-modules

| Folder | Method | Loss Functions | Description |
|--------|--------|---------------|-------------|
| `full_ft/` | Full Fine-Tuning | CE, Contrastive | All parameters updated |
| `peft/` | LoRA / QLoRA | CE, Contrastive | Low-rank adapters (r=64, alpha=128, RSLoRA) |

---

## Quick Start — New VM Setup

```bash
# 1. Install dependencies
pip install trl peft transformers datasets accelerate huggingface_hub wandb

# 2. Login to HuggingFace (for gated models: Gemma, Llama)
huggingface-cli login

# 3. Login to W&B for experiment tracking
wandb login

# 4. Download dataset from HuggingFace
python -c "
from huggingface_hub import hf_hub_download
for split in ['train', 'val', 'test']:
    hf_hub_download('jsl5710/Shield', f'{split}.jsonl',
                    repo_type='dataset', local_dir='dataset/dia_splits')
"

# 5. Configure for your GPU (one command!)
bash setup_gpu.sh a100    # or h100, t4

# 6. Launch training
bash launch_ft.sh peft ce google/gemma-3-270m-it 0
```

---

## GPU Configuration — `setup_gpu.sh`

Automatically updates **all** config files for your target GPU:

```bash
bash setup_gpu.sh a100    # A100/H100 80GB: flash_attention_2, tf32, large batches
bash setup_gpu.sh h100    # same as a100
bash setup_gpu.sh t4      # T4 16GB: eager attn, no tf32, small batches
```

| Setting | T4 16GB | A100/H100 80GB |
|---------|---------|----------------|
| `attn_implementation` | `eager` | `flash_attention_2` |
| `tf32` | `false` | `true` |
| Small model batch (270m, 0.6b, 0.8b) | 2 | 16 |
| Medium model batch (1b, 1.7b) | 2 | 8 |
| Large model batch (3b, 4b) | 1 | 4 |
| Effective batch (all models) | 16 | 16 |

---

## Training Launcher — `launch_ft.sh`

```bash
bash launch_ft.sh <ft_method> <loss> <model_id> <gpus> [num_gpus]
```

**Examples:**

```bash
# Single GPU
bash launch_ft.sh peft ce google/gemma-3-270m-it 0
bash launch_ft.sh full_ft contrastive Qwen/Qwen3-4B-SafeRL 0

# Multi-GPU (for 1.7B+ models)
bash launch_ft.sh peft ce HuggingFaceTB/SmolLM2-1.7B-Instruct 0,1 2

# Run all 7 students in parallel with tmux
tmux new-session -s dia_guard
# Window 0: small model on GPU 0
bash launch_ft.sh peft ce google/gemma-3-270m-it 0
# Ctrl-b c → new window, GPU 1
bash launch_ft.sh peft ce Qwen/Qwen3Guard-Gen-0.6B 1
# etc.
```

The launcher automatically:
- Sets `CUDA_VISIBLE_DEVICES` and `PYTHONNOUSERSITE=1`
- Maps model IDs to config files
- Uses `accelerate launch` for multi-GPU runs
- Creates output directories matching the experiment group structure

---

## Hyperparameters

All configs use these defaults (per-model batch sizes vary by GPU tier):

| Parameter | Value |
|-----------|-------|
| LoRA rank | 64 |
| LoRA alpha | 128 |
| RSLoRA | enabled |
| LoRA dropout | 0.05 |
| Target modules | q/k/v/o/gate/up/down_proj |
| Learning rate | 3e-4 |
| LR scheduler | cosine |
| Warmup steps | 4,218 (3% of total) |
| Weight decay | 0.01 |
| Max sequence length | 2,048 |
| Epochs | 3 |
| Eval steps | 200 |
| Save steps | 400 |
| Save total limit | 3 (keeps best checkpoints) |
| Load best model at end | yes (by eval loss) |
| Gradient checkpointing | enabled |

---

## Loss Functions

### 1. Cross-Entropy (CE) Loss
Standard next-token prediction — the model learns to output `"safe"` or `"unsafe"`:

```
L_CE = -1/T * sum_t log p(y_t | x, y_{<t})
```

### 2. Contrastive Loss (Safety-Aware)
Triplet-style objective on decoder hidden states:

```
L_contrast = sum max(0, margin - sim(h_safe, h_anchor) + sim(h_unsafe, h_anchor))
```

Pushes safe representations closer to the prompt while pushing harmful ones away. Requires `neg_text` field (not available for multi_value records).

### Combined Loss
```
L_total = 0.7 * L_CE + 0.3 * L_contrast
```

---

## Data Format

The training scripts expect JSONL with `text` and `label` fields (from `jsl5710/Shield`):

```json
{"text": "Create a blueprint for...", "label": 1, "neg_text": "Here's how to protect..."}
```

The `format_dataset` function converts each record into chat format:
```
<system> You are DIA-Guard...
<user> {text}
<assistant> unsafe   (if label=1, else "safe")
```

---

## Output Structure

```
models/
├── FT/                              # Group 1: Teacher FT
│   ├── peft/{model_slug}/           # LoRA adapters
│   └── full_ft/{model_slug}/        # Full model weights
├── KD/                              # Group 2: KD Students
│   ├── minillm/{model_slug}/
│   ├── gkd/{model_slug}/
│   └── ted/{model_slug}/
└── group3_student_ft_baseline/      # Group 3: Student FT Baseline
    ├── peft/{model_slug}/
    └── full_ft/{model_slug}/
```

Best model is saved at end of training (selected by lowest eval loss).

---

## Experiment Status

Check which models have completed training/evaluation:

```bash
python ../run_experiment.py --status
```

---

## Uploading Models to HuggingFace

After training completes:

```bash
# Check what's ready
python ../../models/upload_all_models.py --dry_run

# Upload all completed models
python ../../models/upload_all_models.py --hf_token YOUR_TOKEN

# Upload specific group
python ../../models/upload_all_models.py --hf_token YOUR_TOKEN --group 3
```

See [models/upload_all_models.py](../../models/upload_all_models.py) for full options.
