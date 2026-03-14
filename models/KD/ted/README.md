# TED Distilled Models (KD/ted/)

Student models distilled via TED (task-aware layer-wise distillation).

## Expected Checkpoints

Each directory contains both the student weights AND `task_aware_filters.pt`.

| Directory | Student Base | Teacher | Layers Aligned | Status |
|-----------|-------------|---------|---------------|--------|
| `qwen3-guard-0.6b/` | Qwen3Guard-Gen-0.6B | Qwen3-4B-FT | all | — |
| `gemma-270m/` | gemma-3-270m-it | Qwen3-4B-FT | all | — |
| `gemma-1b/` | gemma-3-1b-it | Aya-3B-FT | all | — |
| `qwen3-1.7b/` | Qwen3-1.7B | Qwen3-4B-FT | all | — |

## Training Command

```bash
cd ../../../../Evaluation/Knowledge_Distillation/ted
conda activate dia_ted
python train_ted.py \
    --teacher_model ../../../models/FT/full_ft/qwen3-4b-ft \
    --student_model google/gemma-3-270m-it \
    --train_data /path/to/train.jsonl \
    --output_dir ../../../models/KD/ted/gemma-270m \
    --lam 0.5 \
    --teacher_load_in_4bit  # recommended for memory efficiency
```

## Note on Filters

The `task_aware_filters.pt` file contains the learned projection matrices `W_l` used during training.
These are **not needed for inference** — only the student model weights are needed at inference time.
They are saved for reproducibility and potential further fine-tuning.
