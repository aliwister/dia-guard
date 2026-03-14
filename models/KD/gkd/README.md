# GKD Distilled Models (KD/gkd/)

Student models distilled via GKD (on-policy distillation with JSD/KL loss).

## Expected Checkpoints

| Directory | Student Base | Teacher | Divergence | Status |
|-----------|-------------|---------|-----------|--------|
| `qwen3-guard-0.6b/` | Qwen3Guard-Gen-0.6B | Qwen3-4B-FT | JSD | — |
| `llama-1b/` | Llama-3.2-1B-Instruct | Aya-3B-FT | JSD | — |
| `gemma-1b/` | gemma-3-1b-it | Aya-3B-FT | JSD | — |
| `smollm2-1.7b/` | SmolLM2-1.7B | Qwen3-4B-FT | JSD | — |

## Training Command

```bash
cd ../../../../Evaluation/Knowledge_Distillation/gkd
conda activate dia_gkd
python train_gkd.py \
    --teacher_model ../../../models/FT/full_ft/qwen3-4b-ft \
    --student_model Qwen/Qwen3Guard-Gen-0.6B \
    --train_data /path/to/train.jsonl \
    --output_dir ../../../models/KD/gkd/qwen3-guard-0.6b \
    --lam 0.5 \
    --divergence jsd
```
