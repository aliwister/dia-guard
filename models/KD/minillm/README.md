# MINILLM Distilled Models (KD/minillm/)

Student models distilled via MINILLM (reverse KL divergence).

## Expected Checkpoints

| Directory | Student Base | Teacher | Status |
|-----------|-------------|---------|--------|
| `qwen3-guard-0.6b/` | Qwen3Guard-Gen-0.6B | Qwen3-4B-FT | — |
| `qwen3.5-0.8b/` | Qwen3.5-0.8B | Qwen3-4B-FT | — |
| `llama-1b/` | Llama-3.2-1B-Instruct | Aya-3B-FT | — |
| `deepseek-1.5b/` | DeepSeek-R1-Distill-1.5B | Qwen3-4B-FT | — |

## Training Command

```bash
cd ../../../../Evaluation/Knowledge_Distillation/minillm
conda activate dia_minillm
python train_minillm.py \
    --teacher_model ../../../models/FT/full_ft/qwen3-4b-ft \
    --student_model Qwen/Qwen3Guard-Gen-0.6B \
    --train_data /path/to/train.jsonl \
    --output_dir ../../../models/KD/minillm/qwen3-guard-0.6b \
    --beta 0.5
```
