# Full Fine-Tuning Checkpoints (FT/full_ft/)

Stores fully fine-tuned teacher models where all parameters were updated.

## Expected Checkpoints

| Directory | Model | Loss | Status |
|-----------|-------|------|--------|
| `qwen3-4b-ft/` | Qwen3-4B-SafeRL | Cross-Entropy | — |
| `qwen3-4b-contrastive-ft/` | Qwen3-4B-SafeRL | Contrastive | — |
| `aya-3b-ft/` | tiny-aya-global | Cross-Entropy | — |
| `aya-3b-contrastive-ft/` | tiny-aya-global | Contrastive | — |

## Training Command

```bash
cd ../../../../Evaluation/FineTune/full_ft
conda activate dia_full_ft
python train_ce.py --config configs/qwen3_4b.yaml \
    --train_data /path/to/train.jsonl \
    --output_dir ../../../models/FT/full_ft/qwen3-4b-ft
```
