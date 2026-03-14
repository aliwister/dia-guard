# Evaluation Results — DIA-GUARD

Stores all training and evaluation outputs organized by experiment stage, method, and model.

---

## Directory Structure

```
results/
├── README.md
├── FT/                          ← Fine-tuning results
│   ├── full_ft/
│   │   └── {model}-{timestamp}/
│   │       ├── metrics.json          ← Acc, Precision, Recall, F1
│   │       ├── predictions.jsonl     ← per-sample predictions with sample_id
│   │       ├── confusion_matrix.csv  ← 2×2 confusion matrix
│   │       ├── per_dialect.json      ← metrics broken down by dialect
│   │       └── train_log.jsonl       ← epoch-level training losses
│   └── peft/
│       └── {model}-{timestamp}/
│           └── (same as full_ft)
└── KD/                          ← Knowledge Distillation results
    ├── minillm/
    │   └── {teacher}_to_{student}-{timestamp}/
    │       └── (same structure)
    ├── gkd/
    │   └── {teacher}_to_{student}-{timestamp}/
    │       └── (same structure)
    └── ted/
        └── {teacher}_to_{student}-{timestamp}/
            └── (same structure)
```

---

## File Formats

### `metrics.json`
```json
{
  "stage": "FT",
  "method": "full_ft",
  "loss": "ce",
  "model": "Qwen/Qwen3-4B-SafeRL",
  "split": "test",
  "timestamp": "2024-01-15T10:30:00",
  "overall": {
    "accuracy": 0.912,
    "precision": 0.905,
    "recall": 0.921,
    "f1": 0.913,
    "support": 4800
  },
  "per_class": {
    "safe":   {"precision": 0.918, "recall": 0.903, "f1": 0.910, "support": 2400},
    "unsafe": {"precision": 0.893, "recall": 0.939, "f1": 0.915, "support": 2400}
  }
}
```

### `predictions.jsonl`
Each line:
```json
{
  "sample_id": "salad_bench-aboriginal_english-42",
  "dialect": "aboriginal_english",
  "source_dataset": "Salad_Bench",
  "text_type": "coi_transform",
  "true_label": 1,
  "true_label_str": "unsafe",
  "pred_label": 1,
  "pred_label_str": "unsafe",
  "pred_prob_safe": 0.08,
  "pred_prob_unsafe": 0.92
}
```

### `confusion_matrix.csv`
```
,pred_safe,pred_unsafe
true_safe,2165,235
true_unsafe,189,2211
```

### `per_dialect.json`
```json
{
  "aboriginal_english": {
    "accuracy": 0.923,
    "precision": 0.918,
    "recall": 0.929,
    "f1": 0.923,
    "support": 100
  },
  ...
}
```

### `train_log.jsonl`
Each line:
```json
{
  "epoch": 1,
  "step": 500,
  "train_loss": 0.421,
  "val_loss": 0.389,
  "val_f1": 0.871,
  "lr": 2e-5,
  "timestamp": "2024-01-15T09:12:33"
}
```

---

## Experiment Naming Convention

| Stage | Pattern | Example |
|-------|---------|---------|
| Full FT (CE) | `full_ft_ce-{model_shortname}-{YYYYMMDD_HHMMSS}` | `full_ft_ce-qwen3_4b-20240115_103000` |
| Full FT (Contrastive) | `full_ft_contrastive-{model_shortname}-{timestamp}` | |
| PEFT (LoRA CE) | `peft_ce-{model_shortname}-{timestamp}` | |
| PEFT (LoRA Contrastive) | `peft_contrastive-{model_shortname}-{timestamp}` | |
| MINILLM | `minillm-{teacher_short}_to_{student_short}-{timestamp}` | `minillm-qwen3_4b_to_smollm2-20240115_140000` |
| GKD | `gkd-{teacher_short}_to_{student_short}-{timestamp}` | |
| TED | `ted-{teacher_short}_to_{student_short}-{timestamp}` | |

---

## Comparing Experiments

Use `evaluate.py` to regenerate or compare metrics:

```bash
# Evaluate a saved checkpoint
python evaluate.py \
    --model_dir ../../models/FT/full_ft/qwen3_4b \
    --test_data ../../DIA_Splits/test.jsonl \
    --output_dir results/FT/full_ft/qwen3_4b-20240115_103000 \
    --stage FT --method full_ft

# Compare two experiments
python evaluate.py --compare \
    results/FT/full_ft/qwen3_4b-20240115_103000/metrics.json \
    results/KD/minillm/qwen3_4b_to_smollm2-20240115_140000/metrics.json
```
