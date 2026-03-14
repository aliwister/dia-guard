# Fine-Tuned Teacher Models (FT/)

This directory stores the weights of teacher models that have been fine-tuned on DIA-GUARD safety data. These models are used as the teacher in the knowledge distillation pipeline.

---

## Sub-directories

| Folder | Method | Description |
|--------|--------|-------------|
| `full_ft/` | Full Fine-Tuning | All model parameters updated during training |
| `peft/` | LoRA / QLoRA | Only adapter parameters updated; base model frozen |

---

## Teacher Models

| Name | Base Model | HuggingFace ID | Size |
|------|-----------|----------------|------|
| Dia-Guard-4B (teacher) | Qwen3-4B-SafeRL | `Qwen/Qwen3-4B-SafeRL` | 4B |
| Dia-Guard-3B (teacher) | tiny-aya-global | `CohereLabs/tiny-aya-global` | 3B |

---

## Training Configurations

Refer to the training configs in:
```
../../Evaluation/FineTune/full_ft/configs/
../../Evaluation/FineTune/peft/configs/
```

---

## Usage in KD Pipeline

After fine-tuning, pass the teacher path to any KD script:

```bash
# Full FT teacher
python ../../Evaluation/Knowledge_Distillation/minillm/train_minillm.py \
    --teacher_model FT/full_ft/qwen3-4b-ft \
    --student_model Qwen/Qwen3Guard-Gen-0.6B \
    ...

# PEFT merged teacher (merge LoRA first)
python ../../Evaluation/Knowledge_Distillation/gkd/train_gkd.py \
    --teacher_model FT/peft/qwen3-4b-lora-merged \
    --student_model meta-llama/Llama-3.2-1B-Instruct \
    ...
```
