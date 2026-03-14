# FineTune — DIA-GUARD Teacher Model Training

This module fine-tunes the **teacher models** (>2B parameters) on DIA-GUARD safety data before knowledge distillation.

---

## Why Fine-Tune First?

The distillation pipeline transfers knowledge from a **strong, fine-tuned teacher** to a smaller student. Fine-tuning the teacher on the DIA-GUARD safety task first ensures the student learns:
- Harm detection across 48+ English dialects
- Safe response generation
- Counterfactual safety reasoning (from CounterHarm-SHIELD outputs)

---

## Sub-modules

| Folder | Method | Loss Functions | GPU Requirement |
|--------|--------|---------------|-----------------|
| `full_ft/` | Full Fine-Tuning | Cross-Entropy (CE), Contrastive | 2–4× A100 80GB (4B model) |
| `peft/` | LoRA / QLoRA | Cross-Entropy (CE), Contrastive | 1× A100 40GB (4B model with QLoRA) |

---

## Teacher Models

| Model | Size | HuggingFace ID | Context | Languages |
|-------|------|----------------|---------|-----------|
| Dia-Guard-4B | 4B | `Qwen/Qwen3-4B-SafeRL` | 32K | 100+ |
| Dia-Guard-3B | 3B | `CohereLabs/tiny-aya-global` | — | 70+ |

---

## Loss Functions

### 1. Supervised Cross-Entropy (CE) Loss
Standard next-token prediction on (prompt, safe_response) pairs:

```
L_CE = -1/T * Σ_t log p_θ(y_t | x, y_{<t})
```

Used for generative safety response training. The model learns to produce safe, contextually appropriate responses.

### 2. Contrastive Loss (Safety-Aware)
A triplet-style contrastive objective on decoder hidden states:

```
L_contrast = Σ max(0, margin - sim(h_safe, h_anchor) + sim(h_unsafe, h_anchor))
```

Where:
- `h_anchor` = last hidden state for the input prompt
- `h_safe` = last hidden state for safe/benign response
- `h_unsafe` = last hidden state for harmful response
- `sim(·)` = cosine similarity

This pushes safe response representations closer to the prompt anchor while pushing harmful ones away.

### Combined Loss
```
L_total = α * L_CE + (1 - α) * L_contrast
```
Default `α = 0.7`.

---

## Data Flow

```
DIA-GUARD Dataset (JSONL)
    │
    ├── prompt + response (safe)   → CE Loss Training
    │
    └── prompt + safe + unsafe     → Contrastive Loss Training
```

---

## Quick Start

```bash
# Full fine-tuning with CE loss
cd full_ft
conda env create -f environment.yml && conda activate dia_full_ft
python train_ce.py --config configs/qwen3_4b.yaml

# PEFT / LoRA fine-tuning with contrastive loss
cd ../peft
conda env create -f environment.yml && conda activate dia_peft
python train_contrastive_lora.py --config configs/qwen3_4b_lora.yaml
```

---

## Output

Fine-tuned models are saved to:
```
../../models/FT/full_ft/<model-name>/
../../models/FT/peft/<model-name>/
```

Each saved checkpoint is in HuggingFace format and can be loaded with:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("../../models/FT/full_ft/qwen3-4b-ft")
```
