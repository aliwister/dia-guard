# DIA-GUARD Evaluation Pipeline

This folder contains all training and evaluation code for the **DIA-GUARD** framework — a multilingual, dialectally-aware safety system for LLMs.

---

## Overview

The pipeline has two stages:

```
Stage 1: Fine-Tune (FineTune/)
    └── Train large teacher models (>2B params) on DIA-GUARD safety data
            ├── Full Fine-Tuning (full_ft/) — all weights updated
            └── PEFT / LoRA (peft/) — only adapter weights updated

Stage 2: Knowledge Distillation (Knowledge_Distillation/)
    └── Compress teacher knowledge into smaller student models (<2B params)
            ├── MINILLM — Reverse KL divergence (sequence-level)
            ├── GKD    — On-policy distillation with flexible losses
            └── TED    — Task-aware layer-wise hidden state alignment
```

---

## Model Table

| Role | Model | Size | HuggingFace ID |
|------|-------|------|----------------|
| Teacher | Dia-Guard-4B | 4B | `Qwen/Qwen3-4B-SafeRL` |
| Teacher | Dia-Guard-3B | 3B | `CohereLabs/tiny-aya-global` |
| Student | Dia-Guard-1B§ | ~1B | `meta-llama/Llama-3.2-1B-Instruct` ⭐ |
| Student | Dia-Guard-1B§ | ~1B | `google/gemma-3-1b-it` |
| Student | Dia-Guard-0.8B | 0.8B | `Qwen/Qwen3.5-0.8B` ⭐ |
| Student | Dia-Guard-0.6B | 0.6B | `Qwen/Qwen3Guard-Gen-0.6B` |
| Student | Dia-Guard-270M | 270M | `google/gemma-3-270m-it` |
| Student | — | 1.7B | `Qwen/Qwen3-1.7B` |
| Student | — | 1.5B | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` |
| Student | — | 1.7B | `HuggingFaceTB/SmolLM2-1.7B-Instruct` |

⭐ = top picks for multilingual/dialectal safety focus.

---

## Directory Structure

```
Evaluation/
├── README.md                        ← you are here
├── FineTune/
│   ├── README.md
│   ├── full_ft/                     ← Full fine-tuning (all weights)
│   │   ├── README.md
│   │   ├── environment.yml
│   │   ├── requirements.txt
│   │   ├── train_ce.py              ← Cross-entropy loss training
│   │   ├── train_contrastive.py     ← Contrastive loss training
│   │   └── configs/
│   │       ├── qwen3_4b.yaml
│   │       └── aya_3b.yaml
│   └── peft/                        ← LoRA / QLoRA fine-tuning
│       ├── README.md
│       ├── environment.yml
│       ├── requirements.txt
│       ├── train_ce_lora.py         ← LoRA + CE loss
│       ├── train_contrastive_lora.py ← LoRA + contrastive loss
│       └── configs/
│           ├── qwen3_4b_lora.yaml
│           └── aya_3b_lora.yaml
└── Knowledge_Distillation/
    ├── README.md
    ├── minillm/                     ← Reverse KL distillation (Gu et al., 2024)
    │   ├── README.md
    │   ├── environment.yml
    │   ├── requirements.txt
    │   └── train_minillm.py
    ├── gkd/                         ← On-policy distillation (Agarwal et al., 2024)
    │   ├── README.md
    │   ├── environment.yml
    │   ├── requirements.txt
    │   └── train_gkd.py
    └── ted/                         ← Layer-wise distillation (Liang et al., 2023)
        ├── README.md
        ├── environment.yml
        ├── requirements.txt
        └── train_ted.py
```

---

## Dataset Format

All training scripts expect a JSONL file with the following schema:

```json
{"prompt": "Is it okay to ...", "response": "No, this is harmful because ...", "label": 1}
```

| Field | Type | Description |
|-------|------|-------------|
| `prompt` | str | User input or harmful/dialectal text |
| `response` | str | Model's safe response or judgment |
| `label` | int | `1` = harmful, `0` = benign |
| `dialect` | str | (optional) Dialect tag, e.g., `AAE`, `SAE` |
| `category` | str | (optional) Harm category |

For **contrastive training**, an additional `neg_response` field is required:

```json
{"prompt": "...", "response": "safe response", "neg_response": "unsafe response", "label": 1}
```

---

## Environment Setup

Each sub-module has its own isolated conda environment to avoid dependency conflicts.

| Module | Env Name | Key Packages |
|--------|----------|--------------|
| `full_ft` | `dia_full_ft` | transformers, trl, accelerate, deepspeed |
| `peft` | `dia_peft` | transformers, trl, peft, bitsandbytes |
| `minillm` | `dia_minillm` | transformers, accelerate, torch |
| `gkd` | `dia_gkd` | transformers, trl, accelerate |
| `ted` | `dia_ted` | transformers, accelerate, torch |

Create any environment with:
```bash
conda env create -f <module>/environment.yml
conda activate <env_name>
```

---

## Recommended Run Order

```bash
# 1. Fine-tune teacher (choose full_ft or peft)
cd FineTune/full_ft
conda activate dia_full_ft
python train_ce.py --config configs/qwen3_4b.yaml

# 2. Run distillation on student
cd ../../Knowledge_Distillation/minillm
conda activate dia_minillm
python train_minillm.py \
    --teacher_model ../../models/FT/full_ft/qwen3-4b-ft \
    --student_model Qwen/Qwen3Guard-Gen-0.6B \
    --output_dir ../../models/KD/minillm/qwen3-guard-0.6b

# 3. Upload to HuggingFace
cd ../../..
python models/upload_to_hub.py --model_dir models/KD/minillm/qwen3-guard-0.6b \
    --repo_id your-org/Dia-Guard-0.6B-MINILLM
```

---

## References

- **MINILLM**: Gu et al. (2024). *Knowledge Distillation of Large Language Models*. ICLR 2024. [arXiv:2306.08543](https://arxiv.org/abs/2306.08543)
- **GKD**: Agarwal et al. (2024). *On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes*. ICLR 2024. [arXiv:2306.13649](https://arxiv.org/abs/2306.13649)
- **TED**: Liang et al. (2023). *Less is More: Task-aware Layer-wise Distillation for Language Model Compression*. ICML 2023. [arXiv:2210.01351](https://arxiv.org/abs/2210.01351)
- **DIA-GUARD Survey**: [ACL Anthology 2024.tacl-1.85](https://aclanthology.org/2024.tacl-1.85.pdf)
