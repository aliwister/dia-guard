# GKD — On-Policy Generalized Knowledge Distillation

GKD (Agarwal et al., ICLR 2024) addresses a key weakness in standard KD: the distribution mismatch problem. Standard KD trains the student on teacher-generated sequences, but at inference the student must generate its own sequences — a covariate shift.

GKD trains the student using **a mixture of student-generated and teacher-generated sequences**, with the teacher providing soft probability feedback.

---

## Paper & Code

- **Paper**: Agarwal et al. (2024). *On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes*. ICLR 2024.
  - [arXiv:2306.13649](https://arxiv.org/abs/2306.13649)

---

## Core Idea

### Distribution Mismatch Problem

Standard (offline) KD minimizes:
```
L_offline = E_{y ~ p_teacher}[ D(p_teacher(·|x,y_{<t}) || p_student(·|x,y_{<t})) ]
```

But this trains on teacher-generated sequences. At test time the student generates its own sequences, causing exposure bias.

### GKD Solution: On-Policy Training

GKD trains on a **mixture** of student-generated and teacher-generated sequences:

```
L_GKD = E_{y ~ M_λ}[ D(p_teacher(·|x,y_{<t}) || p_student(·|x,y_{<t})) ]

where:
  M_λ = λ · p_student + (1-λ) · p_teacher    [mixture distribution]
  λ ∈ [0, 1]                                   [student mixing ratio]
```

### Supported Divergence Functions

| Name | Formula | Notes |
|------|---------|-------|
| **Forward KL** (ImitKL) | `Σ p_T log(p_T / p_S)` | Mode-covering |
| **Reverse KL** | `Σ p_S log(p_S / p_T)` | Mode-seeking |
| **JSD** | `½ KL(p_T\|\|M) + ½ KL(p_S\|\|M)` | Symmetric, M=½(p_T+p_S) |
| **TVD** | `½ Σ \|p_T - p_S\|` | Total variation distance |

Default in this implementation: **JSD** (best empirical results from paper).

---

## Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lam` | 0.5 | Mixing ratio: fraction from student |
| `divergence` | `jsd` | Loss function: `fkl`, `rkl`, `jsd`, `tvd` |
| `temperature` | 1.0 | Softmax temperature for distributions |
| `max_new_tokens` | 256 | Max tokens for student/teacher generation |
| `top_p` | 0.9 | Nucleus sampling |
| `learning_rate` | 1e-5 | |

---

## TRL Native Trainer (Alternative)

TRL ≥ 0.9 ships `GKDTrainer` directly:
```python
from trl import GKDTrainer, GKDConfig
trainer = GKDTrainer(
    model=student, teacher_model=teacher,
    args=GKDConfig(lmbda=0.5, ...),
    train_dataset=dataset,
)
trainer.train()
```
`train_gkd.py` in this folder provides a transparent, TRL-version-independent implementation for research/customization.

---

## Environment Setup

```bash
conda env create -f environment.yml
conda activate dia_gkd
```

---

## Usage

```bash
python train_gkd.py \
    --teacher_model ../../../../models/FT/full_ft/qwen3-4b-ft \
    --student_model Qwen/Qwen3Guard-Gen-0.6B \
    --train_data /path/to/train.jsonl \
    --output_dir ../../../../models/KD/gkd/qwen3-guard-0.6b \
    --lam 0.5 \
    --divergence jsd

# Multi-GPU
accelerate launch train_gkd.py \
    --teacher_model ../../../../models/FT/full_ft/qwen3-4b-ft \
    --student_model meta-llama/Llama-3.2-1B-Instruct \
    --train_data /path/to/train.jsonl \
    --output_dir ../../../../models/KD/gkd/llama-1b
```

---

## Divergence Selection Guide

| Use Case | Recommended Divergence |
|----------|----------------------|
| Coverage of all teacher behaviors | Forward KL (`fkl`) |
| Focus on teacher's top predictions | Reverse KL (`rkl`) |
| Balanced (default) | JSD (`jsd`) |
| Robustness to outliers | TVD (`tvd`) |

---

## Hardware Requirements

Same as MINILLM: teacher can be quantized (4-bit) to save memory.

| Teacher | Student | Min VRAM |
|---------|---------|----------|
| Qwen3-4B (bf16) | any <1B | ~26GB |
| Qwen3-4B (4-bit) | any <1B | ~12GB |

---

## Citation

```bibtex
@inproceedings{agarwal2024gkd,
  title={On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes},
  author={Agarwal, Rishabh and Vieillard, Nino and Zhou, Yongchao and Stanczyk, Piotr
          and Ramos, Sabela and Geist, Matthieu and Bachem, Olivier},
  booktitle={ICLR},
  year={2024}
}
```
