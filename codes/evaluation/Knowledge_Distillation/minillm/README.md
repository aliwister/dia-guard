# MINILLM — Reverse KL Knowledge Distillation

MINILLM (Gu et al., 2024) is the first work to study generative knowledge distillation from open-source decoder LLMs. It replaces the standard forward KL divergence with **reverse KL**, which is better suited for generative language models.

---

## Paper & Code

- **Paper**: Gu et al. (2024). *Knowledge Distillation of Large Language Models*. ICLR 2024.
  - [arXiv:2306.08543](https://arxiv.org/abs/2306.08543)
- **Official GitHub**: [microsoft/LMOps/minillm](https://github.com/microsoft/LMOps/tree/main/minillm)

---

## Core Idea

### Why Reverse KL?

Standard forward KL: `KL(p_teacher || p_student)` requires the student to cover all modes of the teacher distribution, leading to **mean-seeking** behavior and overestimation of low-probability regions.

Reverse KL: `KL(p_student || p_teacher)` makes the student **mode-seeking** — it focuses on the high-probability modes of the teacher. For language models, this prevents the student from assigning probability mass to incoherent or unsafe outputs.

### Loss Function

```
L_MINILLM = KL(p_student || p_teacher)
           = E_{y ~ p_student(·|x)} [ log p_student(y|x) - log p_teacher(y|x) ]
           = -E_{y ~ p_student(·|x)} [ r(y|x) ]

where: r(y|x) = log p_teacher(y|x) - log p_student(y|x)   (per-sequence reward)
```

Since this expectation is over `p_student`, direct gradient computation requires the **REINFORCE / policy gradient** trick:

```
∇_θ L ≈ -E_{y ~ p_student}[ r(y|x) · ∇_θ log p_student(y|x) ]
```

with a **value-function baseline** `V(x)` to reduce variance:

```
∇_θ L ≈ -E_{y ~ p_student}[ (r(y|x) - V(x)) · ∇_θ log p_student(y|x) ]
```

---

## Training Algorithm

```
For each batch (x, y_ref) from dataset:
  1. Sample y_s ~ p_student(·|x)       [student generates a response]
  2. Compute r(y_s|x) = log p_teacher(y_s|x) - log p_student(y_s|x)
  3. Compute baseline V(x)             [average reward for x]
  4. Policy gradient loss:
       L_PG = -(r(y_s|x) - V(x)) * log p_student(y_s|x)
  5. Optional SFT regularization:
       L_SFT = -log p_student(y_ref|x)
  6. Total: L = L_PG + beta * L_SFT
```

---

## Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `beta` | 0.5 | SFT regularization weight |
| `max_new_tokens` | 256 | Max tokens for student generation |
| `temperature` | 1.0 | Sampling temperature for student |
| `top_p` | 0.9 | Nucleus sampling for student |
| `baseline_ema` | 0.99 | EMA decay for value baseline |
| `learning_rate` | 1e-5 | (lower than SFT due to RL instability) |
| `kl_coeff` | 1.0 | Coefficient for KL loss term |

---

## Environment Setup

```bash
conda env create -f environment.yml
conda activate dia_minillm
```

---

## Usage

```bash
python train_minillm.py \
    --teacher_model ../../../../models/FT/full_ft/qwen3-4b-ft \
    --student_model Qwen/Qwen3Guard-Gen-0.6B \
    --train_data /path/to/train.jsonl \
    --output_dir ../../../../models/KD/minillm/qwen3-guard-0.6b \
    --num_epochs 3 \
    --beta 0.5

# Multi-GPU
accelerate launch train_minillm.py \
    --teacher_model ../../../../models/FT/full_ft/qwen3-4b-ft \
    --student_model Qwen/Qwen3Guard-Gen-0.6B \
    --train_data /path/to/train.jsonl \
    --output_dir ../../../../models/KD/minillm/qwen3-guard-0.6b
```

---

## Hardware Requirements

| Teacher | Student | Min VRAM |
|---------|---------|----------|
| Qwen3-4B (bf16) | Qwen3-0.6B (bf16) | ~32GB (both on GPU) |
| Qwen3-4B (8-bit) | Qwen3-0.6B (bf16) | ~20GB |
| Qwen3-4B (4-bit) | Qwen3-0.6B (bf16) | ~12GB |

The teacher is loaded in evaluation mode (no gradients) so it can be quantized to save memory.

---

## Output

Student model saved to `../../../../models/KD/minillm/<name>/` in HuggingFace format.

---

## Citation

```bibtex
@inproceedings{gu2024minillm,
  title={Knowledge Distillation of Large Language Models},
  author={Gu, Yuxian and Dong, Li and Wei, Furu and Huang, Minlie},
  booktitle={ICLR},
  year={2024}
}
```
