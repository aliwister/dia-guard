# TED — Task-Aware Layer-wise Distillation

TED (Liang et al., ICML 2023) takes a different approach from MINILLM and GKD: rather than matching output token distributions, TED aligns the student's **intermediate hidden representations** to the teacher's at each layer using learned task-aware filters.

This is especially useful when the student and teacher have different hidden dimensions (e.g., teacher hidden=4096, student hidden=1024).

---

## Paper & Code

- **Paper**: Liang et al. (2023). *Less is More: Task-aware Layer-wise Distillation for Language Model Compression*. ICML 2023.
  - [arXiv:2210.01351](https://arxiv.org/abs/2210.01351)

---

## Core Idea

### Layer-wise Knowledge Transfer

Standard output-distribution KD only transfers knowledge from the final layer. TED additionally transfers knowledge from **every intermediate layer**, providing a richer training signal.

### Task-Aware Filters

For each student layer `l`, a small linear projection `W_l` maps the student's hidden states to the teacher's hidden dimension. These projections are **trained jointly** with the student, and the filter weights `W_l` act as "task-aware" adapters that learn which aspects of the teacher's representations are relevant for the safety task.

### Loss Function (Three Terms)

```
L_TED = α * L_CE + β * L_KD + γ * L_layer

L_CE    = -Σ_t log p_student(y_t | x, y_{<t})
        [cross-entropy on safe responses — task loss]

L_KD    = KL(softmax(z_teacher/τ) || softmax(z_student/τ)) * τ²
        [token-level output distribution matching with temperature τ]

L_layer = (1/L) * Σ_l (1/T) * Σ_t || W_l · h_s_l(t) - h_t_l(t) ||²_F
        [layer-wise hidden state alignment via task-aware filters]

where:
  L           = number of alignment layers
  h_s_l(t)    = student hidden state at layer l, position t
  h_t_l(t)    = teacher hidden state at aligned layer l', position t
  W_l ∈ R^(d_teacher × d_student) = learned task-aware filter for layer l
  τ           = temperature (default 2.0)
  α, β, γ     = loss weights (default 1.0, 1.0, 0.5)
```

### Layer Alignment Strategy

Since teacher and student have different numbers of layers, layers are aligned uniformly:
```
teacher_layer_idx = round(teacher_layer * (l / num_student_layers))
```

---

## Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lam` | 0.5 | Weight for layer alignment loss `λ` |
| `num_align_layers` | -1 | Number of layers to align (-1 = all) |
| `learning_rate` | 2e-5 | (Higher than MINILLM; CE-based) |
| `layer_loss_type` | `mse` | `mse` or `cosine` |

---

## Architecture Notes

| Teacher | Student | Filter Shape W_l |
|---------|---------|-----------------|
| Qwen3-4B (d=2560) | Qwen3-0.6B (d=1024) | (2560, 1024) |
| Qwen3-4B (d=2560) | Qwen3Guard-0.6B (d=1024) | (2560, 1024) |
| tiny-aya-3B (d=2048) | Llama-3.2-1B (d=2048) | (2048, 2048) = identity |
| tiny-aya-3B (d=2048) | gemma-3-270m (d=1152) | (2048, 1152) |

---

## Environment Setup

```bash
conda env create -f environment.yml
conda activate dia_ted
```

---

## Usage

```bash
python train_ted.py \
    --teacher_model ../../../../models/FT/full_ft/qwen3-4b-ft \
    --student_model Qwen/Qwen3Guard-Gen-0.6B \
    --train_data /path/to/train.jsonl \
    --output_dir ../../../../models/KD/ted/qwen3-guard-0.6b \
    --lam 0.5 \
    --num_align_layers 6

# Multi-GPU
accelerate launch train_ted.py \
    --teacher_model ../../../../models/FT/full_ft/qwen3-4b-ft \
    --student_model google/gemma-3-270m-it \
    --train_data /path/to/train.jsonl \
    --output_dir ../../../../models/KD/ted/gemma-270m \
    --lam 0.5
```

---

## Hardware Requirements

Both teacher and student must forward-pass simultaneously (teacher in no-grad mode).

| Teacher | Student | Min VRAM |
|---------|---------|----------|
| Qwen3-4B (bf16) | Gemma-270M (bf16) | ~28GB |
| Qwen3-4B (4-bit) | Gemma-270M (bf16) | ~10GB |

For memory efficiency: load teacher in 4-bit with `--teacher_load_in_4bit`.

---

## Citation

```bibtex
@inproceedings{liang2023ted,
  title={Less is More: Task-aware Layer-wise Distillation for Language Model Compression},
  author={Liang, Chen and Zuo, Simiao and Zhang, Qingru and He, Pengcheng and Chen, Weizhu and Zhao, Tuo},
  booktitle={ICML},
  year={2023}
}
```
