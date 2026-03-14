# Knowledge Distillation — DIA-GUARD

This module distills knowledge from fine-tuned teacher models (>2B params) into smaller student models (<2B params) using three **white-box** distillation methods.

White-box KD is preferred over black-box KD because it enables the student to access the teacher's internal representations (logits, hidden states), leading to higher-performance student models.

---

## Why White-Box KD?

| Approach | Teacher Access | Signal | Performance |
|----------|---------------|--------|-------------|
| Black-Box KD | Output text only | Hard/soft labels | Baseline |
| White-Box KD | Logits + hidden states | Full distribution + representations | **Higher** |

---

## Methods

### 1. MINILLM (Gu et al., 2024)
**Paper**: [arXiv:2306.08543](https://arxiv.org/abs/2306.08543) | [GitHub](https://github.com/microsoft/LMOps/tree/main/minillm)

Uses **reverse KL divergence** to align the student's output distribution with the teacher's:
```
L_MINILLM = KL(p_student || p_teacher) = E_{y~p_student}[log p_student(y|x) - log p_teacher(y|x)]
```
Optimized via policy gradient (REINFORCE with baseline). Prevents the student from overestimating low-probability regions of the teacher's distribution.

**Best for**: Generative safety response learning, distribution matching.

---

### 2. GKD (Agarwal et al., 2024)
**Paper**: [arXiv:2306.13649](https://arxiv.org/abs/2306.13649)

**On-policy** distillation: the student generates sequences during training, and the teacher provides soft probability feedback on those student-generated outputs. Supports multiple divergence functions.

```
L_GKD = E_{y~mix(p_student, p_teacher)}[D(p_teacher(·|x,y_{<t}) || p_student(·|x,y_{<t}))]
```
Mixing ratio `λ` controls the balance between student-generated and teacher-generated sequences.

**Best for**: Handling distribution mismatch; more robust than MINILLM on long sequences.

---

### 3. TED (Liang et al., 2023)
**Paper**: [arXiv:2210.01351](https://arxiv.org/abs/2210.01351)

**Task-aware layer-wise distillation**: trains task-aware filters (linear projections) that align the student's intermediate hidden representations to the teacher's at each layer:

```
L_TED = L_CE + λ * Σ_l ||W_l · h_student_l - h_teacher_l||²_F
```

Where `W_l` is a learnable projection that maps student hidden dim → teacher hidden dim at layer `l`.

**Best for**: Structural knowledge transfer, model compression with architecture differences.

---

## Distillation Pairs

| Teacher (fine-tuned) | Student (to distill into) | Recommended Method |
|---------------------|--------------------------|-------------------|
| Qwen3-4B-SafeRL (FT) | Qwen3Guard-Gen-0.6B | MINILLM, GKD |
| Qwen3-4B-SafeRL (FT) | Qwen3-0.6B | MINILLM, GKD |
| Qwen3-4B-SafeRL (FT) | Qwen3.5-0.8B | MINILLM, TED |
| Qwen3-4B-SafeRL (FT) | Qwen3-1.7B | TED |
| tiny-aya-global (FT) | Llama-3.2-1B-Instruct | GKD, TED |
| tiny-aya-global (FT) | gemma-3-1b-it | GKD, TED |
| tiny-aya-global (FT) | gemma-3-270m-it | TED |
| Qwen3-4B-SafeRL (FT) | SmolLM2-1.7B | GKD |
| Qwen3-4B-SafeRL (FT) | DeepSeek-R1-Distill-1.5B | MINILLM |

---

## Environment Setup

Each method uses its own isolated conda environment:

```bash
# MINILLM
conda env create -f minillm/environment.yml && conda activate dia_minillm

# GKD
conda env create -f gkd/environment.yml && conda activate dia_gkd

# TED
conda env create -f ted/environment.yml && conda activate dia_ted
```

---

## Pipeline

```
1. Fine-tune teacher (see ../FineTune/)
   └── ../../../models/FT/full_ft/qwen3-4b-ft/   OR
       ../../../models/FT/peft/qwen3-4b-lora-merged/

2. Run distillation
   └── python minillm/train_minillm.py \
           --teacher_model ../../../models/FT/full_ft/qwen3-4b-ft \
           --student_model Qwen/Qwen3Guard-Gen-0.6B \
           --train_data /data/train.jsonl \
           --output_dir ../../../models/KD/minillm/qwen3-guard-0.6b

3. Evaluate student
   └── [your evaluation script]

4. Upload to HuggingFace
   └── python ../../../models/upload_to_hub.py \
           --model_dir ../../../models/KD/minillm/qwen3-guard-0.6b \
           --repo_id your-org/Dia-Guard-0.6B-MINILLM
```

---

## References

| Method | Paper | Year | Venue |
|--------|-------|------|-------|
| MINILLM | Gu et al., *Knowledge Distillation of Large Language Models* | 2024 | ICLR |
| GKD | Agarwal et al., *On-Policy Distillation of Language Models* | 2024 | ICLR |
| TED | Liang et al., *Less is More: Task-aware Layer-wise Distillation* | 2023 | ICML |
