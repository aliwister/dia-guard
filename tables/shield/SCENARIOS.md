# DIA-GUARD Shield — Experiment Scenarios Explained

This document explains what each **scenario** in the Shield result tables means, what questions they answer, and how to read them.

---

## Background

**DIA-GUARD** is a safety classifier benchmark spanning **50 English dialects** (AAVE, Australian English, Cameroon English, etc.). Given a text input, a Shield model must classify it as `safe` or `unsafe`.

Two test sets:
- **Holdout**: 181,874 dialect-transformed samples (main benchmark)
- **SAE**: 36,050 standard American English samples (tests whether dialect-tuning hurt plain-English ability)

---

## Experiment scenarios

### 🔹 Baseline (no training)

**What it is**: The 7 student LLMs evaluated **directly from HuggingFace**, without any DIA-GUARD fine-tuning.

**Purpose**: Establishes the zero-shot safety classification ability of the raw pretrained models. Any subsequent FT or KD needs to beat this to justify the compute.

**Models (7)**:
- Gemma-3-270m, Gemma-3-1B (Google)
- Qwen3Guard-Gen-0.6B, Qwen3.5-0.8B, Qwen3-1.7B (Qwen)
- Llama-3.2-1B (Meta), SmolLM2-1.7B (HuggingFace)

**Typical result**: 0.54-0.75 accuracy (most are near-random).

---

### 🔹 Shield PEFT-CE (supervised fine-tuning, parameter-efficient)

**What it is**: LoRA adapters (rank 64) trained on DIA-GUARD using cross-entropy loss on the safe/unsafe label. Base model frozen; only ~3% of parameters are trainable.

**Purpose**: Cheap, fast fine-tuning that keeps the base model intact and only teaches the student the classification format.

**Training**:
- Script: `codes/evaluation/FineTune/peft/train_ce_lora.py`
- 50K-836K dialect samples, 1-3 epochs with early stopping
- QLoRA (4-bit quantized base) for teacher-size models
- Result = LoRA adapter (~100-500 MB each)

**Typical result**: 0.75-0.97 accuracy. This is usually a **strong upper bound** for small models.

---

### 🔹 Shield Full-FT-CE (supervised fine-tuning, all parameters)

**What it is**: All model parameters trained end-to-end with cross-entropy loss on DIA-GUARD.

**Purpose**: More expressive than PEFT but requires more VRAM and risks overfitting/mode collapse.

**Training**:
- Script: `codes/evaluation/FineTune/full_ft/train_ce.py`
- Same data as PEFT-CE
- Liger Kernel for memory-efficient fused softmax+CE where supported
- Result = full model (~2-4 GB each)

**Typical result**: 0.75-0.97 accuracy. Sometimes slightly better than PEFT, sometimes worse (mode collapse risk on small models).

---

### 🔹 Scenario 1 — Out-of-Box Knowledge Distillation (`-OOB`)

**What it is**: Distill a **large off-the-shelf teacher** (that has never seen DIA-GUARD) into a **small off-the-shelf student** (also never seen DIA-GUARD), using 50K DIA-GUARD training samples.

**Pipeline**:
```
Qwen3-4B-SafeRL (off-the-shelf)
         │
         │   50K DIA-GUARD prompts
         │   KD method: MINILLM, GKD, or TED
         ▼
Qwen3Guard-Gen-0.6B (off-the-shelf)
```

**Purpose**: Tests whether knowledge distillation **alone** (without any supervised signal on the student) can transfer safety knowledge from a bigger model to a smaller one.

**Teachers (2)**:
- Qwen3-4B-SafeRL (4B params)
- Qwen3Guard-Gen-8B (8B params, loaded in 8-bit)

**Students (2)** — must share Qwen tokenizer with teachers:
- Qwen3Guard-Gen-0.6B
- Qwen3-1.7B

**KD methods (3)**:
- **MINILLM** — reverse KL divergence + policy gradient + SFT regularizer
- **GKD** — on-policy student generation + JSD divergence over teacher logits
- **TED** — task-aware layer-wise hidden-state alignment + CE + KD losses

**Matrix**: 3 methods × 2 teachers × 2 students = **12 KD runs**

**Typical result**: 0.54-0.69 accuracy. **KD alone is weaker than supervised FT** — the student never sees hard labels, so it can only mimic teacher's distribution.

**Why call it `OOB`?** Because both teacher and student are "out-of-the-box" — no DIA-GUARD training on either side before KD.

---

### 🔹 Scenario 2 — FT Knowledge Distillation (`-FT-S2-...`)

**What it is**: Distill a **DIA-GUARD fine-tuned teacher** into a **student that is also fine-tuned on DIA-GUARD** (or still off-the-shelf for comparison), using 50K samples, **3 epochs** of KD training.

**Pipeline**:
```
Qwen3-4B-SafeRL FT (trained by us on DIA-GUARD via LoRA-CE)
         │
         │   50K DIA-GUARD prompts
         │   KD method: MINILLM, GKD, or TED
         ▼
Qwen3Guard-Gen-0.6B PEFT-CE Shield (already fine-tuned on DIA-GUARD)
```

**Purpose**: Tests whether KD adds **additional value on top of supervised fine-tuning**. Does distilling from a strong FT teacher beat the FT student alone? Can we squeeze more out of our small models?

**Teachers (2)** — trained in-house via QLoRA-CE on DIA-GUARD:
- Qwen3-4B-SafeRL FT (**0.9692 holdout acc** — strong)
- Qwen3Guard-Gen-8B FT (**0.5432 holdout acc** — degenerate "always unsafe" predictor)

**Student starting points (6 per teacher × method)**:

| Tag | Meaning | Comes from |
|-----|---------|-----------|
| `qg-peft`  | Qwen3Guard-Gen-0.6B Shield PEFT-CE | our existing PEFT-CE Shield model |
| `q17-peft` | Qwen3-1.7B Shield PEFT-CE          | our existing PEFT-CE Shield model |
| `qg-full`  | Qwen3Guard-Gen-0.6B Shield Full-FT-CE | our existing Full-FT-CE Shield model |
| `q17-full` | Qwen3-1.7B Shield Full-FT-CE        | our existing Full-FT-CE Shield model |
| `qg-base`  | Qwen3Guard-Gen-0.6B off-the-shelf  | HuggingFace base model (no FT) |
| `q17-base` | Qwen3-1.7B off-the-shelf           | HuggingFace base model (no FT) |

**Matrix**: 3 methods × 2 teachers × 6 starting points = **36 KD runs**

**Typical result**: 0.54-0.96 accuracy. **Strong KD + strong starting student can exceed pure supervised FT** (e.g. GKD 8B-ft → q17-peft reached 0.9577 — beating the teacher and every FT baseline).

**Why call it `S2`?** To distinguish from Scenario 1 — same teacher/student families but both sides have now been fine-tuned on DIA-GUARD before the KD step.

---

## Teacher FT (prerequisite for Scenario 2)

**What it is**: The two teacher models (Qwen3-4B-SafeRL and Qwen3Guard-Gen-8B) are themselves fine-tuned on DIA-GUARD via LoRA-CE before being used as KD teachers in Scenario 2.

**Results** (on Holdout test set):

| Teacher | Acc | Note |
|---------|-----|------|
| Qwen3-4B-SafeRL FT | **0.9692** | Excellent — 2nd-best model in the entire experiment |
| Qwen3Guard-Gen-8B FT | 0.5432 | Degenerate (predicts "always unsafe", 99.8% ASR) |

The 8B teacher's degeneration is a surprising finding — despite healthy training loss (0.15), the model collapsed to a trivial predictor. We left the Scenario 2 runs using this teacher intact to study what happens during KD from a degenerate teacher.

---

## Reading the LaTeX tables

### `shield_ft_holdout.tex` and `shield_ft_sae.tex`
- Columns: Baseline | PEFT-CE | Full-FT-CE (sub-columns: Acc, Prec, Rec, ASR, F1)
- Rows: 7 student models
- These are the **Shield FT results** — the supervised baseline to beat.

### `shield_ft_by_dataset.tex`
- Same as above but compressed: Holdout and SAE side-by-side for each (model, method)
- Lets you quickly see whether dialect-training hurt SAE performance.

### `kd_scenario1.tex`
- **12 OOB KD cells**: 3 methods × 2 teachers × 2 students
- Answer: does off-the-shelf KD alone beat zero-shot baseline?
- Short answer: **marginally**. KD without labels is weak (best ≈ 0.69).

### `kd_scenario2.tex`
- **36 FT KD cells**: 3 methods × 2 teachers × 6 student starts
- Answer: does KD add value on top of FT?
- Short answer: **yes** for the right combination. GKD 8B-ft → q17-peft = 0.9577 beats all FT baselines.

### `shield_per_dialect_top5.tex`
- Top 5 Shield models × 50 dialects
- Shows dialect-specific performance to identify which dialects are hardest.

---

## Glossary

| Term | Definition |
|------|-----------|
| **Shield** | Umbrella name for this family of DIA-GUARD safety classifiers |
| **Holdout** | The 181K-sample dialect test set (primary metric) |
| **SAE** | The 36K-sample Standard American English test set |
| **OOB** | Out-of-box (off-the-shelf, no DIA-GUARD fine-tuning) |
| **FT / PEFT-CE / Full-FT-CE** | Fine-tuning / Parameter-efficient (LoRA) CE / Full-parameter CE |
| **S2** | Scenario 2 KD (FT teacher + FT/base student) |
| **ASR** | Attack Success Rate = TP / (TP + FN) — fraction of unsafe content correctly caught |
| **QLoRA** | Quantized LoRA (4-bit base + trainable LoRA adapters) |
| **MINILLM / GKD / TED** | Three white-box knowledge distillation methods (see respective papers) |

---

## References

- MINILLM: Gu et al., *Knowledge Distillation of Large Language Models* (ICLR 2024)
- GKD: Agarwal et al., *On-Policy Distillation of Language Models* (ICLR 2024)
- TED: Liang et al., *Less is More: Task-aware Layer-wise Distillation* (ICML 2023)
- QLoRA: Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs* (NeurIPS 2023)

---

For the raw numbers, see the `metrics.json` files under `codes/evaluation/results/Shield/` or the HuggingFace model cards at `jsl5710/Shield-*`.
