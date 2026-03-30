# DIA-GUARD Model Weights

This directory stores all trained model weights for the DIA-GUARD pipeline, organized by experiment group and training method.

Models are saved in HuggingFace format and can be uploaded to the Hub with `upload_all_models.py`.

---

## Directory Structure

```
models/
├── README.md                        <- you are here
├── upload_to_hub.py                 <- single model upload to HuggingFace
├── upload_all_models.py             <- batch upload all groups to HuggingFace
│
├── FT/                              <- Group 1: Teacher FT (>2B params)
│   ├── full_ft/                     <- Full fine-tuning weights
│   │   ├── qwen3_4b_saferl/
│   │   └── tiny_aya_global/
│   └── peft/                        <- LoRA adapter weights
│       ├── qwen3_4b_saferl/
│       └── tiny_aya_global/
│
├── KD/                              <- Group 2: KD Students (<2B params)
│   ├── minillm/                     <- MINILLM-distilled
│   ├── gkd/                         <- GKD-distilled
│   └── ted/                         <- TED-distilled
│
└── group3_student_ft_baseline/      <- Group 3: Student FT Baseline (<2B params)
    ├── peft/                        <- LoRA adapters
    │   ├── gemma_3_270m_it/
    │   ├── qwen3guard_gen_0_6b/
    │   ├── qwen3_5_0_8b/
    │   ├── gemma_3_1b_it/
    │   ├── llama_3_2_1b_instruct/
    │   ├── smollm2_1_7b_instruct/
    │   └── qwen3_1_7b/
    └── full_ft/                     <- Full fine-tuning weights
        └── (same 7 models)
```

---

## Experiment Groups

| Group | Directory | Description | Models |
|-------|-----------|-------------|--------|
| **G1** | `FT/` | Teacher FT — fine-tune large models on safety data | Qwen3-4B-SafeRL, Aya-3B |
| **G2** | `KD/` | KD Students — distill from G1 teachers | 7 student models x 3 KD methods |
| **G3** | `group3_student_ft_baseline/` | Student FT Baseline — direct fine-tuning (no KD) | 7 student models x 2 FT methods |

### Student Models (G2 & G3)

| Model | Size | HuggingFace ID | Slug |
|-------|------|----------------|------|
| Gemma-3-270M | 270M | `google/gemma-3-270m-it` | `gemma_3_270m_it` |
| Qwen3Guard-0.6B | 0.6B | `Qwen/Qwen3Guard-Gen-0.6B` | `qwen3guard_gen_0_6b` |
| Qwen3.5-0.8B | 0.8B | `Qwen/Qwen3.5-0.8B` | `qwen3_5_0_8b` |
| Gemma-3-1B | 1B | `google/gemma-3-1b-it` | `gemma_3_1b_it` |
| Llama-3.2-1B | 1B | `meta-llama/Llama-3.2-1B-Instruct` | `llama_3_2_1b_instruct` |
| SmolLM2-1.7B | 1.7B | `HuggingFaceTB/SmolLM2-1.7B-Instruct` | `smollm2_1_7b_instruct` |
| Qwen3-1.7B | 1.7B | `Qwen/Qwen3-1.7B` | `qwen3_1_7b` |

---

## Loading Models

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load a full fine-tuned model
model = AutoModelForCausalLM.from_pretrained(
    "models/KD/minillm/qwen3guard_gen_0_6b",
    torch_dtype="bfloat16",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("models/KD/minillm/qwen3guard_gen_0_6b")

# Load a LoRA adapter (Group 3)
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained("google/gemma-3-270m-it", torch_dtype="bfloat16")
model = PeftModel.from_pretrained(base, "models/group3_student_ft_baseline/peft/gemma_3_270m_it")
```

---

## Uploading to HuggingFace Hub

### Batch upload (recommended)

```bash
# Check status — see which models are complete
python upload_all_models.py --dry_run

# Upload all completed models across all groups
python upload_all_models.py --hf_token YOUR_HF_TOKEN

# Upload only Group 3
python upload_all_models.py --hf_token YOUR_HF_TOKEN --group 3

# Upload a specific model
python upload_all_models.py --hf_token YOUR_HF_TOKEN --only gemma_3_270m_it
```

### HuggingFace Repo Naming

Models are uploaded with this naming convention:
```
{org}/DIA-Guard-{Group}-{ModelShort}-{Method}
```

Examples:
- `jsl5710/DIA-Guard-Teacher-Qwen-4B-LoRA`
- `jsl5710/DIA-Guard-Student-Gemma-270M-LoRA`
- `jsl5710/DIA-Guard-Student-Llama-1B-FullFT`
- `jsl5710/DIA-Guard-KD-QwenGuard-0.6B-MiniLLM`

### Single model upload

```bash
python upload_to_hub.py \
    --model_dir KD/minillm/qwen3guard_gen_0_6b \
    --repo_id jsl5710/Dia-Guard-0.6B-MINILLM
```

---

## Expected Checkpoint Contents

**PEFT (LoRA) adapters:**
```
<model-slug>/
├── adapter_config.json
├── adapter_model.safetensors
├── tokenizer.json
├── tokenizer_config.json
└── special_tokens_map.json
```

**Full fine-tuning:**
```
<model-slug>/
├── config.json
├── model.safetensors (or sharded: model-00001-of-0000N.safetensors)
├── model.safetensors.index.json
├── tokenizer.json
├── tokenizer_config.json
└── special_tokens_map.json
```

**TED distillation (additionally):**
```
├── task_aware_filters.pt
└── alignment_metadata.json
```

---

## Training

See [codes/evaluation/FineTune/README.md](../codes/evaluation/FineTune/README.md) for training instructions, GPU setup, and hyperparameters.
