# DIA-GUARD Model Weights

This directory stores all trained model weights for the DIA-GUARD pipeline, organized by training method.

Models are saved in HuggingFace format and can be loaded directly with `transformers` or uploaded to HuggingFace Hub.

---

## Directory Structure

```
models/
├── README.md                        ← you are here
├── upload_to_hub.py                 ← script to push any model to HuggingFace Hub
│
├── FT/                              ← Fine-tuned teacher models (>2B params)
│   ├── README.md
│   ├── full_ft/                     ← Full fine-tuning weights
│   │   ├── README.md
│   │   ├── qwen3-4b-ft/             ← Dia-Guard-4B (Qwen3-4B teacher, CE loss)
│   │   ├── qwen3-4b-contrastive-ft/ ← Dia-Guard-4B (Qwen3-4B teacher, contrastive loss)
│   │   ├── aya-3b-ft/               ← Dia-Guard-3B (Aya teacher, CE loss)
│   │   └── aya-3b-contrastive-ft/   ← Dia-Guard-3B (Aya teacher, contrastive loss)
│   └── peft/                        ← LoRA adapter weights
│       ├── README.md
│       ├── qwen3-4b-lora/           ← LoRA adapters only (~50MB)
│       ├── qwen3-4b-lora-merged/    ← Merged full model (used as teacher for KD)
│       ├── aya-3b-lora/
│       └── aya-3b-lora-merged/
│
└── KD/                              ← Distilled student models (<2B params)
    ├── README.md
    ├── minillm/                     ← MINILLM-distilled students
    │   ├── README.md
    │   ├── qwen3-guard-0.6b/
    │   ├── llama-1b/
    │   └── gemma-270m/
    ├── gkd/                         ← GKD-distilled students
    │   ├── README.md
    │   ├── qwen3-guard-0.6b/
    │   ├── llama-1b/
    │   └── smollm2-1.7b/
    └── ted/                         ← TED-distilled students
        ├── README.md
        ├── qwen3-guard-0.6b/
        ├── gemma-270m/
        └── gemma-1b/
```

---

## Loading Models

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load a distilled student
model = AutoModelForCausalLM.from_pretrained(
    "models/KD/minillm/qwen3-guard-0.6b",
    torch_dtype="bfloat16",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("models/KD/minillm/qwen3-guard-0.6b")

# Load a LoRA adapter
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B-SafeRL", torch_dtype="bfloat16")
model = PeftModel.from_pretrained(base, "models/FT/peft/qwen3-4b-lora")
```

---

## Uploading to HuggingFace Hub

```bash
# Single model
python upload_to_hub.py \
    --model_dir models/KD/minillm/qwen3-guard-0.6b \
    --repo_id your-org/Dia-Guard-0.6B-MINILLM \
    --private false

# All KD models at once
python upload_to_hub.py --upload_all_kd --org your-org

# Dry run (shows what would be uploaded)
python upload_to_hub.py --model_dir models/KD/ted/gemma-270m --dry_run
```

See `upload_to_hub.py --help` for full options.

---

## Expected Checkpoint Contents

Each saved model directory contains:
```
<model-name>/
├── config.json
├── tokenizer.json
├── tokenizer_config.json
├── special_tokens_map.json
├── model.safetensors                       ← for models <5GB
├── model-00001-of-0000N.safetensors        ← for sharded models
├── model.safetensors.index.json            ← shard index
└── training_config.yaml                    ← hyperparameters used
```

For TED distillation, additionally:
```
├── task_aware_filters.pt     ← saved filter weights
└── alignment_metadata.json   ← layer alignment info
```

For PEFT adapters:
```
├── adapter_config.json       ← LoRA config
└── adapter_model.safetensors ← adapter weights only
```

---

## Model Naming Convention

When uploading to HuggingFace, use this naming convention:

```
{org}/Dia-Guard-{size}-{method}
```

Examples:
- `jsl5710/Dia-Guard-0.6B-MINILLM`
- `jsl5710/Dia-Guard-270M-TED`
- `jsl5710/Dia-Guard-1B-GKD`
- `jsl5710/Dia-Guard-4B-FT`         ← fine-tuned teacher
