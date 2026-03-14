# PEFT / LoRA Fine-Tuning — DIA-GUARD

Parameter-Efficient Fine-Tuning using **LoRA** (Low-Rank Adaptation) and **QLoRA** (4-bit quantized LoRA). Only a small number of adapter weights are trained, drastically reducing GPU memory requirements while achieving near-full-FT performance.

---

## Why PEFT?

| Method | Trainable Params | VRAM (4B model) | Performance |
|--------|-----------------|-----------------|-------------|
| Full FT | 100% | ~32GB | Highest |
| LoRA (r=64) | ~1% | ~16GB | Near-Full FT |
| QLoRA (4-bit + r=64) | ~1% | ~8GB | Slightly below LoRA |

PEFT is recommended when:
- Running on a single GPU (≤24GB VRAM)
- Rapid iteration / hyperparameter search
- Training multiple model variants simultaneously

---

## Scripts

| Script | Loss | Description |
|--------|------|-------------|
| `train_ce_lora.py` | Cross-Entropy | LoRA adapters with standard SFT loss |
| `train_contrastive_lora.py` | Contrastive | LoRA adapters with triplet contrastive loss |

---

## LoRA Target Modules by Model Family

Each decoder model family requires different attention/MLP layer names:

| Model Family | Target Modules |
|---|---|
| **Qwen3 / Qwen3.5** | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` |
| **Llama-3.2** | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` |
| **Gemma-3** | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` |
| **Aya (Cohere)** | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` |
| **SmolLM2** | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` |
| **DeepSeek-R1-Distill** | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` |

The configs automatically set these per `model_name` — see `configs/`.

---

## Environment Setup

```bash
conda env create -f environment.yml
conda activate dia_peft
```

### Hardware Requirements

| Model | Method | Min VRAM |
|-------|--------|----------|
| Qwen3-4B | LoRA (bf16) | ~16GB |
| Qwen3-4B | QLoRA (4-bit) | ~8GB |
| tiny-aya-global (3B) | LoRA (bf16) | ~12GB |
| tiny-aya-global (3B) | QLoRA (4-bit) | ~6GB |

---

## Usage

### LoRA + Cross-Entropy

```bash
python train_ce_lora.py --config configs/qwen3_4b_lora.yaml \
    --train_data /data/train.jsonl \
    --output_dir ../../../models/FT/peft/qwen3-4b-lora

# QLoRA (4-bit quantization)
python train_ce_lora.py --config configs/qwen3_4b_lora.yaml \
    --use_qlora true
```

### LoRA + Contrastive Loss

```bash
python train_contrastive_lora.py --config configs/qwen3_4b_lora.yaml \
    --train_data /data/triplets.jsonl \
    --alpha 0.7 --margin 0.3
```

---

## Merging LoRA Adapters

After training, merge the LoRA adapters into the base model for deployment:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B-SafeRL", torch_dtype="bfloat16")
model = PeftModel.from_pretrained(base, "../../../models/FT/peft/qwen3-4b-lora")
merged = model.merge_and_unload()
merged.save_pretrained("../../../models/FT/peft/qwen3-4b-lora-merged")
```

The merged model is then used as the teacher for distillation.

---

## Output

Adapters saved to `../../../models/FT/peft/<run_name>/`:
```
qwen3-4b-lora/
├── adapter_config.json       ← LoRA config
├── adapter_model.safetensors ← adapter weights only (~50MB for r=64)
├── tokenizer.json
└── training_config.yaml
```

---

## References

- [LoRA paper (Hu et al., 2022)](https://arxiv.org/abs/2106.09685)
- [QLoRA paper (Dettmers et al., 2023)](https://arxiv.org/abs/2305.14314)
- [HuggingFace PEFT docs](https://huggingface.co/docs/peft/index)
- [TRL SFTTrainer + LoRA guide](https://huggingface.co/docs/trl/sft_trainer)
