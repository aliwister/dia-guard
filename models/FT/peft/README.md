# PEFT / LoRA Checkpoints (FT/peft/)

Stores LoRA adapter weights and merged full models.

## Expected Checkpoints

| Directory | Model | Type | Status |
|-----------|-------|------|--------|
| `qwen3-4b-lora/` | Qwen3-4B-SafeRL | LoRA adapters only | — |
| `qwen3-4b-lora-merged/` | Qwen3-4B-SafeRL | Merged full model | — |
| `aya-3b-lora/` | tiny-aya-global | LoRA adapters only | — |
| `aya-3b-lora-merged/` | tiny-aya-global | Merged full model | — |

## Merging LoRA for KD Use

The KD scripts require a full model (not just adapters). Merge before distillation:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-4B-SafeRL", torch_dtype=torch.bfloat16
)
model = PeftModel.from_pretrained(base, "qwen3-4b-lora")
merged = model.merge_and_unload()
merged.save_pretrained("qwen3-4b-lora-merged")

tokenizer = AutoTokenizer.from_pretrained("qwen3-4b-lora")
tokenizer.save_pretrained("qwen3-4b-lora-merged")
```

Or use the merge script:
```bash
cd ../../../../Evaluation/FineTune/peft
python merge_lora.py --adapter_dir ../../../models/FT/peft/qwen3-4b-lora \
    --base_model Qwen/Qwen3-4B-SafeRL \
    --output_dir ../../../models/FT/peft/qwen3-4b-lora-merged
```
