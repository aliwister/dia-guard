# Quantization — DIA-GUARD Group 4

Post-training quantization of distilled student models (Group 2 KD) and optionally
Group 3 (Student FT Baseline) models at multiple precision levels.

---

## Why Quantize?

Quantization reduces model size and inference latency with minimal accuracy loss,
making deployment feasible on edge devices and resource-constrained environments.

| Precision | Bytes/Param | Size Reduction | Typical Accuracy Loss |
|-----------|-------------|----------------|----------------------|
| **fp16** | 2.0 | 1x (baseline) | 0% |
| **int8** | 1.0 | 2x smaller | ~0-0.5% |
| **nf4** | 0.5 | 4x smaller | ~0.5-2% |

---

## Quantization Methods

### FP16 (Float16)
Standard half-precision floating point. Baseline for comparison — no quantization applied.

### INT8 — LLM.int8() (Dettmers et al., 2022)
Mixed-precision decomposition that keeps outlier features in fp16 while quantizing
the rest to 8-bit integers. Near-zero accuracy degradation.

```
Paper: https://arxiv.org/abs/2208.07339
```

### NF4 — NormalFloat 4-bit (Dettmers et al., 2023)
Information-theoretically optimal 4-bit data type for normally distributed weights.
Combined with double quantization (quantizing the quantization constants) for
additional memory savings.

```
Paper: https://arxiv.org/abs/2305.14314 (QLoRA)
```

---

## Usage

```bash
# Quantize all completed KD models at 16, 8, and 4-bit
python quantize_models.py

# Quantize a specific model
python quantize_models.py --model_dir ../../models/KD/minillm/qwen3guard_gen_0_6b

# Only 4-bit quantization
python quantize_models.py --bits 4

# Include Group 3 (Student FT Baseline) models
python quantize_models.py --include_group3

# Dry run — see what would be quantized
python quantize_models.py --dry_run

# Quantize and push to HuggingFace
python quantize_models.py --push_to_hub --hf_org jsl5710 --hf_token YOUR_TOKEN
```

---

## Output Structure

```
models/Quantized/
├── KD/
│   ├── minillm/
│   │   ├── qwen3guard_gen_0_6b/
│   │   │   ├── fp16/          ← float16 baseline
│   │   │   ├── int8/          ← LLM.int8()
│   │   │   └── nf4/           ← NF4 4-bit
│   │   ├── llama_3_2_1b_instruct/
│   │   │   ├── fp16/
│   │   │   ├── int8/
│   │   │   └── nf4/
│   │   └── ...
│   ├── gkd/
│   │   └── ...
│   └── ted/
│       └── ...
└── group3_student_ft_baseline/   ← only with --include_group3
    └── ...
```

Each quantized model directory contains:
```
<precision>/
├── config.json
├── model.safetensors
├── tokenizer.json
├── tokenizer_config.json
├── special_tokens_map.json
└── quantization_config.json    ← metadata (source, bits, method)
```

---

## HuggingFace Naming Convention

When pushed to HuggingFace:
```
{org}/DIA-Guard-{ModelShort}-{KD_METHOD}-{precision}
```

Examples:
- `jsl5710/DIA-Guard-QwenGuard-0.6B-MINILLM-nf4`
- `jsl5710/DIA-Guard-Llama-1B-GKD-int8`
- `jsl5710/DIA-Guard-Gemma-270M-TED-fp16`

---

## Loading Quantized Models

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load 4-bit quantized model
model = AutoModelForCausalLM.from_pretrained(
    "models/Quantized/KD/minillm/qwen3guard_gen_0_6b/nf4",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(
    "models/Quantized/KD/minillm/qwen3guard_gen_0_6b/nf4"
)

# Or from HuggingFace
model = AutoModelForCausalLM.from_pretrained(
    "jsl5710/DIA-Guard-QwenGuard-0.6B-MINILLM-nf4",
    device_map="auto",
)
```

---

## Dependencies

```bash
pip install transformers bitsandbytes accelerate peft
```

---

## References

| Method | Paper | Year |
|--------|-------|------|
| LLM.int8() | Dettmers et al., *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale* | 2022 |
| NF4 / QLoRA | Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs* | 2023 |
