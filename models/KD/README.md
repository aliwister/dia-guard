# Knowledge-Distilled Student Models (KD/)

Stores the smaller student models produced by each white-box distillation method.
These are the final DIA-GUARD deployable models.

---

## Sub-directories

| Folder | Method | Paper |
|--------|--------|-------|
| `minillm/` | Reverse KL distillation | Gu et al., ICLR 2024 |
| `gkd/` | On-policy distillation | Agarwal et al., ICLR 2024 |
| `ted/` | Layer-wise distillation | Liang et al., ICML 2023 |

---

## Student Model Targets

| Model Name | Base | Size | Target Dir |
|-----------|------|------|-----------|
| Dia-Guard-0.6B | Qwen3Guard-Gen-0.6B | 0.6B | `*/qwen3-guard-0.6b/` |
| Dia-Guard-270M | gemma-3-270m-it | 270M | `*/gemma-270m/` |
| Dia-Guard-1B§ | Llama-3.2-1B-Instruct ⭐ | 1B | `*/llama-1b/` |
| Dia-Guard-1B§ | gemma-3-1b-it | 1B | `*/gemma-1b/` |
| — | Qwen3.5-0.8B ⭐ | 0.8B | `*/qwen3.5-0.8b/` |
| — | Qwen3-1.7B | 1.7B | `*/qwen3-1.7b/` |
| — | DeepSeek-R1-Distill-1.5B | 1.5B | `*/deepseek-1.5b/` |
| — | SmolLM2-1.7B | 1.7B | `*/smollm2-1.7b/` |

---

## Uploading to HuggingFace

```bash
cd ..  # go to models/ directory

# Upload one model
python upload_to_hub.py \
    --model_dir KD/minillm/qwen3-guard-0.6b \
    --repo_id jsl5710/Dia-Guard-0.6B-MINILLM

# Upload all at once
python upload_to_hub.py --upload_all_kd --org jsl5710
```
