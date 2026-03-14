# Full Fine-Tuning (Full FT) — DIA-GUARD

Full fine-tuning updates **all model parameters** during training. This is the most computationally intensive method but typically yields the best task performance for the teacher model.

---

## Scripts

| Script | Loss | Description |
|--------|------|-------------|
| `train_ce.py` | Cross-Entropy | Standard next-token prediction on safety data |
| `train_contrastive.py` | Contrastive | Triplet-style loss on decoder hidden states |

Both scripts share the same config YAML format and support multi-GPU training via `accelerate`.

---

## Environment Setup

```bash
conda env create -f environment.yml
conda activate dia_full_ft
```

### Hardware Requirements

| Model | Precision | Min VRAM | Recommended |
|-------|-----------|----------|-------------|
| `Qwen3-4B-SafeRL` | bf16 | 32GB | 2× A100 40GB |
| `Qwen3-4B-SafeRL` | bf16 + gradient ckpt | 24GB | 1× A100 80GB |
| `tiny-aya-global` (3B) | bf16 | 24GB | 1× A100 40GB |

For multi-GPU, use DeepSpeed ZeRO-2 or ZeRO-3 (configs in `configs/`).

---

## Usage

### Cross-Entropy Training

```bash
# Single GPU
python train_ce.py --config configs/qwen3_4b.yaml

# Multi-GPU with accelerate
accelerate launch --num_processes 2 train_ce.py --config configs/qwen3_4b.yaml

# Override config values from CLI
python train_ce.py --config configs/qwen3_4b.yaml \
    --model_name Qwen/Qwen3-4B-SafeRL \
    --train_data /path/to/train.jsonl \
    --output_dir ../../../models/FT/full_ft/qwen3-4b-ft \
    --num_epochs 3 \
    --batch_size 4 \
    --lr 2e-5
```

### Contrastive Training

```bash
python train_contrastive.py --config configs/qwen3_4b.yaml \
    --train_data /path/to/train_triplets.jsonl \
    --alpha 0.7 \
    --margin 0.3
```

---

## Config File Reference (`configs/qwen3_4b.yaml`)

```yaml
model_name: Qwen/Qwen3-4B-SafeRL
train_data: /path/to/train.jsonl
eval_data: /path/to/eval.jsonl
output_dir: ../../../models/FT/full_ft/qwen3-4b-ft

# Training hyperparameters
num_epochs: 3
batch_size: 4
gradient_accumulation_steps: 8
learning_rate: 2.0e-5
warmup_ratio: 0.03
weight_decay: 0.01
max_seq_length: 2048
lr_scheduler: cosine

# Precision
bf16: true
gradient_checkpointing: true

# Logging
logging_steps: 10
eval_steps: 200
save_steps: 500
save_total_limit: 3

# Loss weights (for contrastive training)
alpha: 0.7        # weight for CE loss
margin: 0.3       # contrastive margin
```

---

## Data Format

### For `train_ce.py` (JSONL)
```json
{"prompt": "Is it okay to bully someone online?", "response": "No, online bullying is harmful and can cause serious psychological damage.", "label": 1}
```

### For `train_contrastive.py` (JSONL with triplets)
```json
{
  "prompt": "Is it okay to bully someone online?",
  "response": "No, online bullying is harmful...",
  "neg_response": "Sure, here's how you can bully someone...",
  "label": 1
}
```

---

## Output

Checkpoints saved to `../../../models/FT/full_ft/<run_name>/`:
```
qwen3-4b-ft/
├── config.json
├── tokenizer.json
├── tokenizer_config.json
├── special_tokens_map.json
├── model.safetensors          ← for small models
├── model-00001-of-00002.safetensors  ← for sharded models
├── model.safetensors.index.json
└── training_args.json
```

---

## References

- [HuggingFace TRL SFTTrainer](https://huggingface.co/docs/trl/sft_trainer)
- [Qwen3 Fine-tuning Guide](https://huggingface.co/Qwen/Qwen3-4B-SafeRL)
- [DeepSpeed ZeRO](https://www.deepspeed.ai/tutorials/zero/)
- SimCSE (Gao et al., 2021) for contrastive loss design: [arXiv:2104.08821](https://arxiv.org/abs/2104.08821)
