#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# setup_gpu.sh — Update all training configs for the target GPU tier
#
# Usage:
#   bash setup_gpu.sh a100    # A100 80GB — larger batches, flash_attention_2, tf32
#   bash setup_gpu.sh h100    # H100 80GB — same as A100 but even faster
#   bash setup_gpu.sh t4      # T4  16GB  — small batches, eager attn, no tf32
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

GPU="${1:?Usage: $0 <a100|h100|t4>}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PEFT_CONFIGS="$SCRIPT_DIR/peft/configs"
FULL_CONFIGS="$SCRIPT_DIR/full_ft/configs"

echo "Configuring all training configs for: $GPU"

case "$GPU" in
  a100|h100)
    ATTN="flash_attention_2"
    TF32="true"
    # Batch sizes per model size tier
    SMALL_BS=16    # 270m, 0.6b, 0.8b
    SMALL_GA=1
    MED_BS=8       # 1b, 1.7b
    MED_GA=2
    LARGE_BS=4     # 3b, 4b
    LARGE_GA=4
    ;;
  t4)
    ATTN="eager"
    TF32="false"
    SMALL_BS=2
    SMALL_GA=8
    MED_BS=2
    MED_GA=8
    LARGE_BS=1
    LARGE_GA=16
    ;;
  *)
    echo "ERROR: Unknown GPU type '$GPU'. Use: a100, h100, or t4"
    exit 1
    ;;
esac

update_config() {
  local file="$1" bs="$2" ga="$3"
  if [[ ! -f "$file" ]]; then return; fi

  sed -i "s/^attn_implementation:.*/attn_implementation: $ATTN/" "$file"
  sed -i "s/^tf32:.*/tf32: $TF32/" "$file"
  sed -i "s/^per_device_train_batch_size:.*/per_device_train_batch_size: $bs/" "$file"
  sed -i "s/^per_device_eval_batch_size:.*/per_device_eval_batch_size: $bs/" "$file"
  sed -i "s/^gradient_accumulation_steps:.*/gradient_accumulation_steps: $ga/" "$file"

  echo "  $(basename $file): batch=$bs, grad_acc=$ga, attn=$ATTN, tf32=$TF32"
}

echo ""
echo "PEFT configs:"
# Small models (270m, 0.6b, 0.8b)
for cfg in gemma_270m_lora.yaml qwen_guard_0.6b_lora.yaml qwen_0.8b_lora.yaml; do
  update_config "$PEFT_CONFIGS/$cfg" "$SMALL_BS" "$SMALL_GA"
done
# Medium models (1b, 1.7b)
for cfg in gemma_1b_lora.yaml llama_1b_lora.yaml smollm_1.7b_lora.yaml qwen_1.7b_lora.yaml; do
  update_config "$PEFT_CONFIGS/$cfg" "$MED_BS" "$MED_GA"
done
# Large models (3b, 4b)
for cfg in aya_3b_lora.yaml qwen3_4b_lora.yaml; do
  update_config "$PEFT_CONFIGS/$cfg" "$LARGE_BS" "$LARGE_GA"
done

echo ""
echo "Full FT configs:"
# Small
for cfg in gemma_270m.yaml qwen_guard_0.6b.yaml qwen_0.8b.yaml; do
  update_config "$FULL_CONFIGS/$cfg" "$SMALL_BS" "$SMALL_GA"
done
# Medium
for cfg in gemma_1b.yaml llama_1b.yaml smollm_1.7b.yaml qwen_1.7b.yaml; do
  update_config "$FULL_CONFIGS/$cfg" "$MED_BS" "$MED_GA"
done
# Large (teachers)
for cfg in aya_3b.yaml qwen3_4b.yaml; do
  update_config "$FULL_CONFIGS/$cfg" "$LARGE_BS" "$LARGE_GA"
done

echo ""
echo "Done! Configs updated for $GPU."
echo "Effective batch size is always 16 (batch_size x gradient_accumulation)."
