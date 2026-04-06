#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# launch_all_groups1_3.sh — Orchestrate all Group 1 & 3 training runs
#
# Launches 36 training runs across 8x A100-40GB GPUs using tmux.
# Each run gets its own tmux window for monitoring.
#
# Groups:
#   Group 1: Teacher FT (2 models × 2 methods × 2 losses = 8 runs)
#   Group 3: Student FT Baseline (7 models × 2 methods × 2 losses = 28 runs)
#
# GPU allocation strategy:
#   - Qwen3-4B full FT: 2 GPUs (DeepSpeed ZeRO-2) — too large for single 40GB
#   - Aya-3B full FT: 2 GPUs (DeepSpeed ZeRO-2) — borderline, safer with 2
#   - All PEFT/LoRA runs: 1 GPU each (QLoRA for teachers, LoRA for students)
#   - All student full FT: 1 GPU each (≤1.7B fits on 40GB)
#
# Since we have 8 GPUs and 36 runs, we run in waves:
#   Wave 1-N: 8 concurrent jobs (1 per GPU), rotate through all 36 runs.
#   Multi-GPU jobs (teachers full FT) consume 2 GPUs each.
#
# Usage:
#   bash launch_all_groups1_3.sh
#
# Prerequisites:
#   1. Run: bash setup_gpu.sh a100_40gb
#   2. Data splits available at dataset/dia_splits/{train,val}.jsonl
#   3. All packages installed in /opt/pytorch or active conda env
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TMUX_SESSION="dia-guard-ft"

# ─── Define all runs ─────────────────────────────────────────────────────────
# Format: "method|loss|model_id|gpus|num_gpus|deepspeed_flag"
#
# We organize runs into waves. Each wave fills all 8 GPUs.
# Teachers full FT use 2 GPUs with DeepSpeed; everything else uses 1 GPU.

# ─── WAVE 1: Teachers full FT (4 GPUs) + 4 student PEFT runs ─────────────
WAVE1=(
  # Group 1 — Teacher full FT (DeepSpeed ZeRO-2, 2 GPUs each)
  "full_ft|ce|Qwen/Qwen3-4B-SafeRL|0,1|2|deepspeed"
  "full_ft|ce|CohereLabs/tiny-aya-global|2,3|2|deepspeed"
  # Group 3 — Student PEFT CE (1 GPU each, filling remaining GPUs)
  "peft|ce|google/gemma-3-270m-it|4|1|"
  "peft|ce|Qwen/Qwen3Guard-Gen-0.6B|5|1|"
  "peft|ce|Qwen/Qwen3.5-0.8B|6|1|"
  "peft|ce|google/gemma-3-1b-it|7|1|"
)

# ─── WAVE 2: Teachers full FT contrastive + 4 student PEFT runs ──────────
WAVE2=(
  "full_ft|contrastive|Qwen/Qwen3-4B-SafeRL|0,1|2|deepspeed"
  "full_ft|contrastive|CohereLabs/tiny-aya-global|2,3|2|deepspeed"
  "peft|ce|meta-llama/Llama-3.2-1B-Instruct|4|1|"
  "peft|ce|HuggingFaceTB/SmolLM2-1.7B-Instruct|5|1|"
  "peft|ce|Qwen/Qwen3-1.7B|6|1|"
  "peft|contrastive|google/gemma-3-270m-it|7|1|"
)

# ─── WAVE 3: Teacher PEFT (1 GPU each) + student PEFT contrastive ────────
WAVE3=(
  "peft|ce|Qwen/Qwen3-4B-SafeRL|0|1|"
  "peft|ce|CohereLabs/tiny-aya-global|1|1|"
  "peft|contrastive|Qwen/Qwen3-4B-SafeRL|2|1|"
  "peft|contrastive|CohereLabs/tiny-aya-global|3|1|"
  "peft|contrastive|Qwen/Qwen3Guard-Gen-0.6B|4|1|"
  "peft|contrastive|Qwen/Qwen3.5-0.8B|5|1|"
  "peft|contrastive|google/gemma-3-1b-it|6|1|"
  "peft|contrastive|meta-llama/Llama-3.2-1B-Instruct|7|1|"
)

# ─── WAVE 4: Student PEFT contrastive (remaining) + student full FT CE ───
WAVE4=(
  "peft|contrastive|HuggingFaceTB/SmolLM2-1.7B-Instruct|0|1|"
  "peft|contrastive|Qwen/Qwen3-1.7B|1|1|"
  "full_ft|ce|google/gemma-3-270m-it|2|1|"
  "full_ft|ce|Qwen/Qwen3Guard-Gen-0.6B|3|1|"
  "full_ft|ce|Qwen/Qwen3.5-0.8B|4|1|"
  "full_ft|ce|google/gemma-3-1b-it|5|1|"
  "full_ft|ce|meta-llama/Llama-3.2-1B-Instruct|6|1|"
  "full_ft|ce|HuggingFaceTB/SmolLM2-1.7B-Instruct|7|1|"
)

# ─── WAVE 5: Student full FT CE (remaining) + contrastive ────────────────
WAVE5=(
  "full_ft|ce|Qwen/Qwen3-1.7B|0|1|"
  "full_ft|contrastive|google/gemma-3-270m-it|1|1|"
  "full_ft|contrastive|Qwen/Qwen3Guard-Gen-0.6B|2|1|"
  "full_ft|contrastive|Qwen/Qwen3.5-0.8B|3|1|"
  "full_ft|contrastive|google/gemma-3-1b-it|4|1|"
  "full_ft|contrastive|meta-llama/Llama-3.2-1B-Instruct|5|1|"
  "full_ft|contrastive|HuggingFaceTB/SmolLM2-1.7B-Instruct|6|1|"
  "full_ft|contrastive|Qwen/Qwen3-1.7B|7|1|"
)

# ─── Helper: launch a wave ───────────────────────────────────────────────────

launch_wave() {
  local wave_name="$1"
  shift
  local runs=("$@")

  echo ""
  echo "════════════════════════════════════════════════════════════"
  echo "  Launching ${wave_name} (${#runs[@]} jobs)"
  echo "════════════════════════════════════════════════════════════"

  for run_spec in "${runs[@]}"; do
    IFS='|' read -r method loss model gpus num_gpus ds_flag <<< "${run_spec}"
    local model_short=$(echo "${model}" | awk -F'/' '{print $NF}' | tr '[:upper:]' '[:lower:]' | tr '.' '_')
    local win_name="${method}_${loss}_${model_short}"

    echo "  Starting: ${method} ${loss} ${model} on GPU(s) ${gpus}"

    tmux new-window -t "${TMUX_SESSION}" -n "${win_name}" \
      "bash ${SCRIPT_DIR}/launch_ft.sh ${method} ${loss} '${model}' ${gpus} ${num_gpus} ${ds_flag}; echo 'DONE: ${win_name}'; read"
  done

  echo ""
  echo "  All ${wave_name} jobs launched. Monitor with: tmux attach -t ${TMUX_SESSION}"
  echo "  Waiting for all jobs to complete before next wave..."
  echo "  Press ENTER when all jobs in this wave are done."
  read -r -p "  > "
}

# ─── Main ─────────────────────────────────────────────────────────────────────

echo "DIA-GUARD Groups 1 & 3 — Full Training Orchestrator"
echo "===================================================="
echo "Total runs: 36 (8 teacher + 28 student)"
echo "Waves: 5"
echo "GPUs: 8x A100-40GB"
echo ""

# Create tmux session
tmux kill-session -t "${TMUX_SESSION}" 2>/dev/null || true
tmux new-session -d -s "${TMUX_SESSION}" -n "control"

echo "tmux session created: ${TMUX_SESSION}"
echo "Attach with: tmux attach -t ${TMUX_SESSION}"
echo ""

launch_wave "Wave 1" "${WAVE1[@]}"
launch_wave "Wave 2" "${WAVE2[@]}"
launch_wave "Wave 3" "${WAVE3[@]}"
launch_wave "Wave 4" "${WAVE4[@]}"
launch_wave "Wave 5" "${WAVE5[@]}"

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  All 36 training runs complete!"
echo "════════════════════════════════════════════════════════════"
