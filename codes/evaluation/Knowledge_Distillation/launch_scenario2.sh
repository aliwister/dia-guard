#!/bin/bash
# Scenario 2 KD launcher: FT teacher × {FT-PEFT, FT-Full, Base} students × 3 KD methods
# = 3 × 2 × 6 = 36 runs
#
# This script only generates and prints commands; actual execution happens by
# piping into bash or copy/paste onto tmux windows per GPU.
#
# Usage:
#   bash launch_scenario2.sh print       # print all 36 launch commands
#   bash launch_scenario2.sh run <gpu_list>  # run the N matching commands for given GPUs

set -euo pipefail

KD_BASE=/data/vibe_exp/dia-guard/models/KD
MODELS=/data/vibe_exp/dia-guard/models
TRAIN=/data/vibe_exp/dia-guard/dataset/dia_splits/train_kd_50k.jsonl
PY=/data/pytorch/bin/python
KD_DIR=/data/vibe_exp/dia-guard/codes/evaluation/Knowledge_Distillation

# --- Teachers (merged LoRA → full) ---
T4B=$MODELS/FT/peft/qwen3_4b_saferl-merged
T8B=$MODELS/FT/peft/qwen3guard_gen_8b-merged

# --- Students: (label, path)
# Scenario 2a: FT-PEFT student starting points (merged)
S_QG_PEFT=$MODELS/group3_student_ft_baseline/peft/qwen3guard_gen_0_6b-merged
S_Q17_PEFT=$MODELS/group3_student_ft_baseline/peft/qwen3_1_7b-merged
# Scenario 2b: FT-Full student starting points (already standalone)
S_QG_FULL=$MODELS/group3_student_ft_baseline/full_ft/qwen3guard_gen_0_6b
S_Q17_FULL=$MODELS/group3_student_ft_baseline/full_ft/qwen3_1_7b
# Scenario 2c: Base student starting points (FT-teacher → OOB-student)
S_QG_BASE=Qwen/Qwen3Guard-Gen-0.6B
S_Q17_BASE=Qwen/Qwen3-1.7B

# Job = (method, teacher_slug, teacher_path, student_slug, student_path, starttag, batch_size, 8bit_flag)
JOBS=()
for METHOD in minillm gkd ted; do
  for T in "4b-ft:$T4B:bf16" "8b-ft:$T8B:8bit"; do
    T_TAG=$(echo "$T" | cut -d: -f1)
    T_PATH=$(echo "$T" | cut -d: -f2)
    T_QUANT=$(echo "$T" | cut -d: -f3)
    QFLAG=""
    [[ "$T_QUANT" == "8bit" ]] && QFLAG="--teacher_load_in_8bit"
    for S in "qg-peft:$S_QG_PEFT:4" "q17-peft:$S_Q17_PEFT:2" \
             "qg-full:$S_QG_FULL:4" "q17-full:$S_Q17_FULL:2" \
             "qg-base:$S_QG_BASE:4" "q17-base:$S_Q17_BASE:2"; do
      S_TAG=$(echo "$S" | cut -d: -f1)
      S_PATH=$(echo "$S" | cut -d: -f2)
      BS=$(echo "$S" | cut -d: -f3)
      OUT=$KD_BASE/$METHOD/$T_TAG/$S_TAG
      JOBS+=("$METHOD|$T_TAG|$T_PATH|$S_TAG|$S_PATH|$OUT|$BS|$QFLAG")
    done
  done
done

if [[ "${1:-}" == "print" ]]; then
  echo "Total Scenario 2 KD jobs: ${#JOBS[@]}"
  for i in "${!JOBS[@]}"; do
    IFS='|' read -r METHOD T_TAG T_PATH S_TAG S_PATH OUT BS QFLAG <<< "${JOBS[$i]}"
    printf "%2d. %-8s  %-6s  %-10s  →  %s\n" $((i+1)) "$METHOD" "$T_TAG" "$S_TAG" "$OUT"
  done
  exit 0
fi

echo "Run mode not yet implemented — use 'print' to see the matrix"
