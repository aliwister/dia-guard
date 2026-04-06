#!/bin/bash
# ============================================================================
# Launch all zero-shot + few-shot inference experiments via tmux
#
# 14 Bedrock models x 4 shot counts (0, 2, 4, 8) = 56 experiment configs
# Each model gets its own tmux session running all shot counts sequentially
#
# Usage:
#   export AWS_ACCESS_KEY_ID=...
#   export AWS_SECRET_ACCESS_KEY=...
#   bash launch_all.sh
# ============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFERENCE_DIR="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(cd "$INFERENCE_DIR/../.." && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

TS=$(date +%Y%m%d_%H%M%S)

# Verify credentials
: "${AWS_ACCESS_KEY_ID:?Set AWS_ACCESS_KEY_ID}"
: "${AWS_SECRET_ACCESS_KEY:?Set AWS_SECRET_ACCESS_KEY}"

AWS_KEY="$AWS_ACCESS_KEY_ID"
AWS_SECRET="$AWS_SECRET_ACCESS_KEY"

DATA_DIR="$REPO_ROOT/dataset"
SPLITS_DIR="$REPO_ROOT/dataset/dia_splits"
RESULTS_DIR="$INFERENCE_DIR/results"

# All 15 Bedrock API models
MODELS=(
    bedrock_deepseek
    bedrock_llama4_maverick
    bedrock_llama3_3_70b
    bedrock_mistral_large3
    bedrock_ministral_14b
    bedrock_qwen3_32b
    bedrock_claude_opus
    bedrock_claude_sonnet
    bedrock_gpt_oss_safeguard_20b
    bedrock_gpt_oss_safeguard_120b
    bedrock_gpt_oss_20b
    bedrock_gpt_oss_120b
    bedrock_gemma3_27b
    bedrock_qwen3_coder_30b
    bedrock_qwen3_235b
)

echo "============================================================"
echo "DIA-GUARD Inference: Launching ${#MODELS[@]} model sessions"
echo "Shot counts: 0 2 4 8"
echo "Data: $DATA_DIR"
echo "Splits: $SPLITS_DIR"
echo "Results: $RESULTS_DIR"
echo "Timestamp: $TS"
echo "============================================================"

for MODEL in "${MODELS[@]}"; do
    SESSION="inf-${MODEL#bedrock_}"
    LOG_FILE="$LOG_DIR/${MODEL}_${TS}.log"

    echo "  Launching: $SESSION -> $LOG_FILE"

    tmux new-session -d -s "$SESSION" "\
        export AWS_ACCESS_KEY_ID='$AWS_KEY' AWS_SECRET_ACCESS_KEY='$AWS_SECRET'; \
        cd $INFERENCE_DIR && \
        python few-shot_evaluate_guards.py \
            --data_dir $DATA_DIR \
            --splits_dir $SPLITS_DIR \
            --results_dir $RESULTS_DIR \
            --models $MODEL \
            --num_shots 0 2 4 8 \
        2>&1 | tee $LOG_FILE"
done

echo ""
echo "All ${#MODELS[@]} sessions launched."
echo ""
echo "Monitor:"
echo "  tmux ls | grep inf-"
echo "  tail -f $LOG_DIR/*_${TS}.log"
echo ""
echo "Attach to a session:"
echo "  tmux attach -t inf-deepseek"
