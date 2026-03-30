#!/bin/bash
# ============================================================================
# Model 1: bedrock-deepseek (DeepSeek V3.2)
# Assignment: 7 dialects (Multi-VALUE CounterHarm-SHIELD)
# ============================================================================
set -e

export AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:?Set AWS_ACCESS_KEY_ID}"
export AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:?Set AWS_SECRET_ACCESS_KEY}"

MODEL="bedrock-deepseek"
WORKERS=4

# Local run — navigate directly



cd ~/dia-guard/codes/multi_value_gen/counterharm_shield

DIALECTS=(
    "aboriginal_english"
    "acrolectal_fiji_english"
    "appalachian_english"
    "australian_english"
    "australian_vernacular_english"
    "bahamian_english"
    "black_south_african_english"
)

echo "============================================================"
echo "Model: $MODEL | Workers: $WORKERS | Dialects: ${#DIALECTS[@]}"
echo "============================================================"

for i in "${!DIALECTS[@]}"; do
    DIALECT="${DIALECTS[$i]}"
    NUM=$((i + 1))
    echo ""
    echo "────────────────────────────────────────────────────────────"
    echo "[$NUM/${#DIALECTS[@]}] Starting: $DIALECT"
    echo "────────────────────────────────────────────────────────────"

    python full_generation_parallel.py \
        --model "$MODEL" \
        --workers "$WORKERS" \
        --dialect "$DIALECT"

    echo "[$NUM/${#DIALECTS[@]}] Completed: $DIALECT"
done

echo ""
echo "============================================================"
echo "ALL DONE: $MODEL finished all ${#DIALECTS[@]} dialects"
echo "============================================================"
