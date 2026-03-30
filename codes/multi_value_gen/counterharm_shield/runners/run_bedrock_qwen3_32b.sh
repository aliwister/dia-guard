#!/bin/bash
# ============================================================================
# Model 4: bedrock-qwen3-32b (Qwen3 32B)
# Assignment: 7 dialects (Multi-VALUE CounterHarm-SHIELD)
# ============================================================================
set -e

export AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:?Set AWS_ACCESS_KEY_ID}"
export AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:?Set AWS_SECRET_ACCESS_KEY}"

MODEL="bedrock-qwen3-32b"
WORKERS=4

# Local run — navigate directly



cd ~/dia-guard/codes/multi_value_gen/counterharm_shield

DIALECTS=(
    "indian_english"
    "indian_south_african_english"
    "irish_english"
    "jamaican_english"
    "kenyan_english"
    "liberian_settler_english"
    "malaysian_english"
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
