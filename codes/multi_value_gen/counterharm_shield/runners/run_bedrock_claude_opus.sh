#!/bin/bash
# ============================================================================
# Model 7: bedrock-claude-opus-4.6 (Claude Opus 4.6)
# Assignment: 7 dialects (Multi-VALUE CounterHarm-SHIELD)
# ============================================================================
set -e

export AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:?Set AWS_ACCESS_KEY_ID}"
export AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:?Set AWS_SECRET_ACCESS_KEY}"

MODEL="bedrock-claude-opus-4.6"
WORKERS=4

# Local run — navigate directly



cd ~/dia-guard/codes/multi_value_gen/counterharm_shield

DIALECTS=(
    "st_helena_english"
    "tanzanian_english"
    "tristan_da_cunha_english"
    "ugandan_english"
    "urban_african_american_vernacular_english"
    "welsh_english"
    "white_south_african_english"
    "white_zimbabwean_english"
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
