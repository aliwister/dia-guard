#!/bin/bash
# ============================================================================
# Model 6: bedrock-claude-sonnet-4.6 (Claude Sonnet 4.6)
# Assignment: 7 dialects (Multi-VALUE CounterHarm-SHIELD)
# ============================================================================
set -e

export AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:?Set AWS_ACCESS_KEY_ID}"
export AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:?Set AWS_SECRET_ACCESS_KEY}"

MODEL="bedrock-claude-sonnet-4.6"
WORKERS=4

# Local run — navigate directly



cd ~/dia-guard/codes/multi_value_gen/counterharm_shield

DIALECTS=(
    "pakistani_english"
    "philippine_english"
    "pure_fiji_english_basilectal_fijie"
    "rural_african_american_vernacular_english"
    "scottish_english"
    "southeast_american_enclave_dialects"
    "sri_lankan_english"
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
