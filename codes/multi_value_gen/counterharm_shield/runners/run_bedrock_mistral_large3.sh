#!/bin/bash
# ============================================================================
# Model 3: bedrock-mistral-large3 (Mistral Large 3 675B)
# Assignment: 7 dialects (Multi-VALUE CounterHarm-SHIELD)
# ============================================================================
set -e

export AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:?Set AWS_ACCESS_KEY_ID}"
export AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:?Set AWS_SECRET_ACCESS_KEY}"

MODEL="bedrock-mistral-large3"
WORKERS=4

# Local run — navigate directly



cd ~/dia-guard/codes/multi_value_gen/counterharm_shield

DIALECTS=(
    "east_anglian_english"
    "english_dialects_in_the_north_of_england"
    "english_dialects_in_the_southeast_of_england"
    "english_dialects_in_the_southwest_of_england"
    "falkland_islands_english"
    "ghanaian_english"
    "hong_kong_english"
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
