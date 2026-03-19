#!/bin/bash
# ============================================================================
# Model 3: bedrock-llama4-maverick (Llama 4 Maverick 17B Instruct)
# Assignment: 6 not started dialects
# ============================================================================
set -e

# ── 1. Setup credentials ──
export AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:?Set AWS_ACCESS_KEY_ID}"
export AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:?Set AWS_SECRET_ACCESS_KEY}"

MODEL="bedrock-llama4-maverick"
WORKERS=6

# ── 2. Clone repo (skip if already cloned) ──
if [ ! -d "dia-guard" ]; then
    echo "[Setup] Cloning dia-guard repo..."
    git clone https://github.com/jsl5710/dia-guard.git
fi
cd dia-guard

# ── 3. Install dependencies ──
pip install -q boto3 openai

# ── 4. Navigate to CounterHarm-SHIELD ──
cd codes/dia_llm/counterharm_shield

# ── 5. Run dialects one by one ──
DIALECTS=(
    "chicano_english"
    "colloquial_american_english"
    "colloquial_singapore_english_singlish"
    "earlier_african_american_vernacular_english"
    "east_anglian_english"
    "english_dialects_in_the_north_of_england"
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
