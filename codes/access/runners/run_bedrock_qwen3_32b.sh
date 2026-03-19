#!/bin/bash
# ============================================================================
# Model 7: bedrock-qwen3-32b (Qwen3 32B dense)
# Assignment: 6 not started dialects
# ============================================================================
set -e

# ── 1. Setup credentials ──
export AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:?Set AWS_ACCESS_KEY_ID}"
export AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:?Set AWS_SECRET_ACCESS_KEY}"

MODEL="bedrock-qwen3-32b"
WORKERS=4

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
    "ozark_english"
    "pakistani_english"
    "philippine_english"
    "pure_fiji_english_basilectal_fijie"
    "rural_african_american_vernacular_english"
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
