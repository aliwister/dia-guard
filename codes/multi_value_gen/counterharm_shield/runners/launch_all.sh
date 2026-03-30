#!/bin/bash
# ============================================================================
# Launch all 7 CounterHarm-SHIELD runners in parallel (Multi-VALUE)
#
# 49 dialects across 7 Bedrock models (7 dialects each):
#   1. bedrock-deepseek        → aboriginal, acrolectal_fiji, appalachian, australian, australian_vernacular, bahamian, black_south_african
#   2. bedrock-llama4-maverick → cameroon, cape_flats, channel_islands, chicano, colloquial_american, singapore, earlier_aave
#   3. bedrock-mistral-large3  → east_anglian, northern_england, southeast_england, southwest_england, falkland, ghanaian, hong_kong
#   4. bedrock-qwen3-32b       → indian, indian_south_african, irish, jamaican, kenyan, liberian_settler, malaysian
#   5. bedrock-llama3.3-70b    → maltese, manx, new_zealand, newfoundland, nigerian, orkney_shetland, ozark
#   6. bedrock-claude-sonnet   → pakistani, philippine, fiji_basilectal, rural_aave, scottish, southeast_enclave, sri_lankan
#   7. bedrock-claude-opus     → st_helena, tanzanian, tristan, ugandan, urban_aave, welsh, white_south_african, white_zimbabwean
#
# Usage:
#   export AWS_ACCESS_KEY_ID=...
#   export AWS_SECRET_ACCESS_KEY=...
#   bash launch_all.sh
# ============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "============================================================"
echo "CounterHarm-SHIELD (Multi-VALUE): Launching 7 parallel runners"
echo "Timestamp: $TIMESTAMP"
echo "Logs: $LOG_DIR/"
echo "============================================================"

RUNNERS=(
    "run_bedrock_deepseek.sh"
    "run_bedrock_llama4_maverick.sh"
    "run_bedrock_mistral_large3.sh"
    "run_bedrock_qwen3_32b.sh"
    "run_bedrock_llama3_3_70b.sh"
    "run_bedrock_claude_sonnet.sh"
    "run_bedrock_claude_opus.sh"
)

PIDS=()

for runner in "${RUNNERS[@]}"; do
    NAME="${runner%.sh}"
    LOG_FILE="$LOG_DIR/${NAME}_${TIMESTAMP}.log"

    echo "  Launching: $runner -> $LOG_FILE"
    bash "$SCRIPT_DIR/$runner" > "$LOG_FILE" 2>&1 &
    PIDS+=($!)
done

echo ""
echo "All ${#RUNNERS[@]} runners launched. PIDs: ${PIDS[*]}"
echo ""
echo "Monitor progress:"
echo "  tail -f $LOG_DIR/*_${TIMESTAMP}.log"
echo ""
echo "Check status:"
echo "  ps aux | grep full_generation_parallel"
echo ""

# Wait for all to finish
FAILED=0
for i in "${!PIDS[@]}"; do
    PID=${PIDS[$i]}
    RUNNER=${RUNNERS[$i]}
    if wait $PID; then
        echo "[DONE] $RUNNER (PID $PID) completed successfully"
    else
        echo "[FAIL] $RUNNER (PID $PID) exited with error"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "============================================================"
echo "ALL RUNNERS FINISHED"
echo "  Success: $((${#RUNNERS[@]} - FAILED)) / ${#RUNNERS[@]}"
echo "  Failed:  $FAILED / ${#RUNNERS[@]}"
echo "============================================================"
