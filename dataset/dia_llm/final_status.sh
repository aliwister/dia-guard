#!/bin/bash
echo "================================================"
echo "LLM_DATA GPU METRICS EVALUATION - FINAL STATUS"
echo "================================================"
echo ""
echo "Date: $(date)"
echo ""

# Check if BERTScore/BARTScore is still running
if pgrep -f "run_gpu_metrics.py" > /dev/null; then
    echo "BERTScore/BARTScore: RUNNING"
    bert_completed=$(grep -o '".*_zeroshot_harmfulness_results_with_transforms.csv"' .llm_data_gpu_progress.json 2>/dev/null | wc -l)
    echo "  Completed: $bert_completed/287 files"
    echo "  Remaining: $((287 - bert_completed)) files"
else
    echo "BERTScore/BARTScore: COMPLETED"
    bert_completed=$(grep -o '".*_zeroshot_harmfulness_results_with_transforms.csv"' .llm_data_gpu_progress.json 2>/dev/null | wc -l)
    echo "  Total: $bert_completed/287 files"
fi

echo ""

# Check AlignScore status
align_completed=$(grep -o '".*_zeroshot_harmfulness_results_with_transforms.csv"' .llm_data_alignscore_progress.json 2>/dev/null | wc -l)
echo "AlignScore: COMPLETED (partial)"
echo "  Total: $align_completed/287 files"
echo "  Note: Retry needed for failed files"

echo ""
echo "================================================"
echo "METRICS ADDED TO CSV FILES"
echo "================================================"
echo "- basic_bertscore"
echo "- coi_bertscore"
echo "- basic_bartscore"
echo "- coi_bartscore"
echo "- basic_alignscore (partial)"
echo "- coi_alignscore (partial)"
echo ""
echo "================================================"
echo "GPU STATUS"
echo "================================================"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv

