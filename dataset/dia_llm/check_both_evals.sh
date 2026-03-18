#!/bin/bash
echo "==========================================="
echo "PARALLEL EVALUATION STATUS"
echo "==========================================="
echo ""

# Check processes
echo "Running Processes:"
ps aux | grep "python3.*run_" | grep -v grep | awk '{print $2, $11, $12}'
echo ""

# BERTScore/BARTScore progress
if [ -f .llm_data_gpu_progress.json ]; then
    bert_completed=$(grep -o '".*_zeroshot_harmfulness_results_with_transforms.csv"' .llm_data_gpu_progress.json | wc -l)
    echo "BERTScore/BARTScore: $bert_completed/287 files"
fi

# AlignScore progress
if [ -f .llm_data_alignscore_progress.json ]; then
    align_completed=$(grep -o '".*_zeroshot_harmfulness_results_with_transforms.csv"' .llm_data_alignscore_progress.json | wc -l)
    echo "AlignScore: $align_completed/287 files"
fi

echo ""
echo "GPU Utilization:"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader

