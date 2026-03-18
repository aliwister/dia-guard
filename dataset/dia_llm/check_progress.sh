#!/bin/bash
# Check evaluation progress

echo "========================================="
echo "GPU METRICS EVALUATION PROGRESS"
echo "========================================="
echo ""

# Check if process is still running
if pgrep -f "run_gpu_metrics.py" > /dev/null; then
    echo "Status: RUNNING ✓"
    echo "Process ID: $(pgrep -f run_gpu_metrics.py)"
else
    echo "Status: NOT RUNNING"
fi

echo ""

# Count total CSV files
total_files=$(find . -name "*_zeroshot_harmfulness_results_with_transforms.csv" | wc -l)
echo "Total CSV files: $total_files"

# Check progress file
if [ -f .llm_data_gpu_progress.json ]; then
    completed=$(grep -o '".*_zeroshot_harmfulness_results_with_transforms.csv"' .llm_data_gpu_progress.json | wc -l)
    echo "Completed files: $completed"
    echo "Remaining: $((total_files - completed))"
    echo "Progress: $((completed * 100 / total_files))%"
    
    echo ""
    echo "Last updated:"
    grep "last_updated" .llm_data_gpu_progress.json | head -1
else
    echo "No progress file found yet"
fi

echo ""
echo "========================================="
echo "GPU UTILIZATION"
echo "========================================="
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader

echo ""
echo "========================================="
echo "RECENT LOG OUTPUT"
echo "========================================="
tail -20 full_gpu_eval.log

