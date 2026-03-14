#!/bin/bash

# Run CPU-based evaluation metrics (ROUGE-L, DiffLib)
# Usage: ./run_cpu_eval.sh <input_path> [options]

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CPU_ENV="$SCRIPT_DIR/envs/cpu_eval"

# Check if environment exists
if [ ! -d "$CPU_ENV" ]; then
    echo "CPU environment not found. Run setup_envs.sh first."
    exit 1
fi

# Activate environment
source "$CPU_ENV/bin/activate"

# Change to script directory for imports
cd "$SCRIPT_DIR"

# Run evaluation
python run_evaluation.py \
    --input "$1" \
    --no-gpu-metrics \
    --cpu-metrics \
    --no-feature-accuracy \
    "${@:2}"

deactivate
