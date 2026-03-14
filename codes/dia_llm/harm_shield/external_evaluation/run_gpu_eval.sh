#!/bin/bash

# Run GPU-based evaluation metrics
# Usage: ./run_gpu_eval.sh <input_path> [--batch-size 8] [--dialect dialect_name]

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
GPU_ENV="$SCRIPT_DIR/envs/gpu_eval"

# Check if environment exists
if [ ! -d "$GPU_ENV" ]; then
    echo "GPU environment not found. Run setup_envs.sh first."
    exit 1
fi

# Activate environment
source "$GPU_ENV/bin/activate"

# Change to script directory for imports
cd "$SCRIPT_DIR"

# Run evaluation
python run_evaluation.py \
    --input "$1" \
    --gpu-metrics \
    --no-cpu-metrics \
    --no-feature-accuracy \
    "${@:2}"

deactivate
