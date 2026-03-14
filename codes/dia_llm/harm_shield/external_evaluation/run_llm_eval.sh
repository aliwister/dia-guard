#!/bin/bash

# Run LLM-as-a-Judge evaluation
# Usage: ./run_llm_eval.sh <input_path> --dialect <dialect_name> [--llm-backend azure] [--llm-model gpt-4]

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LLM_ENV="$SCRIPT_DIR/envs/llm_eval"

# Check if environment exists
if [ ! -d "$LLM_ENV" ]; then
    echo "LLM environment not found. Run setup_envs.sh first."
    exit 1
fi

# Activate environment
source "$LLM_ENV/bin/activate"

# Change to parent directory for dialect_transformer imports
cd "$SCRIPT_DIR/.."

# Run evaluation
python -c "
import sys
sys.path.insert(0, '$SCRIPT_DIR')
from run_evaluation import run_llm_evaluation
run_llm_evaluation('$1', dialect_name='$2')
"

deactivate
