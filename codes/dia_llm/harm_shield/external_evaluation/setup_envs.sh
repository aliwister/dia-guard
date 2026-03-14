#!/bin/bash

# Setup script for evaluation environments
# Creates separate virtual environments for GPU and CPU metrics

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ENV_DIR="$SCRIPT_DIR/envs"

echo "=========================================="
echo "Setting up evaluation environments"
echo "=========================================="

# Create envs directory
mkdir -p "$ENV_DIR"

# ==========================================
# GPU Environment (BERTScore, BARTScore, AlignScore, METEOR)
# ==========================================
echo ""
echo "Setting up GPU environment..."
echo "=========================================="

GPU_ENV="$ENV_DIR/gpu_eval"

if [ ! -d "$GPU_ENV" ]; then
    python -m venv "$GPU_ENV"
    source "$GPU_ENV/bin/activate"

    pip install --upgrade pip
    pip install -r "$SCRIPT_DIR/requirements_gpu.txt"

    # Download NLTK data for METEOR
    python -c "import nltk; nltk.download('wordnet'); nltk.download('punkt'); nltk.download('omw-1.4')"

    deactivate
    echo "GPU environment created at: $GPU_ENV"
else
    echo "GPU environment already exists at: $GPU_ENV"
fi

# ==========================================
# CPU Environment (ROUGE-L, DiffLib)
# ==========================================
echo ""
echo "Setting up CPU environment..."
echo "=========================================="

CPU_ENV="$ENV_DIR/cpu_eval"

if [ ! -d "$CPU_ENV" ]; then
    python -m venv "$CPU_ENV"
    source "$CPU_ENV/bin/activate"

    pip install --upgrade pip
    pip install -r "$SCRIPT_DIR/requirements_cpu.txt"

    deactivate
    echo "CPU environment created at: $CPU_ENV"
else
    echo "CPU environment already exists at: $CPU_ENV"
fi

# ==========================================
# LLM Evaluation Environment (uses dialect_transformer)
# ==========================================
echo ""
echo "Setting up LLM evaluation environment..."
echo "=========================================="

LLM_ENV="$ENV_DIR/llm_eval"

if [ ! -d "$LLM_ENV" ]; then
    python -m venv "$LLM_ENV"
    source "$LLM_ENV/bin/activate"

    pip install --upgrade pip
    pip install openai anthropic pandas tqdm

    deactivate
    echo "LLM environment created at: $LLM_ENV"
else
    echo "LLM environment already exists at: $LLM_ENV"
fi

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Usage:"
echo "  GPU metrics:  source $GPU_ENV/bin/activate"
echo "  CPU metrics:  source $CPU_ENV/bin/activate"
echo "  LLM metrics:  source $LLM_ENV/bin/activate"
echo ""
echo "Or use the runner scripts:"
echo "  ./run_gpu_eval.sh <input_path>"
echo "  ./run_cpu_eval.sh <input_path>"
echo "  ./run_llm_eval.sh <input_path>"
