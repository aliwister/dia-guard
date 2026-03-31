#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# launch_ft.sh  —  DIA-GUARD fine-tuning launcher (Groups 1 & 3)
#
# Usage:
#   bash launch_ft.sh <ft_method> <loss> <model_id> <gpus> [num_gpus]
#
# Examples:
#   # Group 3 — Student FT Baseline (single GPU)
#   bash launch_ft.sh peft ce google/gemma-3-270m-it 0
#
#   # Group 3 — Student FT Baseline (multi-GPU for 1.7B)
#   bash launch_ft.sh peft ce HuggingFaceTB/SmolLM2-1.7B-Instruct 0,1 2
#
#   # Group 1 — Teacher FT
#   bash launch_ft.sh peft ce Qwen/Qwen3-4B-SafeRL 0,1 2
#   bash launch_ft.sh full_ft ce CohereLabs/tiny-aya-global 0,1,2,3 4
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

FT_METHOD="${1:?Usage: $0 <ft_method> <loss> <model_id> <gpus> [num_gpus]}"
LOSS="${2:?}"
MODEL_ID="${3:?}"
GPUS="${4:?}"
NUM_GPUS="${5:-1}"

# Auto-detect python — use conda env if available, else system python
if [[ -x /anaconda/envs/azureml_py38_PT_TF/bin/python ]]; then
    PYTHON=/anaconda/envs/azureml_py38_PT_TF/bin/python
    ACCELERATE=/anaconda/envs/azureml_py38_PT_TF/bin/accelerate
else
    PYTHON=$(which python3)
    ACCELERATE=$(which accelerate)
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACCEL_CFG="${SCRIPT_DIR}/configs/accel_2gpu.yaml"

export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES="${GPUS}"
export WANDB_PROJECT="dia-guard"
# Set WANDB_API_KEY in your environment or run: wandb login

# Pick training script
if [[ "${FT_METHOD}" == "peft" ]]; then
    if [[ "${LOSS}" == "contrastive" ]]; then
        SCRIPT="${SCRIPT_DIR}/peft/train_contrastive_lora.py"
    else
        SCRIPT="${SCRIPT_DIR}/peft/train_ce_lora.py"
    fi
else
    if [[ "${LOSS}" == "contrastive" ]]; then
        SCRIPT="${SCRIPT_DIR}/full_ft/train_contrastive.py"
    else
        SCRIPT="${SCRIPT_DIR}/full_ft/train_ce.py"
    fi
fi

# Pick config file
CONFIGS_DIR="${SCRIPT_DIR}/${FT_METHOD}/configs"
MODEL_SLUG=$(echo "${MODEL_ID}" | awk -F'/' '{print $NF}' | tr '[:upper:]' '[:lower:]' | tr '-' '_' | tr '.' '_')

# Map model slug to config filename
declare -A CFG_MAP=(
    # Students
    ["gemma_3_270m_it"]="gemma_270m"
    ["gemma_3_1b_it"]="gemma_1b"
    ["llama_3_2_1b_instruct"]="llama_1b"
    ["qwen3guard_gen_0_6b"]="qwen_guard_0.6b"
    ["qwen3_5_0_8b"]="qwen_0.8b"
    ["smollm2_1_7b_instruct"]="smollm_1.7b"
    ["qwen3_1_7b"]="qwen_1.7b"
    # Teachers
    ["qwen3_4b_saferl"]="qwen3_4b"
    ["tiny_aya_global"]="aya_3b"
)

# Teacher model slugs — used to determine output group
declare -A TEACHERS=(
    ["qwen3_4b_saferl"]=1
    ["tiny_aya_global"]=1
)

CFG_KEY="${CFG_MAP[$MODEL_SLUG]:-}"
if [[ -n "${CFG_KEY}" ]]; then
    if [[ "${FT_METHOD}" == "peft" ]]; then
        CONFIG="${CONFIGS_DIR}/${CFG_KEY}_lora.yaml"
    else
        CONFIG="${CONFIGS_DIR}/${CFG_KEY}.yaml"
    fi
else
    CONFIG=""
fi

SPLITS_DIR="${SCRIPT_DIR}/../../.."
SPLITS_DIR="$(cd "${SPLITS_DIR}" && pwd)/dataset/dia_splits"
TRAIN_DATA="${SPLITS_DIR}/train.jsonl"
EVAL_DATA="${SPLITS_DIR}/val.jsonl"

# Output dir — Group 1 (teachers) vs Group 3 (students)
MODEL_SHORT=$(echo "${MODEL_ID}" | awk -F'/' '{print $NF}' | tr '[:upper:]' '[:lower:]' | tr '-' '_' | tr '.' '_')
MODELS_DIR="${SCRIPT_DIR}/../../.."
MODELS_DIR="$(cd "${MODELS_DIR}" && pwd)/models"

if [[ -n "${TEACHERS[$MODEL_SLUG]:-}" ]]; then
    GROUP="1"
    GROUP_LABEL="Teacher FT"
    OUTPUT_DIR="${MODELS_DIR}/FT/${FT_METHOD}/${MODEL_SHORT}"
else
    GROUP="3"
    GROUP_LABEL="Student FT Baseline"
    OUTPUT_DIR="${MODELS_DIR}/group3_student_ft_baseline/${FT_METHOD}/${MODEL_SHORT}"
fi

mkdir -p "${OUTPUT_DIR}"

echo "========================================================"
echo "DIA-GUARD Group ${GROUP} — ${GROUP_LABEL}"
echo "  Model    : ${MODEL_ID}"
echo "  Method   : ${FT_METHOD} | Loss: ${LOSS}"
echo "  GPUs     : ${GPUS} (${NUM_GPUS} process(es))"
echo "  Output   : ${OUTPUT_DIR}"
echo "  Config   : ${CONFIG:-<none>}"
echo "========================================================"

EXTRA_ARGS=""
[[ -n "${CONFIG}" ]] && EXTRA_ARGS="--config ${CONFIG}"

if [[ "${NUM_GPUS}" -gt 1 ]]; then
    echo "Launching with accelerate (${NUM_GPUS} GPUs)..."
    exec "${ACCELERATE}" launch \
        --config_file "${ACCEL_CFG}" \
        --num_processes "${NUM_GPUS}" \
        "${SCRIPT}" \
        --model_name "${MODEL_ID}" \
        --train_data "${TRAIN_DATA}" \
        --eval_data  "${EVAL_DATA}" \
        --output_dir "${OUTPUT_DIR}" \
        ${EXTRA_ARGS}
else
    echo "Launching single-GPU..."
    exec "${PYTHON}" "${SCRIPT}" \
        --model_name "${MODEL_ID}" \
        --train_data "${TRAIN_DATA}" \
        --eval_data  "${EVAL_DATA}" \
        --output_dir "${OUTPUT_DIR}" \
        ${EXTRA_ARGS}
fi
