#!/usr/bin/env bash
# monitor.sh — Hourly status monitor + auto-launch teachers when GPUs free
# Usage: bash monitor.sh (runs in foreground, Ctrl-C to stop)

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_DATA="/data/vibe_exp/dia-guard/dataset/dia_splits/train.jsonl"
EVAL_DATA="/data/vibe_exp/dia-guard/dataset/dia_splits/val.jsonl"
LOG_FILE="/data/vibe_exp/dia-guard/logs/monitor.log"
TEACHER_AYA_LAUNCHED=0
TEACHER_QWEN_LAUNCHED=0
TEACHER_QWEN8B_LAUNCHED=0

# Check if teacher launch flags exist from previous runs
[[ -f /data/vibe_exp/dia-guard/logs/.aya_launched ]] && TEACHER_AYA_LAUNCHED=1
[[ -f /data/vibe_exp/dia-guard/logs/.qwen4b_launched ]] && TEACHER_QWEN_LAUNCHED=1
[[ -f /data/vibe_exp/dia-guard/logs/.qwen8b_launched ]] && TEACHER_QWEN8B_LAUNCHED=1

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

print_status() {
    log "============================================================"
    log "  DIA-GUARD Training Status Report"
    log "============================================================"
    log ""

    # GPU table header
    printf "%-4s | %-30s | %-10s | %-6s | %-8s | %-5s | %-4s | %-10s | %-7s | %-7s | %-10s\n" \
        "GPU" "Job" "VRAM" "VRAM%" "GPU Util" "Batch" "GA" "Step" "Loss" "Acc" "ETA" | tee -a "$LOG_FILE"
    printf "%s\n" "$(printf '%.0s-' {1..120})" | tee -a "$LOG_FILE"

    # Get GPU info
    mapfile -t gpu_info < <(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits)

    # Get all tmux windows
    mapfile -t windows < <(tmux list-windows -t dia-guard-ft -F '#{window_name}' 2>/dev/null)

    for line in "${gpu_info[@]}"; do
        IFS=',' read -r gpu_id mem_used mem_total gpu_util <<< "$line"
        gpu_id=$(echo "$gpu_id" | xargs)
        mem_used=$(echo "$mem_used" | xargs)
        mem_total=$(echo "$mem_total" | xargs)
        gpu_util=$(echo "$gpu_util" | xargs)

        vram_pct=$((mem_used * 100 / mem_total))
        vram_str="${mem_used}/${mem_total}MB"

        # Find which job is on this GPU
        job_name="-"
        step_info="-"
        loss="-"
        acc="-"
        eta="-"
        batch="-"
        ga="-"

        if [[ $mem_used -gt 100 ]]; then
            # Find the job running on this GPU by checking window outputs
            for w in "${windows[@]}"; do
                output=$(tmux capture-pane -t "dia-guard-ft:$w" -p -S -50 2>/dev/null)
                # Check if this window mentions this GPU
                if echo "$output" | grep -q "CUDA_VISIBLE_DEVICES.*${gpu_id}" 2>/dev/null || \
                   echo "$w" | grep -qv "control" 2>/dev/null; then
                    # Get progress
                    prog=$(echo "$output" | grep -oP "\d+/\d+\s*\[" | tail -1 | tr -d ' [')
                    if [[ -n "$prog" ]]; then
                        step_info="$prog"
                    fi
                    loss_val=$(echo "$output" | grep -oP "'loss': '[0-9.]+'" | tail -1 | grep -oP "[0-9.]+$")
                    acc_val=$(echo "$output" | grep -oP "'mean_token_accuracy': '[0-9.]+'" | tail -1 | grep -oP "[0-9.]+$")
                    [[ -n "$loss_val" ]] && loss="$loss_val"
                    [[ -n "$acc_val" ]] && acc="$acc_val"
                fi
            done
        fi

        printf "%-4s | %-30s | %-10s | %-5s%% | %-7s%% | %-5s | %-4s | %-10s | %-7s | %-7s | %-10s\n" \
            "$gpu_id" "$job_name" "$vram_str" "$vram_pct" "$gpu_util" "$batch" "$ga" "$step_info" "$loss" "$acc" "$eta" | tee -a "$LOG_FILE"
    done

    log ""

    # Per-job detailed status
    log "--- Per-Job Detail ---"
    for w in "${windows[@]}"; do
        [[ "$w" == "control" ]] && continue
        output=$(tmux capture-pane -t "dia-guard-ft:$w" -p -S -10 2>/dev/null)
        done_check=$(echo "$output" | grep "DONE" | tail -1)
        progress=$(echo "$output" | grep -E "it/s\]|s/it" | tail -1 | head -c 120)
        if [[ -n "$done_check" ]]; then
            # Check if it was error or success
            error_check=$(tmux capture-pane -t "dia-guard-ft:$w" -p -S -30 2>/dev/null | grep -E "Error|OOM|Saving.*adapter|Saving.*model|saved" | tail -1)
            if echo "$error_check" | grep -qi "error\|oom"; then
                log "  FAILED: $w — $error_check"
            else
                log "  COMPLETED: $w — $error_check"
            fi
        elif [[ -n "$progress" ]]; then
            log "  RUNNING: $w — $progress"
        else
            log "  LOADING: $w"
        fi
    done

    log ""

    # Completed models check
    log "--- Completed Models ---"
    completed=0
    for d in /data/vibe_exp/dia-guard/models/group3_student_ft_baseline/peft/*/; do
        count=$(find "$d" -type f 2>/dev/null | wc -l)
        if [[ $count -gt 0 ]]; then
            log "  SAVED: $(basename $d) ($count files)"
            completed=$((completed + 1))
        fi
    done
    for d in /data/vibe_exp/dia-guard/models/FT/full_ft/*/; do
        count=$(find "$d" -type f 2>/dev/null | wc -l)
        if [[ $count -gt 0 ]]; then
            log "  SAVED: $(basename $d) ($count files)"
            completed=$((completed + 1))
        fi
    done
    [[ $completed -eq 0 ]] && log "  (none yet)"
    log ""
}

check_and_launch_teachers() {
    # Count free GPUs (< 100MB used)
    free_gpus=()
    while IFS=',' read -r gpu_id mem_used; do
        gpu_id=$(echo "$gpu_id" | xargs)
        mem_used=$(echo "$mem_used" | xargs)
        if [[ $mem_used -lt 100 ]]; then
            free_gpus+=("$gpu_id")
        fi
    done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)

    num_free=${#free_gpus[@]}
    log "Free GPUs: ${num_free} (${free_gpus[*]:-none})"

    # Need 4 free GPUs for a teacher
    if [[ $num_free -ge 4 ]]; then
        # Pick first 4 free GPUs
        gpu_str="${free_gpus[0]},${free_gpus[1]},${free_gpus[2]},${free_gpus[3]}"

        if [[ $TEACHER_AYA_LAUNCHED -eq 0 ]]; then
            log ">>> Launching Aya-3B full FT CE on GPUs ${gpu_str} (ZeRO-3, 4 GPUs)"
            FIRST_GPU=${free_gpus[0]}
            DS_PORT=$((29500 + FIRST_GPU * 10))

            # Create config with correct port
            cat > /tmp/accel_ds_zero3_aya.yaml <<EOFCFG
compute_environment: LOCAL_MACHINE
deepspeed_config:
  gradient_accumulation_steps: auto
  gradient_clipping: auto
  zero_optimization:
    stage: 3
    overlap_comm: true
    contiguous_gradients: true
    reduce_scatter: true
    reduce_bucket_size: 5.0e+08
    allgather_bucket_size: 5.0e+08
    offload_optimizer:
      device: cpu
      pin_memory: true
    offload_param:
      device: cpu
      pin_memory: true
  zero3_init_flag: true
  zero3_save_16bit_model: true
distributed_type: DEEPSPEED
downcast_bf16: 'no'
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 4
rdzv_backend: static
same_network: true
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
main_process_port: ${DS_PORT}
EOFCFG

            tmux new-window -t dia-guard-ft -n "aya3b_4gpu" \
              "export PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=${gpu_str} WANDB_PROJECT=dia-guard HF_HOME=/data/huggingface HF_TOKEN=\${HF_TOKEN:-\$(cat /data/huggingface/token 2>/dev/null)} TMPDIR=/data/tmp PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
               /opt/pytorch/bin/accelerate launch --config_file /tmp/accel_ds_zero3_aya.yaml --num_processes 4 \
               ${SCRIPT_DIR}/full_ft/train_ce.py --model_name 'CohereLabs/tiny-aya-global' --train_data '${TRAIN_DATA}' --eval_data '${EVAL_DATA}' \
               --output_dir '/data/vibe_exp/dia-guard/models/FT/full_ft/tiny_aya_global' --config ${SCRIPT_DIR}/full_ft/configs/aya_3b.yaml \
               2>&1 | tee /data/vibe_exp/dia-guard/logs/aya3b_4gpu_full_ce.log; echo 'DONE'; read"

            TEACHER_AYA_LAUNCHED=1
            touch /data/vibe_exp/dia-guard/logs/.aya_launched
            log ">>> Aya-3B launched on GPUs ${gpu_str}"

        elif [[ $TEACHER_QWEN_LAUNCHED -eq 0 ]]; then
            log ">>> Launching Qwen3-4B full FT CE on GPUs ${gpu_str} (ZeRO-3, 4 GPUs)"
            FIRST_GPU=${free_gpus[0]}
            DS_PORT=$((29500 + FIRST_GPU * 10))

            cat > /tmp/accel_ds_zero3_qwen.yaml <<EOFCFG
compute_environment: LOCAL_MACHINE
deepspeed_config:
  gradient_accumulation_steps: auto
  gradient_clipping: auto
  zero_optimization:
    stage: 3
    overlap_comm: true
    contiguous_gradients: true
    reduce_scatter: true
    reduce_bucket_size: 5.0e+08
    allgather_bucket_size: 5.0e+08
    offload_optimizer:
      device: cpu
      pin_memory: true
    offload_param:
      device: cpu
      pin_memory: true
  zero3_init_flag: true
  zero3_save_16bit_model: true
distributed_type: DEEPSPEED
downcast_bf16: 'no'
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 4
rdzv_backend: static
same_network: true
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
main_process_port: ${DS_PORT}
EOFCFG

            tmux new-window -t dia-guard-ft -n "qwen4b_4gpu" \
              "export PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=${gpu_str} WANDB_PROJECT=dia-guard HF_HOME=/data/huggingface HF_TOKEN=\${HF_TOKEN:-\$(cat /data/huggingface/token 2>/dev/null)} TMPDIR=/data/tmp PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
               /opt/pytorch/bin/accelerate launch --config_file /tmp/accel_ds_zero3_qwen.yaml --num_processes 4 \
               ${SCRIPT_DIR}/full_ft/train_ce.py --model_name 'Qwen/Qwen3-4B-SafeRL' --train_data '${TRAIN_DATA}' --eval_data '${EVAL_DATA}' \
               --output_dir '/data/vibe_exp/dia-guard/models/FT/full_ft/qwen3_4b_saferl' --config ${SCRIPT_DIR}/full_ft/configs/qwen3_4b.yaml \
               2>&1 | tee /data/vibe_exp/dia-guard/logs/qwen4b_4gpu_full_ce.log; echo 'DONE'; read"

            TEACHER_QWEN_LAUNCHED=1
            touch /data/vibe_exp/dia-guard/logs/.qwen4b_launched
            log ">>> Qwen3-4B launched on GPUs ${gpu_str}"

        elif [[ $TEACHER_QWEN8B_LAUNCHED -eq 0 ]]; then
            log ">>> Launching Qwen3Guard-8B full FT CE on GPUs ${gpu_str} (ZeRO-3, 4 GPUs)"
            FIRST_GPU=${free_gpus[0]}
            DS_PORT=$((29500 + FIRST_GPU * 10))

            cat > /tmp/accel_ds_zero3_qwen8b.yaml <<EOFCFG
compute_environment: LOCAL_MACHINE
deepspeed_config:
  gradient_accumulation_steps: auto
  gradient_clipping: auto
  zero_optimization:
    stage: 3
    overlap_comm: true
    contiguous_gradients: true
    reduce_scatter: true
    reduce_bucket_size: 5.0e+08
    allgather_bucket_size: 5.0e+08
    offload_optimizer:
      device: cpu
      pin_memory: true
    offload_param:
      device: cpu
      pin_memory: true
  zero3_init_flag: true
  zero3_save_16bit_model: true
distributed_type: DEEPSPEED
downcast_bf16: 'no'
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 4
rdzv_backend: static
same_network: true
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
main_process_port: ${DS_PORT}
EOFCFG

            tmux new-window -t dia-guard-ft -n "qwen8b_4gpu" \
              "export PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=${gpu_str} WANDB_PROJECT=dia-guard HF_HOME=/data/huggingface HF_TOKEN=\${HF_TOKEN:-\$(cat /data/huggingface/token 2>/dev/null)} TMPDIR=/data/tmp PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
               /opt/pytorch/bin/accelerate launch --config_file /tmp/accel_ds_zero3_qwen8b.yaml --num_processes 4 \
               ${SCRIPT_DIR}/full_ft/train_ce.py --model_name 'Qwen/Qwen3Guard-Gen-8B' --train_data '${TRAIN_DATA}' --eval_data '${EVAL_DATA}' \
               --output_dir '/data/vibe_exp/dia-guard/models/FT/full_ft/qwen3guard_gen_8b' --config ${SCRIPT_DIR}/full_ft/configs/qwen3guard_8b.yaml \
               2>&1 | tee /data/vibe_exp/dia-guard/logs/qwen8b_4gpu_full_ce.log; echo 'DONE'; read"

            TEACHER_QWEN8B_LAUNCHED=1
            touch /data/vibe_exp/dia-guard/logs/.qwen8b_launched
            log ">>> Qwen3Guard-8B launched on GPUs ${gpu_str}"
        else
            log "All teachers already launched."
        fi
    else
        teachers_remaining=""
        [[ $TEACHER_AYA_LAUNCHED -eq 0 ]] && teachers_remaining="Aya-3B "
        [[ $TEACHER_QWEN_LAUNCHED -eq 0 ]] && teachers_remaining="${teachers_remaining}Qwen3-4B "
        [[ $TEACHER_QWEN8B_LAUNCHED -eq 0 ]] && teachers_remaining="${teachers_remaining}Qwen3Guard-8B"
        if [[ -n "$teachers_remaining" ]]; then
            log "Need 4 free GPUs for teacher(s): ${teachers_remaining}. Only ${num_free} free."
        fi
    fi
}

# Main loop
log "Monitor started. Checking every hour. Ctrl-C to stop."
log "Teachers pending: Aya=$((1 - TEACHER_AYA_LAUNCHED)) Qwen4B=$((1 - TEACHER_QWEN_LAUNCHED)) Qwen8B=$((1 - TEACHER_QWEN8B_LAUNCHED))"

while true; do
    print_status
    check_and_launch_teachers
    log "Next check in 1 hour..."
    log "============================================================"
    sleep 3600
done
