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

    # Header
    printf "%-25s | %-9s | %-9s | %-15s | %-9s | %-7s | %-7s | %-9s | %-10s\n" \
        "Job" "VRAM%" "GPU Util" "Step" "Epoch/3" "Loss" "Acc" "EvalLoss" "ETA" | tee -a "$LOG_FILE"
    printf "%s\n" "$(printf '%.0s-' {1..120})" | tee -a "$LOG_FILE"

    mapfile -t windows < <(tmux list-windows -t dia-guard-ft -F '#{window_name}' 2>/dev/null)

    for w in "${windows[@]}"; do
        [[ "$w" == "control" || "$w" == "monitor" ]] && continue

        # Live tmux capture (latest progress bar / DONE)
        out_short=$(tmux capture-pane -t "dia-guard-ft:$w" -p -S -10 2>/dev/null)

        # Skip dead windows
        done_check=$(echo "$out_short" | grep "DONE" | tail -1)
        if [[ -n "$done_check" ]]; then
            log_path=$(ls /data/vibe_exp/dia-guard/logs/${w}*.log 2>/dev/null | tail -1)
            err_check=""
            [[ -f "$log_path" ]] && err_check=$(grep -E "Error|OOM|FAILED" "$log_path" 2>/dev/null | tail -1)
            if [[ -n "$err_check" ]]; then
                printf "%-25s | %-95s\n" "${w:0:25}" "FAILED: ${err_check:0:90}" | tee -a "$LOG_FILE"
            else
                printf "%-25s | %-95s\n" "${w:0:25}" "COMPLETED" | tee -a "$LOG_FILE"
            fi
            continue
        fi

        # Find log file for this window (full history of training output)
        log_path=$(ls /data/vibe_exp/dia-guard/logs/${w}*.log 2>/dev/null | tail -1)

        # Find GPU from launch banner ("GPUs     : 0,1") or CUDA_VISIBLE_DEVICES
        gpu_id="?"
        if [[ -f "$log_path" ]]; then
            gpu_id=$(grep -oP "^\s+GPUs\s+:\s+\K[0-9,]+" "$log_path" 2>/dev/null | tail -1)
            [[ -z "$gpu_id" ]] && gpu_id=$(grep -oP "CUDA_VISIBLE_DEVICES=\K[0-9,]+" "$log_path" 2>/dev/null | tail -1)
            [[ -z "$gpu_id" ]] && gpu_id="?"
        fi
        first_gpu=$(echo "$gpu_id" | cut -d',' -f1)

        # GPU stats
        if [[ "$first_gpu" =~ ^[0-9]+$ ]]; then
            gpu_stat=$(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits -i "$first_gpu" 2>/dev/null)
            mem_used=$(echo "$gpu_stat" | awk -F',' '{gsub(/ /,""); print $1}')
            mem_total=$(echo "$gpu_stat" | awk -F',' '{gsub(/ /,""); print $2}')
            gpu_util=$(echo "$gpu_stat" | awk -F',' '{gsub(/ /,""); print $3}')
            vram_pct=$((mem_used * 100 / mem_total))
        else
            vram_pct="-"; gpu_util="-"
        fi

        # Step info — only count real training steps (phase 2)
        step_info=""
        if [[ -f "$log_path" ]]; then
            step_info=$(grep -oP "\d+/156930|\d+/209241|\d+/139493|\d+/313860" "$log_path" 2>/dev/null | tail -1)
        fi
        [[ -z "$step_info" ]] && step_info=$(echo "$out_short" | grep -oP "\d+/\d+\s*\[" | tail -1 | tr -d ' [')

        # Epoch — directly from SFTTrainer log line ('epoch': '0.6576')
        epoch_val=""
        if [[ -f "$log_path" ]]; then
            epoch_val=$(grep -oP "'epoch': '[0-9.]+'" "$log_path" 2>/dev/null | tail -1 | grep -oP "[0-9.]+" | head -1)
        fi
        if [[ -n "$epoch_val" ]]; then
            epoch_str=$(printf "%.2f/3" "$epoch_val")
        else
            # Contrastive script: parse "Epoch N: step/per_epoch"
            ep_line=$(echo "$out_short" | grep -oP "Epoch \d+:\s+\d+/\d+" | tail -1)
            if [[ -n "$ep_line" ]]; then
                ep_num=$(echo "$ep_line" | grep -oP "Epoch \K\d+")
                ep_step=$(echo "$ep_line" | grep -oP "\d+/\d+" | head -1 | cut -d'/' -f1)
                ep_total=$(echo "$ep_line" | grep -oP "\d+/\d+" | head -1 | cut -d'/' -f2)
                if [[ -n "$ep_num" && -n "$ep_step" && -n "$ep_total" && "$ep_total" -gt 0 ]]; then
                    frac=$(awk "BEGIN {printf \"%.2f\", ($ep_num - 1) + $ep_step / $ep_total}")
                    epoch_str="${frac}/3"
                else
                    epoch_str="-"
                fi
            else
                epoch_str="-"
            fi
        fi

        # Loss / acc / eval loss — parsed from log file with strict patterns
        loss_val=""; acc_val=""; evloss_val=""
        if [[ -f "$log_path" ]]; then
            # Training loss only — exclude eval_loss by requiring leading {' or comma
            loss_val=$(grep -oP "(?<=^|[{,] )'loss': '[0-9.]+'" "$log_path" 2>/dev/null | tail -1 | grep -oP "[0-9.]+(?=')" | head -1)
            [[ -z "$loss_val" ]] && loss_val=$(grep -oP "'loss': '[0-9.]+'" "$log_path" 2>/dev/null | tail -1 | grep -oP "[0-9.]+(?=')" | head -1)
            acc_val=$(grep -oP "'mean_token_accuracy': '[0-9.]+'" "$log_path" 2>/dev/null | tail -1 | grep -oP "[0-9.]+(?=')" | head -1)
            evloss_val=$(grep -oP "'eval_loss': '[0-9.]+'" "$log_path" 2>/dev/null | tail -1 | grep -oP "[0-9.]+(?=')" | head -1)
        fi
        # Contrastive uses ce= field
        [[ -z "$loss_val" ]] && loss_val=$(echo "$out_short" | grep -oP "ce=[0-9.]+" | tail -1 | grep -oP "[0-9.]+")

        # ETA — from current step, total, and speed
        eta_str="-"
        if [[ -n "$step_info" ]]; then
            cur=$(echo "$step_info" | cut -d'/' -f1)
            tot=$(echo "$step_info" | cut -d'/' -f2)
            speed=$(echo "$out_short" | grep -oP "[0-9.]+it/s|[0-9.]+s/it" | tail -1)
            if [[ -n "$speed" && "$cur" =~ ^[0-9]+$ && "$tot" =~ ^[0-9]+$ ]]; then
                remaining=$((tot - cur))
                if [[ "$speed" == *"it/s"* ]]; then
                    its=$(echo "$speed" | grep -oP "[0-9.]+")
                    eta_sec=$(awk "BEGIN {printf \"%.0f\", $remaining / $its}")
                else
                    sit=$(echo "$speed" | grep -oP "[0-9.]+")
                    eta_sec=$(awk "BEGIN {printf \"%.0f\", $remaining * $sit}")
                fi
                eta_h=$((eta_sec / 3600))
                eta_m=$(((eta_sec % 3600) / 60))
                eta_str="${eta_h}h${eta_m}m"
            fi
        fi

        printf "%-25s | %-7s%% | %-7s%% | %-15s | %-9s | %-7s | %-7s | %-9s | %-10s\n" \
            "${w:0:25}" "${vram_pct:--}" "${gpu_util:--}" "${step_info:--}" "${epoch_str:--}" "${loss_val:--}" "${acc_val:--}" "${evloss_val:--}" "${eta_str:--}" | tee -a "$LOG_FILE"
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
    # Teacher auto-launch disabled by user request
    # check_and_launch_teachers
    log "Next check in 1 hour..."
    log "============================================================"
    sleep 3600
done
