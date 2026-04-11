#!/bin/bash
# GPU Watchdog: polls nvidia-smi, launches pending jobs on idle GPUs
#
# Usage: bash gpu_watchdog.sh <jobs_queue_file>
#
# jobs_queue_file format: one job per line
#   PRIORITY|NAME|CMD_TEMPLATE
# where CMD_TEMPLATE contains {GPU} placeholder for the assigned GPU index.
# Lower PRIORITY numbers run first.
#
# State is tracked in /tmp/watchdog_state/ — delete that dir to reset.
# Logs go to /tmp/watchdog.log

set -uo pipefail

QUEUE="${1:-}"
[[ -z "$QUEUE" || ! -f "$QUEUE" ]] && { echo "Usage: $0 <queue_file>"; exit 1; }

STATE_DIR=/tmp/watchdog_state
LOG=/tmp/watchdog.log
IDLE_MEM_THRESHOLD_MB=1000   # GPU is "free" if used memory < this
POLL_INTERVAL=60             # seconds
LAUNCH_COOLDOWN=30           # wait after launching so memory registers

mkdir -p "$STATE_DIR"

log() { echo "[$(date +'%F %T')] $*" | tee -a "$LOG"; }

get_idle_gpus() {
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | awk -F',' -v t="$IDLE_MEM_THRESHOLD_MB" '$2+0 < t {gsub(/ /,""); print $1}'
}

get_next_pending() {
    # Return the highest-priority (lowest number) job that hasn't been launched yet.
    sort -t'|' -k1,1n "$QUEUE" | while IFS='|' read -r PRI NAME CMD; do
        [[ -z "$NAME" || -z "$CMD" ]] && continue
        [[ -f "$STATE_DIR/$NAME.launched" ]] && continue
        echo "$PRI|$NAME|$CMD"
        return 0
    done | head -1
}

launch_on_gpu() {
    local GPU="$1"
    local JOB
    JOB=$(get_next_pending)
    [[ -z "$JOB" ]] && return 1

    local NAME CMD RESOLVED
    NAME=$(echo "$JOB" | cut -d'|' -f2)
    CMD=$(echo "$JOB"  | cut -d'|' -f3-)
    RESOLVED=${CMD//\{GPU\}/$GPU}

    # Mark launched before sending to avoid double-launch races
    touch "$STATE_DIR/$NAME.launched"
    echo "$GPU" > "$STATE_DIR/$NAME.gpu"

    # Kill any stale window with this name and create fresh
    tmux kill-window -t dia-guard-ft:"$NAME" 2>/dev/null || true
    tmux new-window -t dia-guard-ft -n "$NAME"
    tmux send-keys -t dia-guard-ft:"$NAME" "$RESOLVED" Enter

    log "LAUNCH  GPU=$GPU  $NAME"
    return 0
}

log "Watchdog started. Queue: $QUEUE"

while true; do
    TOTAL=$(grep -cv '^\s*$\|^#' "$QUEUE" 2>/dev/null || echo 0)
    LAUNCHED=$(ls "$STATE_DIR"/*.launched 2>/dev/null | wc -l | tr -d ' ')
    REMAINING=$((TOTAL - LAUNCHED))

    if [[ $REMAINING -le 0 ]]; then
        log "All $TOTAL jobs launched. Watchdog exiting."
        exit 0
    fi

    IDLE=$(get_idle_gpus)
    if [[ -z "$IDLE" ]]; then
        log "POLL    pending=$REMAINING  idle=0  (waiting...)"
    else
        for G in $IDLE; do
            if launch_on_gpu "$G"; then
                sleep "$LAUNCH_COOLDOWN"
            else
                log "Queue empty."
                break
            fi
        done
    fi

    sleep "$POLL_INTERVAL"
done
