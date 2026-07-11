#!/usr/bin/env bash
set -euo pipefail

TEMPLATE_ID="${TEMPLATE_ID:-runpod-torch-v280}"
CMD="${1:?usage: remote_train.sh <command>}"

# Override with GPU_ID=... to force a specific GPU.
# DATA_CENTER_IDS=EU-SE-1 optionally pins placement (helps when stock is regional).
GPU_ID="${GPU_ID:-}"
DATA_CENTER_IDS="${DATA_CENTER_IDS:-}"

# gpu_id|datacenter_ids — tried in order when GPU_ID is unset.
DEFAULT_GPU_ATTEMPTS=(
    "NVIDIA GeForce RTX 4090|"
    "NVIDIA GeForce RTX 5090|EU-RO-1"
    "NVIDIA A40|EU-SE-1"
)

SSH_TIMEOUT_SECS="${SSH_TIMEOUT_SECS:-180}"
SSH_POLL_SECS="${SSH_POLL_SECS:-5}"
REMOTE_DIR="${REMOTE_DIR:-/workspace/ftstudio}"

POD_ID=""

cleanup() {
    if [[ -n "$POD_ID" ]]; then
        echo ">>>> Terminating pod $POD_ID"
        runpodctl pod delete "$POD_ID" || echo "WARN: delete failed - CHECK DASHBOARD"
    fi
}
trap cleanup EXIT INT TERM

create_pod() {
    local gpu="$1"
    local dc_ids="${2:-}"
    local -a args=(
        pod create
        --template-id "$TEMPLATE_ID"
        --gpu-id "$gpu"
    )

    if [[ -n "$dc_ids" ]]; then
        args+=(--data-center-ids "$dc_ids")
    fi

    local output=""
    if ! output=$(runpodctl "${args[@]}" -o json 2>&1); then
        :
    fi

    local pod_id=""
    pod_id=$(printf '%s\n' "$output" | jq -r 'select(type == "object") | .id // empty' | head -n 1)

    if [[ -n "$pod_id" ]]; then
        echo "$pod_id"
        return 0
    fi

    local err=""
    err=$(printf '%s\n' "$output" | jq -r 'select(type == "object") | .error // empty' | head -n 1)
    if [[ -z "$err" ]]; then
        err="unknown create error"
    fi

    echo "WARN: ${gpu}${dc_ids:+ @ ${dc_ids}} unavailable: ${err}" >&2
    return 1
}

wait_for_ssh() {
    local pod_id="$1"
    local deadline=$((SECONDS + SSH_TIMEOUT_SECS))

    while ((SECONDS < deadline)); do
        local ssh_json=""
        ssh_json=$(runpodctl ssh info "$pod_id" -o json 2>/dev/null || true)

        SSH_HOST=$(printf '%s\n' "$ssh_json" | jq -r '.ip // empty')
        SSH_PORT=$(printf '%s\n' "$ssh_json" | jq -r '.port // empty')
        SSH_KEY=$(printf '%s\n' "$ssh_json" | jq -r '.ssh_key.path // empty')

        if [[ -n "$SSH_HOST" && -n "$SSH_PORT" && -n "$SSH_KEY" && -f "$SSH_KEY" ]]; then
            return 0
        fi

        echo ">>> Waiting for SSH (${SECONDS}s / ${SSH_TIMEOUT_SECS}s)..."
        sleep "$SSH_POLL_SECS"
    done

    echo "ERROR: timed out waiting for SSH on pod $pod_id" >&2
    return 1
}

echo ">>>> Creating pod"
if [[ -n "$GPU_ID" ]]; then
    POD_ID=$(create_pod "$GPU_ID" "$DATA_CENTER_IDS")
else
    for attempt in "${DEFAULT_GPU_ATTEMPTS[@]}"; do
        IFS='|' read -r gpu dc_ids <<<"$attempt"
        if POD_ID=$(create_pod "$gpu" "$dc_ids"); then
            echo ">>>> Using GPU: ${gpu}${dc_ids:+ @ ${dc_ids}}"
            break
        fi
    done
fi

if [[ -z "${POD_ID:-}" ]]; then
    echo "ERROR: no GPU capacity found. Try:" >&2
    echo "  runpodctl gpu list" >&2
    echo "  runpodctl datacenter list" >&2
    echo "  GPU_ID='NVIDIA A40' DATA_CENTER_IDS=EU-SE-1 make cuda-check" >&2
    exit 1
fi

echo ">>>> Pod created: $POD_ID"
wait_for_ssh "$POD_ID"

SSH_OPTS=(-i "$SSH_KEY" -p "$SSH_PORT" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null)

echo ">>>> Syncing repo up..."
rsync -avz \
    --exclude '.git' \
    --exclude '.venv' \
    --exclude 'data/raw' \
    --exclude 'llama.cpp' \
    --exclude '__pycache__' \
    --exclude '.pytest_cache' \
    -e "ssh ${SSH_OPTS[*]}" \
    ./ "root@${SSH_HOST}:${REMOTE_DIR}/"

echo ">>>> Running job..."
quoted_remote_dir=$(printf '%q' "$REMOTE_DIR")
quoted_cmd=$(printf '%q' "$CMD")
ssh "${SSH_OPTS[@]}" "root@${SSH_HOST}" \
    "REMOTE_DIR=$quoted_remote_dir CMD=$quoted_cmd bash -s" <<'REMOTE'
set -euo pipefail

cd "$REMOTE_DIR"
python3 -m venv .remote-venv --system-site-packages
.remote-venv/bin/pip install -q transformers

echo ">>>> Remote command: $CMD"
if [[ "$CMD" == python\ * ]]; then
    PYTHONUNBUFFERED=1 .remote-venv/bin/python -u ${CMD#python }
else
    bash -c "$CMD"
fi
REMOTE

if ssh "${SSH_OPTS[@]}" "root@${SSH_HOST}" "test -d ${REMOTE_DIR}/outputs"; then
    echo ">>>> Pulling artifacts home..."
    mkdir -p ./outputs
    rsync -avz -e "ssh ${SSH_OPTS[*]}" \
        "root@${SSH_HOST}:${REMOTE_DIR}/outputs/" ./outputs/
fi

echo ">>>> Done. Pod will be terminated by trap."
