#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --account=marlowe-m000120
#SBATCH --partition=preempt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/m000120-pm05/ekb/sky_harbor/logs/ot-eval-%x_%j.out
#SBATCH --error=/scratch/m000120-pm05/ekb/sky_harbor/logs/ot-eval-%x_%j.out

# ==============================================================================
# Unified Eval Harbor — Marlowe (Stanford DGX H100 SuperPod)
#
# Starts vLLM, runs Harbor eval via Apptainer, checks errors.
# Adapted from eval/sherlock/unified_eval_harbor.sbatch.
#
# Key differences from Sherlock:
#   - No module system for vLLM — uses uv from sky_harbor project
#   - Harbor installed in sky_harbor/.venv (not harbor_runner.sif)
#   - H100 80GB GPUs (8 per node)
#   - Apptainer SIF images pre-pulled with tmux baked in
#   - Compute nodes have internet access
#
# Positional args:
#   $1 = MODEL (HF model name, e.g. nvidia/Nemotron-Terminal-8B)
#   $2 = REPO_ID (HF dataset repo or local path, e.g. DCAgent/dev_set_v2)
#   $3 = BENCHMARK_ID (optional, DB benchmark UUID)
#   $4 = RUN_TAG_ARG (optional, override run tag)
#
# Env vars:
#   EVAL_N_CONCURRENT       (default: 16)
#   EVAL_GPU_MEMORY_UTIL    (default: 0.95)
#   EVAL_VLLM_TENSOR_PARALLEL_SIZE (default: 1)
# ==============================================================================

set -eo pipefail
ulimit -c 0
ulimit -n 65536 2>/dev/null || true

TIMESTAMP=$(date +'%Y%m%d_%H%M%S')

# --- Parse positional args ---
MODEL="${1:-nvidia/Nemotron-Terminal-8B}"
REPO_ID="${2:-/scratch/m000120-pm05/ekb/sky_harbor/terminal-bench-2}"
BENCHMARK_ID="${3:-}"
RUN_TAG_ARG="${4:-}"

# --- Env vars (with defaults) ---
N_CONCURRENT="${EVAL_N_CONCURRENT:-16}"
GPU_MEMORY_UTIL="${EVAL_GPU_MEMORY_UTIL:-0.95}"

# Strip slashes and special chars for file-safe names
SAFE_MODEL=$(echo "$MODEL" | tr '/:' '_')
if [[ "$REPO_ID" == /* ]]; then
    SAFE_REPO=$(basename "$REPO_ID")
else
    SAFE_REPO=$(echo "$REPO_ID" | tr '/:' '_')
fi

echo "=============================================="
echo "Marlowe Eval Harbor"
echo "=============================================="
echo "Model: $MODEL"
echo "Dataset: $REPO_ID"
echo "Benchmark ID: ${BENCHMARK_ID:-<auto>}"
echo "N concurrent: $N_CONCURRENT"
echo "GPU memory util: $GPU_MEMORY_UTIL"
echo "=============================================="

# ==============================================================================
# Marlowe Environment Setup
# ==============================================================================

PROJ_DIR="${PROJ_DIR:-/users/ekb/Projects/sky_harbor}"
SCRATCH_DIR="/scratch/m000120-pm05/ekb"
OT_AGENT_DIR="${PROJ_DIR}/modules/OpenThoughts-Agent"
HARBOR_CONFIG="${EVAL_HARBOR_CONFIG:-${OT_AGENT_DIR}/eval/marlowe/dcagent_eval_config.yaml}"

unset PYTHONPATH

# Source secrets
DC_AGENT_SECRET_ENV="${DC_AGENT_SECRET_ENV:-$HOME/.env}"
if [ -f "$DC_AGENT_SECRET_ENV" ]; then
    source "$DC_AGENT_SECRET_ENV"
fi

# Apptainer cache (SIF images with tmux pre-baked)
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-${SCRATCH_DIR}/.apptainer-cache}"

# vLLM / Ray env vars
export VLLM_USE_V1=1
export VLLM_CONFIG_ROOT="${VLLM_CONFIG_ROOT:-${SCRATCH_DIR}/.cache/vllm}"
export TRITON_CACHE_DIR="/tmp/triton_cache_${USER}"
export FLASHINFER_WORKSPACE_BASE="/tmp/flashinfer_cache_${USER}"
export HF_HOME="${HF_HOME:-${SCRATCH_DIR}/sky_harbor/hf_cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${SCRATCH_DIR}/.uv-cache}"
export PATH="$HOME/.local/bin:$PATH"

echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"

# ==============================================================================
# Cleanup trap
# ==============================================================================
cleanup() {
    echo "Cleaning up..."
    if [ -n "${VLLM_PID:-}" ]; then
        kill "$VLLM_PID" 2>/dev/null || true
    fi
    echo "Cleanup done."
}
trap cleanup EXIT

# ==============================================================================
# Start vLLM Server
# ==============================================================================
LOGS_DIR="${SCRATCH_DIR}/sky_harbor/logs"
mkdir -p "$LOGS_DIR"

# Download model (Marlowe compute nodes have internet)
echo "Downloading/caching model: $MODEL"
cd "$PROJ_DIR"
uv run --no-sync python -c "
from huggingface_hub import snapshot_download
path = snapshot_download('$MODEL')
print(f'Model cached at: {path}')
"
if [ $? -ne 0 ]; then
    echo "ERROR: Model download failed for $MODEL. Exiting."
    exit 1
fi

echo "Starting vLLM server for model: $MODEL"
cd "$OT_AGENT_DIR"
source "eval/build_vllm_cmd.sh"
# Use uv-managed Python from project venv
PYTHON_BIN="$PROJ_DIR/.venv/bin/python"
build_vllm_cmd "$PYTHON_BIN" "$MODEL" "$GPU_MEMORY_UTIL"

env TORCHDYNAMO_DISABLE=1 \
    TMPDIR="/tmp" \
    TRITON_CACHE_DIR="/tmp/triton_cache_${USER}" \
    TORCH_COMPILE_CACHE_DIR="/tmp/torch_cache_${USER}" \
    TORCHINDUCTOR_CACHE_DIR="/tmp/torchinductor_${USER}" \
"${VLLM_CMD[@]}" \
    > "${LOGS_DIR}/ot-vllm_${SLURM_JOB_ID}.log" 2>&1 &
VLLM_PID=$!

# Health check loop
MAX_RETRIES=20
RETRY_INTERVAL=30
for i in $(seq 1 $MAX_RETRIES); do
    if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
        echo "vLLM server is ready"
        break
    fi
    echo "Waiting for vLLM server (attempt $i/$MAX_RETRIES)..."
    sleep $RETRY_INTERVAL
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "ERROR: vLLM server died"
        tail -50 "${LOGS_DIR}/ot-vllm_${SLURM_JOB_ID}.log" 2>/dev/null || true
        exit 1
    fi
    if [ $i -eq $MAX_RETRIES ]; then
        echo "ERROR: vLLM server failed to start"
        tail -50 "${LOGS_DIR}/ot-vllm_${SLURM_JOB_ID}.log" 2>/dev/null || true
        exit 1
    fi
done

# ==============================================================================
# Resolve dataset
# ==============================================================================
if [[ "$REPO_ID" == /* ]]; then
    echo "Using local dataset path: $REPO_ID"
    DATASET_PATH="$REPO_ID"
    if [ ! -d "$DATASET_PATH" ]; then
        echo "ERROR: Local dataset path does not exist: $DATASET_PATH"
        exit 1
    fi
    TASK_COUNT=$(ls -d "$DATASET_PATH"/*/instruction.md 2>/dev/null | wc -l)
    echo "Found $TASK_COUNT tasks with instruction.md"
else
    echo "Downloading/locating dataset: $REPO_ID"
    DATASET_LOCAL_DIR="${SCRATCH_DIR}/sky_harbor/eval_datasets/${SAFE_REPO}"
    DOWNLOAD_LOG=$(mktemp /tmp/download_XXXXXX.log)
    cd "$OT_AGENT_DIR"
    "$PYTHON_BIN" "eval/sherlock/snapshot_download.py" "$REPO_ID" --local-dir "$DATASET_LOCAL_DIR" > "$DOWNLOAD_LOG" 2>&1
    DOWNLOAD_EXIT=$?
    cat "$DOWNLOAD_LOG"
    DATASET_PATH=$(grep DATASET_PATH "$DOWNLOAD_LOG" | tail -n 1 | cut -d'=' -f2)
    rm -f "$DOWNLOAD_LOG"
    if [ $DOWNLOAD_EXIT -ne 0 ] || [ -z "${DATASET_PATH:-}" ]; then
        echo "ERROR: Failed to get dataset path (exit code: $DOWNLOAD_EXIT)"
        exit 1
    fi
fi
echo "Using dataset path: $DATASET_PATH"

# ==============================================================================
# Construct run tag and directory
# ==============================================================================
if [ -n "$RUN_TAG_ARG" ]; then
    RUN_TAG="$RUN_TAG_ARG"
else
    RUN_TAG="${SAFE_REPO}_${SAFE_MODEL}"
fi
EVAL_JOBS_DIR="${EVAL_JOBS_DIR:-${SCRATCH_DIR}/sky_harbor/ot_eval_jobs}"
RUN_DIR="${EVAL_JOBS_DIR}/${RUN_TAG}"
mkdir -p "$RUN_DIR"

echo "Run tag: $RUN_TAG"
echo "Run dir: $RUN_DIR"

# ==============================================================================
# Run Harbor Eval
#
# On Marlowe, harbor is installed in the sky_harbor project venv — no
# harbor_runner.sif container needed. Harbor spawns Apptainer sandboxes
# for each task directly.
# ==============================================================================
set +e

cd "$PROJ_DIR"
uv run --no-sync harbor jobs start \
    -p "$DATASET_PATH" \
    --n-concurrent "$N_CONCURRENT" \
    --agent terminus-2 \
    --model "hosted_vllm/$MODEL" \
    --env apptainer \
    --agent-kwarg "api_base=http://localhost:8000/v1" \
    --agent-kwarg "key=fake_key" \
    --n-attempts 3 \
    --job-name "$RUN_TAG" \
    --config "$HARBOR_CONFIG" \
    --jobs-dir "$EVAL_JOBS_DIR"

SB_EXIT=$?
set -e

# ==============================================================================
# Save meta.env
# ==============================================================================
mkdir -p "$RUN_DIR"
{
    echo "MODEL=$MODEL"
    echo "REPO_ID=$REPO_ID"
    echo "TIMESTAMP=$TIMESTAMP"
    echo "SLURM_JOB_ID=$SLURM_JOB_ID"
    echo "BENCHMARK_ID=${BENCHMARK_ID:-}"
    echo "N_CONCURRENT=$N_CONCURRENT"
    echo "GPU_MEMORY_UTIL=$GPU_MEMORY_UTIL"
} > "$RUN_DIR/meta.env"

if [ ${SB_EXIT:-0} -ne 0 ]; then
    echo "Harbor eval exited with non-zero status: ${SB_EXIT}."
    exit ${SB_EXIT}
fi

# ==============================================================================
# Check results
# ==============================================================================
RESULT_FILE="$RUN_DIR/result.json"

if [ -f "$RESULT_FILE" ]; then
    echo "Results:"
    "$PYTHON_BIN" -c "
import json
with open('$RESULT_FILE') as f:
    data = json.load(f)
stats = data.get('stats', {})
print(f'  Mean reward: {stats.get(\"mean_reward\", \"N/A\")}')
print(f'  Total trials: {stats.get(\"total_trials\", \"N/A\")}')
print(f'  Errors: {stats.get(\"total_errors\", \"N/A\")}')
" 2>/dev/null || echo "  (could not parse result.json)"
else
    echo "Warning: result.json not found at $RESULT_FILE"
fi

echo "=============================================="
echo "Eval finished. Results: $RUN_DIR"
echo "=============================================="
