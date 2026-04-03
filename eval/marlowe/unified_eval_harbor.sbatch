#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --account=marlowe-m000120
#SBATCH --partition=preempt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/m000120-pm05/ekb/data/coding_agent/outputs/logs/ot-eval-%x_%j.out
#SBATCH --error=/scratch/m000120-pm05/ekb/data/coding_agent/outputs/logs/ot-eval-%x_%j.out

# ==============================================================================
# Unified Eval Harbor — Marlowe (Stanford DGX H100 SuperPod)
#
# Starts vLLM, runs Harbor eval via Apptainer, checks errors.
# Adapted from eval/sherlock/unified_eval_harbor.sbatch.
#
# Key differences from Sherlock:
#   - vLLM runs in Singularity container (no module, no native install)
#   - Harbor runs natively from scripts/harbor/.venv (not harbor_runner.sif)
#   - No fakeroot needed (Ubuntu 22.04, kernel 5.15)
#   - H100 80GB GPUs (8 per node)
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

# --- Env vars (with defaults) ---
N_CONCURRENT="${EVAL_N_CONCURRENT:-16}"
GPU_MEMORY_UTIL="${EVAL_GPU_MEMORY_UTIL:-0.95}"

# --- Parse args: read from YAML config, positional args override ---
_harbor_cfg="${EVAL_HARBOR_CONFIG:-}"
_yaml_model=$(grep 'model_name:' "$_harbor_cfg" 2>/dev/null | head -1 | sed 's/.*model_name: *//' | sed 's|hosted_vllm/||')
MODEL="${1:-${_yaml_model:-nvidia/Nemotron-Terminal-8B}}"
REPO_ID="${2:-DCAgent2/terminal_bench_2}"
BENCHMARK_ID="${3:-}"
RUN_TAG_ARG="${4:-}"

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

PROJ_DIR="${PROJ_DIR:-$HOME/Projects/tbench_sherlock}"
SCRATCH_DIR="/scratch/m000120-pm05/ekb"
OT_AGENT_DIR="$PROJ_DIR/modules/OpenThoughts-Agent"
HARBOR_CONFIG="${EVAL_HARBOR_CONFIG:-${OT_AGENT_DIR}/eval/marlowe/dcagent_eval_config.yaml}"

unset PYTHONPATH

# Source secrets
DC_AGENT_SECRET_ENV="${DC_AGENT_SECRET_ENV:-$HOME/.env}"
if [ -f "$DC_AGENT_SECRET_ENV" ]; then
    source "$DC_AGENT_SECRET_ENV"
fi
SECRET_ENV="${OT_AGENT_DIR}/eval/marlowe/secret.env"
if [ -f "$SECRET_ENV" ]; then
    source "$SECRET_ENV"
fi

# Apptainer cache (SIF images with tmux pre-baked)
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-${SCRATCH_DIR}/.cache/singularity}"

# vLLM / Ray env vars
export VLLM_USE_V1=1
export VLLM_CONFIG_ROOT="${VLLM_CONFIG_ROOT:-${SCRATCH_DIR}/.cache/vllm}"
export TRITON_CACHE_DIR="/tmp/triton_cache_${USER}"
export FLASHINFER_WORKSPACE_BASE="/tmp/flashinfer_cache_${USER}"
export HF_HOME="${HF_HOME:-${SCRATCH_DIR}/.cache/huggingface}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${SCRATCH_DIR}/.cache/uv}"
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
# Start vLLM Server (via Singularity container)
#
# Marlowe has no native vLLM install. We use the containerized vLLM
# (same approach as scripts/harbor/serve_qwen3_8b.sh).
# ==============================================================================
LOGS_DIR="${SCRATCH_DIR}/data/coding_agent/outputs/logs"
mkdir -p "$LOGS_DIR"

HARBOR_VENV="$PROJ_DIR/scripts/harbor/.venv"
HARBOR_CLI="$HARBOR_VENV/bin/harbor"
PYTHON_BIN="$HARBOR_VENV/bin/python"

VLLM_IMAGE="${VLLM_IMAGE:-docker://vllm/vllm-openai:latest}"
VLLM_PORT=8000

# Build vLLM args
VLLM_TP="${EVAL_VLLM_TENSOR_PARALLEL_SIZE:-1}"
VLLM_MAX_LEN="${EVAL_VLLM_MAX_MODEL_LEN:-}"
VLLM_TCP="${EVAL_VLLM_TOOL_CALL_PARSER:-}"

VLLM_ARGS="--model $MODEL --host 0.0.0.0 --port $VLLM_PORT"
VLLM_ARGS="$VLLM_ARGS --served-model-name $MODEL"
VLLM_ARGS="$VLLM_ARGS --tensor-parallel-size $VLLM_TP"
VLLM_ARGS="$VLLM_ARGS --gpu-memory-utilization $GPU_MEMORY_UTIL"
VLLM_ARGS="$VLLM_ARGS --trust-remote-code"
[ -n "$VLLM_MAX_LEN" ] && VLLM_ARGS="$VLLM_ARGS --max-model-len $VLLM_MAX_LEN"
[ -n "$VLLM_TCP" ] && VLLM_ARGS="$VLLM_ARGS --enable-auto-tool-choice --tool-call-parser $VLLM_TCP"

echo "Starting vLLM server for model: $MODEL"
echo "  Image: $VLLM_IMAGE"
echo "  Args: $VLLM_ARGS"

singularity exec --nv \
    --bind /scratch:/scratch \
    "$VLLM_IMAGE" \
    python3 -m vllm.entrypoints.openai.api_server $VLLM_ARGS \
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
    DATASET_LOCAL_DIR="${SCRATCH_DIR}/data/coding_agent/eval_datasets/${SAFE_REPO}"
    DOWNLOAD_LOG=$(mktemp /tmp/download_XXXXXX.log)
    cd "$OT_AGENT_DIR"
    "$PYTHON_BIN" "eval/marlowe/snapshot_download.py" "$REPO_ID" --local-dir "$DATASET_LOCAL_DIR" > "$DOWNLOAD_LOG" 2>&1
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
EVAL_JOBS_DIR="${EVAL_JOBS_DIR:-${SCRATCH_DIR}/data/coding_agent/outputs/harbor/eval_jobs}"
RUN_DIR="${EVAL_JOBS_DIR}/${RUN_TAG}"
mkdir -p "$RUN_DIR"

echo "Run tag: $RUN_TAG"
echo "Run dir: $RUN_DIR"

# ==============================================================================
# Run Harbor Eval
#
# On Marlowe, harbor is installed in scripts/harbor/.venv — no
# harbor_runner.sif container needed.
#
# The YAML config controls everything: agent, model, environment, kwargs.
# We don't pass --agent/--model on CLI because that replaces the YAML agent
# config entirely, losing model_info, enable_summarize, enable_thinking, etc.
# Only pass what the YAML doesn't know: dataset path (local), job name (per-run), config path.
# ==============================================================================
set +e

HARBOR_EXTRA_ARGS="${EVAL_HARBOR_EXTRA_ARGS:-}"
HARBOR_ARGS="jobs start -p $DATASET_PATH --job-name $RUN_TAG --config $HARBOR_CONFIG $HARBOR_EXTRA_ARGS"

echo "[DEBUG] Harbor args: $HARBOR_ARGS"

"$HARBOR_CLI" $HARBOR_ARGS

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
