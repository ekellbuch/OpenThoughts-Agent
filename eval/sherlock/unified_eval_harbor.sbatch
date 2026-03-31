#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --partition=swl1,gpu,owners
#SBATCH --gres=gpu:1
#SBATCH --constraint="GPU_MEM:80GB"
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/groups/swl1/ekb/logs/eval_sherlock/%x_%j.out

# ==============================================================================
# Unified Eval Harbor — Sherlock (Stanford Research Computing)
#
# Starts vLLM, runs Harbor eval via Apptainer, checks errors, uploads results.
# Adapted from eval/perlmutter/unified_eval_harbor.sbatch.
#
# Key differences from other clusters:
#   - Uses Apptainer (not Docker/Daytona) as environment backend
#   - Harbor runs inside a Singularity container (harbor_runner.sif)
#   - Modules: cuda, python/3.12, py-vllm/0.7.0_py312
#   - Compute nodes have internet access
#
# Positional args:
#   $1 = MODEL (HF model name, e.g. mlfoundations-dev/some_model)
#   $2 = REPO_ID (HF dataset repo, e.g. DCAgent/dev_set_v2)
#   $3 = BENCHMARK_ID (optional, DB benchmark UUID)
#   $4 = RUN_TAG_ARG (optional, override run tag)
#
# Env vars from listener:
#   EVAL_N_CONCURRENT       (default: 16)
#   EVAL_GPU_MEMORY_UTIL    (default: 0.95)
#   EVAL_DAYTONA_THRESHOLD  (default: 3)
#   EVAL_HARBOR_CONFIG      (default: $DCFT/eval/sherlock/dcagent_eval_config.yaml)
#   EVAL_DB_JOB_ID          (optional, pre-created Pending job ID)
# ==============================================================================

set -eo pipefail
ulimit -c 0
ulimit -n 65536 2>/dev/null || true

TIMESTAMP=$(date +'%Y%m%d_%H%M%S')

# --- Parse positional args ---
MODEL="${1:-mlfoundations-dev/claude_3_7_20250219_tbench_traces_sharegptv1}"
REPO_ID="${2:-DCAgent/dev_set_71_tasks}"
BENCHMARK_ID="${3:-}"
RUN_TAG_ARG="${4:-}"

# --- Read env vars from listener (with defaults) ---
N_CONCURRENT="${EVAL_N_CONCURRENT:-16}"
GPU_MEMORY_UTIL="${EVAL_GPU_MEMORY_UTIL:-0.95}"
DAYTONA_ERROR_THRESHOLD="${EVAL_DAYTONA_THRESHOLD:-999999}"
HARBOR_CONFIG="${EVAL_HARBOR_CONFIG:-${DCFT:-$GROUP_HOME/ekb/Projects/coding_agent/modules/OpenThoughts-Agent}/eval/sherlock/dcagent_eval_config.yaml}"
DB_JOB_ID="${EVAL_DB_JOB_ID:-}"

# Strip slashes and special chars for file-safe names
SAFE_MODEL=$(echo "$MODEL" | tr '/:' '_')
if [[ "$REPO_ID" == /* ]]; then
    SAFE_REPO=$(basename "$REPO_ID")
else
    SAFE_REPO=$(echo "$REPO_ID" | tr '/:' '_')
fi

echo "=============================================="
echo "Sherlock Eval Harbor"
echo "=============================================="
echo "Model: $MODEL"
echo "Dataset: $REPO_ID"
echo "Benchmark ID: ${BENCHMARK_ID:-<auto>}"
echo "N concurrent: $N_CONCURRENT"
echo "GPU memory util: $GPU_MEMORY_UTIL"
echo "Harbor config: $HARBOR_CONFIG"
echo "Daytona threshold: $DAYTONA_ERROR_THRESHOLD"
echo "DB Job ID (pending): ${DB_JOB_ID:-<none>}"
echo "=============================================="

# ==============================================================================
# Sherlock Environment Setup
# ==============================================================================

DCFT="${DCFT:-$GROUP_HOME/ekb/Projects/coding_agent/modules/OpenThoughts-Agent}"
CODING_AGENT_DIR="$GROUP_HOME/ekb/Projects/coding_agent"
HARBOR_DIR="$CODING_AGENT_DIR/modules/harbor"
HARBOR_VENV="$CODING_AGENT_DIR/scripts/harbor/.venv"

# Reset PYTHONPATH to avoid inheriting submitter's environment
unset PYTHONPATH

# Load Sherlock modules
# Note: vLLM runs in Singularity container (not system module) for newer model support
module load cuda
module load python/3.12
module load py-pyarrow/18.1.0_py312

# Source secrets
DC_AGENT_SECRET_ENV="${DC_AGENT_SECRET_ENV:-$HOME/.env}"
if [ -f "$DC_AGENT_SECRET_ENV" ]; then
    source "$DC_AGENT_SECRET_ENV"
fi

# Apptainer cache
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-$GROUP_SCRATCH/$USER/.cache/singularity}"

# --- vLLM / Ray env vars ---
export VLLM_USE_V1=1
export VLLM_CONFIG_ROOT="${VLLM_CONFIG_ROOT:-$GROUP_SCRATCH/$USER/.cache/vllm}"
export TRITON_CACHE_DIR="/tmp/triton_cache_${USER}"
export FLASHINFER_WORKSPACE_BASE="/tmp/flashinfer_cache_${USER}"
export HYDRA_FULL_ERROR=1
export TORCH_DISABLE_ADDR2LINE=1

# PYTHONPATH: project root for database.unified_db imports
export PYTHONPATH="${DCFT}:${PYTHONPATH:-}"

PYTHON_BIN=$(which python3)
echo "Python: $($PYTHON_BIN --version)"
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
# The system py-vllm/0.7.0_py312 module has transformers 4.48 which is too old
# for Qwen3 and other recent architectures. Use the containerized vLLM instead.
# ==============================================================================
mkdir -p eval/sherlock/logs

VLLM_IMAGE="${VLLM_IMAGE:-docker://vllm/vllm-openai:latest}"
VLLM_PORT=8000

# Build vLLM args from EVAL_VLLM_* env vars
# Note: --swap-space removed (not supported in vLLM 0.18+)
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
    "$VLLM_IMAGE" \
    python3 -m vllm.entrypoints.openai.api_server $VLLM_ARGS \
    > "/scratch/groups/swl1/ekb/logs/eval_sherlock/vllm_${SLURM_JOB_ID}.log" 2>&1 &
VLLM_PID=$!

# Health check loop
# Health check: wait up to 30 minutes for vLLM to be ready
VLLM_TIMEOUT=1800
VLLM_START=$SECONDS
while ! curl -s http://localhost:8000/v1/models > /dev/null 2>&1; do
    ELAPSED=$(( SECONDS - VLLM_START ))
    if [ $ELAPSED -ge $VLLM_TIMEOUT ]; then
        echo "ERROR: vLLM server failed to start after ${VLLM_TIMEOUT}s"
        tail -50 "/scratch/groups/swl1/ekb/logs/eval_sherlock/vllm_${SLURM_JOB_ID}.log" 2>/dev/null || true
        exit 1
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "ERROR: vLLM process died"
        tail -50 "/scratch/groups/swl1/ekb/logs/eval_sherlock/vllm_${SLURM_JOB_ID}.log" 2>/dev/null || true
        exit 1
    fi
    echo "Waiting for vLLM server (${ELAPSED}s/${VLLM_TIMEOUT}s)..."
    $PYTHON_BIN -c "import time; time.sleep(30)"
done
echo "vLLM server is ready (took $(( SECONDS - VLLM_START ))s)"

# ==============================================================================
# Download dataset
# ==============================================================================
if [[ "$REPO_ID" == /* ]]; then
    echo "Using local dataset path: $REPO_ID"
    DATASET_PATH="$REPO_ID"
    if [ ! -d "$DATASET_PATH" ]; then
        echo "ERROR: Local dataset path does not exist: $DATASET_PATH"
        exit 1
    fi
    TASK_COUNT=$(ls -d "$DATASET_PATH"/*/instruction.md 2>/dev/null | wc -l)
    echo "[DEBUG] Found $TASK_COUNT tasks with instruction.md"
else
    echo "Downloading/locating dataset: $REPO_ID"
    DATASET_LOCAL_DIR="$GROUP_SCRATCH/$USER/eval_datasets/${SAFE_REPO}"
    echo "[DEBUG] DATASET_LOCAL_DIR=$DATASET_LOCAL_DIR"
    DOWNLOAD_LOG=$(mktemp /tmp/download_XXXXXX.log)
    $PYTHON_BIN "$DCFT/eval/sherlock/snapshot_download.py" "$REPO_ID" --local-dir "$DATASET_LOCAL_DIR" > "$DOWNLOAD_LOG" 2>&1
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
echo "[DEBUG] Task count in dataset: $(ls -d "$DATASET_PATH"/*/instruction.md 2>/dev/null | wc -l)"

# ==============================================================================
# Construct run tag and directory
# ==============================================================================
if [ -n "$RUN_TAG_ARG" ]; then
    RUN_TAG="${RUN_TAG_ARG}_${SLURM_JOB_ID}"
else
    RUN_TAG="${SAFE_REPO}_${SAFE_MODEL}_${SLURM_JOB_ID}"
fi
EVAL_JOBS_DIR="${EVAL_JOBS_DIR:-$GROUP_SCRATCH/ekb/data/coding_agent/outputs/harbor/eval_jobs}"
RUN_DIR="${EVAL_JOBS_DIR}/${RUN_TAG}"
mkdir -p "$RUN_DIR"

echo "Run tag: $RUN_TAG"
echo "Run dir: $RUN_DIR"

# ==============================================================================
# Run Harbor Eval via Apptainer
#
# Harbor runs inside harbor_runner.sif with bind-mounts for Apptainer-in-Apptainer.
# ==============================================================================
set +e

HARBOR_RUNNER_IMAGE="${HARBOR_RUNNER_IMAGE:-$GROUP_SCRATCH/$USER/.cache/harbor/harbor_runner.sif}"
HARBOR_SITE_PACKAGES="$HARBOR_VENV/lib/python3.13/site-packages"
HARBOR_SRC="$HARBOR_DIR/src"

HARBOR_EXTRA_ARGS="${EVAL_HARBOR_EXTRA_ARGS:-}"
HARBOR_ARGS="jobs start -p $DATASET_PATH --n-concurrent $N_CONCURRENT --agent terminus-2 --model hosted_vllm/$MODEL --env apptainer --agent-kwarg api_base=http://localhost:8000/v1 --agent-kwarg key=fake_key --n-attempts 3 --job-name $RUN_TAG --config $HARBOR_CONFIG --jobs-dir $EVAL_JOBS_DIR $HARBOR_EXTRA_ARGS"

echo "[DEBUG] Harbor args: $HARBOR_ARGS"

singularity exec \
    --bind /home/groups/swl1:/home/groups/swl1 \
    --bind /home/users:/home/users \
    --bind /scratch:/scratch \
    --bind /usr/bin/apptainer:/usr/bin/apptainer \
    --bind "$CODING_AGENT_DIR/scripts/harbor/etc_apptainer":/etc/apptainer \
    --bind /usr/libexec/apptainer:/usr/libexec/apptainer \
    --bind /var/lib/apptainer:/var/lib/apptainer \
    --env "PYTHONPATH=$HARBOR_SITE_PACKAGES:$HARBOR_SRC" \
    "$HARBOR_RUNNER_IMAGE" \
    bash -c 'unset APPTAINER_BIND SINGULARITY_BIND APPTAINER_CONTAINER SINGULARITY_CONTAINER APPTAINER_NAME SINGULARITY_NAME APPTAINER_COMMAND SINGULARITY_COMMAND APPTAINER_ENVIRONMENT SINGULARITY_ENVIRONMENT; exec /usr/local/bin/python3 -m harbor.cli.main '"$HARBOR_ARGS"

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
    echo "DB_JOB_ID=${DB_JOB_ID:-}"
    echo "BENCHMARK_ID=${BENCHMARK_ID:-}"
    echo "N_CONCURRENT=$N_CONCURRENT"
    echo "GPU_MEMORY_UTIL=$GPU_MEMORY_UTIL"
} > "$RUN_DIR/meta.env"

if [ ${SB_EXIT:-0} -ne 0 ]; then
    echo "Harbor eval exited with non-zero status: ${SB_EXIT}. Skipping upload."
    exit ${SB_EXIT}
fi

# ==============================================================================
# Check results
# ==============================================================================
RESULT_FILE="$RUN_DIR/result.json"

if [ -f "$RESULT_FILE" ]; then
    echo "Results:"
    $PYTHON_BIN -c "
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
