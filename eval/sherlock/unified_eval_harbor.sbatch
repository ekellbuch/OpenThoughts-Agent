#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --partition=swl1,gpu,owners
#SBATCH --gres=gpu:1
#SBATCH --constraint="GPU_SKU:H100_SXM5"
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/groups/swl1/ekb/logs/eval_sherlock/eval_%j.out

# ==============================================================================
# Unified Eval Harbor — Sherlock (Stanford Research Computing)
#
# Starts vLLM, runs Harbor eval. Everything else comes from the YAML config:
#   model, agent, environment, datasets, task_names, job_name, concurrency,
#   retries, timeouts, jobs_dir, etc.
#
# Env vars:
#   EVAL_HARBOR_CONFIG      Harbor YAML config path (required)
#   EVAL_VLLM_CONFIG        vLLM serving YAML config path (required)
#   EVAL_HARBOR_EXTRA_ARGS  Extra CLI args for Harbor (optional)
# ==============================================================================

set -eo pipefail
ulimit -c 0
ulimit -n 65536 2>/dev/null || true

HARBOR_CONFIG="${EVAL_HARBOR_CONFIG:?ERROR: EVAL_HARBOR_CONFIG not set}"
VLLM_CONFIG="${EVAL_VLLM_CONFIG:?ERROR: EVAL_VLLM_CONFIG not set}"

# Read model from YAML
MODEL=$(grep 'model_name:' "$HARBOR_CONFIG" 2>/dev/null | head -1 | sed 's/.*model_name: *//' | sed 's|hosted_vllm/||')
if [ -z "$MODEL" ]; then
    echo "ERROR: Could not read model_name from $HARBOR_CONFIG" >&2
    exit 1
fi

echo "=============================================="
echo "Sherlock Eval Harbor"
echo "=============================================="
echo "Model: $MODEL"
echo "Harbor config: $HARBOR_CONFIG"
echo "vLLM config: $VLLM_CONFIG"
echo "=============================================="

# ==============================================================================
# Sherlock Environment Setup
# ==============================================================================

CODING_AGENT_DIR="${GROUP_HOME:-/home/groups/swl1}/ekb/Projects/coding_agent"
HARBOR_DIR="$CODING_AGENT_DIR/modules/harbor"
HARBOR_VENV="$CODING_AGENT_DIR/scripts/harbor/.venv"

unset PYTHONPATH

module load cuda
module load python/3.12
module load py-pyarrow/18.1.0_py312

# Source secrets
DC_AGENT_SECRET_ENV="${DC_AGENT_SECRET_ENV:-$HOME/.env}"
if [ -f "$DC_AGENT_SECRET_ENV" ]; then
    source "$DC_AGENT_SECRET_ENV"
fi

export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-${GROUP_SCRATCH:-/scratch/groups/swl1}/$USER/.cache/singularity}"
export VLLM_USE_V1=1
export VLLM_CONFIG_ROOT="${VLLM_CONFIG_ROOT:-${GROUP_SCRATCH:-/scratch/groups/swl1}/$USER/.cache/vllm}"
export TRITON_CACHE_DIR="/tmp/triton_cache_${USER}"
export FLASHINFER_WORKSPACE_BASE="/tmp/flashinfer_cache_${USER}"
export HYDRA_FULL_ERROR=1
export TORCH_DISABLE_ADDR2LINE=1

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
# ==============================================================================
VLLM_IMAGE="${VLLM_IMAGE:-docker://vllm/vllm-openai:latest}"
VLLM_PORT=8000

# Parse vLLM config from YAML
PARSE_SCRIPT="$CODING_AGENT_DIR/scripts/oa/eval/parse_vllm_config.py"
eval "$($PYTHON_BIN "$PARSE_SCRIPT" "$VLLM_CONFIG")"

VLLM_TP="${EVAL_VLLM_TENSOR_PARALLEL_SIZE:-1}"
VLLM_MAX_LEN="${EVAL_VLLM_MAX_MODEL_LEN:-}"
VLLM_TCP="${EVAL_VLLM_TOOL_CALL_PARSER:-}"
GPU_MEMORY_UTIL="${EVAL_GPU_MEMORY_UTIL:-0.95}"

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

VLLM_LOG="/scratch/groups/swl1/ekb/logs/eval_sherlock/vllm_${SLURM_JOB_ID}.log"
singularity exec --nv \
    "$VLLM_IMAGE" \
    python3 -m vllm.entrypoints.openai.api_server $VLLM_ARGS \
    > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!

# Health check: wait up to 30 minutes for vLLM to be ready
VLLM_TIMEOUT=1800
VLLM_START=$SECONDS
while ! curl -s http://localhost:8000/v1/models > /dev/null 2>&1; do
    ELAPSED=$(( SECONDS - VLLM_START ))
    if [ $ELAPSED -ge $VLLM_TIMEOUT ]; then
        echo "ERROR: vLLM server failed to start after ${VLLM_TIMEOUT}s"
        tail -50 "$VLLM_LOG" 2>/dev/null || true
        exit 1
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "ERROR: vLLM process died"
        tail -50 "$VLLM_LOG" 2>/dev/null || true
        exit 1
    fi
    echo "Waiting for vLLM server (${ELAPSED}s/${VLLM_TIMEOUT}s)..."
    $PYTHON_BIN -c "import time; time.sleep(30)"
done
echo "vLLM server is ready (took $(( SECONDS - VLLM_START ))s)"

# ==============================================================================
# Run Harbor Eval via Singularity
# ==============================================================================
set +e

HARBOR_RUNNER_IMAGE="${HARBOR_RUNNER_IMAGE:-${GROUP_SCRATCH:-/scratch/groups/swl1}/$USER/.cache/harbor/harbor_runner.sif}"
HARBOR_SITE_PACKAGES="$HARBOR_VENV/lib/python3.13/site-packages"
HARBOR_SRC="$HARBOR_DIR/src"
HARBOR_EXTRA_ARGS="${EVAL_HARBOR_EXTRA_ARGS:-}"

# Everything comes from the YAML.
HARBOR_ARGS="jobs start --config $HARBOR_CONFIG $HARBOR_EXTRA_ARGS"

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

if [ ${SB_EXIT:-0} -ne 0 ]; then
    echo "Harbor eval exited with non-zero status: ${SB_EXIT}."
    exit ${SB_EXIT}
fi

echo "=============================================="
echo "Eval finished."
echo "=============================================="
