#!/bin/bash
#SBATCH -p boost_usr_prod
#SBATCH --time=24:00:00
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --account AIFAC_5C0_290
#SBATCH --output=eval/leonardo/logs/%x_%j.out
#SBATCH --job-name=eval

# ==============================================================================
# Unified Eval Harbor — Leonardo Cluster (CINECA A100-64GB)
#
# Starts vLLM, runs Harbor eval, checks errors, uploads results.
# Adapted from eval/jupiter/unified_eval_harbor.sbatch for Leonardo x86_64.
#
# Positional args:
#   $1 = MODEL (HF model name, e.g. mlfoundations-dev/some_model)
#   $2 = REPO_ID (HF dataset repo, e.g. DCAgent/dev_set_v2)
#   $3 = BENCHMARK_ID (optional, DB benchmark UUID)
#   $4 = RUN_TAG_ARG (optional, override run tag)
#
# Env vars from listener:
#   EVAL_N_CONCURRENT       (default: 64)
#   EVAL_GPU_MEMORY_UTIL    (default: 0.90)
#   EVAL_DAYTONA_THRESHOLD  (default: 3)
#   EVAL_HARBOR_CONFIG      (default: $DCFT/eval/leonardo/dcagent_eval_config.yaml)
#   EVAL_DB_JOB_ID          (optional, pre-created Pending job ID)
# ==============================================================================

set -eo pipefail
ulimit -c 0  # Disable core dumps
ulimit -n 65536 2>/dev/null || true

TIMESTAMP=$(date +'%Y%m%d_%H%M%S')

# --- Parse positional args ---
MODEL="${1:-mlfoundations-dev/claude_3_7_20250219_tbench_traces_sharegptv1}"
REPO_ID="${2:-DCAgent2/terminal_bench_2}"
BENCHMARK_ID="${3:-}"
RUN_TAG_ARG="${4:-}"

# --- Read env vars from listener (with defaults) ---
N_CONCURRENT="${EVAL_N_CONCURRENT:-64}"
GPU_MEMORY_UTIL="${EVAL_GPU_MEMORY_UTIL:-0.90}"
DAYTONA_ERROR_THRESHOLD="${EVAL_DAYTONA_THRESHOLD:-999999}"
HARBOR_CONFIG="${EVAL_HARBOR_CONFIG:-$DCFT/eval/leonardo/dcagent_eval_config.yaml}"
DB_JOB_ID="${EVAL_DB_JOB_ID:-}"

# Strip slashes and special chars for file-safe names
SAFE_MODEL=$(echo "$MODEL" | tr '/:' '_')
if [[ "$REPO_ID" == /* ]]; then
    SAFE_REPO=$(basename "$REPO_ID")
else
    SAFE_REPO=$(echo "$REPO_ID" | tr '/:' '_')
fi

echo "=============================================="
echo "Leonardo Eval Harbor"
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
# Leonardo Environment Setup
# ==============================================================================

# Hardcode WORK — Leonardo sets WORK=/leonardo_work/AIFAC_5C0_290 system-wide
# which is the project dir, not the user dir. Override unconditionally.
WORK="/leonardo_work/EUHPC_E03_068/bfeuer00"
DCFT="${WORK}/code/OpenThoughts-Agent"

# Reset PYTHONPATH
unset PYTHONPATH

# Conda activation (x86_64 — no LD_LOADER wrapper needed)
source "${WORK}/miniforge3/etc/profile.d/conda.sh"
conda activate otagent

# All build deps (CUDA, libstdc++, etc.) managed via conda — no module loads.
# Prepend conda's lib to ensure its libstdc++ wins over any system/module versions.
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Source Leonardo environment
source "$DCFT/hpc/dotenv/leonardo.env"

# Source secrets
DC_AGENT_SECRET_ENV="${DC_AGENT_SECRET_ENV:-$HOME/secrets.env}"
if [ -f "$DC_AGENT_SECRET_ENV" ]; then
    source "$DC_AGENT_SECRET_ENV"
fi

# --- Daytona API key for evals (3:1 weighted: 3/4 org2, 1/4 org1) ---
DAYTONA_KEY_ORG1="dtn_17868a1955b56a52cb367af6dd3c6e93ee531b2073df801784273435c0e0fc6c"
DAYTONA_KEY_ORG2="dtn_ecfb759d2abcbf0f044413182677b3cc7624af495cb48409e8873b5283bda313"
if (( RANDOM % 4 == 0 )); then
    export DAYTONA_API_KEY="$DAYTONA_KEY_ORG1"
    echo "Selected Daytona org: org1 (${DAYTONA_API_KEY:0:12}...)"
else
    export DAYTONA_API_KEY="$DAYTONA_KEY_ORG2"
    echo "Selected Daytona org: org2 (${DAYTONA_API_KEY:0:12}...)"
fi

# --- vLLM / Triton / torch cache dirs (home quota is small; use WORK) ---
export VLLM_CACHE_ROOT="${WORK}/vllm_cache"
export VLLM_CONFIG_ROOT="${WORK}/vllm_cache"
export TRITON_CACHE_DIR="/tmp/triton_cache_${USER}_${SLURM_JOB_ID}"
export FLASHINFER_WORKSPACE_BASE="/tmp/flashinfer_cache_${USER}_${SLURM_JOB_ID}"
export UV_CACHE_DIR="${SCRATCH_FAST}/uv_cache_dir"
export TORCH_EXTENSIONS_DIR="${WORK}/torch_extensions"
export TORCH_HOME="${WORK}/torch_home"
export XDG_CACHE_HOME="${WORK}/.cache"
export TORCHINDUCTOR_CACHE_DIR="${WORK}/.cache/torchinductor"
export HOME="${WORK}"
export HYDRA_FULL_ERROR=1
export HF_HUB_OFFLINE=0  # Login nodes pre-downloaded; set to 1 after download

# PYTHONPATH: harbor src + project root
HARBOR_SRC="${WORK}/code/harbor/src"
export PYTHONPATH="${HARBOR_SRC}:${DCFT}:${PYTHONPATH:-}"

PYTHON_BIN="$(which python)"
echo "Python: $($PYTHON_BIN --version)"
echo "PYTHONPATH: $PYTHONPATH"

# ==============================================================================
# Proxychains Setup (no internet on compute nodes)
# Leonardo blocks SSH tunnel auth (pubkey disabled for intra-cluster).
# Instead, a standalone SOCKS5 proxy runs on the login node (in tmux).
# Start it before submitting jobs:
#   python3 eval/leonardo/socks5_proxy.py --port 17003
# ==============================================================================

# Resolve login node IP at submission time (proxychains needs numeric IP)
PROXY_IP="${PROXY_IP:-10.99.0.5}"
PROXY_PORT="${PROXY_PORT:-17003}"
PROXYCHAINS_BIN="$(which proxychains4 2>/dev/null || echo "${WORK}/miniforge3/envs/otagent/bin/proxychains4")"

setup_proxy() {
    echo "[proxy] Configuring proxychains to use SOCKS5 proxy at ${PROXY_IP}:${PROXY_PORT}..."

    # Verify connectivity to the proxy
    if python3 -c "
import socket, sys
try:
    s = socket.create_connection(('${PROXY_IP}', ${PROXY_PORT}), timeout=10)
    s.close()
    print('[proxy] Login node proxy is reachable')
except Exception as e:
    print(f'[proxy] ERROR: Cannot reach proxy at ${PROXY_IP}:${PROXY_PORT}: {e}')
    sys.exit(1)
"; then
        echo "[proxy] Proxy connectivity confirmed"
    else
        echo "[proxy] WARNING: Proxy not reachable. External connectivity will fail."
        echo "[proxy] Ensure socks5_proxy.py is running on login node port ${PROXY_PORT}"
        return 1
    fi

    # Generate proxychains config (proxy_dns required for domain resolution via proxy)
    CFG_PATH="$HOME/.proxychains/proxychains_${SLURM_JOB_ID}.conf"
    mkdir -p "$HOME/.proxychains"

    cat > "$CFG_PATH" <<PCEOF
strict_chain
proxy_dns
quiet_mode
tcp_read_time_out 30000
tcp_connect_time_out 15000
localnet 127.0.0.0/255.0.0.0
[ProxyList]
socks5 ${PROXY_IP} ${PROXY_PORT}
PCEOF

    export PROXYCHAINS_CONF_FILE="$CFG_PATH"
    export PROXYCHAINS_SOCKS5_HOST="${PROXY_IP}"
    export PROXYCHAINS_SOCKS5_PORT="${PROXY_PORT}"

    echo "[proxy] Config at $CFG_PATH (socks5://${PROXY_IP}:${PROXY_PORT})"

    # Daytona timeout/retry settings
    export DAYTONA_MAX_RETRIES=5
    export DAYTONA_RETRY_DELAY=30
    export DAYTONA_BACKOFF_FACTOR=2
    export DAYTONA_TIMEOUT=1800
    export AIOHTTP_CLIENT_TIMEOUT=900
    export AIOHTTP_CONNECTOR_TIMEOUT=900
    export AIOHTTP_SOCK_CONNECT_TIMEOUT=300
    export AIOHTTP_TOTAL_TIMEOUT=1800

    # Disable SSL verification (cert chain issues through tunnel)
    export PYTHONHTTPSVERIFY=0
    unset SSL_CERT_FILE
    unset CURL_CA_BUNDLE
    unset REQUESTS_CA_BUNDLE
    unset SSL_CERT_DIR

    # Test connectivity
    if [ -x "$PROXYCHAINS_BIN" ]; then
        if "$PROXYCHAINS_BIN" -f "$CFG_PATH" curl -s --connect-timeout 10 https://huggingface.co -o /dev/null 2>/dev/null; then
            echo "[proxy] Connectivity test passed"
        else
            echo "[proxy] WARNING: Connectivity test failed (may still work)"
        fi
    fi

    echo "[proxy] Setup complete"
    return 0
}

# Helper: run command through proxychains
proxied() {
    if [ -x "$PROXYCHAINS_BIN" ] && [ -n "${PROXYCHAINS_CONF_FILE:-}" ]; then
        "$PROXYCHAINS_BIN" -f "$PROXYCHAINS_CONF_FILE" "$@"
    else
        "$@"
    fi
}

# Setup proxy
setup_proxy || echo "[proxy] Continuing without proxy (some operations may fail)"

# ==============================================================================
# Cleanup trap
# ==============================================================================
cleanup() {
    echo "Cleaning up..."
    if [ -n "${VLLM_PID:-}" ]; then
        kill "$VLLM_PID" 2>/dev/null || true
    fi
    rm -f "${PROXYCHAINS_CONF_FILE:-}" 2>/dev/null || true
    echo "Cleanup done."
}
trap cleanup EXIT

# ==============================================================================
# Start vLLM Server (single node, TP=4 for 4 A100-64GB GPUs)
# ==============================================================================
mkdir -p eval/leonardo/logs

# Pre-download model via proxychains (compute nodes have no direct internet)
echo "Pre-downloading model: $MODEL"
proxied $PYTHON_BIN -c "
from huggingface_hub import snapshot_download
import os
cache = os.environ.get('HF_HUB_CACHE')
path = snapshot_download('$MODEL', cache_dir=cache)
print(f'Model cached at: {path}')
"
if [ $? -ne 0 ]; then
    echo "ERROR: Model pre-download failed for $MODEL. Exiting."
    exit 1
fi

export HF_HUB_OFFLINE=1  # Prevent further network access during vLLM startup

echo "Starting vLLM server for model: $MODEL"
source "$DCFT/eval/build_vllm_cmd.sh"
build_vllm_cmd "$PYTHON_BIN" "$MODEL" "$GPU_MEMORY_UTIL"

srun --overlap --gres=gpu:4 "${VLLM_CMD[@]}" \
    > "eval/leonardo/logs/vllm_${SLURM_JOB_ID}.log" 2>&1 &
VLLM_PID=$!

# Health check loop
MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -n1)
VLLM_HEALTH_URL="http://${MASTER_ADDR}:8000/v1/models"
echo "Health check URL: $VLLM_HEALTH_URL"
MAX_RETRIES=30
RETRY_INTERVAL=100
for i in $(seq 1 $MAX_RETRIES); do
    if env -u LD_PRELOAD curl -s "$VLLM_HEALTH_URL" > /dev/null 2>&1; then
        echo "vLLM server is ready"
        break
    fi
    echo "Waiting for vLLM server (attempt $i/$MAX_RETRIES)..."
    sleep $RETRY_INTERVAL
    if [ $i -eq $MAX_RETRIES ]; then
        echo "ERROR: vLLM server failed to start"
        tail -50 "eval/leonardo/logs/vllm_${SLURM_JOB_ID}.log" 2>/dev/null || true
        exit 1
    fi
done

# ==============================================================================
# Download dataset (via proxychains for HF access)
# ==============================================================================
EVAL_JOBS_DIR="${EVAL_JOBS_DIR:-${WORK}/eval_jobs}"

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
    DATASET_LOCAL_DIR="${WORK}/data/datasets/${SAFE_REPO}"
    echo "[DEBUG] DATASET_LOCAL_DIR=$DATASET_LOCAL_DIR"
    if [ -d "$DATASET_LOCAL_DIR" ]; then
        TASK_COUNT=$(ls -d "$DATASET_LOCAL_DIR"/*/instruction.md 2>/dev/null | wc -l)
        echo "[DEBUG] Found $TASK_COUNT tasks with instruction.md in local dir"
    fi
    DOWNLOAD_LOG=$(mktemp /tmp/download_XXXXXX.log)
    proxied $PYTHON_BIN "$DCFT/eval/jupiter/snapshot_download.py" "$REPO_ID" --local-dir "$DATASET_LOCAL_DIR" > "$DOWNLOAD_LOG" 2>&1
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
RUN_DIR="${EVAL_JOBS_DIR}/${RUN_TAG}"
mkdir -p "$RUN_DIR"

echo "Run tag: $RUN_TAG"
echo "Run dir: $RUN_DIR"

# --- Compute canonical benchmark name for DB uploads ---
declare -A BENCHMARK_NAME_MAP=(
    ["DCAgent2_aider_polyglot"]="aider_polyglot"
    ["DCAgent_dev_set_v2"]="dev_set_v2"
    ["DCAgent_dev_set_71_tasks"]="dev_set_71_tasks"
    ["DCAgent2_terminal_bench_2"]="terminal_bench_2"
    ["DCAgent_swebench_verified_eval_set"]="swebench-verified-random-100-folders"
    ["DCAgent2_bfcl-parity"]="bfcl-parity"
)
BASE_NAME="${BENCHMARK_NAME_MAP[$SAFE_REPO]:-$SAFE_REPO}"

BENCHMARK_SUFFIX=""
if [ -n "${EVAL_OVERRIDE_MEMORY_MB:-}" ] && [ "$EVAL_OVERRIDE_MEMORY_MB" != "1024" ]; then
    mem_gb=$(( EVAL_OVERRIDE_MEMORY_MB / 1024 ))
    BENCHMARK_SUFFIX="${BENCHMARK_SUFFIX}_${mem_gb}gb"
fi
if [ -n "${EVAL_TIMEOUT_MULTIPLIER:-}" ] && [ "$EVAL_TIMEOUT_MULTIPLIER" != "1" ] && [ "$EVAL_TIMEOUT_MULTIPLIER" != "1.0" ]; then
    BENCHMARK_SUFFIX="${BENCHMARK_SUFFIX}_${EVAL_TIMEOUT_MULTIPLIER}x"
fi
if [ -n "$BENCHMARK_SUFFIX" ]; then
    BENCHMARK_NAME="${BASE_NAME}${BENCHMARK_SUFFIX}"
else
    BENCHMARK_NAME="$BASE_NAME"
fi
export BENCHMARK_NAME
echo "Benchmark name: $BENCHMARK_NAME"

# ==============================================================================
# Update DB: Started (using database/unified_db)
# ==============================================================================
echo "Creating/updating DB job entry..."
export MODEL REPO_ID RUN_TAG SLURM_JOB_ID BENCHMARK_NAME

HARBOR_VERSION=$($PYTHON_BIN -c "import harbor; print(harbor.__version__)" 2>/dev/null || echo "unknown")
export HARBOR_VERSION

proxied $PYTHON_BIN - <<'PY' || true
import os, sys

from database.unified_db.utils import create_job_entry_started

model_hf = os.environ["MODEL"]
dataset_hf = os.environ["REPO_ID"]
run_tag = os.environ["RUN_TAG"]
slurm_job_id = os.environ["SLURM_JOB_ID"]
harbor_version = os.environ.get("HARBOR_VERSION", "unknown")

result = create_job_entry_started(
    model_hf_name=model_hf,
    benchmark_hf_name=dataset_hf,
    job_name=run_tag,
    username=os.environ.get("USER", "leonardo"),
    slurm_job_id=slurm_job_id,
    harbor_package_version=harbor_version,
    agent_name="terminus-2",
    config={"agent": "terminus-2", "env": "daytona"},
    n_trials=128,
    n_rep_eval=3
)

if not result.get("success"):
    print(f"WARNING: DB create failed: {result.get('error')}", file=sys.stderr)
    sys.exit(0)

db_job_id = result["job"]["id"]
print(f"DB job created with ID: {db_job_id}")
PY

# Get the DB job ID
DB_JOB_ID=$(proxied $PYTHON_BIN - <<'PY' || true
import os, sys

from database.unified_db.utils import get_latest_job_for_model_benchmark

model_hf = os.environ["MODEL"]
dataset_hf = os.environ["REPO_ID"]

try:
    result = get_latest_job_for_model_benchmark(model_hf, dataset_hf)
    if result and result.get("id"):
        print(result["id"])
except Exception as e:
    print(f"WARNING: DB lookup failed: {e}", file=sys.stderr)
PY
)

if [ -z "${DB_JOB_ID:-}" ]; then
    echo "WARNING: Failed to get DB job ID"
else
    echo "DB job entry: $DB_JOB_ID"
fi

# ==============================================================================
# Run Harbor Eval (via proxychains for Daytona access)
# ==============================================================================
set +e

EXTRA_HARBOR_ARGS=""
if [ -n "${EVAL_SNAPSHOT_NAME:-}" ]; then
    echo "Using Daytona snapshot: $EVAL_SNAPSHOT_NAME"
    EXTRA_HARBOR_ARGS="--environment-kwarg snapshot_template_name=$EVAL_SNAPSHOT_NAME --no-force-build"
fi
if [ -n "${EVAL_TIMEOUT_MULTIPLIER:-}" ]; then
    echo "Using timeout multiplier: $EVAL_TIMEOUT_MULTIPLIER"
    EXTRA_HARBOR_ARGS="$EXTRA_HARBOR_ARGS --timeout-multiplier $EVAL_TIMEOUT_MULTIPLIER"
fi
if [ -n "${EVAL_OVERRIDE_MEMORY_MB:-}" ]; then
    echo "Using memory override: ${EVAL_OVERRIDE_MEMORY_MB}MB"
    EXTRA_HARBOR_ARGS="$EXTRA_HARBOR_ARGS --override-memory-mb $EVAL_OVERRIDE_MEMORY_MB"
fi

echo "[DEBUG] EXTRA_HARBOR_ARGS=$EXTRA_HARBOR_ARGS"
echo "[DEBUG] Full harbor command:"
echo "  harbor jobs start -p $DATASET_PATH --n-concurrent $N_CONCURRENT --agent terminus-2 --model hosted_vllm/$MODEL --env daytona ... $EXTRA_HARBOR_ARGS"

# Check if a previous job dir exists for this run tag (for resume)
EXISTING_JOB_DIR="${EVAL_JOBS_DIR}/${RUN_TAG}"
if [ -d "$EXISTING_JOB_DIR" ] && [ -f "$EXISTING_JOB_DIR/config.json" ]; then
    echo "Found existing job dir, resuming: $EXISTING_JOB_DIR"
    proxied harbor jobs resume \
        -p "$EXISTING_JOB_DIR" \
        --filter-error-type EnvironmentStartTimeoutError \
        --filter-error-type DaytonaError \
        --filter-error-type DaytonaRateLimitError
else
    echo "Starting new job"
    proxied harbor jobs start \
        -p "$DATASET_PATH" \
        --n-concurrent "$N_CONCURRENT" \
        --agent terminus-2 \
        --model "hosted_vllm/$MODEL" \
        --env "daytona" \
        --agent-kwarg "api_base=http://${MASTER_ADDR}:8000/v1" \
        --agent-kwarg "key=fake_key" \
        --n-attempts 3 \
        --job-name "$RUN_TAG" \
        --config "$HARBOR_CONFIG" \
        $EXTRA_HARBOR_ARGS
fi
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

if [ ! -d "$RUN_DIR" ]; then
    echo "Expected run directory not found: $RUN_DIR"
    exit 2
fi

# ==============================================================================
# Check for DaytonaErrors before upload
# ==============================================================================
RESULT_FILE="$RUN_DIR/result.json"
ERROR_LOG="${EVAL_JOBS_DIR}/daytona_errors.log"

if [ -f "$RESULT_FILE" ]; then
    echo "Checking for DaytonaErrors in $RESULT_FILE..."

    DAYTONA_COUNT=$($PYTHON_BIN -c "
import json, sys
try:
    with open('$RESULT_FILE', 'r') as f:
        data = json.load(f)

    total_errors = 0
    if 'stats' in data and 'evals' in data['stats']:
        for eval_key, eval_data in data['stats']['evals'].items():
            if 'exception_stats' in eval_data and 'DaytonaError' in eval_data['exception_stats']:
                daytona_errors = eval_data['exception_stats']['DaytonaError']
                if isinstance(daytona_errors, list):
                    total_errors += len(daytona_errors)
    print(total_errors)
except Exception as e:
    print(f'Error parsing result.json: {e}', file=sys.stderr)
    print('0')
" 2>&1 | tail -n 1)

    echo "DaytonaError count: ${DAYTONA_COUNT}"

    if [ "${DAYTONA_COUNT:-0}" -gt "$DAYTONA_ERROR_THRESHOLD" ]; then
        echo "Job has ${DAYTONA_COUNT} DaytonaErrors (> ${DAYTONA_ERROR_THRESHOLD}), skipping upload"
        echo "Job completed but not uploaded due to excessive DaytonaErrors"
        exit 0
    fi
else
    echo "Warning: result.json not found, continuing with upload"
fi

# ==============================================================================
# Upload results to DB (via proxychains for HF upload)
# ==============================================================================
export RUN_DIR
export UPLOAD_USERNAME="${UPLOAD_USERNAME:-$USER}"
export UPLOAD_MODE="${UPLOAD_MODE:-skip_on_error}"
export RUN_TAG

UPLOAD_LOG="eval/leonardo/logs/upload_${SLURM_JOB_ID}.log"
mkdir -p "$(dirname "$UPLOAD_LOG")"

echo "Uploading results from: $RUN_DIR" | tee -a "$UPLOAD_LOG"
echo "Using username=${UPLOAD_USERNAME}, mode=${UPLOAD_MODE}" | tee -a "$UPLOAD_LOG"

proxied $PYTHON_BIN - <<'PY' 2>&1 | tee -a "$UPLOAD_LOG"
import os, sys, re, hashlib

from database.unified_db.utils import upload_eval_results


def sanitize_hf_repo_id(repo_id: str, max_length: int = 96) -> str:
    def collapse(s: str) -> str:
        prev = None
        while s != prev:
            prev = s
            s = s.replace("--", "-").replace("..", ".")
        return s

    org, name = repo_id.split("/", 1) if "/" in repo_id else (None, repo_id)
    name = re.sub(r"[^A-Za-z0-9._-]", "-", name)
    name = collapse(name).strip("-.")
    if not name:
        name = "repo"

    limit = max_length - (len(org) + 1 if org else 0)
    if len(name) > limit:
        digest = hashlib.sha1(name.encode()).hexdigest()[:8]
        keep = max(1, limit - len(digest))
        base = name[:keep].rstrip("-.")
        if not base:
            base = "r"
        name = f"{base}{digest}"
        name = collapse(name).strip("-.")

    name = collapse(name).strip("-.")
    if name[0] in "-.":
        name = "r" + name[1:]
    if name[-1] in "-.":
        name = name[:-1] + "0"

    return f"{org}/{name}" if org else name


run_dir = os.environ["RUN_DIR"]
run_tag = os.environ["RUN_TAG"]
username = os.environ.get("UPLOAD_USERNAME", os.environ.get("USER", "leonardo"))
error_mode = os.environ.get("UPLOAD_MODE", "skip_on_error")
hf_repo_id = sanitize_hf_repo_id(f"DCAgent2/{run_tag}")
hf_token = os.environ["HF_TOKEN"]

print(f"[uploader] upload_eval_results(path={run_dir!r}, username={username!r}, "
      f"error_mode={error_mode!r}, hf_repo_id={hf_repo_id!r})")
dataset_hf = os.environ.get("REPO_ID", "")
benchmark_name = os.environ.get("BENCHMARK_NAME", "")
if not benchmark_name:
    benchmark_name = dataset_hf.split("/")[-1] if "/" in dataset_hf else dataset_hf

import hashlib
benchmark_version_hash = hashlib.sha256(benchmark_name.encode()).hexdigest()
print(f"[uploader] benchmark_name={benchmark_name!r}, version_hash={benchmark_version_hash[:16]}...")

upload_eval_results(
    run_dir,
    username=username,
    error_mode=error_mode,
    hf_token=hf_token,
    hf_repo_id=hf_repo_id,
    register_benchmark=True,
    benchmark_name=benchmark_name,
    benchmark_version_hash=benchmark_version_hash,
)
print("[uploader] done.")
PY
UPLOAD_EXIT=${PIPESTATUS[0]}

if [ $UPLOAD_EXIT -ne 0 ]; then
    echo "Upload failed with exit code: $UPLOAD_EXIT"
    exit $UPLOAD_EXIT
fi

echo "=============================================="
echo "Eval and upload finished successfully."
echo "=============================================="
