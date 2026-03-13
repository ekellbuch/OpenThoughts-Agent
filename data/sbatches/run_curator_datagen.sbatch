#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:4
#SBATCH --output=data/sbatches/logs/curator_datagen_%j.out
#SBATCH --job-name=curator_datagen

set -eo pipefail

# ============================================
# Curator Non-Agentic Datagen
# ============================================
# Starts a vLLM server on the node, then runs
# curator_datagen.py to generate completions
# from a HuggingFace dataset of prompts.
#
# Uses a dedicated 'curator' conda env to avoid
# dependency conflicts with other environments.
#
# Pre-requisites (run on login node before submitting):
#   conda create -n curator python=3.12 -y
#   conda activate curator
#   pip install 'bespokelabs-curator>=0.1.0' 'vllm>=0.11.2' \
#       'datasets>=2.0.0' 'huggingface_hub>=0.20.0'
#
# Usage:
#   sbatch run_curator_datagen.sbatch \
#     <model> <input_dataset> <output_repo> [limit]
#
# Example:
#   sbatch run_curator_datagen.sbatch \
#     Qwen/Qwen3-32B \
#     open-thoughts/OpenThoughts3-1.2M \
#     mlfoundations-dev/OT3-1.2M-Qwen3-32B-completions
# ============================================

MODEL="${1:?Usage: sbatch run_curator_datagen.sbatch <model> <input_dataset> <output_repo> [limit]}"
INPUT_DATASET="${2:?Missing input dataset}"
OUTPUT_REPO="${3:?Missing output repo}"
LIMIT="${4:-}"  # Optional: limit number of prompts

echo "============================================"
echo "Curator Datagen"
echo "============================================"
echo "Model:          $MODEL"
echo "Input dataset:  $INPUT_DATASET"
echo "Output repo:    $OUTPUT_REPO"
echo "Limit:          ${LIMIT:-all}"
echo "Job ID:         $SLURM_JOB_ID"
echo "Node:           $(hostname)"
echo "============================================"

# -----------------------------------------------
# Step 1: Environment setup
# -----------------------------------------------
echo ""
echo "=== Step 1: Environment Setup ==="

# Activate the dedicated curator conda env
source ~/.bashrc
conda activate curator

# Source secrets (HF_TOKEN, etc.)
if [ -f ~/secrets.env ]; then
    source ~/secrets.env
fi

# Source cluster dotenv for HF_HUB_CACHE, etc.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
if [ -f "$REPO_DIR/hpc/dotenv/jupiter.env" ]; then
    source "$REPO_DIR/hpc/dotenv/jupiter.env"
fi

# Load modules (Jupiter needs CUDA for vLLM)
module purge 2>/dev/null || true
module load nvidia-compilers/25.9-CUDA-13 2>/dev/null || true

export VLLM_USE_V1=1

mkdir -p "$REPO_DIR/data/sbatches/logs"
cd "$REPO_DIR"

echo "HF_HUB_CACHE: $HF_HUB_CACHE"
echo "Python: $(python3 --version 2>&1)"
echo "Conda env: $CONDA_DEFAULT_ENV"

# -----------------------------------------------
# Step 2: Start vLLM server
# -----------------------------------------------
echo ""
echo "=== Step 2: Starting vLLM Server ==="

VLLM_PORT=8000
VLLM_API_KEY="token-curator-${SLURM_JOB_ID}"
VLLM_LOG="$REPO_DIR/data/sbatches/logs/vllm_${SLURM_JOB_ID}.log"

python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --port $VLLM_PORT \
    --api-key "$VLLM_API_KEY" \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.92 \
    --max-model-len 32768 \
    --max-num-seqs 64 \
    --enable-prefix-caching \
    --trust-remote-code \
    >> "$VLLM_LOG" 2>&1 &

VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"
echo "vLLM log: $VLLM_LOG"

# Wait for vLLM to become healthy
echo "Waiting for vLLM server to start..."
for i in {1..120}; do
    if curl -s "http://localhost:${VLLM_PORT}/v1/models" -H "Authorization: Bearer ${VLLM_API_KEY}" > /dev/null 2>&1; then
        echo "vLLM server is ready! (took ~$((i * 10))s)"
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "ERROR: vLLM process died. Check $VLLM_LOG"
        tail -30 "$VLLM_LOG"
        exit 1
    fi
    if [ "$i" -eq 120 ]; then
        echo "ERROR: vLLM server failed to start after 20 minutes"
        tail -30 "$VLLM_LOG"
        kill $VLLM_PID 2>/dev/null || true
        exit 1
    fi
    sleep 10
done

# -----------------------------------------------
# Step 3: Run Curator datagen
# -----------------------------------------------
echo ""
echo "=== Step 3: Running Curator Datagen ==="

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="$REPO_DIR/data/sbatches/logs/curator_output_${SLURM_JOB_ID}_${TIMESTAMP}"

CURATOR_ARGS=(
    --input-dataset "$INPUT_DATASET"
    --model "$MODEL"
    --base-url "http://localhost:${VLLM_PORT}/v1"
    --api-key "$VLLM_API_KEY"
    --output-repo "$OUTPUT_REPO"
    --output-dir "$OUTPUT_DIR"
    --max-tokens 16384
    --temperature 0.7
    --top-p 0.95
)

if [ -n "$LIMIT" ]; then
    CURATOR_ARGS+=(--limit "$LIMIT")
fi

echo "Running: python3 scripts/datagen/curator_datagen.py ${CURATOR_ARGS[*]}"

python3 scripts/datagen/curator_datagen.py "${CURATOR_ARGS[@]}"
CURATOR_EXIT=$?

# -----------------------------------------------
# Step 4: Cleanup
# -----------------------------------------------
echo ""
echo "=== Step 4: Cleanup ==="

kill $VLLM_PID 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true

echo ""
echo "============================================"
if [ $CURATOR_EXIT -eq 0 ]; then
    echo "Curator Datagen COMPLETED"
    echo "  Input:  $INPUT_DATASET"
    echo "  Output: $OUTPUT_REPO"
    echo "  Local:  $OUTPUT_DIR"
else
    echo "Curator Datagen FAILED (exit code: $CURATOR_EXIT)"
fi
echo "============================================"

exit $CURATOR_EXIT
