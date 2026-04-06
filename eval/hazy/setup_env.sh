#!/usr/bin/env bash
# One-time setup: create eval venv with vLLM + Harbor on Hazy.
#
# Usage:
#   ./modules/OpenThoughts-Agent/eval/hazy/setup_env.sh
#
set -euo pipefail

EVAL_ROOT="/data/home/macs/eval"
VENV_DIR="$EVAL_ROOT/venv"

echo "Creating eval environment at $VENV_DIR"

# Create directory structure
mkdir -p "$EVAL_ROOT"/{datasets,jobs,logs,cache/vllm}

# Create venv
uv venv "$VENV_DIR" --python 3.12

# Install vLLM (CUDA 12.8)
uv pip install --python "$VENV_DIR/bin/python" \
    vllm \
    --find-links https://download.pytorch.org/whl/cu128

# Install Harbor from submodule + Daytona backend + dependencies
REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
uv pip install --python "$VENV_DIR/bin/python" \
    "$REPO_ROOT/modules/harbor" \
    "daytona>=0.121.0" \
    socksio \
    huggingface_hub \
    Jinja2 \
    pyyaml

echo ""
echo "Done. Verify with:"
echo "  $VENV_DIR/bin/python -c 'import vllm; print(vllm.__version__)'"
echo "  $VENV_DIR/bin/harbor --help"
