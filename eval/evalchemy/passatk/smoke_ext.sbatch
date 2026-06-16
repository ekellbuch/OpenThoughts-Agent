#!/bin/bash
#SBATCH --account=AIFAC_5C0_290
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH --job-name=passatk-ext-smoke2
#SBATCH --output=/leonardo_work/AIFAC_5C0_290/bfeuer00/experiments/delphi-eval/passatk-sft/logs/%x-%j.log
set -euo pipefail
ROOT=/leonardo_work/AIFAC_5C0_290/bfeuer00/experiments/delphi-eval/passatk-sft
source /leonardo_work/AIFAC_5C0_290/bfeuer00/miniforge3/etc/profile.d/conda.sh
conda activate evalchemy-marin
cd /leonardo_work/AIFAC_5C0_290/bfeuer00/code/evalchemy
source ~/secrets.env
export HF_HOME=/leonardo_work/AIFAC_5C0_290/bfeuer00/data/hub
export HF_HUB_CACHE=/leonardo_work/AIFAC_5C0_290/bfeuer00/data/hub
export HF_HUB_OFFLINE=1
export HF_ALLOW_CODE_EVAL=1
export TOKENIZERS_PARALLELISM=false
CACHE_ROOT=$ROOT/../.cache
mkdir -p "$CACHE_ROOT"/{flashinfer,triton,inductor,vllm,xdg,hf_local}
export HOME="$CACHE_ROOT/hf_local"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
export FLASHINFER_WORKSPACE_BASE="$CACHE_ROOT/flashinfer"
export TRITON_CACHE_DIR="$CACHE_ROOT/triton"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_ROOT/inductor"
export VLLM_CACHE_ROOT="$CACHE_ROOT/vllm"
mkdir -p "$HOME/.cache/flashinfer"
OUT=$ROOT/_smoke_ext/qwen3-0.6b-base-AIME24-smoke2
rm -rf "$OUT"; mkdir -p "$OUT"
echo SMOKE2: qwen3-0.6b-base / AIME24 / limit15 / no-chat / CAPPED 8192/4096 mechanism-validation
python $ROOT/passatk_driver.py --task AIME24 --model-repo Qwen/Qwen3-0.6B-Base --out "$OUT"   --n 128 --tp 4 --max-model-len 8192 --max-gen-toks 4096   --temperature 0.7 --top-p 1.0 --seed 42 --limit 15
echo SMOKE2_DONE
cat "$OUT/passatk_results.json"
