#!/bin/bash
#SBATCH --account=AIFAC_5C0_290
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --time=23:59:00
#SBATCH --job-name=passatk-moefast
#SBATCH --output=/leonardo_work/AIFAC_5C0_290/bfeuer00/experiments/delphi-eval/passatk-sft/logs/%x-%A_%a.log
#
# FAST-config split relaunch of the throughput-bound deep-MoE pass@k cells (Delphi #231).
# Validated 2026-06-19 (probe 47375584): --gpu-mem-util 0.85 --max-num-seqs 64 (CUDA graphs ON,
# NO --enforce-eager) inits clean on one 4xA100-64GB node (KV 1.69M tok, 51.54x concurrency for 32k
# requests) and runs ~770 toks/s output vs ~35 baseline (~22x). NOT device-mem bound at the old
# max-num-seqs=8; that was the throttle.
#
# Driven by env: MODEL_REPO, RUN, TASK, CHAT(0|1), and a per-array OFFSET/LIMIT map via SHARDS.
# SHARDS = space-separated "off:lim" pairs; array task i picks pair i (lim 0 = to end). --shard mode;
# merge with passatk_merge_shards.py afterward (--expect-problems = full cell size).
#   sbatch --array=0-N --export=ALL,MODEL_REPO=..,RUN=..,TASK=..,CHAT=0,SHARDS="0:250 250:0" passatk_moe_fast_split.sbatch
set -euo pipefail

source /leonardo_work/AIFAC_5C0_290/bfeuer00/miniforge3/etc/profile.d/conda.sh
conda activate evalchemy-marin
cd /leonardo_work/AIFAC_5C0_290/bfeuer00/code/evalchemy-marin
source ~/secrets.env

export HF_HOME=/leonardo_work/AIFAC_5C0_290/bfeuer00/data/hub
export HF_HUB_CACHE=/leonardo_work/AIFAC_5C0_290/bfeuer00/data/hub
export HF_HUB_OFFLINE=1
export HF_ALLOW_CODE_EVAL=1
export TOKENIZERS_PARALLELISM=false

CACHE_ROOT=/leonardo_work/AIFAC_5C0_290/bfeuer00/experiments/delphi-eval/.cache
mkdir -p "$CACHE_ROOT"/{flashinfer,triton,inductor,vllm,xdg,hf_local}
export HOME="$CACHE_ROOT/hf_local"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
export FLASHINFER_WORKSPACE_BASE="$CACHE_ROOT/flashinfer"
export TRITON_CACHE_DIR="$CACHE_ROOT/triton"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_ROOT/inductor"
export VLLM_CACHE_ROOT="$CACHE_ROOT/vllm"
mkdir -p "$HOME/.cache/flashinfer"

DRIVER=/leonardo_work/AIFAC_5C0_290/bfeuer00/experiments/delphi-eval/passatk-sft/passatk_driver.py
OUT=/leonardo_work/AIFAC_5C0_290/bfeuer00/experiments/delphi-eval/passatk-sft/${TASK}/${RUN}
mkdir -p "$OUT"

# pick this task index's off:lim pair from SHARDS
read -ra PAIRS <<< "$SHARDS"
PAIR=${PAIRS[${SLURM_ARRAY_TASK_ID:-0}]}
OFFSET=${PAIR%%:*}
LIMIT=${PAIR#*:}

CHAT_FLAG=""
[ "${CHAT:-0}" = "1" ] && CHAT_FLAG="--apply-chat-template"

echo ">>> MOEFAST RUN=$RUN TASK=$TASK repo=$MODEL_REPO offset=$OFFSET limit=$LIMIT chat=${CHAT:-0} task_id=${SLURM_ARRAY_TASK_ID:-NA}"
python "$DRIVER" \
  --task "$TASK" \
  --model-repo "$MODEL_REPO" \
  --out "$OUT" \
  --n 128 --tp 4 \
  --max-model-len 32768 --max-gen-toks 30720 \
  --temperature 0.7 --top-p 1.0 --seed 42 \
  --gpu-mem-util 0.85 --max-num-seqs 64 \
  --offset "$OFFSET" --limit "$LIMIT" --shard \
  $CHAT_FLAG \
  --evalchemy-root /leonardo_work/AIFAC_5C0_290/bfeuer00/code/evalchemy-marin
echo "DONE shard $RUN/$TASK off=$OFFSET lim=$LIMIT"
