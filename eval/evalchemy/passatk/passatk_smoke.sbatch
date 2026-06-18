#!/bin/bash
#SBATCH --account=AIFAC_5C0_290
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --job-name=passatk-smoke
#SBATCH --output=/leonardo_work/AIFAC_5C0_290/bfeuer00/experiments/delphi-eval/passatk-sft/logs/%x-%j.log
#
# pass@k MECHANISM SMOKE TEST (EVAL_CONVENTION pass@k §3): Qwen3-0.6B / MATH500.
# Validates: 128 samples/problem generated + graded, pass@1 ~ single-sample run,
# and pass@128 >= pass@32 >= pass@8 >= pass@1 (monotone, non-trivial spread).
# Uses --limit 30 x 128 samples. NOTE: this smoke caps max_gen_toks=4096 / max_model_len=8192 ONLY
# to bound the run (Qwen3-0.6B thinking-on at the production 30720 budget runs many samples to the
# full length -> 4h+ even at limit=50; the FIRST attempt job 46871681 was cancelled for this). The
# REDUCED budget is a SMOKE-ONLY speed knob to validate the MECHANISM (128 graded samples/problem +
# monotone pass@k); the production grid uses the full 30720 (qwen3) / 3584 (base) budgets unchanged.
set -euo pipefail

PR=/leonardo_work/AIFAC_5C0_290/bfeuer00/experiments/delphi-eval/passatk-sft
DRIVER=$PR/passatk_driver.py

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

OUT=$PR/_smoke/MATH500/qwen3-0.6b
rm -rf "$OUT"; mkdir -p "$OUT"

echo ">>> SMOKE: Qwen3-0.6B / MATH500 / n=128 / limit=30 / gen=4096 (smoke speed cap)"
python "$DRIVER" \
  --task MATH500 \
  --model-repo Qwen/Qwen3-0.6B \
  --out "$OUT" \
  --n 128 --tp 4 \
  --max-model-len 8192 --max-gen-toks 4096 \
  --temperature 0.7 --top-p 1.0 --seed 42 \
  --apply-chat-template \
  --limit 30

echo "=== SMOKE RESULT JSON ==="
cat "$OUT/passatk_results.json"
echo
echo "=== single-sample (greedy-ish, n=1) cross-check for pass@1 sanity ==="
OUT1=$PR/_smoke/MATH500/qwen3-0.6b-n1
rm -rf "$OUT1"; mkdir -p "$OUT1"
python "$DRIVER" \
  --task MATH500 \
  --model-repo Qwen/Qwen3-0.6B \
  --out "$OUT1" \
  --n 1 --tp 4 \
  --max-model-len 8192 --max-gen-toks 4096 \
  --temperature 0.7 --top-p 1.0 --seed 4242 \
  --apply-chat-template \
  --limit 30
cat "$OUT1/passatk_results.json"
echo
echo "DONE SMOKE"
