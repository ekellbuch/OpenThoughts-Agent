#!/bin/bash
#SBATCH --job-name=marinskyrl_gsm8k_canary
#SBATCH --account=AIFAC_5C0_290
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=480G
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out
#
# MarinSkyRL NON-AGENTIC GSM8K GRPO canary on Leonardo (1 node x 4 A100-64GB).
# apptainer exec --nv the marinskyrl.sif and run run_gsm8k_canary.sh with
# LOCAL pre-staged model + dataset, fully offline.
set -euxo pipefail

WORK=/leonardo_work/AIFAC_5C0_290/bfeuer00
SF=/leonardo_scratch/fast/AIFAC_5C0_290/bfeuer00
SIF=$WORK/containers/marinskyrl.sif
MARIN=$WORK/code/MarinSkyRL/skyrl-train
CFG=$WORK/code/OpenThoughts-Agent/hpc/skyrl_yaml/leonardo

export DATA_DIR=$WORK/data/gsm8k
export MODEL_PATH=Qwen/Qwen2.5-1.5B-Instruct      # resolved from offline HF cache
export NUM_GPUS=4
export CKPT_DIR=$SF/ckpts/gsm8k_canary

# Offline / cache env (compute nodes have no internet)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=$WORK/data/hub
export HF_HUB_CACHE=$WORK/data/hub
export WANDB_MODE=offline
export VLLM_CACHE_ROOT=$SF/vllm_cache
export TRITON_CACHE_DIR=$SF/vllm_cache/triton
export FLASHINFER_WORKSPACE_BASE=$SF/vllm_cache/flashinfer
mkdir -p "$CKPT_DIR" "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$FLASHINFER_WORKSPACE_BASE"

# Bind WORK + scratch so cache/model/data are visible inside the container.
BINDS="$WORK,$SF"

nvidia-smi || true

# Run from the skyrl-train dir so the installed skyrl_train package + hydra configs resolve.
singularity exec --nv \
  --bind "$BINDS" \
  --pwd "$MARIN" \
  --env HF_HUB_OFFLINE=1,TRANSFORMERS_OFFLINE=1,HF_HOME=$HF_HOME,HF_HUB_CACHE=$HF_HUB_CACHE,WANDB_MODE=offline,VLLM_CACHE_ROOT=$VLLM_CACHE_ROOT,TRITON_CACHE_DIR=$TRITON_CACHE_DIR,FLASHINFER_WORKSPACE_BASE=$FLASHINFER_WORKSPACE_BASE,DATA_DIR=$DATA_DIR,MODEL_PATH=$MODEL_PATH,NUM_GPUS=$NUM_GPUS,CKPT_DIR=$CKPT_DIR,LOGGER=console \
  "$SIF" bash "$CFG/run_gsm8k_canary.sh"

echo "CANARY_EXIT=$?"
