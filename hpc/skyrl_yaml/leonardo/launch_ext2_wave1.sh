#!/bin/bash
# EXTENSION 2 — AXIS A (big models, multi-node) + AXIS B (context-length stress).
# Submits Wave 1 (<=8 cells) on Leonardo. All MATH/aime, tbs64 x n4, cudagraph ON,
# fresh per-cell ckpt (resume_mode=null), account AIFAC_5C0_290.
#
#   AXIS A : Qwen3-32B dense, 2 nodes / 8 GPU (FSDP across nodes).
#   AXIS B : Qwen2.5-7B, max_model_len sweep {16k, 32k, 65k, 131k} single-node 4xA100.
#
# The 30B-A3B MoE cell (Axis A) is submitted separately by launch_ext2_moe.sh
# once its shards finish staging (it uses the EP/grouped-GEMM path).
set -euxo pipefail
L=/leonardo_work/AIFAC_5C0_290/bfeuer00/code/OpenThoughts-Agent/hpc/skyrl_yaml/leonardo
cd "$L"

ACCT=AIFAC_5C0_290
PART=boost_usr_prod
QOS=normal
TIME=00:40:00
COMMON="trainer.max_steps=25"

# ---------------- AXIS A1: Qwen3-32B dense, 2 nodes / 8 GPU -----------------
# FSDP across both nodes (policy/ref/critic num_nodes=2, 4 GPU/node), 8xTP1 vLLM
# engines. cpu_offload=true on FSDP (EXTENSION 2 single-node OOM was in the FSDP
# backward) + gradient_checkpointing=true to fit the 32B weights+grads+activations.
sbatch --nodes=2 --time=$TIME --account=$ACCT --partition=$PART --qos=$QOS \
  --job-name=mathmn_32b_nd2 \
  --export=ALL,MODEL_PATH=Qwen/Qwen3-32B,NUM_GPUS=4 \
  sbatch_math_grid_multinode.sh \
    $COMMON \
    generator.num_inference_engines=8 generator.inference_engine_tensor_parallel_size=1 \
    trainer.placement.policy_num_gpus_per_node=4 trainer.placement.policy_num_nodes=2 \
    trainer.placement.ref_num_gpus_per_node=4 trainer.placement.ref_num_nodes=2 \
    trainer.placement.critic_num_gpus_per_node=4 trainer.placement.critic_num_nodes=2 \
    trainer.placement.colocate_all=true \
    trainer.policy.fsdp_config.cpu_offload=true \
    trainer.ref.fsdp_config.cpu_offload=true \
    trainer.gradient_checkpointing=true

# ---------------- AXIS B: Qwen2.5-7B max_model_len stress -------------------
# Single node, 4xA100-64GB, 4xTP1 vLLM. Workload FIXED (prompt 1024 / gen 4096 =
# ~5k used); only max_model_len (the vLLM KV reservation) grows. 16k/32k are
# within the 7B's native 32768 ctx; 65k/131k need YaRN rope_scaling (factor 2/4),
# which vLLM uses to auto-set max_model_len = factor * 32768.
B_MODEL=Qwen/Qwen2.5-7B-Instruct

# 16k (native, explicit cap)
sbatch --nodes=1 --time=$TIME --account=$ACCT --partition=$PART --qos=$QOS \
  --gres=gpu:4 --job-name=math_ctx16k \
  --export=ALL,MODEL_PATH=$B_MODEL \
  sbatch_math_grid.sh \
    $COMMON +generator.engine_init_kwargs.max_model_len=16384

# 32k (native max)
sbatch --nodes=1 --time=$TIME --account=$ACCT --partition=$PART --qos=$QOS \
  --gres=gpu:4 --job-name=math_ctx32k \
  --export=ALL,MODEL_PATH=$B_MODEL \
  sbatch_math_grid.sh \
    $COMMON +generator.engine_init_kwargs.max_model_len=32768

# 65k (YaRN factor 2 -> max_model_len 65536)
sbatch --nodes=1 --time=$TIME --account=$ACCT --partition=$PART --qos=$QOS \
  --gres=gpu:4 --job-name=math_ctx65k \
  --export=ALL,MODEL_PATH=$B_MODEL \
  sbatch_math_grid.sh \
    $COMMON \
    "trainer.rope_scaling={rope_type:yarn,factor:2.0,original_max_position_embeddings:32768}"

# 131k (YaRN factor 4 -> max_model_len 131072) — expected KV-capacity ceiling
sbatch --nodes=1 --time=$TIME --account=$ACCT --partition=$PART --qos=$QOS \
  --gres=gpu:4 --job-name=math_ctx131k \
  --export=ALL,MODEL_PATH=$B_MODEL \
  sbatch_math_grid.sh \
    $COMMON \
    "trainer.rope_scaling={rope_type:yarn,factor:4.0,original_max_position_embeddings:32768}"

echo "WAVE1 SUBMITTED"
squeue -u "$USER" -o "%.10i %.18j %.8T %.10M %.6D"
