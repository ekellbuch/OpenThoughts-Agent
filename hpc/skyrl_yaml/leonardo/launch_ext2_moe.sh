#!/bin/bash
# EXTENSION 2 — AXIS A: Qwen3-30B-A3B MoE (128 experts top-8) on MATH/aime.
# MarinSkyRL main (9bb6d5e) HAS the full EP/grouped-GEMM/router-replay MoE port
# (Stages 3b-6; moe_swap.py supports Qwen3MoeSparseMoeBlock). Proven flag combo =
# tests/gpu/test_e2e_moe_rl_step.py (EP x FSDP trainer + EP vLLM engine, colocated;
# grouped-GEMM needs micro_train_batch=1 + gradient_checkpointing=false).
#
# Cell A2a: single node, 4xA100-64GB. Trainer EP=2 x FSDP=2 (3-D mesh ddp1/ep2/fsdp2);
# 1 vLLM engine TP4 + enable_expert_parallel. cpu_offload on FSDP for the 60GB weights.
set -euxo pipefail
L=/leonardo_work/AIFAC_5C0_290/bfeuer00/code/OpenThoughts-Agent/hpc/skyrl_yaml/leonardo
cd "$L"
ACCT=AIFAC_5C0_290; PART=boost_usr_prod; QOS=normal; TIME=00:40:00
MOE=Qwen/Qwen3-30B-A3B-Instruct-2507

sbatch --nodes=1 --time=$TIME --account=$ACCT --partition=$PART --qos=$QOS \
  --gres=gpu:4 --job-name=math_moe30b_ep2 \
  --export=ALL,MODEL_PATH=$MOE \
  sbatch_math_grid.sh \
    trainer.max_steps=25 \
    generator.num_inference_engines=1 generator.inference_engine_tensor_parallel_size=4 \
    +generator.engine_init_kwargs.enable_expert_parallel=true \
    trainer.policy.fsdp_config.moe_grouped_gemm=true \
    trainer.policy.fsdp_config.moe_router_replay=true \
    trainer.policy.fsdp_config.expert_model_parallel_size=2 \
    trainer.ref.fsdp_config.moe_grouped_gemm=true \
    trainer.ref.fsdp_config.moe_router_replay=true \
    trainer.ref.fsdp_config.expert_model_parallel_size=2 \
    trainer.policy.fsdp_config.cpu_offload=true \
    trainer.ref.fsdp_config.cpu_offload=true \
    trainer.gradient_checkpointing=false \
    trainer.micro_train_batch_size_per_gpu=1 \
    trainer.micro_forward_batch_size_per_gpu=1

echo "MOE SUBMITTED"
squeue -u "$USER" -o "%.10i %.18j %.8T %.10M %.6D"
