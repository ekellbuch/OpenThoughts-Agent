#!/bin/bash
# EXTENSION 2 — Wave 3: 32B + MoE re-run with the Wave-2 sizing fixes.
#   32B 2-node : Wave-2 vLLM aborted KV init — at gmu 0.6 only 1.70 GiB KV left
#                after 32B(TP2) weights, < the 5 GiB needed for the 40960 native
#                ctx. FIX = cap engine max_model_len to 8192 (our workload is
#                1024+4096=5120) + raise gmu to 0.75. One sizing fix.
#   MoE 30B-A3B: Wave-2 vLLM EP engine came up fine (32/128 experts/rank) but the
#                FSDP trainer mesh asserted world_size=4 % (ep2*fsdp4)=8. FIX = set
#                fsdp_size=2 so ep2*fsdp2=4=world (the e2e test's exact mesh).
set -euxo pipefail
L=/leonardo_work/AIFAC_5C0_290/bfeuer00/code/OpenThoughts-Agent/hpc/skyrl_yaml/leonardo
cd "$L"
ACCT=AIFAC_5C0_290; PART=boost_usr_prod; QOS=normal; TIME=00:40:00
COMMON="trainer.max_steps=25"

# ---- 32B dense, 2 nodes / 8 GPU, 4xTP2, gmu 0.75, max_model_len 8192 ----
sbatch --nodes=2 --time=$TIME --account=$ACCT --partition=$PART --qos=$QOS \
  --job-name=mathmn_32b_nd2tp2 \
  --export=ALL,MODEL_PATH=Qwen/Qwen3-32B,NUM_GPUS=4 \
  sbatch_math_grid_multinode.sh \
    $COMMON \
    generator.num_inference_engines=4 generator.inference_engine_tensor_parallel_size=2 \
    generator.gpu_memory_utilization=0.75 generator.enforce_eager=true \
    +generator.engine_init_kwargs.max_model_len=8192 \
    trainer.placement.policy_num_gpus_per_node=4 trainer.placement.policy_num_nodes=2 \
    trainer.placement.ref_num_gpus_per_node=4 trainer.placement.ref_num_nodes=2 \
    trainer.placement.critic_num_gpus_per_node=4 trainer.placement.critic_num_nodes=2 \
    trainer.placement.colocate_all=true \
    trainer.policy.fsdp_config.cpu_offload=true trainer.ref.fsdp_config.cpu_offload=true \
    trainer.gradient_checkpointing=true

# ---- MoE 30B-A3B, 1 node / 4 GPU, ep2 x fsdp2 mesh ----
sbatch --nodes=1 --time=$TIME --account=$ACCT --partition=$PART --qos=$QOS \
  --gres=gpu:4 --job-name=math_moe30b_ep2 \
  --export=ALL,MODEL_PATH=Qwen/Qwen3-30B-A3B-Instruct-2507 \
  sbatch_math_grid.sh \
    $COMMON \
    generator.num_inference_engines=1 generator.inference_engine_tensor_parallel_size=4 \
    generator.inference_engine_expert_parallel_size=4 \
    generator.gpu_memory_utilization=0.6 \
    +generator.engine_init_kwargs.max_model_len=8192 \
    trainer.policy.fsdp_config.moe_grouped_gemm=true \
    trainer.policy.fsdp_config.moe_router_replay=true \
    trainer.policy.fsdp_config.expert_model_parallel_size=2 \
    trainer.policy.fsdp_config.fsdp_size=2 \
    trainer.ref.fsdp_config.moe_grouped_gemm=true \
    trainer.ref.fsdp_config.moe_router_replay=true \
    trainer.ref.fsdp_config.expert_model_parallel_size=2 \
    trainer.ref.fsdp_config.fsdp_size=2 \
    trainer.policy.fsdp_config.cpu_offload=true trainer.ref.fsdp_config.cpu_offload=true \
    trainer.gradient_checkpointing=false \
    trainer.micro_train_batch_size_per_gpu=1 trainer.micro_forward_batch_size_per_gpu=1

echo "WAVE3 SUBMITTED"
squeue -u "$USER" -o "%.10i %.18j %.8T %.10M %.6D"
