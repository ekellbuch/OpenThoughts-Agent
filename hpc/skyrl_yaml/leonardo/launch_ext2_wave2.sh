#!/bin/bash
# EXTENSION 2 — Wave 2: the 3 Wave-1 FAILED cells re-run with ONE genuine fix each.
#   32B 2-node : Wave-1 OOM was in the vLLM ENGINE INIT (8xTP1 replicates the full
#                64GB 32B per GPU -> can't fit). FIX = shard the engine: 4xTP2 +
#                gmu 0.6 + enforce_eager (skip CUDA-graph capture mem). FSDP trainer
#                still 2-node (8 GPU). cudagraph stays ON for the trainer policy.
#   MoE 30B-A3B: Wave-1 died on a flag COLLISION (+engine_init_kwargs.enable_expert
#                _parallel duplicated SkyRL's own kwarg). FIX = use the proper knob
#                generator.inference_engine_expert_parallel_size=4 (EP across TP4 engine).
#   ctx65k/131k: vLLM rejected max_model_len>32768 (7B native). FIX = explicit
#                max_model_len via engine_init_kwargs + VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
#                (set in env). Capacity stress only; outputs ~600 tok so RoPE-nan moot.
set -euxo pipefail
L=/leonardo_work/AIFAC_5C0_290/bfeuer00/code/OpenThoughts-Agent/hpc/skyrl_yaml/leonardo
cd "$L"
ACCT=AIFAC_5C0_290; PART=boost_usr_prod; QOS=normal; TIME=00:40:00
COMMON="trainer.max_steps=25"

# ---- 32B dense, 2 nodes / 8 GPU, engine sharded 4xTP2, gmu 0.6, eager ----
sbatch --nodes=2 --time=$TIME --account=$ACCT --partition=$PART --qos=$QOS \
  --job-name=mathmn_32b_nd2tp2 \
  --export=ALL,MODEL_PATH=Qwen/Qwen3-32B,NUM_GPUS=4 \
  sbatch_math_grid_multinode.sh \
    $COMMON \
    generator.num_inference_engines=4 generator.inference_engine_tensor_parallel_size=2 \
    generator.gpu_memory_utilization=0.6 generator.enforce_eager=true \
    trainer.placement.policy_num_gpus_per_node=4 trainer.placement.policy_num_nodes=2 \
    trainer.placement.ref_num_gpus_per_node=4 trainer.placement.ref_num_nodes=2 \
    trainer.placement.critic_num_gpus_per_node=4 trainer.placement.critic_num_nodes=2 \
    trainer.placement.colocate_all=true \
    trainer.policy.fsdp_config.cpu_offload=true trainer.ref.fsdp_config.cpu_offload=true \
    trainer.gradient_checkpointing=true

# ---- MoE 30B-A3B, 1 node / 4 GPU, EP via proper knob ----
sbatch --nodes=1 --time=$TIME --account=$ACCT --partition=$PART --qos=$QOS \
  --gres=gpu:4 --job-name=math_moe30b_ep2 \
  --export=ALL,MODEL_PATH=Qwen/Qwen3-30B-A3B-Instruct-2507 \
  sbatch_math_grid.sh \
    $COMMON \
    generator.num_inference_engines=1 generator.inference_engine_tensor_parallel_size=4 \
    generator.inference_engine_expert_parallel_size=4 \
    generator.gpu_memory_utilization=0.6 \
    trainer.policy.fsdp_config.moe_grouped_gemm=true \
    trainer.policy.fsdp_config.moe_router_replay=true \
    trainer.policy.fsdp_config.expert_model_parallel_size=2 \
    trainer.ref.fsdp_config.moe_grouped_gemm=true \
    trainer.ref.fsdp_config.moe_router_replay=true \
    trainer.ref.fsdp_config.expert_model_parallel_size=2 \
    trainer.policy.fsdp_config.cpu_offload=true trainer.ref.fsdp_config.cpu_offload=true \
    trainer.gradient_checkpointing=false \
    trainer.micro_train_batch_size_per_gpu=1 trainer.micro_forward_batch_size_per_gpu=1

# ---- ctx65k (explicit max_model_len + allow-long env) ----
sbatch --nodes=1 --time=$TIME --account=$ACCT --partition=$PART --qos=$QOS \
  --gres=gpu:4 --job-name=math_ctx65k \
  --export=ALL,MODEL_PATH=Qwen/Qwen2.5-7B-Instruct,VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  sbatch_math_grid.sh \
    $COMMON +generator.engine_init_kwargs.max_model_len=65536

# ---- ctx131k (explicit max_model_len + allow-long env) — expected KV ceiling ----
sbatch --nodes=1 --time=$TIME --account=$ACCT --partition=$PART --qos=$QOS \
  --gres=gpu:4 --job-name=math_ctx131k \
  --export=ALL,MODEL_PATH=Qwen/Qwen2.5-7B-Instruct,VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  sbatch_math_grid.sh \
    $COMMON +generator.engine_init_kwargs.max_model_len=131072

echo "WAVE2 SUBMITTED"
squeue -u "$USER" -o "%.10i %.18j %.8T %.10M %.6D"
