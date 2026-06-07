#!/bin/bash
# MarinSkyRL NON-AGENTIC RL — Leonardo (CINECA) A100-64GB.
# HARD-MATH (Hendrycks MATH, Level 3-5) GRPO with NATURAL long CoT.
# Reuses the existing `aime` env (Minerva/boxed answer-match verifier).
# Runs via skyrl_train.entrypoints.main_base. Invoked inside the marinskyrl
# sandbox via singularity from sbatch_math_grid.sh. Fully offline, no Harbor.
#
# Goal: scale MODEL_SIZE on a task that naturally emits long CoT to find the
# GENERATION-BOUND saturation point (timing/generate / timing/step crossing
# timing/policy_train / timing/step). Fixed modest batch (tbs64 x n4) so model
# size is the clean variable. cudagraph ON (enforce_eager=false). max_generate
# 4096 so natural CoT is NOT truncated; model emits EOS naturally.
#
# Env (from sbatch): DATA_DIR MODEL_PATH NUM_GPUS TP CKPT_DIR
set -x

: "${DATA_DIR:?set DATA_DIR}"
: "${MODEL_PATH:?set MODEL_PATH}"
: "${NUM_GPUS:=4}"
: "${TP:=1}"
: "${CKPT_DIR:?set CKPT_DIR}"
: "${LOGGER:=console}"
: "${INFERENCE_BACKEND:=vllm}"
: "${VENV_PY:=python}"

# engines = total GPUs / TP (colocated)
ENGINES=$(( NUM_GPUS / TP ))

"$VENV_PY" -m skyrl_train.entrypoints.main_base \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path="$MODEL_PATH" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.critic_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  generator.num_inference_engines=$ENGINES \
  generator.inference_engine_tensor_parallel_size=$TP \
  trainer.epochs=5 \
  trainer.max_steps=25 \
  trainer.eval_batch_size=64 \
  trainer.eval_before_train=false \
  trainer.eval_interval=0 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=64 \
  trainer.policy_mini_batch_size=64 \
  trainer.micro_forward_batch_size_per_gpu=4 \
  trainer.micro_train_batch_size_per_gpu=2 \
  trainer.ckpt_interval=999999 \
  trainer.max_prompt_length=1024 \
  generator.sampling_params.max_generate_length=4096 \
  generator.max_num_batched_tokens=8192 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=false \
  generator.backend=$INFERENCE_BACKEND \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.batched=true \
  generator.enforce_eager=false \
  environment.env_class=aime \
  generator.n_samples_per_prompt=4 \
  generator.gpu_memory_utilization=0.85 \
  generator.vllm_stats_interval=1 \
  trainer.logger="$LOGGER" \
  trainer.project_name="math_longcot_grid" \
  trainer.run_name="leonardo_math_grid" \
  trainer.resume_mode=null \
  trainer.ckpt_path="$CKPT_DIR" \
  trainer.export_path="$CKPT_DIR/exports" \
  "$@"
