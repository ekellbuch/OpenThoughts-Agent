# Stage 2 — gpu-rl-megatron image spike

- **Date:** 2026-07-13 · **Status:** `scoped GO`
- **Companion:** `megatron_migration_plan.md`

## Why this is the right next step
Every GPU stage (5/6/7/8/9) needs a runtime with **TransformerEngine 2.11 + megatron-core + Megatron-Bridge + Ray 2.56** that our current gpu-rl image lacks. Build + prove the image early so it isn't the long pole later. Can run in parallel with Stages 3–5 once 0/1 are GO.

## Change-set
`docker/Dockerfile.gpu-rl-megatron` (new variant, in the isolated OT-Agent clone) adapting upstream `docker/Dockerfile.megatron` (`FROM anyscale/ray:2.56.0-slim-py312-cu128`, `torch==2.11.0`, `transformer-engine[pytorch]==2.11.0` + `-torch`/`-cu12==2.11.0`, flash-attn 2.8.3, flashinfer 0.6.12) onto our gpu-rl base — plus megatron-core + Megatron-Bridge (pinned rev per `[tool.uv.sources]`), harbor baked non-editably (as today), and the vLLM per Stage 3's decision. Built **in-cluster via kaniko on CoreWeave** (the Mac can't; multi-layer, `SINGLE_SNAPSHOT=0`, max layer <8 GB — see `build-gpu-rl-image-iris` + `.claude/ops/iris/coreweave_gpu_ops.md`). TE builds from source → needs the nccl-dev headers + build toolchain (upstream Dockerfile's toolchain block).

## Validation gate (GO / NO-GO)
- **GO:** kaniko build succeeds + digest captured; in-pod `import megatron`, `import transformer_engine`, `import megatron.bridge` succeed and a trivial megatron model init runs on 1 node (`NVTE_FLASH_ATTN=0` for any test path). Bump a `DEFAULT_RL_MEGATRON_DOCKER_IMAGE` (new; does NOT touch the FSDP2 default).
- **NO-GO:** TE 2.11 or megatron-core won't build/coexist with our vLLM choice on CUDA 12.8 → revisit Stage 3 (vLLM) or pin adjustments.
- **Cost:** one CoreWeave kaniko build job (512 GB node).

## Composes with / depends on
Depends on Stage 3 (which vLLM to bake) and Stage 1 (which tree the megatron backend lives in for the editable install). Gates all GPU stages.
