# Stage 5 — Small-MoE megatron smoke (Qwen3-30B-A3B, full-attn)

- **Date:** 2026-07-13 · **Status:** `scoped GO`
- **Companion:** `megatron_migration_plan.md`

## Why this is the right next step
Cheapest **real** proof the megatron trainer runs OUR agentic Harbor/Daytona rollout + fully-async loop end-to-end, on a model WITHOUT the GDN complication. Qwen3-30B-A3B is full-attention MoE (no GatedDeltaNet) and is a registered megatron example (`run_megatron_dapo_qwen3_30b_a3b.sh`) → isolates "does our stack drive the megatron backend" from "does GDN work" (Stage 6).

## Change-set
**Config + launch only; no source (assuming Stage 4 wiring landed).** A `hpc/skyrl_yaml/iris/megatron/30b_a3b_smoke.yaml` (small ctx, short run) on the gpu-rl-megatron image, 2×8 H100, agentic terminal_bench rollout, `NVTE_FLASH_ATTN=0` where required.

## Validation gate (GO / NO-GO)
- **GO:** reaches `global_step=1` with a clean `fwd_logprobs`, healthy metrics (entropy stable, `raw_grad_norm` sane, no NCCL/optimizer hang), a checkpoint saved, and rollouts producing non-degenerate outputs. Weight-sync (megatron→vLLM) succeeds (no token-salad — watch the `w13` reload equivalent).
- **NO-GO:** crash / hang / degenerate rollouts → diagnose (image vs weight-sync vs our agentic-path assumptions) before touching GDN or 80B.
- **Cost:** 2×8 H100, short.

## Composes with / depends on
Depends on Stage 4. De-risks Stage 6 (adds GDN) and Stage 7 (adds R3/TIS/seqnorm). Independent of Stage 0/8 (no GDN here).
