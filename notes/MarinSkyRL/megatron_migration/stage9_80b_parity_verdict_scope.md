# Stage 9 — 80B end-to-end parity gate + migrate/don't-migrate verdict ★ (critical path, terminal)

- **Date:** 2026-07-13 · **Status:** `scoped GO`
- **Companion:** `megatron_migration_plan.md`

## Why this is the right next step
The whole point: does the megatron 80B path actually beat FSDP2-EP on the pain points that motivated this (host-RAM OOM, gs1 SavedTensorHooks assert, fwd_logprobs perf, bring-up flakes) OR at least match reward/loss at competitive throughput? This is the decision gate that either greenlights production migration or records why we stay on FSDP2.

## Change-set
**Config + launch + measurement; no source (all landed by Stage 8).** A full 5D-parallel megatron 80B config (`hpc/skyrl_yaml/iris/megatron/128GPU_qwen3_next_80b_a3b.yaml`) — TP/PP/EP/ETP/CP geometry chosen per the megatron parallelism-strategy notes (the megatron equivalent of our EP8×FSDP8×CP1) — with R3/TIS/seqnorm ON. Run alongside (or A/B against) the FSDP2-EP baseline; measure reward, loss, per-step wall-time breakdown, host-RAM/fd, and bring-up reliability.

## Validation gate (GO / NO-GO = the migration verdict)
- **GO (migrate):** megatron 80B reaches **≥ global_step 2** with healthy metrics (entropy, log-space TIS, router-replay grad-norm sane) AND either (a) throughput ≥ the FSDP2-EP path, OR (b) it cleanly clears at least one FSDP2 structural pain point (no host-RAM optimizer OOM, no gs1 SavedTensorHooks assert, no R3 head-plasma overflow, fewer 16-node bring-up flakes) at acceptable throughput. → recommend staged production migration; open the upstreaming path for our features.
- **NO-GO (stay on FSDP2):** megatron 80B fails to reach gs2, regresses reward/loss, or is materially slower without clearing an FSDP2 pain point → record the verdict + evidence; the FSDP2-EP path remains production and this evaluation is archived as "megatron not yet worth it, revisit at <trigger>".
- **Cost:** 16 nodes (128×H100), against the ≤6-RUNNING-RL-jobs-per-cluster guardrail — schedule explicitly, never displace a RUNNING job without permission.

## Composes with / depends on
Depends on ALL prior stages (0–8). Terminal stage; produces the documented migrate/don't-migrate recommendation the plan exists to deliver.
