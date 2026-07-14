# Stage 3 — vLLM decision spike

- **Date:** 2026-07-13 · **Status:** `scoped GO` (discovery; no fix)
- **Companion:** `megatron_migration_plan.md`

## Why this is the right next step
The megatron trainer's **weight-sync surface** (megatron sharded state → vLLM inference engines) may differ from our FSDP2 `broadcast_to_inference_engines` path, and upstream's megatron backend is validated against **`vllm==0.23.0`**, while we run our **fork `@76259c63`**. Deciding vLLM early is a hard input to the image (Stage 2) and the parity semantics (Stage 7).

## Change-set
**Read-only / decision doc.** Inventory what our fork `@76259c63` carries that 0.23.0 may not: DCP / GQA-LSE fix, routed-experts HTTP surface (R3 capture over `/chat/completions`), FusedMoE `w13`/`SKYRL_W13_RELOAD_BRACKET` reload semantics, any MoE bits. Cross against 0.23.0's changelog + the megatron weight-sync path's expectations. Evaluate:
1. **Adopt `vllm==0.23.0`** for the megatron path (aligns with upstream; risks losing fork-only features → check R3-HTTP + w13 reload + DCP survive or have 0.23.0 equivalents).
2. **Keep our fork `@76259c63`** with the megatron trainer (preserves fork features; risks weight-sync surface mismatch vs what upstream megatron expects).
3. **Rebase our fork onto 0.23.0** (best of both; largest vLLM-side effort — its own mini-plan).

## Validation gate (GO / NO-GO)
- **GO:** a decision + a compatibility matrix (megatron weight-sync API × vLLM version × each fork-only feature: present / equivalent / lost) + the required vLLM ref for Stage 2's image.
- **NO-GO:** neither vLLM option supports both megatron weight-sync AND R3-HTTP routed-experts capture without a fork-rebase → escalate: R3 on megatron may need a different capture transport (feeds Stage 7).
- **Cost:** CPU (read + reasoning); optional tiny weight-sync smoke on the Stage-2 image.

## Composes with / depends on
Depends on Stage 1 (adoption tree). Feeds Stage 2 (image) and Stage 7 (R3 capture transport).
