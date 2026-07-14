# Stage 8 — Qwen3-Next-80B-A3B megatron bridge authoring ★ (critical path)

- **Date:** 2026-07-13 · **Status:** `scoped GO` (CONDITIONAL on Stage 0 verdict)
- **Companion:** `megatron_migration_plan.md`

## Why this is the right next step
The 80B is our production target and is **NOT a registered megatron bridge** (zero `qwen3_next` refs). Only after the GDN-MoE path is proven on the registered 35B (Stage 6) and our features re-implemented (Stage 7) does authoring the 80B bridge pay off. Gated on Stage 0: if Stage 0 said "needs authoring", this is where it happens; if Stage 0 said "infeasible", this stage is cut.

## Change-set
`model_bridges.py` (in the chosen tree): register a `Qwen3NextBridge` for Qwen3-Next-80B-A3B — analogous to `GLM47FlashBridge` (DeepSeek-V3-like) and `Qwen35MoELMBridge` — mapping the HF `architectures` string (recorded in Stage 0) onto the megatron GDN + 512-expert-top10 + 1-shared MoE spec, with the correct rope/norm/expert-router config translation. Reuse the Qwen3.5 GDN `language_model_only` GPTModel path where the arch matches.

## Validation gate (GO / NO-GO)
- **GO:** the 80B loads via the bridge (weights convert HF→megatron without shape/key errors) AND a **1-step forward-logprob parity** vs an HF/FSDP2 reference passes at a tight fp32 tol (`torch.allclose` at the bf16 floor; do NOT loosen silently). Expert count (512), top-k (10), shared-expert, and GDN layers all resolve.
- **NO-GO:** the bridge cannot faithfully convert (e.g. shared-expert or GDN-variant mismatch vs megatron-core's spec) → the 80B needs a megatron-core model-spec contribution (upstream-scale effort) or the migration targets the 35B GDN model as the deliverable instead.
- **Cost:** 1-GPU load + forward (weights fit sharded); a short multi-GPU forward for the parity check.

## Composes with / depends on
Depends on Stage 0 (feasibility + arch details), Stage 6 (GDN-MoE proven), Stage 7 (features). Gates Stage 9.
