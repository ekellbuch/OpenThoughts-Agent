# Stage 0 — Megatron-Bridge GDN-80B coverage spike ★ (critical path, RECOMMENDED FIRST)

- **Date:** 2026-07-13 · **Status:** `scoped GO` (discovery; no fix)
- **Companion:** `megatron_migration_plan.md`

## Why this is the right next step (cheapest de-risk that can NO-GO the whole project)
The entire migration is worthless if Megatron cannot represent **Qwen3-Next-80B-A3B** (GatedDeltaNet linear-attn + 512-expert top-10 + 1 shared). We KNOW the GDN+MoE machinery exists (Qwen3.5-35B-A3B routes to the native `GPTModel` GDN path via a registered bridge), but the 80B's exact arch has **no bridge and unconfirmed representability**. Answer this on CPU (config-only) before spending a single GPU-hour or touching the launcher.

## Change-set
**Read-only / spike; no `MarinSkyRL/` source touched.** In the isolated clone (or a throwaway venv with megatron-core + Megatron-Bridge installed): attempt `AutoBridge.from_hf_pretrained(Qwen3-Next-80B-A3B config)` with **no weights** (or a tiny random-init), inspecting whether Megatron-Bridge resolves the architecture (`Qwen3NextForCausalLM` or its actual `architectures` string) to a `GPTModel`/Mamba-hybrid spec, and whether Megatron-core exposes a GDN linear-attention layer + a 512-expert MoE spec (top-k=10 + shared-expert). Compare against the GLM47/Qwen35 bridge templates. Optionally a 1-GPU tiny-config forward to confirm the GDN kernel path instantiates.

## Validation gate (GO / NO-GO)
- **GO:** Megatron-Bridge/-core can represent Qwen3-Next-80B-A3B — either an existing bridge resolves it, OR a concrete authoring path is identified (a `Qwen3NextBridge` analogous to `GLM47FlashBridge`/`Qwen35MoELMBridge` over existing GDN + MoE-EP megatron-core specs). Record the exact `architectures` string, the megatron layer specs (GDN + shared-expert MoE), and the top-k/shared-expert config knobs. → proceed; feeds Stage 8.
- **NO-GO:** Megatron-core has no GDN-hybrid + 512-expert-with-shared representation → the migration is a **megatron model-spec authoring project** (or infeasible for the 80B), not a backend swap. Report this loudly; the plan re-scopes (possibly target the 35B GDN model as the migration proof-of-value instead of the 80B).
- **Cost:** CPU (a few hours) + optional 1×H100 tiny-load.

## Composes with / depends on
None. This is the entry point. Its verdict sizes Stage 8 (and can cancel the project).
