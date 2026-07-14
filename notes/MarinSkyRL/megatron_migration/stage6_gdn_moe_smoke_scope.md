# Stage 6 — GDN-MoE megatron smoke (Qwen3.5/3.6-35B-A3B, registered) ★ (critical path)

- **Date:** 2026-07-13 · **Status:** `scoped GO`
- **Companion:** `megatron_migration_plan.md`

## Why this is the right next step
Prove the **GatedDeltaNet + MoE** megatron path works end-to-end for us on the arch that IS **already in the registry** (Qwen3.5-35B-A3B → native `GPTModel` GDN via `language_model_only=True`), BEFORE authoring the unregistered 80B bridge (Stage 8). This isolates "does GDN-MoE train on megatron in our stack" from "can we author the 80B bridge". Note: on FSDP2 the 35B is **CP=1 only** (GDN has no CP-aware kernel) — the megatron GDN path (native GPTModel) may lift this, which is itself a datapoint.

## Change-set
**Config + launch only.** A `hpc/skyrl_yaml/iris/megatron/35b_a3b_gdn_smoke.yaml` routing Qwen3.5/3.6-35B-A3B through the megatron GDN path (`language_model_only=True`, respecting the wrapper's refusal of sample-packing/microbatch-padding-removal that corrupts GDN `cu_seqlens`), 2×8 H100, `NVTE_FLASH_ATTN=0`.

## Validation gate (GO / NO-GO)
- **GO:** gs1 completes on the GDN path with healthy metrics + a checkpoint; the GDN cu_seqlens handling holds (no forward tensor-size crash of the FSDP2-CP2 flavor). Record whether megatron GDN supports CP>1 (an FSDP2 limitation the migration might clear).
- **NO-GO:** GDN forward/backward fails on megatron for the *registered* 35B → the 80B (also GDN) is unlikely to work; re-scope Stage 8 / reconsider the migration for GDN models.
- **Cost:** 2×8 H100.

## Composes with / depends on
Depends on Stage 5 (megatron trainer proven) + Stage 2 (image). Strongly de-risks Stage 8 (80B is the same GDN+MoE family, larger). Feeds Stage 7 (R3/TIS on a GDN model).
