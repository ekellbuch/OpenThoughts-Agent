# Stage 7 — Feature parity on megatron: R3 / TIS / seqnorm ★ (critical path, HIGHEST RISK)

- **Date:** 2026-07-13 · **Status:** `scoped GO` (expect sub-staging)
- **Companion:** `megatron_migration_plan.md`

## Why this is the right next step
Our correctness-critical fork features have **zero upstream references** (`moe_router_replay`=0, `align_logprobs_by_token_ids`=0, `seq_mean_token_sum_norm_global`=0). They must be re-implemented against the megatron surface. R3 today rides the **FSDP2 grouped-GEMM swap + DTensor `[Shard_fsdp,Shard_ep]` expert layout**, which has **no direct megatron-EP analogue** — this is the single biggest technical risk in the whole migration. Do it AFTER the plain megatron path (Stages 5/6) is proven, so failures here are attributable to the feature, not the backend.

## Change-set (expect 3 sub-stages, each own gate)
1. **seqnorm-global** (`seq_mean_token_sum_norm_global`): re-derive the global loss denom `Z` on the megatron path. The FSDP2 fix computes `Z` collective-free on the driver (`compute_global_loss_denom`) to avoid the fully-async non-co-arrival deadlock — the megatron DP/EP mesh has a different rank layout; recompute `Z` bit-identically for the megatron geometry. **Sub-gate:** megatron `Z` bit-matches a reference + no `NumelIn=1` collective desync.
2. **TIS exact-align** (`align_logprobs_by_token_ids`): the alignment is generator-side (zips vLLM logprobs onto training tokens by Harbor `completion_token_ids`) → largely backend-agnostic, but the logprob EXTRACTION on megatron (fused LM-head `GPTModel.output_processor` / `FusedLinearChunkedDistributedLogprob`) must feed the same token-id-aligned logprobs. **Sub-gate:** `policy/tis/imp_ratio_mean≈1.0`, `log_ratio_abs_mean~0.06 nats`, `exact_match_fraction≈0.99` on a megatron smoke — matching FSDP2 (log-space, NOT the benign linear `rollout_train_prob_diff_mean`).
3. **R3 router-replay** (`moe_router_replay`, THE hard one): re-implement capture→replay for megatron-EP. Megatron routes experts via its own MoE token-dispatch (not the FSDP2 grouped-GEMM swap); the replay must inject captured `rollout_routed_experts` into megatron's router selection faithfully (the #6335 alignment guardrail). Also decide the capture TRANSPORT under megatron (R3 decentral/`_relocate_chunk_to_node` may or may not port; couples to Stage 3's vLLM-HTTP capture). **Sub-gate:** healthy router-replay signature (`raw_grad_norm~4e-5`, `log_ratio_abs_max~0.2`; NOT the broken `~1e5 / ~19`) + flag-off byte-identical (R3 off ⇒ standard megatron MoE).

## Validation gate (GO / NO-GO — the stage as a whole)
- **GO:** all three features produce healthy metrics on a megatron GDN-MoE smoke (Stage 6 model) matching FSDP2 semantics; each is flag-off byte-identical.
- **NO-GO:** R3 replay cannot be made faithful on megatron-EP (e.g. megatron's token-dispatch can't accept an externally-captured routing) → R3-dependent 80B RL is not viable on megatron; escalate a design decision (is R3 required for the 80B objective, or can TIS alone carry it?).
- **Cost:** 2×8 H100 per sub-stage smoke + CPU dev.

## Composes with / depends on
Depends on Stages 5 + 6 (megatron trainer + GDN proven) and Stage 3 (R3 capture transport). Gates Stage 9 (80B parity needs these correctness features).
