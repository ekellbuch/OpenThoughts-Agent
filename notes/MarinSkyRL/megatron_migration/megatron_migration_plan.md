# MarinSkyRL → upstream SkyRL Megatron backend — STAGED MIGRATION PLAN (parent)

- **Date:** 2026-07-13
- **Status:** `scoped — propose-only; no code yet`
- **Target repo:** `marin-community/MarinSkyRL`
  - **Canonical local path:** `/Users/benjaminfeuer/Documents/MarinSkyRL` (branch `penfever/working`, editable-installed — LEAVE ON `penfever/working`, do NOT branch here)
  - **Isolated working-copy path (to be created at execution, NOT now):** `/Users/benjaminfeuer/Documents/staged-work/megatron-migration/MarinSkyRL` (rsync-clone → branch `feuer/megatron-migration`). Cross-repo: also `.../staged-work/megatron-migration/{OpenThoughts-Agent,vllm}` if Stages 2–4 touch the launcher / vLLM fork.
  - **Upstream reference:** `NovaSky-AI/SkyRL` `main` (read-only via `gh api`; megatron backend lives at `skyrl/backends/skyrl_train/{workers,distributed}/megatron/`).
- **Evidence log:** `agent_logs/2026-07-13_megatron_migration_scoping.md`
- **Companion skill:** run via `code-execute-staged-plan` (gate-by-gate).

---

## Goal (precise, testable end state)

A **`trainer.strategy=megatron`** training path for MarinSkyRL that (a) leaves the existing FSDP2-EP path **byte-identical when the flag is off** (`strategy=fsdp`, the default), (b) runs our **agentic Harbor/terminal_bench + Daytona + CoreWeave iris** rollout/launch stack unchanged, (c) preserves our **correctness-critical fork features** (R3 router-replay, TIS exact-by-id alignment, seqnorm-global loss, fully-async trainer, OPD teacher-logits), and (d) trains **Qwen3-Next-80B-A3B** (GatedDeltaNet + 512-expert top-10 + 1-shared MoE) to a **documented parity verdict** vs the FSDP2-EP baseline (reward / loss / throughput), producing a migrate / don't-migrate recommendation.

**This plan does not commit to migrating.** It commits to a de-risked, staged *evaluation* whose early stages can return NO-GO cheaply. The two load-bearing unknowns (GDN-80B representability in Megatron; R3/TIS/seqnorm parity on Megatron) are front-loaded as discovery stages with their own gates.

---

## Why / the mechanism (motivation — verified this session)

Our FSDP2-EP path for Qwen3-Next-80B-A3B keeps hitting structural walls: host-RAM optimizer OOM (mitigated via `foreach:false`), the gs1-backward **SavedTensorHooks assert** (`use_reentrant` TLS bug, under test), a `fwd_logprobs` perf question (CPU-offload + micro_forward=1), R3 head-plasma object-store pressure (Fix A/B), and repeated 16-node Ray-join / HF-prestage bring-up flakes (see `agent_logs/2026-07-13_80b-*`). Upstream SkyRL's **Megatron backend** is now **dependency-compatible** with our runtime and is **actively maintained for big MoE** (5D parallelism TP/PP/DP/CP/EP+ETP; example scripts for Qwen3.5/3.6-35B-A3B, Qwen3-30B-A3B, Qwen3-235B-A22B, GLM-4.7), making it a genuine alternative worth a staged trial rather than a rewrite of our FSDP2 pain points one by one.

### Verified facts this session (confirm/extend at impl time — anchors drift)
1. **Backend coexistence already exists upstream.** `skyrl/train/config/config.py` has `trainer.strategy: Optional[str]` selecting `"fsdp"` vs `"megatron"`, with a dedicated `MegatronConfig` (TP/PP/EP/ETP/CP, `moe_router_load_balancing`, DDP, LoRA, fake-INT4-QAT, Megatron-Bridge passthrough). → the **flag-off byte-identical** invariant is *natively supported* by the upstream design, not something we must invent.
2. **Megatron deps match our runtime on the big pins.** `docker/Dockerfile.megatron` = `FROM anyscale/ray:2.56.0-slim-py312-cu128` (Ray 2.56, py3.12, CUDA 12.8) + `torch==2.11.0`, TE `[pytorch]==2.11.0`, flash-attn `2.8.3`, flashinfer `0.6.12`, `vllm==0.23.0`. Our gpu-rl image already carries torch 2.11 + flash-attn 2.8.3 + CUDA 12.8 (but our **own vLLM fork `@76259c63`**, NOT 0.23.0, and no TE/megatron-core).
3. **HF→Megatron conversion is via Megatron-Bridge** (`AutoBridge.from_hf_pretrained`; custom bridges registered in `model_bridges.py`). GDN is handled: Qwen3.5-35B-A3B routes to the **native `GPTModel` GDN path** (`language_model_only=True`; the wrapper explicitly manages GatedDeltaNet `cu_seqlens`, and *refuses* sample-packing/microbatch-padding-removal that would corrupt them). Registered bridges today: **GLM-4.7-Flash** (`Glm4MoeLiteForCausalLM`, DeepSeek-V3-like MLA+MoE) and **Qwen3.5 MoE/dense** (`Qwen3_5MoeTextForCausalLM` / `Qwen3_5TextForCausalLM`).
4. **Qwen3-Next-80B-A3B is NOT a registered megatron bridge.** Zero `qwen3_next` / `Qwen3Next` references in the megatron backend. The GDN+MoE *machinery* provably exists (Qwen3.5-35B), but the 80B's exact arch (512 experts top-10 + 1 shared, its GDN variant) has **no bridge and unconfirmed Megatron-core/Megatron-Bridge representability** → discrete authoring stage gated on a discovery spike.
5. **Our correctness-critical fork features do NOT exist upstream.** `gh` code-search on `NovaSky-AI/SkyRL`: `moe_router_replay`=0, `align_logprobs_by_token_ids`=0, `seq_mean_token_sum_norm_global`=0 (`routed_experts`=20, but those are megatron/MoE internals, not our R3 capture→replay). → R3 / TIS-exact / seqnorm-global must be **re-implemented against the megatron surface**; R3 today rides the FSDP2 grouped-GEMM swap + DTensor `[Shard_fsdp,Shard_ep]` expert layout, which has **no direct megatron-EP analogue** → highest-risk stage.
6. **Layout reorg is a first-order cost.** Upstream reorganized to `skyrl/backends/skyrl_train/…` (post-reorg); our fork is on the **pre-reorg `skyrl-train/skyrl_train/…`** layout. The megatron backend only exists in the post-reorg tree. This is the same divergence blocking our OPD upstream PRs.

---

## Stage map (dependency-ordered) — critical path marked ★

| Stage | Title | What | Feature(s) | Layer | Cost (CPU / 1-GPU / N-GPU) | Gate (GO/NO-GO) |
|---|---|---|---|---|---|---|
| ★0 | Megatron-Bridge GDN-80B coverage spike | Can Megatron-Bridge/-core represent Qwen3-Next-80B-A3B (GDN + 512-expert)? | model coverage (Q2) | model-bridge | CPU + optional 1-GPU tiny-load | Bridge resolves arch OR a concrete authoring path exists (analogous to GLM47/Qwen35 GDN); else migration = model-spec project or infeasible |
| ★1 | Adoption-strategy + reorg divergence map | Rebase-onto-post-reorg vs port-into-fork vs upstream-main+re-apply; map `skyrl-train/skyrl_train/`↔`skyrl/backends/` | adoption (Q1) | repo/layout | CPU | A chosen strategy + written migration path + feature-reapply cost estimate |
| 2 | gpu-rl-megatron image spike | Dockerfile variant: TE 2.11 + megatron-core + Megatron-Bridge + Ray 2.56, kaniko on CoreWeave | runtime/image (Q5) | build/infra | N-GPU build job | Image builds + `import megatron`/TE + trivial megatron init on 1 node |
| 3 | vLLM decision spike | Adopt vllm 0.23.0 vs keep fork `@76259c63` with megatron trainer; inventory fork-only vLLM features | vLLM (Q4) | inference-engine | CPU + read | Decision + compat matrix (megatron weight-sync surface × vLLM version × DCP/GQA-LSE/routed-experts-HTTP/MoE-w13) |
| ★4 | Backend coexistence wiring (flag-off byte-identical) | Wire `strategy=megatron` into our launcher/configs; `strategy=fsdp` default unchanged | global invariant | launcher/config | CPU + 1-GPU flag-off smoke | Flag-off byte-identical (all FSDP2 arms unchanged) + a megatron config renders |
| 5 | Small-MoE megatron smoke (30B full-attn) | Qwen3-30B-A3B (no GDN) on megatron via our agentic path, 2-node | trainer path | GPU | 2×8 H100 | gs1 completes, healthy metrics (entropy/grad-norm), no crash |
| ★6 | GDN-MoE megatron smoke (35B, registered) | Qwen3.5/3.6-35B-A3B GDN on native GPTModel path, 2-node — de-risks 80B before authoring | GDN path | GPU | 2×8 H100 | gs1 completes on GDN path (`language_model_only`), healthy metrics |
| ★7 | Feature parity on megatron: R3 / TIS / seqnorm | Re-implement each fork-only correctness feature vs megatron surface; per-feature sub-gate | R3, TIS, seqnorm (Q3) | trainer/loss/dispatch | 2×8 H100 | Each: flag-off byte-identical + parity vs FSDP2 (log-space TIS ratio, router-replay grad-norm sanity, seqnorm denom bit-match) |
| ★8 | Qwen3-Next-80B-A3B bridge authoring | Register `Qwen3NextBridge` (GDN + 512-expert), gated on Stage 0 verdict | model coverage | model-bridge | 1-GPU load + fwd | 80B loads via bridge + 1-step fwd-logprob parity vs HF/FSDP2 (tight fp32 tol) |
| ★9 | 80B end-to-end parity gate + verdict | Full 5D-parallel megatron 80B vs FSDP2-EP baseline; migrate/don't-migrate | parity (Q6) | full run | 16 nodes | Megatron 80B ≥ gs2 healthy AND (throughput ≥ FSDP2 OR clears an FSDP2 pain point) → documented verdict |

**Critical path:** 0 → 1 → 4 → 6 → 7 → 8 → 9 (with 2 gating all GPU stages, 3 feeding 4). Stages 2, 3, 5 can proceed in parallel once 0/1 return GO.

---

## Global invariants (assert in EVERY stage)

- **Flag-off / default-off byte-identical.** `trainer.strategy=fsdp` (default) is a no-op relative to today: no config churn reaches the FSDP2 arms, no import of megatron/TE on the fsdp path, existing YAMLs render identically. Upstream's `strategy` field makes this natural — keep it that way. First gate of every code stage is the flag-off byte-identical check BEFORE any behavior-on test.
- **Parity gate (load-bearing equivalence).** The megatron path must match the FSDP2 path where semantics are defined: forward-logprob parity at tight fp32 tol (bf16 cross-kernel diffs at the ~5e-2 floor are the quantization floor, not error — do NOT loosen a tol silently, per the CP Stage-2 discipline); TIS in **log space** (`policy/tis/imp_ratio_mean≈1.0`, `log_ratio_abs_mean` ~0.06 nats, NOT the benign `rollout_train_prob_diff_mean` linear-space artifact); seqnorm global denom `Z` bit-matched to the FSDP2 driver-computed value.
- **Regression bounds.** Do not break the dense-8B FSDP2 arms, the 30B/35B FSDP2-EP arms, or the agentic eval path. The running r4b and any in-flight FSDP2 job read the canonical clone — this plan never branches it.
- **Minimal diff.** No gratuitous API/config renames; reuse upstream's `MegatronConfig` fields rather than inventing parallel ones. Prefer upstream bridge patterns (register a bridge like `GLM47FlashBridge`) over bespoke conversion code.
- **RL correctness / no reward-hacking regression.** The migration must not weaken any ground-truth reward or verifier path; R3-replay faithfulness (the #6335 capture→train alignment guardrail) is a correctness invariant, not an optimization.

---

## Borrow map (don't reinvent — anchors are a 2026-07-13 dated read; RECONFIRM at impl time)

Upstream (`NovaSky-AI/SkyRL` `main`, post-reorg):
- `skyrl/train/config/config.py` — `trainer.strategy`, `MegatronConfig`, `MegatronDDPConfig`, `MegatronLoraConfig`, `DEFAULT_MEGATRON_OPTIMIZER_KWARGS`. **The coexistence-flag anchor.**
- `skyrl/backends/skyrl_train/workers/megatron/megatron_worker.py` — the megatron policy/ref worker (mirror of our `fsdp_worker.py`).
- `skyrl/backends/skyrl_train/workers/megatron/megatron_model_wrapper.py` — GPTModel output_processor hook (fused LM-head logprob), GDN cu_seqlens handling, `language_model_only` GDN routing, VLM asserts, `PolicyLossRegistry`.
- `skyrl/backends/skyrl_train/workers/megatron/model_bridges.py` — **bridge registration pattern** (`GLM47FlashBridge`, `Qwen35MoELMBridge`, `Qwen35DenseLMBridge`); the template for the Stage-8 `Qwen3NextBridge`.
- `skyrl/backends/skyrl_train/distributed/megatron/{model_utils,optimizer,optimizer_dtype,megatron_utils,packing_utils,fused_linear_logprob_triton}.py` — distributed + fused-logprob (Liger-style chunked / triton).
- `docker/Dockerfile.megatron` — image recipe (adapt for our kaniko/CoreWeave build).
- `examples/train/megatron/run_megatron_dapo_qwen3{.5_35b_a3b,_30b_a3b,_235b_a22b_lora}.sh` — parallelism-geometry references.
- `.claude/docs/backends/megatron.md` — Megatron-Bridge + parallelism-strategy notes; `NVTE_FLASH_ATTN=0` test requirement.

Ours (`marin-community/MarinSkyRL` `penfever/working`, PRE-reorg — the features to preserve):
- `skyrl-train/skyrl_train/model_wrapper.py` — FSDP2 model wrapper, GDN mask path (`_gdn_mask_fla_enabled`), attn-backend resolution.
- `skyrl-train/skyrl_train/worker.py` — TIS metrics emission (`align_logprobs_by_token_ids`), seqnorm-global (`compute_global_loss_denom`), `WORKER_FORWARD_ENTER`.
- `skyrl-train/skyrl_train/distributed/dispatch.py` — R3 `MeshDispatch` (by_value/resident/decentral transport, `_relocate_chunk_to_node`).
- `skyrl-train/skyrl_train/{trainer,fully_async_trainer}.py` — fully-async loop, `compute_global_loss_denom`, log-ratio diagnostics.
- `skyrl-train/skyrl_train/dataset/preprocess.py` — routed_experts dtype narrowing (num_experts→int16).
- `rl/cloud/launch_rl_iris.py` (OT-Agent) — the runtime-knob flags + `--daytona-api-key-env`; where a `strategy`/megatron-image selector wires in.
- `hpc/skyrl_yaml/iris/*.yaml` — config authoring rules (`n_concurrent_trials` ratio, `num_parallel_generation_workers: 128`, no container block).

---

## Safety / RL considerations
- **R3-replay faithfulness** is the correctness anchor (not perf): any megatron R3 re-impl must preserve the capture→train alignment (the benign-vs-broken router-replay signature: healthy `raw_grad_norm~4e-5` + `log_ratio_abs_max~0.2` vs broken `~1e5 / ~19`).
- **Cross-user / cluster safety:** all edits in the isolated clone → commit → push → cluster `git pull`; **never patch a cluster**; vLLM always **built from source**, never rsync'd. gpu-rl-megatron image built **in-cluster via kaniko** (Mac can't).
- **Guardrails unchanged:** ≤6 RUNNING RL jobs/cluster; `enable_db_registration:false`; never kill the running r4b or any RUNNING job without explicit permission. The parity runs (Stages 5/6/9) consume the RL job budget — schedule against it.

---

## Validation discipline (per stage)
Every code stage: **(1) flag-off byte-identical FIRST** (`strategy=fsdp` unchanged — the cheapest gate, catches import/config leakage), **(2) behavior-on** (megatron path runs), **(3) parity** (fp32 logprob `torch.allclose` at the bf16 floor; log-space TIS; seqnorm denom bit-match), **(4) GPU smoke** on the gpu-rl-megatron image with `NVTE_FLASH_ATTN=0` for tests. Name the measurement; never loosen a tol silently. Discovery stages (0, 1, 3) gate on a **written verdict + evidence**, not a passing test.
