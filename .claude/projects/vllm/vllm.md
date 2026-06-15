# vLLM fork — dependency overview

Our **`mlfoundations/vllm`** fork: the OpenAI-compatible inference engine for RL rollouts, datagen, and
eval (spawned by `hpc/vllm_utils.py`). Written 2026-06-14 from notes + the local fork
(`/Users/benjaminfeuer/Documents/vllm`). Two divergences from upstream carry the project: **R3 routed-experts
capture** and the in-progress **DCP GQA-LSE fix**.

> **vLLM is our fork (`mlfoundations/vllm`, own upstream) — the local clone is ground truth.** Unlike
> Harbor/SkyRL/OT-Agent (editable + `git pull`), vLLM is **compiled**, so it's **built from source on each
> cluster (per-arch) from the committed fork**, or **baked into a SIF** from a committed commit. Edit the
> fork locally → commit → push → build on the cluster from that commit. **Never** rsync working-tree edits
> or hand-patch a cluster (no patch-by-rsync). Every cluster keeps at least one env with our fork built for
> it; some envs may run **vanilla** vLLM, which is fine. Version-bump the fork only when necessary.

---

## Version lines we run

| Line | torch | Runtime | Notes |
|---|---|---|---|
| **vLLM 0.16.0** | 2.9 | RL venv + `*_r3baked.sif` (Jupiter) | the dense-RL + MoE/80B stack; carries the R3 patch |
| **vLLM 0.20.2rc0** | 2.11 | `skyrl_megatron_vllm0202rc0_r3.sif` + otagent | the new SIF; R3 upstreamed; getting the DCP fix |

- **The router patch is commit-sensitive on the 0.16 line:** `084aa19f0` is the newest torch-2.9.1-pinned fork commit that still carries the `routed_experts` **RL-emission** path — later commits bump torch (2.10→2.11). The older torch-2.9.1 bump predates the patch (has routed_experts only in upstream MoE infra, NOT the RL emission). Verify a build by grepping `gpu_model_runner`/`scheduler`/`output_processor` for the emission path, not just `vllm.__version__` (which reports `dev`/`0.1.dev…`). See `.claude/ops/jupiter/ENVIRONMENT_MAP.md` §0 (torch is the reliable discriminator).

---

## R3 routed-experts capture (the MoE router-replay transport)

Lets RL replay MoE routing: vLLM serializes which experts each token routed to, over `/chat/completions`.

- **Flag:** `enable_return_routed_experts` (engine `--return-routed-experts`); default off.
- **Protocol:** a top-level choice field `routed_experts`, shape `[gen_len, num_layers, top_k]` (int) — harvested like `token_ids` (litellm → `provider_specific_fields` → Harbor `RolloutDetail.extra["routed_experts"]` → SkyRL `extract_routed_experts_from_rollout_details` → `router_replay.py`/`moe.py`, which asserts `shape[-1]==top_k`).
- **Files:** `config/model.py` (flag), `model_executor/layers/fused_moe/{routed_experts_capturer.py,layer.py}` (GPU capture buffer + D2H copy), `v1/worker/gpu_model_runner.py`, `v1/core/sched/scheduler.py`, `v1/engine/output_processor.py`, `entrypoints/.../chat_completion/serving.py` + `protocol.py` (serialization), `outputs.py`.
- **Qwen3-Next gotcha:** the Ray **Compiled-DAG** backend deadlocks on the hybrid arch when capture is on → run with the **mp executor backend** (`generator.inference_engine_mp_backend: true`), validated clean. Plus an undersized hybrid-kv-buffer fix + defensive clip (`gmr_fix`/`scheduler_fix`/`capturer_fix` single-file binds). Full detail in `.claude/projects/marinskyrl/marinskyrl.md` / `.claude/ops/jupiter/ENVIRONMENT_MAP.md`.
- **Status:** RESOLVED on the existing prod SIF (no rebuild) — only `enable_return_routed_experts=False` ever blocked it. The FSDP2 router-replay hook exists and ran a full GRPO backprop step on the 80B (do NOT repeat the "Megatron-only, no FSDP2 replay" claim).

---

## DCP GQA-LSE fix (active — branch `feuer/dcp-gqa-lse-fix`)

Decode-Context-Parallel shards the decode KV cache across ranks to cut KV memory; there's a genuine math
defect recombining multi-rank attention outputs + log-sum-exp (LSE) **under grouped-query attention (GQA)**.

- **Signature:** ~0.1 max|logprob Δ| + argmax flips after a short matching prefix; error grows as sharded KV history grows; **GQA-specific** (MHA/MLA unaffected); identical across runtimes (so it's the math, not a version mismatch).
- **Suspected locus:** the query all-gather (`dim=1`) + LSE all-gather (`dim=0`) + reduce-scatter (`dim=1`) orientation/head-accounting in `v1/attention/backends/flash_attn.py` (`_forward_with_dcp`, the `context_lse.transpose(0,1)` handoff) and the cross-rank LSE math in `v1/attention/ops/common.py` (`cp_lse_ag_out_rs`, the Triton rescale kernel). Anchors drift — reconfirm at impl time.
- **Staged plan** (`notes/vllm/stage{0..6}_*.md`): 0 repro harness → 1 root-cause → 2 the fix → 3 unit parity (`dcp=N`==`dcp=1`) → 4 e2e rollout parity (Qwen2.5-1.5B tp4 dcp2 vs dcp1 → greedy **bit-identical**, logprobs atol 1e-2 — the ship gate) → 5 regression (MLA + GSM8K unaffected) → **6 merge to `v2-migration` + build-from-source deploy on each cluster, NO upstream PR** (per the 2026-06-13 revision). Invariants: G1 `dcp=1` byte-identical, G2 the parity gate, G3 no MLA/GSM8K regression, G4 minimal diff.
- After it lands, resume the MarinSkyRL **rollout-DCP** workstream (task #222): Stage-3 parity should flip to PASS, then long-ctx OOM→OK.

---

## Build + branches

- **From-source build env:** `SETUPTOOLS_SCM_PRETEND_VERSION=<ver>` (required when building from a source tree without `.git` — setuptools-scm can't derive `_version.py`), `MAX_JOBS=<N>`, `TORCH_CUDA_ARCH_LIST="9.0"` (GH200/H100) / `8.0` (A100), built against the SIF's own torch for ABI match (~60–75 min compile). Full recipe + the GCC/PATH scrubbing gotchas are in `.claude/ops/jupiter/ENVIRONMENT_MAP.md` §2c (the `skyrl_megatron_vllm.sif` build notes).
- **Branches on the fork:** `v2-migration` (the 0.20.2rc0/torch-2.11 mainline; R3 present; the DCP-fix merge target), `feuer/dcp-gqa-lse-fix` (active, Stage 0–2 instrumentation commits), plus older debug branches (`penfever-debug-layer-split-v0.16.0`, `dp1-debug-instrumentation-*`). Local tree currently on `feuer/dcp-gqa-lse-fix`.
- **0.20.2rc0 SIF gotchas** (run-time, from the env map): set `VLLM_USE_FLASHINFER_SAMPLER=0` (SIF has no flashinfer), `LIBRARY_PATH=/.singularity.d/libs` for tp>1 Triton linking, and `VLLM_ATTENTION_BACKEND` is ignored on 0.20.2rc0.
