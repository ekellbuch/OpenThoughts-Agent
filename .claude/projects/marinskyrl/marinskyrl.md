# MarinSkyRL — framework facts & gotchas

The RL training framework. Cluster runtimes/SIFs live in `.claude/ops/<cluster>/`; live
project trackers (rollout-fanout, PR tracking, log-ratio v3, ray-workercrashed) stay as
memories.

---

## Source of truth

- **SoT = `marin-community/MarinSkyRL` branch `penfever/working`** (strict superset of `main`).
- **`github.com/penfever/SkyRL` is OBSOLETE** (archived at `archive/skyrl-pre-marin-consolidation-20260608`, tip `5376d8f`). Every production feature it had (seqnorm-global loss, `policy_strict_spread_pg`, Stage-7 streamed-EP weight-sync, Qwen3-Next GDN kernel routing) is already in marin under squashed SHAs. marin *additionally* has TIS exact-alignment (`align_logprobs_by_token_ids`) + mp-backend the fork lacked. Do NOT merge the fork — it re-applies OLD copies (worker.py/trainer.py conflicts) and risks regressing the SoT.
- **Cluster clones** (all track marin `penfever/working`, editable-installed):
  - Jupiter: `/e/scratch/jureap59/feuer1/OpenThoughts-Agent/SkyRL` (editable from `.../SkyRL/skyrl-train`).
  - Leonardo: `/leonardo_work/AIFAC_5C0_290/bfeuer00/code/MarinSkyRL`.
  - Perlmutter / NYU Torch: realign when next used.
- **Sync:** commit+push from the laptop (Leonardo can't push), then `git pull` on the cluster. See `.claude/ops/jupiter/ENVIRONMENT_MAP.md` for baked-SIF facts.

---

## Config rules

### `n_concurrent_trials` vs `num_parallel_generation_workers` — heuristic, NOT a law

The `2 * num_parallel_generation_workers + 32` ratio is a rule of thumb, **not** a derived relationship. The two knobs are **independent** and govern different layers:
- `num_parallel_generation_workers` = trainer's `max_concurrent_generation_groups` bound (`fully_async_trainer.py:374`) + gen-buffer queue `maxsize`, constrained `≥ mini_batch_size`.
- `n_concurrent_trials` = the Daytona-sandbox TrialQueue semaphore, divided across `num_coordinators` (K) worker processes (`rollout_coordinator.py:117`).

The 2:1 idea only models head-plasma footprint + "keep engines fed"; it ignores `batch_size`, `n_samples_per_prompt`, and engine serving capacity. **Tune the two empirically + independently against the engine-saturation tuple (Waiting/KV/power/mem-BW), not locked at 2:1.** Do not auto-rewrite configs to this ratio.

### The engine "sawtooth" trough (Running→1) under fully_async is BENIGN backpressure while policy_train-bound

Per-engine `Running` sawtoothing PEAK(=engine cap)→TROUGH 1 with `Waiting=0` + KV near-floor (period ~30–90 s) is the gen-buffer's **backpressure working as designed** — NOT a supply bug, NOT parse-bound, NOT under-provisioned workers. **The one decider: `timing/wait_for_generation_buffer`.** ≈0 (e.g. 0.0009 s) ⇒ the training consumer NEVER waits for rollouts ⇒ generation is OFF the critical path ⇒ the trough is benign and the run is compute/policy_train-bound. Confirm it with: gen buffer sitting FULL at its cap most of the step (`Saved N generation buffer items`, N=`maxsize`=npgw), `staleness_mean` ≪ cap, `discard_rate 0` — the bulk of the `npgw` workers are parked in `await buffer.put()`; only a residual cohort drives the engines, and their agentic turns are phase-correlated within a group → instantaneous demand beats 1↔cap instead of averaging out. The weight-sync `pause_generation` barrier fires once per step (not at the sawtooth period).
- **`npgw < n_concurrent_trials` is NOT a starve.** `npgw × n_samples_per_prompt ≫ n_concurrent_trials` (e.g. 338×8=2704 ≫ 675) → the Harbor orchestrator can always be kept full; the submission/staleness gate is disabled, so the only live throttle is the buffer cap. **Raising npgw does NOT lift the trough** — generation is already ahead; more workers only deepen the buffer.
- **When a cheaper-training architecture flips the run inference-bound, the levers in order:** (1) `harbor.n_concurrent_trials` — the real engine-demand knob (subscription ≈ concurrent_trials × per-trial LLM duty cycle); (2) per-trial DUTY CYCLE — trials spend most wall-time in Daytona tool-exec, not awaiting the LLM (see §"Engine saturation in agentic fully_async RL" — the harbor poll-loop floor, fixed `ef42e75e`; + sandbox reuse). **NOT** npgw / `max_staleness_steps` / `max_buffered_groups` (those bound off-policy lead depth, not instantaneous engine subscription). Evidence: `agent_logs/2026-07-15_async-inference-sawtooth-rootcause.md`.

### Hydra struct — every run-YAML key must be declared in `ppo_base_config.yaml`

A run-YAML key not declared in the base struct `skyrl-train/skyrl_train/config/ppo_base_config.yaml` (at the launch's `--skyrl-ref`) is **REJECTED at config-parse**. OmegaConf loads the base in struct mode → `Could not override '<key>'. Key '<key>' is not in struct` → `HydraException` → driver exits 1 ~8–30 s after Ray-attach, **before `init_model`** (no NCCL, no training). Grep the finelog for `not in struct`.

- `yaml.safe_load` passing validates **syntax only**, not the struct. Before launching a config with any new/moved key, diff against the base: `git show <ref>:skyrl-train/skyrl_train/config/ppo_base_config.yaml` — every key you set must be declared there OR sit inside a passthrough (`{}`-typed) dict field.
- A knob needs BOTH a runtime reader AND a schema declaration. Adding only `.get("<key>", default)` makes it readable-if-present but **un-settable** (struct rejects at load before the `.get` runs). Example: `use_grouped_mm` reader without the `fsdp_config` schema decl (fixed `56cad1d7`, which declares `use_grouped_mm: false` in every `fsdp_config` block).
- **`optimizer_config` valid keys** = `optimizer, lr, adam_betas, weight_decay, max_grad_norm, offload_after_step, num_warmup_steps, scheduler, optimizer_kwargs`. Optimizer-specific params (e.g. `foreach: false`, the host-RAM CPU-Adam fix) go **nested under `optimizer_kwargs:`** (the `{}` passthrough), NOT bare — a bare `foreach:` is struct-rejected.

### `gradient_checkpointing_use_reentrant` — EVERY RL config must set it `true`

Non-reentrant (`use_reentrant: false`) uses PyTorch saved-tensor pack/unpack hooks; under **activation CPU-offload + SkyRL's async-actor threading** the thread-local hook stack push/pop mismatches → `SavedTensorHooks.cpp:69 INTERNAL ASSERT FAILED (is_initialized && !tls.stack.empty())` at the **gs1 backward** (pytorch#84864 / #90481). **Fix = `trainer.gradient_checkpointing_use_reentrant: true`** (reentrant checkpoint doesn't use the saved-tensor hooks). The `ppo_base_config.yaml` default is **STILL `false`** (`@56cad1d7`) — a config that forgets the line will hit the assert. Set it explicitly in every iris/jupiter RL YAML that uses cpu_offload.

---

## MoE + EP sharding — `fsdp_size` MUST divide `num_experts // ep_size`

When SkyRL shards the expert dim (dim-0 = `num_experts`) over BOTH the EP mesh axis AND FSDP:

> **`fsdp_size` must evenly divide `num_experts // ep_size`.**

For Qwen3-Next-80B-A3B: `num_experts=512`, `ep_size=8` → 64 experts/EP-rank. Valid `fsdp_size` ∈ {1,2,4,8,16,32,64}. **`fsdp_size=6` is INVALID** (64/6 uneven → [11,11,11,11,11,9]).

- **Failure signature:** completes rollout + all policy_train fwd/bwd, then dies at the FIRST Adam step with `RuntimeError: The size of tensor a (10) must match the size of tensor b (9) at non-singleton dimension 0` (`adam.py _single_tensor_adam`, `exp_avg.lerp_(grad)`) — a=even-PADDED FSDP2 local shard, b=UNpadded EP-backward reduce-scatter grad, disagreeing by one on a boundary rank. Deterministic, ~2.3h in.
- **Fix = FSDP=8** (64/8=8 even, also more sharding → fixes the OOM).
- **Guard:** `distributed/fsdp_utils.py` (`63cd2eb`) raises a fail-fast assertion at `create_device_mesh` if `(num_experts // ep_size) % fsdp_size != 0`.

---

## GDN (GatedDeltaNet) specifics

GDN layers exist in **Qwen3-Next-80B-A3B** (36 GDN layers) and **Qwen3.6-35B-A3B** (30 GDN layers). **Qwen3-Coder-30B-A3B is full-attention (no GDN)** — it does NOT hit any of the GDN issues below.

### 80B gs1 death = pure-torch GDN Python loop holds the GIL ~2h → SIGABRT. FIX = FlashQLA.

GDN runs the pure-torch chunk-recurrence `torch_chunk_gated_delta_rule` (`transformers/models/qwen3_next/modeling_qwen3_next.py:442`) — two nested Python loops (`:502` loop-carried state recurrence + `:486` intra-chunk), ~`S/chunk_size` iters each. At `max_prompt_length=98304`, chunk=64 → ~1536 iters/layer × 36 layers ≈ **55,000 sequential Python iterations per forward micro-batch**. This **holds the process GIL ~continuously ~2h** (forward) and again in the gradient-checkpoint backward recompute during `policy_train` → the c10d HeartbeatMonitor (needs the GIL to verify liveness) logs `watchdog got stuck … Could not acquire GIL within 300 ms` at 1× `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC` (3600s) and **fatally SIGABRTs at 2× (7200s)**. The `_ALLGATHER_BASE` "remote process exited" on peers is the CASCADE after rank0 aborted.

- **Both a stability bug AND the dominant throughput cost** — the ~2h forward is Python-dispatch-bound, not GPU-bound (the module self-documents the pure-torch path ~27× slower at S=8192; worse at 98k).
- The slow path is active because: `mask_fla()` forces the (broken) fla wheel OFF; the fused FlashQLA kernel is gated OFF (`SKYRL_GDN_FLASHQLA` unset); and `flash_qla` is not in the gpu-rl image.
- **Do NOT "fix" by raising `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC`** — it only masks the abort while leaving the 2h dominant cost; a genuinely GIL-locked process must not be tolerated. The 1× log looks benign but the 2× abort is real.
- **THE REAL FIX = the fused FlashQLA tilelang GDN kernel** (`SKYRL_GDN_FLASHQLA=1` → `engage_flashqla` → `_build_flashqla_chunk`, `models/qwen3_next_gdn.py`): one fused GPU kernel/GDN-layer (GIL-releasing) replaces the Python loop. Needs **`flash_qla 0.1.0+6ef4858` + tilelang 0.1.8 + apache-tvm-ffi 0.1.9 baked into the gpu-rl image, built for x86_64+sm_90** (Jupiter's `fla_tilelang_overlay.img` is arm64 — can't copy), THEN a CP1 logprob-parity smoke (fwd/bwd numerics change), THEN relaunch with `--gdn-flashqla on`.
- **Diagnostic distinction:** real wedge (SIGABRT + stale logs + GPUs 0%) vs healthy compute-bound backward (GPUs high, timers advancing). On Qwen3-Next the GIL-hold IS the bug to fix, not an artifact to tolerate.

### 35B GDN + Context-Parallel (CP>1) HARD-CRASHES at forward

The Qwen3.6-35B-A3B GDN layers have **no CP-aware kernel** → `context_parallel_size > 1` **hard-crashes at the forward pass** (not merely numerically wrong): `RuntimeError: The expanded size of the tensor (33784) must match the existing size (16892) at non-singleton dimension 3` in `FSDPPolicyWorkerBase.forward()` → `fwd_logprobs_values_reward` — 33784 = 2×16892 = the CP=2 sequence-shard doubling the GDN attention mask; dies at step-1 forward (0 training steps).

⇒ **the 35B (GDN) model is CP=1 ONLY**; its CP>1 configs (EP8×CP2, EP8×CP4, EP16×CP2) are infeasible-by-inference. A trainable 35B-CP config needs a CP-aware GDN kernel + a CP1-vs-CP2 logprob-parity smoke (`tests/gpu/test_cp_logprob_parity.py`). The Stage-2 attn-backend pivot (sdpa/flex under CP) covers the *full-attn* path — it does NOT make GDN CP-correct.

### grouped-GEMM path logging

`fsdp_config.use_grouped_mm: true` swaps MoE blocks to the grouped-GEMM path at MODEL-LOAD and logs `[MoE] grouped-GEMM swap active` once per policy worker. **That log proves the swap was ARMED, not that grouped_mm actually RUNS the forward** — with EP the expert weights are DTensors routing to `_run_experts_grouped_mm`, but a cpu_offload/DTensor-strip can silently fall back to an eager for-loop.

- The proof is the once-per-forward `[MoE-PATH] …` log (added `56cad1d7`), but it **emits LATE in the forward** (~1.5h into the 80B's ~2h `fwd_logprobs` forward — it fires at the expert-forward, deep in the forward, NOT at `WORKER_FORWARD_ENTER`/forward-start). A probe BEFORE the expert-forward runs sees the swap flag but no `[MoE-PATH]` yet — that is NOT a broken instrument, just too-early. To check grouped_mm on the 80B, probe AFTER the forward is well underway (≥~1.5h in).
- vLLM's GENERATION engines use their own FusedMoE kernels, separate from this training grouped_mm path — engine throughput says nothing about the training path.

---

## TIS + TITO

### `trainer.algorithm.tito_full` (token-in-token-out) — resolution order

Declared `null` in `ppo_base_config.yaml` (`56cad1d7`). Resolution (`generators/utils.py::_tito_full_enabled`): (1) env `SKYRL_TITO_FULL` wins if set; else (2) explicit `tito_full: true/false`; else (3) **auto → defaults to `use_tis`** (TITO-full ON whenever TIS is on). TITO-full assembles the training response from the exact sampled token-ids end-to-end (vs the LCS text round-trip), driving `generate/tis/lcs_fallback_fraction` → ~0 (residual ~12% is irreducible masked-context retokenization, benign; a HIGH rate = target corruption). Non-TIS runs byte-identical (the whole TITO branch is skipped).

### TIS exact-alignment (validated)

`align_logprobs_by_token_ids()` zips vLLM logprobs onto training tokens **by token id** (Harbor `completion_token_ids`); `extract_token_ids_from_rollout_details()`; float logprob format no longer disables TIS; LCS is last-resort and RECORDS every fallback in `AlignmentStats` → metrics `generate/tis/{exact_match_fraction,lcs_fallback_fraction,unaligned_fraction,alignment_fail_count}`; `worker.py` emits per-step `tis/{imp_ratio_mean,imp_ratio_capped_fraction,log_ratio_abs_mean}` **keyset-identical on every rank** (all_reduce-safe). vLLM needs NO change (already emits token_ids+logprobs over `/chat/completions`).

- Commits: MarinSkyRL `11285333` + `d32022ee`; harbor `8737426c` (per-turn logprob/token-id length-parity guard in `Chat._accumulate_rollout_details` → empty-list-on-mismatch keeps index alignment).
- At an on-policy step (`staleness_max=0`, `log_ratio_abs_mean=0.0`): `tis/imp_ratio_mean≈0.79`, `tis/log_ratio_abs_mean≈0.094` nats (= inherent vLLM↔FSDP bf16 precision gap, what TIS corrects — not misalignment), `imp_ratio_capped_fraction≈0`. Smoke config: `hpc/skyrl_yaml/jupiter/extra/tis_smoke_0p6b.yaml`.
- `concatenate_generator_outputs` must merge via token-weighted merge (not `get_rollout_metrics`, which is reward/len only and drops `generate/tis/*` on the fully-async path).

---

## Collectives — `strategy.all_reduce(status)` requires IDENTICAL keys on every rank

`DistributedStrategy.all_reduce(data: dict)` iterates `for k,v in data.items(): ret[k]=all_reduce(v)` — a separate NCCL all_reduce per key. If keys differ across ranks, the per-key calls don't line up → NCCL **watchdog timeout** and process-group abort.

- **Symptom:** `Watchdog caught collective operation timeout: WorkNCCL(SeqNum=N, OpType=ALLREDUCE, NumelIn=1, NumelOut=1, Timeout(ms)=600000)` → `NCCL communicator was aborted`. The **`NumelIn=1` scalar** is the giveaway.
- Fix pattern (`compute_log_ratio_diagnostics` v4, `69294ba5`): a `_log_ratio_diag_zero_metrics()` 16-key zeros fallback used on early return + try/except at the worker call site.
- **Rule whenever you touch `status` dict keys in `worker.py`:** (1) every rank contributes the SAME key set every iteration — no conditional skips; (2) data-dependent values that might fail get a sentinel (0.0/NaN) under the same key, never omitted; (3) wrap risky helpers in try/except with a full-keyset fallback; (4) unit-test the helper with empty / all-padded / normal input and assert `sorted(keys())` is identical.

### Do NOT fire a full-PG collective from INSIDE `ppo_train` under fully_async + R3-decentral (FIXED `68ea066e`)

Same `NumelIn=1` fingerprint, DIFFERENT mechanism than the keyset diff. Under **fully_async + R3-decentral** the policy ranks do NOT co-arrive at ppo_train's top-of-body — the staggered R3-chunk relocation (`MeshDispatch` derefs each dp group's `_relocate_chunk_to_node` ObjectRef at different times) means some ranks are IN ppo_train while others sit idle in asyncio `select` → a scalar all_reduce deadlocks. The keyset is UNIFORM — the bug is **collective PARTICIPATION desync**.

- **Fix (`68ea066e`, Option B, objective-preserving):** precompute `Z` on the **DRIVER, collective-free** (`ppo_utils.compute_global_loss_denom`: `Z = ranks_per_dp_group × nonzero-adv-count(full batch) × max_seq_len`, BIT-IDENTICAL to the summed reduce because MeshDispatch replicates each dp-chunk across its dp-group) in `trainer.train_critic_and_policy`, stash in `data.metadata["global_loss_denom"]`; the worker READS it (no collective), falls back to the legacy all_reduce only when the key is absent. Gated on `is_fully_async` → sync RL byte-identical.
- **Durable rule:** any NEW cross-DP reduction needed for the loss/optimizer step must be hoisted to a **guaranteed co-arrival point** (the driver, or a barrier every rank provably reaches) — NEVER fired inline from `ppo_train`/the async loss loop, which fully_async makes a non-co-arrival section. The forward (`fwd_logprobs_values_reward`) IS a co-arrival point (all ranks reach it); ppo_train's body is NOT.

---

## Runtime knobs — now FLAGS on the iris RL launcher (`rl/cloud/launch_rl_iris.py`)

The `SKYRL_*` runtime knobs are first-class CLI flags (argparse group "MarinSkyRL runtime knobs"). The env var is retained as an override — precedence is **env/`extra_env` > flag > code default**. Every flag defaults to *unspecified*, so an all-defaults launch injects `{}` and the pod env is byte-identical to before; a config's `extra_env:` still wins over a flag. Footgun defaults were flipped ON so a config that FORGETS the line is now safe.

- **GDN path — `--gdn-mask-fla {auto,on,off}` (env `SKYRL_GDN_MASK_FLA`), default AUTO.** Default-ON, auto-derived from the model arch (`model_wrapper._gdn_mask_fla_enabled`): the pure-torch GatedDeltaNet path (the `fla` wheel is broken) auto-engages for GDN archs (Qwen3-Next / Qwen3.6-35B-A3B's GDN layers) and is a strict **no-op on dense** models. Set `--gdn-mask-fla on/off` (or the env) to force.
- **TIS served-id splice — `--tis-splice {on,off}` (canonical env `SKYRL_TIS_SPLICE`), default ON.** Uses vLLM's raw served `completion_token_ids` as the generated region so TIS tier-1 exact-by-id alignment holds (closes the think-block re-tokenization divergence). The two old knobs (`SKYRL_TIS_SERVED_ID_SPLICE` generalized + `SKYRL_QWEN3_5_TIS_SPLICE` empty-think special case) → one `_tis_splice_enabled()` policy (both legacy env names still honored). Byte-identical on non-thinking turns.
- **NCCL timeout — `--nccl-timeout-s` (env `SKYRL_WORKER_NCCL_TIMEOUT_IN_S`), default 1800.** Single accessor `constants.get_worker_nccl_timeout_s()` / `DEFAULT_WORKER_NCCL_TIMEOUT_IN_S=1800`.
- **R3 transport — `--r3-transport {by_value,resident,decentral}`, default `decentral`** (folds `SKYRL_R3_RESIDENT` + `SKYRL_R3_DECENTRAL`); `--r3-put-timeout-s` = `SKYRL_DISPATCH_PUT_TIMEOUT_S`.
- **Correctness knobs (default-ON), each now a flag:** `--forward-dispatch-fix`, `--weightsync-drain-barrier`, `--cp-require-right-align`, `--w13-reload-bracket` (pass `off` only for an A/B). Observability: `--host-ram-monitor`(+`-interval-s`). Feature: `--gdn-flashqla`, `--ep-loader-chunk-rows`.
- **REMOVED as dead:** `SKYRL_FWD_UNSHARD_FENCE` (superseded by the drain barrier), the EPDIAG/`SKYRL_GROUPMM_DIAG`/`SKYRL_NUM_EXPERTS`/`SKYRL_EP_LOADER_DEBUG`/`SKYRL_ROUTER_REPLAY_DEBUG` diagnostic family, and `SKYRL_WEIGHT_SYNC_SERIALIZE`. Jupiter `extra/` configs still carry now-inert EPDIAG env lines (harmless — no reader).

---

## Known crashes & their fixes (factual)

### uvloop/libuv SIGABRT → force stock asyncio (BOTH driver + actors)

The libuv epoll SIGABRT under Daytona sandbox-teardown socket churn is fixed by **forcing CPython's stock asyncio SelectorEventLoop**, not by changing libuv. The version chase was a dead end (uvloop 0.19→0.22.1, custom 1.49 — all still abort; `UV_USE_IO_URING=0` doesn't gate it). Ray installs uvloop in every worker via `try_install_uvloop()` (gated by `RAY_USE_UVLOOP`, default True); the SkyRL orchestrator is RTT-bound, so uvloop's throughput edge is moot.

- **Driver fix:** reset the policy at the top of **`BasePPOExp.run()`** (`main_base.py`), before its `asyncio.run()` calls — SkyRL `77fb0074`:
  ```python
  import asyncio
  asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
  ```
  **Placement is the trap:** must be on the SHARED `BasePPOExp.run()`, NOT a per-example `skyrl_entrypoint` (there are 26+; terminal_bench jobs use `TerminalBenchExp` which doesn't override run()).
- **Actor fix (SkyRL `9e04851`):** the driver reset does NOT protect Ray ACTOR processes. Also set `env_vars["RAY_USE_UVLOOP"]="0"` in `prepare_runtime_environment` (utils.py) so `try_install_uvloop` is a no-op in every worker/actor. **Use BOTH.**
- `set_event_loop_policy()` is deprecated Py3.12+; when removed, switch to `asyncio.run(coro, loop_factory=asyncio.SelectorEventLoop)`.
- Related but SEPARATE bug class: the refcount-SIGABRT fix (harbor `ec508562` orphan-task reap + SkyRL `3b1708a0` gc backstop) on AgentTimeout-heavy datasets.

### MoE-RL first-step forward wedge → R3 transport (resident → decentral)

The MoE agentic-RL wedge at the first training-step policy forward (R3 shipped BY VALUE through every per-forward Ray task arg → Ray object-store spill → silent hang, a Ray-store wait **NOT** an NCCL watchdog).

- **Root cause:** `rollout_routed_experts` `[B,seq,L,K]` was shipped BY VALUE through every per-forward Ray task arg — ~3.0 GB/dp-chunk even at uint8 (~24 GB at int64), and the naive dispatch loop re-serialized it per actor (~16×/dp-group) → 96 GB plasma cap overflow.
- **Fix (`ac4b3806`) — BOTH levers load-bearing:** (1) `dataset/preprocess.py` narrows routed_experts to uint8/int16 keyed to num_experts → int16 for Qwen3-Next's 512 (`39faff7d`; uint8 is UNSAFE at >255 experts → truncate + rank-divergent → NCCL size-mismatch hang); (2) `distributed/dispatch.py MeshDispatch.dispatch` `ray.put`s each dp-chunk ONCE + shares the ObjectRef across the dp-group (`SKYRL_R3_RESIDENT`, marker `R3_RESIDENT_SET`). uint8 alone would NOT fit.
- **The "worker-resident" title is a MISNOMER:** `ray.put` runs on the **DRIVER (head node)**, so R3 lands in head plasma — just deduped to 1 copy/dp-group. The driver-centralization re-manifested at 80B.
- **The REAL structural fix = `SKYRL_R3_DECENTRAL` (default ON as of `e9b2f10d`):** route routed-experts **generation-worker → node-resident consumer** so the head holds ~0 R3 (O(1) in model scale). `_relocate_chunk_to_node` re-puts the chunk with a `NodeAffinitySchedulingStrategy` (soft) pass-through, byte-identical, bounded-put fallback. Marker `R3_DECENTRAL_SET method=ppo_train …`. Set `=0` to force the old centralized driver-put for A/B.
- **Config-only mitigation:** `trainer.fully_async.max_buffered_groups` knob (`8b065938`; default None ⇒ = num_parallel_generation_workers, byte-identical); **`num_parallel_generation_workers: 128` across ALL iris configs** caps buffer + worker-held groups → footprint O(1) in model scale, fits the 96 GB store. The `OT_AGENT_RAY_OBJECT_STORE_CAP_GIB=160` band-aid was reverted.
- **A generation "worker" owns a GROUP, not a rollout** (`_run_generate_for_a_group_loop`; `max_concurrent_generation_groups = num_parallel_generation_workers`; a `GeneratedOutputGroup` = one prompt × n_samples, ~126 MiB with R3 at 80B). Workers ≫ vLLM working set (`num_inference_engines · max_num_seqs / n_samples`) only accumulate STALE groups — bound it, don't max it.
- **The dp=1 put stall (gs1) loud-fails via `DispatchPutTimeoutError`** (`d13c3586`, 600s bounded-put) at global_step 1 — turned the silent 4.8h NCCL-watchdog wedge into a 10-min failure. NOT a GDN-kernel deadlock; `del`/`ray free` after dispatch does NOT help (the chunk is pinned by the LIVE `forward.remote()` task).

### MoE served-policy token-salad on the RL update path → `SKYRL_W13_RELOAD_BRACKET` (FIXED `2bb70a88`)

A served MoE policy emitting incoherent CJK token-salad (100% reward-0) after a disaggregated weight sync = the FusedMoE **`w13` gate/up halves are in the wrong kernel order**. vLLM's initial from-disk load runs `process_weights_after_loading`, which for FusedMoE under **FlashInfer-CUTLASS / TRTLLM** (auto-selected on H100) applies `swap_w13_to_w31` (`[gate;up]→[up;gate]`). The RL update path did per-chunk `model.load_weights` with **no finalize**, reverting `w13` to checkpoint `[gate;up]` → the kernel reads the wrong halves → salad. **TRITON / AITER backends do NOT swap** → the same skip is harmless there.

- **Env var `SKYRL_W13_RELOAD_BRACKET`** (default `1`): brackets the multi-chunk sync in `fsdp_worker.broadcast_to_inference_engines` with `WorkerWrap.skyrl_begin/finish_weight_reload` (= vLLM `initialize/finalize_layerwise_reload`) so `process_weights_after_loading` runs **exactly once** post-sync. **Swap-inert on triton/dense → byte-identical there**, so leave it on.
- Scope: only the non-IPC, non-`_fuse_weights` broadcast path is bracketed. Diagnosis was MoE-specific × FlashInfer-CUTLASS × disaggregated-RL-update — NOT NCCL, NOT the gather, NOT placement.
- **Bring-up check:** engine log shows `initialize_layerwise_reload` / `finish_weight_reload`.

### 80B placement init-OOM = two-PACK-PG race → `policy_strict_spread_pg`

Qwen3-Next-80B-A3B init-OOM was a **two-PACK-PG race** (inference PG + lazy policy PG, no anti-affinity, exactly-full 24 nodes → a policy worker lands on a vLLM-occupied GPU), **NOT** a ref-model issue (ref is correctly not instantiated, `use_ref_model=False`). Fix = opt-in **`policy_strict_spread_pg`** flag (SkyRL `6e3afc34`, OT-Agent `96df706f`): reserves the policy PG up front with STRICT_SPREAD. Default-off → all other RL byte-identical. The 80B yaml `hpc/skyrl_yaml/jupiter/extra/128GPU_qwen3_next_80b_a3b.yaml` sets it true.

### 80B gs1 optimizer-step NCCL hang (NEW code path)

At EP8×FSDP8×CP1, the FIRST backward+optimizer at 80B hangs at the **gs1 OPTIMIZER step** (`policy_train`): a `default_pg` ALLREDUCE `SeqNum=288606 NumelIn=1` (a barrier/grad-norm reduce) hangs → timeout → `mesh_fsdp _ALLGATHER_BASE SeqNum=6936` timeout → SIGABRT on FSDP policy worker → raylet worker-death cascade (failure_count stayed 0 → holds the gang wedged indefinitely; state-poll "running/16pods" is the colocated-vLLM-engine deception, not real progress). global_step never reaches 1. This is a NEW/untested code path, not an R3-store regression. (R3-store overflow is CLEARED — zero `DispatchPutTimeout`/`ObjectStoreFull`/`_ray_put_bounded` at any forward across the full run.)

---

## Resume / checkpoints

### Checkpoint pathing — ckpts are NESTED at `<rundir>/<job_name>/checkpoints/`, NOT the rundir top-level

```
<rundir>/                              # configs/ exports/ logs/ ray_logs/ sbatch/ wandb/ + an EMPTY top-level exports/
└── <job_name>/                        # the run subdir (same name as rundir leaf)
    ├── checkpoints/                   # ← FULL RESUMABLE CKPTS LIVE HERE (trainer.ckpt_path)
    │   ├── global_step_2/  global_step_4/  ...   # each: policy/ (FSDP shards) + trainer_state.pt + data_consumption_state.pt + generation_buffer_state.pt (~34MB)
    │   └── latest_ckpt_global_step.txt           # ← the step the restart resumes from
    ├── exports/                       # HF-format exports (trainer.export_path; cadence = hf_save_interval)
    └── trace_jobs/                    # per-rollout traces (HUNDREDS of thousands of files — NEVER find/du it)
```
- **Cadences are independent:** `trainer.ckpt_interval` (full resumable, e.g. 2) vs `trainer.hf_save_interval` (HF export, e.g. 5). Rendered values live in `configs/<job>_rl_config.json`.
- A glob of the rundir top-level finds nothing — the top-level `exports/` is empty. To check resume state: `cat <rundir>/<job_name>/checkpoints/latest_ckpt_global_step.txt`.
- Each ckpt persists `generation_buffer_state.pt` (the async rollout buffer), so `resume_mode=latest` restores the buffer too — relevant if a hang is *in* the buffer state (resume can re-trigger it).

### Resume overshoots `max_steps` — a step past the data ceiling is spurious

For the pymethods2test-large RL family: `epochs=2` × dataset/`train_batch_size=64` = exactly **80 optimizer steps**; configs set `max_steps=80` because that's the data ceiling. With `resume_mode=latest` + chained restarts, `global_step` does **not** hard-stop at 80 — it **overshoots** (observed 86). Steps 81→86 are **spurious** (re-runs exhausted data, "eternal-retry").

- A run reaching **step ≥80 is COMPLETE, not failed** → RL Cleanup Checklist, NOT fix-and-requeue.
- **Best-checkpoint selection must CAP candidates at step ≤80** (use ≤78 if a step-79+ greedy/eval-pass reward jump is present — that's an eval-checkpoint artifact, not learning).
- Errors in the 81–86 tail are noise (e.g. `VLLMValidationError: 32769 > 32768` near-budget BPE +1) — don't mis-diagnose.

### a3 RL resume (`--dry_run` regenerates the dedup config — RESOLVED `0b01a273`)

The launcher now (1) auto-resumes from the canonical run dir's `checkpoints/global_step_*` when present and `--overwrite_output_dir` was NOT passed (pins `ckpt_path`/`export_path`/`resume_mode=latest` as last-wins hydra overrides), and (2) routes `--dry_run` to a `<name>__dryrun` sibling so it can't seed the real dedup config. Pass `--overwrite_output_dir true` to force a fresh fork. (a3 series is CONCLUDED — no relaunch.)

---

## 80B RL is TRAINING-bound, and SkyRL FSDP is ALWAYS cross-node

The Qwen3-Next-80B-A3B production RL step (EP=8×FSDP=8, 32k, R3+TIS) is **training-bound, NOT gen-bound**. Measured step ≈ 17,000s (~4.7h): `policy_train` ~48%, `fwd_logprobs_values_reward` ~31%, `sync_weights` ~13%, `wait_for_generation_buffer` only ~7.5%.

- **SkyRL FSDP is cross-node regardless of `fsdp_size`.** `create_device_mesh` (`fsdp_utils.py:814`) builds the mesh with **`ep` innermost/contiguous** (`mesh_shape=(ddp,fsdp,ep)`), so on 4-GPU/node Jupiter an FSDP group = 1 GPU per node spanning `fsdp_size` nodes; EP is the intra-node dim. The yaml comments claiming FSDP is "intra-node" are FACTUALLY WRONG. The ordering is deliberate for a **correctness** reason (fsdp must precede ep so the composed expert DTensor `[Shard_fsdp, Shard_ep]` slices ascending; reverse → `KeyError: Mesh dim indices should be in ascending order`), NOT topology.
- **EP=16×FSDP=4 does NOT restore intra-node FSDP** — launchable but throughput-neutral-to-worse (FSDP stays cross-node; EP all-to-all widens 8→16). Do NOT switch EP/FSDP for the speed objective.
- **The real speed lever** = make FSDP intra-node via a mesh-dim-reorder CODE change (fsdp last, working around the DTensor ascending-slice constraint) — attacks the dominant 48% `policy_train`. Secondary: the 13% `sync_weights` (`broadcast_to_inference_engines` full_tensor gather) and 31% `fwd_logprobs`.

---

## FSDP2 Context-Parallel Stage 2 — attn-backend pivot

Branch `feuer/fsdp2-cp`. Added `trainer.attn_backend ∈ {auto,flash_attention_2,sdpa,flex}` (default `auto` = byte-identical to pre-Stage-2). `model_wrapper.py`: guarded flash import (`_HAS_FLASH` + shims that raise only if called) + `resolve_attn_implementation(...)`; CP (`context_parallel_size>1`) forces sdpa/flex, rejects flash varlen. Wired through policy/critic/ref. Tests: `test_attn_backend.py`, `test_sdpa_flash_parity.py`.

- **Parity:** sdpa@fp32 vs eager@fp32 logp 2.29e-3 (tight = correct); bf16 cross-kernel diffs (5e-2) are the bf16-quantization floor, not flash error.
- **GOTCHAS for GPU CP runs:**
  - **`/opt/SkyRL` baked-module shadow:** the SIF bakes SkyRL at `/opt/SkyRL`; bare `python script.py` imports it (not a worktree clone) → new kwargs silently ignored via `**kwargs`, both arms fall to eager (false pass). FIX: `apptainer exec --env PYTHONPATH=<worktree>/skyrl-train`; the GPU test asserts `model.config._attn_implementation` actually engaged.
  - Triton JIT gcc `-l:libcuda.so.1` link fail on compute node: `--env LIBRARY_PATH=/.singularity.d/libs`.
  - HF offline: prefetch into `/e/scratch/jureap59/feuer1/hf_cache`, `HF_HUB_OFFLINE=1`; `-p no:cacheprovider --confcutdir tests/cpu/<sub>` dodges the session-autouse `ray_init()` login-node hang.

---

## Engine saturation in agentic fully_async RL — how to READ it

The vLLM inference engines are starved by **HARBOR'S CLIENT-SIDE 1-SECOND POLL LOOP**, not Daytona, not an intrinsic agentic duty cycle, not the engine count. **Raising `n_concurrent_trials` (n) cannot help** — doubling n=128→256 stayed STARVED cleanly (tok/s FLAT). Per-turn (mean 101.4 s): generate 21.2 s (21%) + agent-requested command 0.7 s (<1% intrinsic) + **RESIDUAL 79.5 s (78%)**.

- **Direct measurement:** Daytona is FAST from CoreWeave (full `exec` 0.15 s median, each HTTP call ~0.03 s, sandbox create ~0.05 s, bare API RTT ~0.15 s — FASTER than the Mac's 0.78 s). CoreWeave↔Daytona network + Daytona control-plane are BOTH REFUTED (network ≈ 2–4 s/turn total).
- **The residual is harbor's `_poll_response` `asyncio.sleep(1)` loop** (`harbor/src/harbor/environments/environment.py:_poll_response`): per-`exec` ≈ `ceil(command_runtime)` + ~0.14 s network — every non-instant command is floored at ≥1 s. With ~15–24 execs/turn on ~3–4 s commands → 52–84 s/turn ≈ the observed 79.5 s. The "Daytona exec round-trip latency" label was a MISATTRIBUTION — it is client-side poll rounding.
- **FIX (harbor-side, shipped):** (1) poll loop → `sleep(0.001 + random.uniform(0,0.001))` (harbor `ef42e75e`); (2) setup-timeout → tmux/asciinema baked into the snapshot via `_AGENT_TOOLING_LAYER`/`bake_agent_tooling()` in `snapshots.py` (harbor `0729a3e9`), so episodes no longer apt-install/compile per-episode. **Measured payoff: `agent_setup` ~401 s → mean 0.41 s (~1000×); `AgentSetupTimeout` 63% → 0%.** Re-minting the snapshot requires manually deleting the stale `harbor__<hash>__snapshot` first (env-dir hash is unchanged by a library-code change).
- On CoreWeave harbor is baked NON-editably into the gpu-rl image → harbor fixes need a kaniko rebuild + digest bump (SLURM harbor is editable → live on `git pull`). Build the gpu-rl image MULTI-LAYER (`SINGLE_SNAPSHOT=0`, max layer <8 GB) — a single >8 GB layer can't cold-pull over CW→ghcr.

### How to read saturation — `nvidia-smi` SM-util% is a TRAP

A point-in-time `nvidia-smi` SM-util reads **84–89% even when the engine is idle-waiting** — it only means "a kernel was resident during the sample interval." **Do NOT trust SM-util% alone.** The trustworthy saturation tuple is: **`Waiting`-queue depth (vLLM scheduler) + GPU KV-cache usage + power draw + memory-BW util**, cross-checked against nvidia-smi.

- **Saturated ⇔** `Waiting > 0` AND KV usage well off the floor AND power near TDP (~700 W H100) AND mem-BW util high.
- **Under-fed ⇔** `Waiting = 0` always (engines instantly drain what trickles in) + KV ≈ 0 + power ≈ ⅓ TDP + mem-BW 2–5%.
- `Σ Running` (concurrent-request count) is a **valid** proxy. Use KV/power/mem-BW, not SM-util%.

### Daytona ORG routing

The agentic-RL grid must run on the **dedicated RL org (`DAYTONA_RL_API_KEY`, DataCompRL)**, NOT the shared `DAYTONA_API_KEY` (eval/datagen) org — the shared org's control plane self-saturates and throttles the eval campaign. The iris RL launcher re-sources `secrets.env` after any shell `export` (`hpc/iris/env.py`: file overrides shell), so a pre-launch `export DAYTONA_API_KEY=$DAYTONA_RL_API_KEY` is CLOBBERED — use the committed **`--daytona-api-key-env DAYTONA_RL_API_KEY`** flag (`rl/cloud/launch_rl_iris.py`, c6001bc1). VERIFY in-pod: `kubectl exec … printenv DAYTONA_API_KEY | sha1sum` == the RL key hash (NOT the shell before launch).

---

## CI — the `SkyRL-GPU-E2E-CI` convergence test is DISABLED; RESTORE when Marin CI can substitute

The daily-scheduled `SkyRL-GPU-E2E-CI` (`.github/workflows/gpu_e2e_ci.yaml` → `Basic convergence test`) `anyscale job submit`s to **upstream SkyRL's Anyscale account**, which the `marin-community/MarinSkyRL` fork has no `ANYSCALE_CLI_TOKEN`/`WANDB_API_KEY` for → every scheduled run fast-fails (~30s) at CLI auth (NOT a signal about our changes). The `schedule:` cron is disabled (kept `workflow_dispatch`); the disable must land on **`main`** to take effect. **RESTORE** the `schedule:` trigger once a Marin-owned GPU-E2E CI exists.

---

## Diagnostics — how to read metrics

### `policy/rollout_train_prob_diff_mean` in the MILLIONS is a BENIGN SCALE ARTIFACT

Seen `policy/rollout_train_prob_diff_mean ≈ 7.5e6` on MoE grid runs — looks alarming but is a diagnostic-metric artifact; training is correct, R3 replay is faithful. Root: `trainer.py:~1521` computes `prob_diff = (rollout_logprobs[mask] − action_log_probs[mask]).exp().abs()` then `.mean()` — i.e. **`E[exp(logprob_diff)]` in LINEAR space**, which any fat tail dominates (a single token → millions; that's the std≫mean signature). Masked correctly (response tokens only), just exp-then-linear-mean.

- **The ACTUAL gradient signal is the log-space, aligned, cap-2.0 TIS path:** `policy/tis/imp_ratio_mean≈1.017`, `imp_ratio_capped_fraction≈1e-4`, `log_ratio_abs_mean≈0.06 nats`, `log_ratio_abs_p99≈0.21`, `generate/tis/exact_match_fraction≈0.99`, `raw_grad_norm≈4e-5` (stable — a BROKEN MoE router-replay instead gives `log_ratio_abs_max~19, policy_loss~1e4, raw_grad_norm~1e5`, per `trainer.py:1406-1417`). **Correctness check = the `policy/tis/*` + `raw_grad_norm` metrics, NOT `rollout_train_prob_diff_mean`.** Listed benign in the `monitor-job-tables` skill.

### Rollout/generate path is uniformly `AttributeError`-guarded

The ENTIRE agentic rollout/generate path is robustly guarded against `AttributeError`:
- `examples/terminal_bench/terminal_bench_generator.py`: `_process_trial_result` wrapped in `except (KeyError, AttributeError, TypeError)` (~L1284); an outer handler (~L799, `fb102ed`) catches errors raised DURING processing (e.g. jinja2 `TemplateError`) and coerces them to masked.
- `skyrl_train/generators/utils.py`: `get_response_ids_and_loss_mask_from_messages` (1031); `extract_{logprobs,token_ids,routed_experts}_from_rollout_details` (757-895) are None/dict/object-safe via `getattr`+`isinstance`.
- harbor `agents/terminus_2/terminus_2.py` + `llms/lite_llm.py`: `LLMResponse` fields all declared; response parsing `getattr`/`.get`-safe.

**Heuristic:** a deterministic rollout `AttributeError` is almost certainly raised inside a rollout DEPENDENCY (litellm / openai-SDK / daytona / uvloop) under the image's package pins, returned as an Exception from `asyncio.gather` and classified to `generate/errors/AttributeError`. Treat it as an IMAGE/env issue (a deps rebuild likely fixes it), NOT a MarinSkyRL/harbor source bug, unless a verbatim traceback points at first-party code. Confirm on the rebuilt gpu-rl image with rollout logs captured **live, in-window** — CoreWeave finelog retains only the init-phase log post-mortem.

### Rank-0 logging — "only rank 0 logged X" ≠ "only rank 0 RAN X"

Most skyrl-train worker diagnostic logs are **gated to rank 0** — `if torch.distributed.get_rank() == 0:` (e.g. `init_weight_sync_state`'s lines, `worker.py:452-463`) or tqdm `disable=not self.strategy.is_rank_0()` (`worker.py:1038,1533`). The collective itself runs on **all** ranks; only rank 0 emits the message. The iris finelog AGGREGATES every Ray actor's stdout into one stream tagged by actor `ip=`/`pid=`. **You CANNOT infer "only rank 0 reached code X" from "only rank 0 logged X" for a gated log.**

- Per-NODE pod logs (`pod_rank1..N`) only show the `start_rl_iris_controller` bootstrap — the actual rank-actors run on those nodes but log to the HEAD/finelog, so per-pod logs are NOT per-FSDP-rank views.
- **The one RELIABLE per-rank signal: `WORKER_FORWARD_ENTER rank={self._rank}` (`worker.py:534`) is deliberately UNGATED** — every rank that reaches `worker.forward` prints its own rank. So "only rank 0 printed `WORKER_FORWARD_ENTER`" genuinely DOES mean only rank 0 reached the forward. Trust THIS marker; do not trust the gated ones.
- **To localize a per-rank hang reliably:** (1) use UNGATED instrumentation (log `self._rank` unconditionally); (2) capture per-rank FR dumps for ALL pods, not just rank 0 (`TORCH_NCCL_DEBUG_INFO_TEMP_FILE` writes `/tmp/nccl_fr_rank<N>` on every node — `kubectl cp` from each pod BEFORE the kill reaps them); (3) per-rank faulthandler/SIGUSR1 py-stacks. A SUCCESSFUL run's logs make the best diff.
