# 2026-06-20 — #232 CP2 smoke FINALLY reached gs1: new blocker = CP ring-SDPA "No available kernel"

## TL;DR
The `rl_30b_a3b_32k_cp2_SMOKE` chain (latest job **932229**, dir `…/rl_30b_a3b_32k_cp2_SMOKE_6`,
FAILED 2026-06-20T11:27:20, 30:46 elapsed) **cleared every prior pre-gs1 blocker** (batch-size
asserts, Ray-bootstrap hangs, Daytona/FD rollout blips) and **reached global_step 1** — the first
time the CP-MoE forward/backward path actually executed. The two deployed SkyRL CP-MoE fixes
(`b02d758` arch-gate dict `attention_mask=None`; `6a1379f` `cp_position_ids` built before the CP
context + registered in `_cp_buffers`) **held** — the forward got past them. A **new, genuine
blocker** then surfaced inside the CP attention kernel selection.

## The new failure (gs1 forward, real)
```
Train loop failed at global_step 1: ray::FSDPPolicyWorkerBase.forward()
  …/skyrl_train/model_wrapper.py:777
    output = self.model(sequences_fwd, attention_mask=None, position_ids=cp_position_ids)   # <- both fixes active here
  …/transformers/models/qwen3_moe/modeling_qwen3_moe.py:181  attention_interface(...)
  …/transformers/integrations/sdpa_attention.py:92  F.scaled_dot_product_attention(...)
  …/torch/distributed/tensor/experimental/_context_parallel/_attention.py:966  inner_fn → target_fn(...)
RuntimeError: No available kernel. Aborting execution.
  select_sdp_backend  (aten/src/ATen/native/transformers/cuda/sdp_utils.cpp:992)
```
- Config in play: `attn_backend=sdpa`, `flash_attn=false`, `use_sample_packing=false`,
  CP=2 `cp_style=ring_sdpa` `cp_rotate_method=allgather`, EP4×FSDP2×CP2=16 policy GPU,
  gradient_checkpointing + use_reentrant=false. SIF `_cp_fixb3.sif` (#237-merged). Qwen3-Coder-30B-A3B.
- Meaning: PyTorch's **context-parallel SDPA wrapper** (`_context_parallel/_attention.py`) calls
  `F.scaled_dot_product_attention` and `select_sdp_backend` finds **no usable SDP kernel** for this
  (dtype, shape, mask) combo on GH200/cu130/torch2.11. FlashAttention is force-disabled (CP requires
  sdpa), and the math/mem-efficient backends are being rejected under the CP path.

## Status of the two questions this smoke was meant to answer
1. **Do the CP-MoE dict + cp_position_ids fixes work?** → YES, validated: gs1 forward reached and
   passed `model_wrapper.py:777`. No `dict.ndim`, no CheckpointError, no `_validate_cp_cfg`/mesh error.
2. **Does CP=2 ring-SDPA train end-to-end?** → NO (new wall): SDP backend selection fails at the
   first attention under the CP context-parallel wrapper. This is the actual remaining #232 blocker.

## Likely fix direction (NOT yet attempted)
Force an explicit SDP backend around the CP attention so `select_sdp_backend` has a legal choice —
e.g. wrap the forward in `torch.nn.attention.sdpa_kernel([SDPBackend.MATH])` (or EFFICIENT_ATTENTION),
or ensure the CP `_attention.py` dispatch is given an enabled backend. Investigate whether the CP
ring wrapper disables all backends by default and which one is valid for non-causal/masked + the
GH200 dtype. This is a SkyRL/torch-CP code change (local `MarinSkyRL` penfever/working), then rebuild
is NOT needed (editable) — but it needs a smoke to validate, and **Jupiter locks today**.

## Jupiter cutoff caveat
Access ends today for ~1 month. A CP=2 smoke takes ~30 min to reach gs1. There may be time for ONE
more fix→smoke cycle IF the SDP-backend fix is identified confidently — but this is iterative CP
kernel debugging that may not land first try. Surfaced to the user for a go/no-go rather than burning
the last window on a guess. If deferred: #232 CP-MoE resumes when Jupiter returns; #237 is already
merged + baked (`_cp_fixb3.sif`), so the rung restarts from this exact "No available kernel" point.

## Not at risk
No weights lost (smoke `ckpt_interval=999`, no HF/DB). All production checkpoints already secured by
the pre-deadline harvest (lever1 gs60, swesmith Muon gs25, coder #217 gs40, symclip_loopshape gs45).
lever1 927677 RUNNING but has banked nothing past the already-uploaded gs60.

---

## 2026-06-20 (later) — FIX-4: root cause found + fixed (the FIX-3 flash-ONLY pin WAS the regression)

### Root cause (definitive, from the .out kernel-rejection reasons + an in-SIF micro-repro)
The gs1 "No available kernel" was **caused by FIX-3 itself**. FIX-3 wrapped the CP forward in
`_cp_force_flash_sdpa()` = `sdpa_kernel([SDPBackend.FLASH_ATTENTION], set_priority=True)`. The full
kernel-rejection block in `…_932229.out` (previously truncated) is decisive:
```
UserWarning: Flash Attention does not support non-null attn_mask.      (sdp_utils_cpp.h:262)
UserWarning: Memory Efficient attention has been runtime disabled.     (sdp_utils_cpp.h:552)
UserWarning: cuDNN attention has been runtime disabled.                (sdp_utils.cpp:706)
RuntimeError: No available kernel.                                     (select_sdp_backend, sdp_utils.cpp:992)
```
Two facts FIX-3 got wrong:
1. **HF DOES build a non-null 4D SDPA mask** for the Qwen3-MoE forward even with
   `attention_mask=None` on this SIF (transformers 5.10.1) — `create_causal_mask` did NOT return
   None. So flash (which cannot take a non-null `attn_mask`) is ineligible at that attention call.
2. **`sdpa_kernel([FLASH], set_priority=True)` runtime-DISABLES every other backend.**
   `torch.nn.attention._sdpa_kernel` first does `enabled = (backend in list)` for ALL backends —
   so memory-efficient + cuDNN (the ones that DO accept a non-null mask) were turned OFF.
   `set_priority` only *reorders the survivors*; it does NOT keep the others as fallbacks (the FIX-3
   docstring's "leaving the others as fallbacks" claim was false). Flash rejected + eff/cuDNN
   disabled ⇒ nothing available ⇒ "No available kernel".

### Why the old job-930793 `aten.expand` crash does NOT recur with eff/cuDNN re-enabled
torch 2.11's CP dispatch is `_DispatchMode.MONKEY_PATCH` (confirmed in
`…/_context_parallel/_attention.py`): it replaces `F.scaled_dot_product_attention` itself with a
wrapper (`inner_fn`, line 966) that shards q/k/v **and the mask** to LOCAL ring chunks (`input_fn`)
and then calls the ORIGINAL `F.sdpa` on those local tensors. The 4D-bias `aten.expand` therefore
happens on already-local chunks, NOT on a CP-sharded DTensor — so the 930793 sharding-prop expand
mismatch (`S_kv/cp → S_kv`) is structurally absent on this path. (The CP ring `call_maps` supports
flash/efficient/cuDNN; MATH is NOT CP-legal.)

### In-SIF micro-repro (skyrl_megatron_vllm0202rc0_r3_cp_fixb3.sif, torch 2.11.0+cu130, GH200)
`F.scaled_dot_product_attention(q,k,v, attn_mask=<4D bf16 [B,1,S,S]>, is_causal=False)`:
- under `sdpa_kernel([FLASH], set_priority=True)`        → **RuntimeError: No available kernel** (reproduces prod)
- under `sdpa_kernel([EFFICIENT], set_priority=True)`     → **OK**
- under `sdpa_kernel([FLASH,EFFICIENT,CUDNN], priority)`  → **OK** (flash still wins null-mask is_causal chunks)

### The fix (FIX-4) — local MarinSkyRL `penfever/working`, commit `baa085f`
`skyrl-train/skyrl_train/model_wrapper.py` `_cp_force_flash_sdpa()` (≈ lines 120–172): instead of
`sdpa_kernel([FLASH], …)`, enable the full CP-legal backend list
`[FLASH_ATTENTION, EFFICIENT_ATTENTION, CUDNN_ATTENTION]` with `set_priority=True` (flash first).
Flash still serves the null-mask `is_causal=True` i==0 ring step; the HF-masked chunks fall back to
memory-efficient/cuDNN. Both CP call sites (forward + the no-grad scoring path) are unchanged — they
call the same renamed-behavior helper. Gated to `cp_size > 1` only ⇒ CP1/non-CP byte-identical.
- Pushed: **YES** (`marin/penfever/working` bee47ca → **baa085f**).
- On cluster: **YES** — `/e/scratch/jureap59/feuer1/OpenThoughts-Agent/SkyRL` fast-forwarded to
  `baa085f` (was clean: no tracked drift, only `core.*` crash dumps untracked). Editable install ⇒ live, NO SIF rebuild.

### Smoke relaunch (validation)
Relaunched via `/e/scratch/jureap59/feuer1/cp2_smoke_relaunch.sh` (tmux `cp2smoke`), same smoke YAML +
`_cp_fixb3.sif`. **Job 932932**, dir `…/ot-baf/rl_30b_a3b_32k_cp2_SMOKE_7`, log
`…/logs/rl_30b_a3b_32k_cp2_SMOKE_932932.out`. Confirmed at submit: **RUNNING**, NumNodes=6,
Reservation=reformo, SIF=`…_cp_fixb3.sif`, `cp_style=ring_sdpa`, `context_parallel_size=2`,
`enable_db_registration=false`. RL count at launch = 0 RUNNING (≤6 cap satisfied; no RUNNING job
touched — lever1 927677 had already terminated per sacct).

### gs1 VERDICT — kernel wall CLEARED, but a DEEPER CP-mask blocker surfaced (FIX-4 half-true)
Job 932932 progressed FAR past where 932229 died: Ray bootstrap OK, rollout generation buffer
filled cleanly to 16/16 (multiple "Batch generation complete: 2/2 successful"), then hit the gs1
training forward at 2026-06-20T12:14:03.

**"No available kernel" is GONE** — `grep -c "No available kernel"` on the 932932 .out = **0**. The
FIX-4 backend change did exactly what it was meant to at the SDPA-selection layer: a kernel is now
available. So FIX-3's flash-ONLY pin WAS the cause of the kernel-selection wall, confirmed.

**But the SAME gs1 forward (`model_wrapper.py:798`, MoE `attention_mask=None` path) now raises the
job-930793 `aten.expand` DTensor sharding-prop crash** — the very pathology FIX-3 was trying to dodge:
```
File ".../transformers/integrations/sdpa_attention.py", line 92, in sdpa_attention_forward
  attn_output = torch.nn.functional.scaled_dot_product_attention(
File ".../_context_parallel/_attention.py", line 966, in inner_fn
  outputs = target_fn(*args, **kwargs)
File ".../torch/distributed/tensor/_dispatch.py", line 251, in _propagate_op_sharding_dispatch_slow_path
RuntimeError: The expanded size of the tensor (10448) must match the existing size (5224)
  at non-singleton dimension 3. Target sizes: [2, 32, 10448, 10448]. Tensor sizes: [2, 1, 10448, 5224]
Sharding propagation failed for aten.expand.default(
  Spec(bf16[2, 1, 10448, 5224](S(2))), [2, 32, 10448, 10448]) on DeviceMesh((cp=2))
Train loop failed at global_step 1
```
Reading: HF's Qwen3-MoE DOES build a non-null 4D additive mask `[B,1,S_q,S_kv]`; under CP the kv
dim is sharded to S/cp (`5224` = `10448/2`, Spec `S(2)` = sharded on dim 2/kv) while q stays full
(`10448`); the memory-efficient/cuDNN SDPA path then `aten.expand`s the mask to all 32 heads + full
kv `[2,32,10448,10448]`, and DTensor sharding-prop rejects `5224 -> 10448`. This is **exactly**
job-930793. **MY MONKEY_PATCH ANALYSIS WAS WRONG**: the mask is NOT pre-sharded to a plain local
chunk before `target_fn` — it reaches the SDPA call as a CP-sharded DTensor (Spec `S(2)`), so the
expand happens at the DTensor sharding-prop layer, not on a local tensor. Re-enabling efficient/
cuDNN therefore re-exposed the bias-expand crash.

**Net**: FIX-4 cleared the kernel-selection wall (real, verified progress) but the underlying #232
CP-MoE blocker is now precisely characterized as: *HF builds a 4D additive attention mask for
Qwen3-MoE that cannot survive CP kv-sharding under the efficient/cuDNN SDPA expand* — and flash
(the only backend that avoids the 4D-bias expand) is rejected because it cannot take that non-null
mask. So neither legal CP backend works as-is with the HF-built 4D mask. The fix must STOP HF from
building the 4D mask at all on the MoE path (so SDPA gets `attn_mask=None, is_causal=True` → flash),
NOT toggle backends. Candidate directions (UNVALIDATED — for next session):
  * patch/monkeypatch Qwen3-MoE `create_causal_mask` (or the eager-mask attr) so the CP forward
    truly passes `attn_mask=None` + `is_causal=True` (the dict escape hatch `_cp_mask_dict_supported`
    is False for MoE — that's the gap; FIX-1/2 territory revisited with the new evidence that
    `attention_mask=None` alone is NOT sufficient on transformers 5.10.1 for MoE);
  * or set the model's `config._attn_implementation`/mask-interface so HF emits no 4D SDPA mask under CP;
  * or pre-shard/avoid the mask expand in the CP path.

### STOP — reported, not looped (per task STOP conditions)
This is iterative CP-kernel/mask debugging that did NOT land first try, and Jupiter access ends today.
Per the brief's go/no-go discipline, NOT burning the closing window on another unvalidated guess.
FIX-4 is committed/pushed/synced and is a STRICT IMPROVEMENT (clears the kernel wall; gated to CP>1
so flag-off is byte-identical), so it stays in. The #232 CP-MoE rung now restarts from this exact
`aten.expand` sharding-prop crash (HF 4D-mask vs CP-sharded kv) when Jupiter returns. No weights at
risk (ckpt_interval=999, no HF/DB). No RUNNING job was touched; 932932 self-failed at gs1.
Evidence: `/e/data1/datasets/playground/ot-baf/rl_30b_a3b_32k_cp2_SMOKE_7/logs/rl_30b_a3b_32k_cp2_SMOKE_932932.out`.

---

## 2026-06-20 (continuation) — FIX-5: kill HF's 4D mask on the MoE CP forward via create_causal_mask monkeypatch

### Root-cause confirmation (in-SIF, definitive)
Investigated transformers 5.10.1 `masking_utils` inside `_cp_fixb3.sif`. The Qwen3-MoE forward
(`modeling_qwen3_moe.py` `Qwen3MoeModel.forward`) does
`mask_function = create_causal_mask if config.sliding_window is None else create_sliding_window_causal_mask`
(Qwen3-Coder-30B-A3B has `sliding_window=None` → `create_causal_mask`). The is_causal skip lives in
`sdpa_mask` → `_ignore_causal_mask_sdpa`, which returns True (→ mask None) when **no padding** and
**q_length==kv_length** AND **`not is_tracing(...)`**.
- In ISOLATION (single rank, no FSDP2/GC): `create_causal_mask(attention_mask=None, monotonic
  position_ids)` returns **None** — verified via in-SIF repro (also inside a 1-rank
  `context_parallel` ctx → still None, `is_tracing(pos)=False`). So FIX-1's `attention_mask=None`
  premise is correct in isolation.
- In the REAL multi-rank FSDP2 + gradient-checkpointed CP forward it does NOT: smoke 932932 (FIX-4)
  built a 4D bias `bf16[2,1,10448,5224](S(2))` and crashed at the SDPA head-broadcast expand
  (`aten.expand [2,1,10448,5224] -> [2,32,10448,10448]`, kv sharded S/cp=5224 vs q full 10448).
  Python frames from the 932932 .out: `model_wrapper.py:798` → CP `_attention.py:966 inner_fn` →
  `sdpa_attention.py:92` → `modeling_qwen3_moe.py:{181,337,509,670}` — i.e. the 4D mask was built by
  HF's `create_causal_mask` at model-forward time, then mis-expanded under CP-sharded kv. The skip is
  suppressed because `is_tracing()` trips under the CP MONKEY_PATCH SDPA wrapper + GC recompute
  (fake-tensor / stream-capture), forcing `_ignore_causal_mask_sdpa` False.

### The fix (FIX-5) — local MarinSkyRL `penfever/working`, commit `80f1875` (pushed)
`skyrl-train/skyrl_train/model_wrapper.py`: idempotent import-time monkeypatch
`_install_cp_moe_mask_patch()` wraps the MoE modeling module's `create_causal_mask` AND
`create_sliding_window_causal_mask` (rebinding the NAME in `modeling_qwen3_moe`, since it imported
by value) so they return **None while a thread-local switch `_cp_moe_force_no_mask.active` is set**.
The `_cp_moe_no_mask()` context manager flips the switch ON for the duration of the MoE forward, and
is wrapped around BOTH MoE `cp_size>1` call sites (HFModelWrapper training forward ~L798; the
value/scoring `base_model_prefix` path ~L1322). Switch OFF ⇒ wrapper delegates verbatim to HF.
- Gating: switch is only ever ON inside the MoE `cp_size>1 and not _cp_mask_dict_supported` branch.
  Non-CP / CP1 / dense-Qwen3 (dict path) / generation forwards are byte-identical (switch stays OFF).
- In-SIF validation (transformers 5.10.1): OFF + a padded 2D mask → HF materializes 4D `[2,1,16,16]`
  (delegation works); ON + same input → **None**. Patch target + gating confirmed correct.
- Pushed: marin `penfever/working` `baa085f → 80f1875`. Cluster
  `/e/scratch/jureap59/feuer1/OpenThoughts-Agent/SkyRL` fast-forwarded to `80f1875` (no tracked
  drift; pre-existing stashes untouched). Editable ⇒ live, NO SIF rebuild.

### Smoke relaunch (validation)
Relaunched via `/e/scratch/jureap59/feuer1/cp2_smoke_relaunch.sh` (tmux `cp2smoke`). **Job 933207**,
dir `…/ot-baf/rl_30b_a3b_32k_cp2_SMOKE_8`. Confirmed RUNNING, 6 nodes, SIF `_cp_fixb3.sif`, rendered
config: `context_parallel_size=2` `cp_style=ring_sdpa` `cp_rotate_method=allgather` (policy+ref),
`attn_backend=sdpa` `use_sample_packing=false` `enable_db_registration=false`. RL RUNNING count at
launch = 0 (≤6 satisfied; the 8 `eval_` jobs are EVALS, untouched).

### gs1 VERDICT (FIX-5) — FORWARD PASSED; new blocker = CP×gradient-checkpointing BACKWARD recompute
Job 933207 progressed FURTHER than any prior attempt. Rollout filled 16/16 cleanly (all "Batch
generation complete: 2/2 successful, 0 failed, 0 masked"), then at gs1:
- **`[CP FIX-5] Installed Qwen3-MoE no-4D-mask CP patch on: [...]`** logged on the FSDPPolicyWorker
  actors at 13:11:47 — the patch armed.
- **The gs1 FORWARD COMPLETED CLEAN**: `grep` counts on the .out = `expanded size of the tensor`:0,
  `No available kernel`:0, `Sharding propagation failed`:0. FIX-5 KILLED the aten.expand wall — the
  CP-MoE forward ran the full ring SDPA with `attn_mask=None`+`is_causal=True`. **This is the first
  time the #232 CP-MoE forward executed end-to-end.** `global_step` advanced to [0, 1].
- **It then FAILED at the gs1 BACKWARD** with a NEW, deeper blocker:
  `torch.utils.checkpoint.CheckpointError: Recomputed values for the following tensors have different
  metadata than during the forward pass` —
  saved `q/k/v [1,32,10404,128]` vs recomputed `[1,32,5202,128]` (FULL 10404 vs CP-sharded 5202=10404/2).
  Python frames: `worker.py:1071 strategy.backward` → `fsdp_strategy.py:186 backward` →
  `torch/utils/checkpoint.py:921 check_recomputed_tensors_match`.

Root cause (FIX-6): torch's `context_parallel` CM monkey-patches `F.scaled_dot_product_attention` with
the ring wrapper (which ALL-GATHERS k/v so each rank's SDPA sees the FULL kv length) and UNPATCHES on
exit. Under FSDP2 gradient checkpointing the per-layer forward is RECOMPUTED during `backward()`, which
runs AFTER `model_wrapper.forward` exited the CP CM — so SDPA is plain again and the recomputed
attention keeps q/k/v at the LOCAL CP-sharded length (5202) while the original forward saved them
ring-gathered to full (10404) → metadata mismatch.

### The fix (FIX-6) — local MarinSkyRL `penfever/working`, commit `82b54f3` (pushed)
Keep the ring-SDPA patch installed across backward so the GC recompute dispatches to the SAME ring
attention. `distributed/cp_utils.py::cp_sdpa_dispatcher_span(cp_mesh)` re-installs ONLY the SDPA patch
(via torch's `_enable/_disable_context_parallel_dispatcher_impl`, MONKEY_PATCH mode) — NOT the buffer
sharding (inputs already stay sharded via `no_restore`; re-sharding would halve them again).
`model_wrapper.forward` arms `self._cp_needs_backward_sdpa_span` on a CP grad-building forward and
exposes `cp_backward_dispatcher_span()`; `worker.py::training_step` wraps `strategy.backward()` in it.
Returns a literal `nullcontext` for CP1 / non-CP / no-grad backward (byte-identical) and if the torch
primitives are absent (guarded). In-SIF validation: `_enable_..._impl(seq_dim=2, mesh)` swaps
`F.sdpa` identity to the ring wrapper; `_disable_..._impl()` restores it.
- Pushed: marin `penfever/working` `80f1875 → 82b54f3`. Cluster pulled (editable ⇒ live, NO rebuild).

### Smoke relaunch (FIX-6 validation) — job 933336
Relaunched via `/e/scratch/jureap59/feuer1/cp2_smoke_relaunch.sh` (tmux `cp2smoke`). **Job 933336**,
dir `…/ot-baf/rl_30b_a3b_32k_cp2_SMOKE_9`. Confirmed RUNNING, 6 nodes, SIF `_cp_fixb3.sif`, rendered
config: `context_parallel_size=2` `cp_style=ring_sdpa` `cp_rotate_method=allgather` (policy+ref),
`attn_backend=sdpa` `gradient_checkpointing=true` (the GC that triggers the recompute FIX-6 fixes),
`use_sample_packing=false` `enable_db_registration=false`. RL RUNNING count at launch = 0 (≤6).

### gs1 VERDICT (FIX-5 + FIX-6) — ✅ VALIDATED SUCCESS: forward AND backward PASSED, trainer advanced to gs2
Rollout filled 16/16 cleanly (all "Batch generation complete: 2/2 successful, 0 failed, 0 masked"),
then at gs1 (13:46:29):
- **`[CP FIX-5] Installed Qwen3-MoE no-4D-mask CP patch on: ['create_causal_mask',
  'create_sliding_window_causal_mask']`** logged on the FSDPPolicyWorker actors (verbatim).
- **gs1 FORWARD CLEAN**: `expanded size of the tensor`=0, `No available kernel`=0,
  `Sharding propagation failed`=0 (FIX-5 held — no aten.expand wall).
- **gs1 BACKWARD CLEAN (FIX-6 validated)**: `CheckpointError`=0, `different metadata`=0. The policy
  training step completed: `Policy Train epoch [1/1]: 100%|██████████| 2/2 [00:33<00:00, 16.68s/it,
  pg=0, glen=7988.0, policy_lr=8e-6, ent=0.26, grad_norm=0]` — forward+BACKWARD across both
  micro-batches with a finite grad_norm. The prior run's GC-recompute CheckpointError is GONE.
- **TRAINER ADVANCED PAST gs1**: fully_async_trainer phase transitions (verbatim):
  `Started: 'run_training' → Finished: 'run_training' → Started: 'sync_weights'` (13:47:11) — the gs1
  training step finished and the weight-sync to vLLM for the NEXT step started (vLLM engines logged
  `abort_generation` + NCCL AllReduce receiving synced weights at 13:47:16). i.e. the trainer left the
  gs1 step and entered the gs2 cycle (sync_weights → generation). `Train loop failed`=0 throughout;
  job RUNNING, no error.

**NET: #232 CP-MoE is VALIDATED.** CP=2 ring-SDPA on Qwen3-Coder-30B-A3B (EP4×FSDP2×CP2, GC on) now
runs a full GRPO training step — gs1 forward AND backward — and the fully-async trainer advances to the
next step. The complete CP-MoE fix stack that landed it: arch-gate `attention_mask=None` (`b02d758`),
`cp_position_ids` pre-context + registered (`6a1379f`), FIX-4 CP-legal SDPA backend list (`baa085f`),
FIX-5 `create_causal_mask`→None MoE monkeypatch (`80f1875`), FIX-6 ring-SDPA dispatcher span across
backward for GC recompute (`82b54f3`). All gated to `cp_size>1` ⇒ CP1/non-CP byte-identical.
Evidence: `/e/data1/datasets/playground/ot-baf/rl_30b_a3b_32k_cp2_SMOKE_9/logs/rl_30b_a3b_32k_cp2_SMOKE_933336.out`.
No weights at risk (ckpt_interval=999, no HF/DB; enable_db_registration=false). No RUNNING job touched
(the 8 `eval_` jobs are EVALS, untouched). The smoke is a SMOKE — it'll TIMEOUT/end on its own; left
running to keep accumulating CP steps as further confirmation.

