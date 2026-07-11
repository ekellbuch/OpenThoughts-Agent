# MarinSkyRL DESLOP — Plan of Record

Operator-approved refactor: turn live caller-facing `SKYRL_*` env vars into first-class
flags/config-fields, flip footgun defaults on-by-default, strip dead code.

Repo: `/Users/benjaminfeuer/Documents/MarinSkyRL` (branch `penfever/working`, base `e9b2f10d`).
Launcher/configs: OT-Agent `/Users/benjaminfeuer/Documents/OpenThoughts-Agent`.

Invariant: **flag-OFF / config-unset behavior must be byte-identical to today** EXCEPT the
intended footgun-default flips (Stage 2). Env var retained as override in Stage 3 (env > flag/config).

Live job `80b-next-cp1-r3d2` + flawed_summ evals imported code at process start ⇒ editing files is
SAFE (only the NEXT launch picks up changes). Do NOT relaunch/kill.

---

## Env-var inventory + disposition

| env var | read-site(s) | current default | action | stage | validation |
|---|---|---|---|---|---|
| `SKYRL_FWD_UNSHARD_FENCE` | worker.py:594 (+comment 561-604) | "0" (OFF) | **REMOVE** (superseded by drain barrier; own comment says it only "relocated the hang") | 1 | grep no refs; IDE clean |
| `EPDIAG` | moe.py:106 (`_epdiag_enabled`), worker.py:539/1012/1174/1491 | "0" | **REMOVE** (probe+phase counters); EP/CP bug fixed | 1 | grep no refs; IDE clean |
| `EPDIAG_CP`/`EPDIAG_EP`/`EPDIAG_SP` | moe.py:142-146,306-308 (coord decode) | 0/0/1 | **REMOVE** (only used by EPDIAG/GRPMM probes) | 1 | grep no refs |
| `SKYRL_GROUPMM_DIAG` | moe.py:272 (`_grpmm_diag_enabled`),375 | "0" | **REMOVE** (probe); grouped-mm offsets bug fixed | 1 | grep no refs |
| `SKYRL_NUM_EXPERTS` | moe.py:310 (inside `_grpmm_diag` only) | "-1" | **REMOVE** (consumed only in GRPMM block) | 1 | grep no refs |
| `SKYRL_EP_LOADER_DEBUG` | fsdp_utils.py:288 (`_dbg`),458,470 | "" | **REMOVE** reads + `if _dbg:` logging; keep loader logic | 1 | grep no refs |
| `SKYRL_ROUTER_REPLAY_DEBUG` | router_replay.py:108 (`self._debug`),193 | unset | **REMOVE** read + `if self._debug:` logging | 1 | grep no refs |
| `R3_EPTRACE` | (none — already absent) | — | N/A (task listed it; not in codebase) | 1 | grep confirms absent |
| `SKYRL_WEIGHT_SYNC_SERIALIZE` | fsdp_worker.py:199 (+comment 190-196) | "0" (OFF) | **VERIFY dead → REMOVE** (superseded by gather-order fix ac440797 + drain barriers; only in 4 DEPRECATED configs) | 1 | confirm no live dep; grep |
| --- | --- | --- | --- | --- | --- |
| `SKYRL_GDN_MASK_FLA` | model_wrapper.py:373,531 | "0" (OFF) | **DEFAULT ON**, auto-derive from arch (Qwen3-Next/GDN); no-op on dense → flag | 2/3 | dense smoke = no-op; GDN engages |
| `SKYRL_TIS_SERVED_ID_SPLICE` | utils.py:1291 | unset (OFF) | **DEFAULT ON**; MERGE with `SKYRL_QWEN3_5_TIS_SPLICE` into one `tis_splice` policy | 2/3 | non-thinking model byte-identical |
| `SKYRL_QWEN3_5_TIS_SPLICE` | utils.py:1176 | "1" (ON) | **MERGE** into `tis_splice` (served-id path supersedes empty-think path) | 2/3 | splice reconcile |
| `SKYRL_WORKER_NCCL_TIMEOUT_IN_S` | constants.py:9 (def 600), utils.py:1143 (max(1200,…)) | 600 / 1200 | **collapse to ONE accessor, default ≥1800** | 2/3 | single source; ≥1800 |
| `SKYRL_FORWARD_DISPATCH_FIX` | fsdp_worker.py:1092,1167,988 | "1" (ON) | keep default-ON, expose as flag | 3 | flag-off byte-identical |
| `SKYRL_WEIGHTSYNC_DRAIN_BARRIER` | worker.py:658 | "1" (ON) | keep default-ON, expose as flag | 3 | " |
| `SKYRL_R3_RESIDENT` | dispatch.py:345 | "1" (ON) | keep default-ON, expose as flag | 3 | " |
| `SKYRL_CP_REQUIRE_RIGHT_ALIGN` | model_wrapper.py:761 | "1" (ON) | keep default-ON, expose as flag | 3 | " |
| `SKYRL_W13_RELOAD_BRACKET` | fsdp_worker.py:809 | "1" (ON) | keep default-ON, expose as flag | 3 | " |
| `SKYRL_R3_DECENTRAL` | dispatch.py:366 | "1" (ON) | already default-ON — expose in r3_transport enum | 3 | " |
| `SKYRL_DISPATCH_PUT_TIMEOUT_S` | dispatch.py:358 | "600" | → `trainer.fully_async.r3_put_timeout_s` | 3 | " |
| `SKYRL_POLICY_HOST_RAM_MONITOR` (+`_INTERVAL`) | fsdp_worker.py:723,727 | "1"/"60" | → observability flags | 3 | " |
| `SKYRL_GDN_FLASHQLA` | qwen3_next_gdn.py:140 | "0" | expose as flag (leave default OFF) | 3 | " |
| `SKYRL_EP_LOADER_CHUNK_ROWS` | fsdp_utils.py:329 | "8" | expose as flag | 3 | " |
| `SKYRL_LD_LIBRARY_PATH_EXPORT` / `SKYRL_PYTHONPATH_EXPORT` | constants.py:15,26 | False | LEAVE (pure infra-plumbing constants) | 3 | note only |
| `SKYRL_FUSE_WEIGHTS` | main_base.py:535 etc (set by FP8 path) | "0" | LEAVE (internally set by fp8 path, not a caller knob) | — | note |
| `SKYRL_ENABLE_NUMA_AFFINITY` | numa.py:29 | "0" | LEAVE (advanced, not in caller configs) | — | note |
| `SKYRL_HF_LOAD_*` | hf_load_retry.py | 5/2.0/32.0 | LEAVE (retry tunables, not caller-facing) | — | note |
| `SKYRL_QWEN3_5_VLM_UNWRAP` | qwen3_5_vlm.py | "1" | LEAVE (already default-on internal) | — | note |
| `SKYRL_RAY_PG_TIMEOUT_IN_S` | constants.py:4 | 180 | LEAVE (infra timeout, not footgun) | — | note |

---

## Stage status

- [x] **Stage 1** — strip dead EPDIAG/fence/serialize knobs — DONE, commit `2f20bf91` (pushed). IDE clean, no dangling refs. Note: `SKYRL_WEIGHT_SYNC_SERIALIZE` confirmed dead (only 4 deprecated configs); `R3_EPTRACE` was already absent.
- [x] **Stage 2** — footgun defaults on-by-default — DONE, commit `06756e36` (pushed). GDN_MASK_FLA default-ON auto-derived from arch; TIS splice merged into one default-ON `SKYRL_TIS_SPLICE` policy; NCCL timeout collapsed to `get_worker_nccl_timeout_s()` default 1800. All live iris configs already set these explicitly ⇒ no-op for them. IDE clean; helper semantics unit-checked.
- [x] **Stage 3** — promote SKYRL_* env vars to flags — DONE (OT-Agent). `rl/cloud/launch_rl_iris.py`
  gained a "MarinSkyRL runtime knobs" argparse group + `build_skyrl_flag_env(args)`; all flags default
  to `None` so an all-defaults launch injects `{}` (byte-identical), and a config's `extra_env:` overlays
  ON TOP of the flag env (precedence env/extra_env > flag > code default). Validated: all-defaults ⇒ `{}`,
  mappings + R3-transport folding correct, IDE clean, all iris configs parse.
  - Flags added: `--r3-transport {by_value,resident,decentral}` (folds SKYRL_R3_RESIDENT+SKYRL_R3_DECENTRAL),
    `--r3-put-timeout-s`, `--nccl-timeout-s`, `--host-ram-monitor`(+`-interval-s`), `--tis-splice`,
    `--gdn-mask-fla {auto,on,off}`, `--gdn-flashqla`, `--forward-dispatch-fix`, `--weightsync-drain-barrier`,
    `--cp-require-right-align`, `--w13-reload-bracket`, `--ep-loader-chunk-rows`.
  - Config cleanup (byte-identical): dropped code-forced `VLLM_USE_FLASHINFER_SAMPLER: "0"` from 7 iris
    configs (vllm_engine.py:185 forces it), the duplicate `TORCH_FR_DUMP_TEMP_FILE` alias from the 2 80B
    configs (utils.py re-derives it from `TORCH_NCCL_DEBUG_INFO_TEMP_FILE`), and the now-inert EPDIAG env
    arm from `64GPU_qwen3_coder_30b_a3b.yaml`.
  - DEFERRED (deliberately, not forced): the `trainer.fully_async.r3_transport` / `r3_put_timeout_s`
    **skyrl pydantic config-FIELD** route — implemented as launcher flags instead (the task's primary
    "argparse flag" route), to avoid touching the SkyRL pydantic schema while an 80B job is live. The
    launcher flag + env-override fully covers the knob surface; adding config fields later is additive.
  - LEFT as-is (per plan): `SKYRL_LD_LIBRARY_PATH_EXPORT` / `SKYRL_PYTHONPATH_EXPORT` (pure infra plumbing
    constants); `SKYRL_FUSE_WEIGHTS` (set internally by the FP8 path, not a caller knob); `SKYRL_HF_LOAD_*`,
    `SKYRL_ENABLE_NUMA_AFFINITY`, `SKYRL_RAY_PG_TIMEOUT_IN_S`, `SKYRL_QWEN3_5_VLM_UNWRAP` (advanced/internal).
  - NOT staged by me: two SMOKE iris configs (`2node_qwen3_coder_30b_r3decentral_SMOKE.yaml`,
    `64GPU_qwen3_6_35b_a3b_SMOKE_r3decentral.yaml`) were untracked pre-existing files; their byte-identical
    VLLM_USE_FLASHINFER_SAMPLER drop is in the working tree but I did not `git add` new files I didn't author.
  - Minor: the obsolete `# ... EPDIAG env below ... DO NOT DROP IT` prose comment in the coder config is now
    stale (probe removed in stage 1) but harmless; left in place to bound the diff.

Commit per stage; push to marin `penfever/working` (Stages 1-2, MarinSkyRL) / OT-Agent `penfever/working` (Stage 3).
