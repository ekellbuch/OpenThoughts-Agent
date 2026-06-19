# #232 cp2 — residual uvloop SSL abort fix (job 930208 → relaunch) — 2026-06-20

This log tracks the #232 cp2 production run's uvloop/libuv abort saga. The
CP-MoE `dict.ndim` forward bug and the actor-concurrency-group uvloop abort were
already fixed (SkyRL `b02d758`/`6a1379f` and `47bf11f` respectively). This entry
covers the REMAINING uvloop leak that re-triggered the libuv-1.48.0 io_uring
abort via a DIFFERENT path in job 930208.

## Symptom (job 930208, CANCELLED, exp dir rl_30b_a3b_32k_cp2_3)
- Reached **genbuf 2/64**, then a `RolloutCoordinator` actor (pid 237783 /
  235352, ip 10.128.27.62/.64) hit `Fatal Python error: Aborted`.
- Traceback terminal frame: **`uvloop/sslproto.pyx:517 SSLProtocol._on_handshake_complete`**
  — the litellm→Daytona HTTPS handshake running on a **uvloop SSL transport**.
- The actor's loaded extension modules include both **`uvloop.loop`** and
  **`aiohttp._http_*`** → a uvloop.Loop() was live in the actor and aiohttp
  (litellm's transport) was driving SSL on it.
- This is DESPITE the prior fixes: BasePPOExp.run() driver reset, the
  `worker_process_setup_hook` `_force_stock_asyncio_in_worker` (`47bf11f`, sets
  the asyncio *policy*), `RAY_USE_UVLOOP=0`, and the `RolloutCoordinator.__init__`
  reset. A uvloop loop STILL existed in the actor.

## Definitive root cause (two independent facts)
1. **A policy reset cannot stop a uvloop loop.** `uvloop.new_event_loop()`
   constructs `uvloop.Loop()` DIRECTLY — it does NOT consult asyncio's
   event-loop policy. So `set_event_loop_policy(DefaultEventLoopPolicy())` (the
   `47bf11f` hook + the `__init__` reset) only governs `asyncio.new_event_loop()`
   / `get_event_loop()`; anything that calls `uvloop.new_event_loop()` outright
   (Ray's C++ CoreWorker, or aiohttp/litellm bringing up a loop) still gets a
   uvloop loop → the SSL transport runs on libuv 1.48.0 → the io_uring
   `uv__epoll_ctl_prep` abort path is reachable from the SSL handshake.
   Verified locally + in-SIF: pre-fix `uvloop.new_event_loop()` returns
   `uvloop.Loop`.
2. **`UV_USE_IO_URING=0` was inert because it never reached the actor's env.**
   libuv 1.48.0 DOES honor `UV_USE_IO_URING` — `uv__use_io_uring()` reads it via
   `getenv()` at first use and caches it atomically (confirmed against libuv
   v1.48.0 `src/unix/linux.c`). `"0"` → io_uring is never armed and the buggy
   path is dead. But the var lived only in the driver/launcher (host) env
   (`rl_launch_utils.py` only *comments* that host env should survive into
   apptainer; it is never actually exported, and grep found `UV_USE_IO_URING`
   nowhere in any of the three repos as a real export). Ray ACTOR processes
   derive their environment from `runtime_env["env_vars"]`, NOT arbitrary driver
   env — so even a host-set value never reached the RolloutCoordinator's libuv.

## The fix (belt-and-suspenders, no SIF rebuild) — SkyRL `0554dae` (penfever/working)
File: `skyrl-train/skyrl_train/utils/utils.py`
- `prepare_runtime_environment`: add `env_vars["UV_USE_IO_URING"] = "0"` (right
  after `RAY_USE_UVLOOP="0"`, ~line 1048) so it reaches EVERY actor process env
  before libuv's first `uv__use_io_uring()` getenv+cache.
- `_force_stock_asyncio_in_worker` (the worker-boot hook): (a) set
  `os.environ["UV_USE_IO_URING"]="0"` FIRST (before any libuv init; guards an
  import-order race vs the runtime-env injection); (b) after the policy reset,
  NEUTRALIZE uvloop in-process — alias `uvloop.new_event_loop` /
  `uvloop.install` / `uvloop.EventLoopPolicy` / `uvloop.Loop` to the stock
  asyncio equivalents (SelectorEventLoop / DefaultEventLoopPolicy), so NO uvloop
  loop can be created at all — covering the SSL path the bare policy reset
  missed. Guarded for uvloop-not-imported + best-effort on attribute drift;
  idempotent.

File: `skyrl-train/examples/terminal_bench/rollout_coordinator.py`
- `RolloutCoordinator.__init__`: the secondary backstop now CALLS the hardened
  `_force_stock_asyncio_in_worker()` (env var + uvloop neutralization) instead of
  only resetting the policy.

(2) makes sure no uvloop loop exists; (1) is the fallback that disables the
buggy libuv io_uring path even if some uvloop loop survives. Either alone kills
the abort; together they cover the C++ CoreWorker AND the litellm/aiohttp SSL
paths.

## Validation
- `ast.parse` clean on both files.
- Local (uvloop 0.22.1): after the hook, `UV_USE_IO_URING=0` is set;
  `uvloop.new_event_loop()`, `uvloop.Loop()` → `_UnixSelectorEventLoop`;
  `uvloop.EventLoopPolicy` → stock `DefaultEventLoopPolicy`.
- **In-SIF on Jupiter** (`_cp_fixb3.sif`, uvloop 0.22.1 = the SIF's actual
  version, editable skyrl_train via PYTHONPATH): baseline `uvloop.new_event_loop()`
  = uvloop loop; after the hook `UV_USE_IO_URING=0` and
  `uvloop.new_event_loop()`/`uvloop.Loop()` both yield SelectorEventLoop. VERIFY_OK.

## Sync
- Commit `0554dae` on marin `penfever/working`, pushed.
- `git pull` on Jupiter SkyRL clone (`/e/scratch/jureap59/feuer1/OpenThoughts-Agent/SkyRL`),
  fast-forward `47bf11f..0554dae`, HEAD = `0554dae`. Editable install → live.

## Relaunch
- cp2 (32k / CP2 / R3-off), `_cp_fixb3.sif`, `6node_qwen3_30b_a3b_32k_cp2.yaml`,
  detached tmux `cp2prod`, `--num_nodes 6 --reservation reformo
  --time_limit 11:59:00 --max_restarts 5`, db_reg false (auto-injected).
- RL concurrency at launch: 2 RUNNING (927673 lever1, 925740 swesmith) → 3 with
  cp2, under the ≤6 cap. (eval_SERA jobs are eval, not RL.)
- **Head jobid 930367** (chain 930368-930372 = 5 afterany restart links). Exp
  dir forked to `rl_30b_a3b_32k_cp2_4`. RUNNING 2026-06-20T00:57:05 on
  jpbo-004-[01,03-07]. Rendered sbatch confirmed `_cp_fixb3.sif`, --nodes=6,
  --reservation=reformo, and (host-level) `export UV_USE_IO_URING=0` (line 182)
  — but the FIX is the runtime_env env_vars + worker-boot hook injection, which
  is what actually reaches the actor; the host export was already present in the
  sbatch yet inert at the actor, exactly the gap diagnosed above.

## Verdict
PENDING — monitoring 930367 for genbuf advance past 2/64 (where 930208 died)
with NO `uvloop/sslproto` / `Fatal Python error: Aborted` in any
RolloutCoordinator. Will update with the genbuf milestone + abort-free
confirmation.
