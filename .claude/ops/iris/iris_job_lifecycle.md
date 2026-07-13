# Iris jobs on Google TPU cloud — launch, monitor, tear down

Operational lifecycle for OpenThoughts-Agent jobs on Marin's Iris-managed Google
TPU cloud. Hardware specs (per-chip HBM/FLOPs, slice totals) live in
`iris_google_tpu_cloud_hardware.md`; canonical upstream ops:
`marin:lib/iris/OPS.md`.

Conventions: `IRIS = /Users/benjaminfeuer/Documents/marin/.venv/bin/iris`
(or `conda activate marin && uv run iris`); launch from the **otagent py3.12
conda env**; `source "$DC_AGENT_SECRET_ENV"` first.

---

## 1. Launch

> **ℹ NEW federated submission `iris.oa.dev` (2026-07-09) — see `coreweave_gpu_ops.md` §iris.oa.dev.** An
> easier path is coming: submit to `iris.oa.dev` (needs `iris login` w/ the openathena.ai gmail); an **H100
> request auto-routes to a CW cluster via a meta-scheduler**, `--target-cluster` pins one. It will become the
> default as bugs settle; **our existing paths (this doc's `--cluster=marin` TPU submission + the CW
> controller-tunnel launcher) still work** and are NOT the rejected "old-style SSH tunnel" (validated
> 2026-07-09). On the new path always pass an explicit cluster (no-target ⇒ random dispatch).

Two entrypoints (both submit `--no-wait`; a launchd fetch daemon mirrors outputs
back — see §2):
- **datagen / tracegen** → `data/cloud/launch_tracegen_iris.py`
- **eval** → `eval/cloud/launch_eval_iris.py`

Both forward `--harbor_config`, `--model` (or infer from `--datagen_config`),
`--tpu`, `--n_concurrent`, `--secrets-env`, `--upload_hf_repo`,
`--gcs-output-dir`, `--no-wait`, and auto-inject
`--harbor_extra_arg=--jobs-dir=<gcs_output_dir>/<job>` so harbor writes through
fsspec/UPath straight to GCS. (Full templates: `run-datagen-iris` / `run-eval-iris`
skills.)

**Eval on `dev_set_v2` — pass `--hf-offline-mode off`.** The default `auto` runs
an inline `snapshot_download` of the full `dev_set_v2` tree (300 task-environment
folders, hundreds of tiny files) + GCS mirror **before** submit → a 15–25 min
submit-stall. Only heavy unmirrored datasets bite; model side stays inert.
Safe to kill a stalled launcher mid-`snapshot_download` (GCS upload only starts
after the snapshot completes) — clean the tmp mirror dir.

### Before you submit — region, disk, node shape

**Region (cross-region egress is the #1 cost footgun).** Model **weight** buckets
stay multi-region: `gs://marin-models-us/...` and `gs://marin-models-eu/...`
(durable inputs). Transient **outputs** (trace dirs, eval outputs, `xla_cache`)
now route to a co-located **single-region** bucket (`gs://marin-us-east5`,
`gs://marin-eu-west4`, …) — ~half the multi-region cost, still read/write-local.
Cross-continent reads are a major cost driver and project policy forbids them
(`AGENTS.md`).
- Keep **model weight bucket + worker region in the same multi-region** (all US or
  all EU); the launcher handles output placement.
- The launcher auto-pins the job to the region with most capacity for the TPU type
  (`hpc/iris/regions.py:discover_region_for_tpu`) and routes output to that
  region's **single-region** bucket (`output_bucket_for_region`). It records the
  chosen output URI in the registry, so readers resolve it via
  `hpc.iris.job_output_resolver` (never a hardcoded bucket). **Static default
  `DEFAULT_GCS_OUTPUT_ROOT` is `gs://marin-eu-west4/...`** (single-region EU) — the
  discovery-failed fallback; a US placement that lands here reads EU = egress, so
  let the pin run.
- `--gcs-output-dir gs://marin-models-us/ot-agent` **opts out of the region pin AND
  the single-region routing** (forces that multi-region bucket; places on first
  free worker in any US region — the fix for a collapsed single-region pool, §3
  stuck-PENDING). A deliberate override; prefer leaving the pin on.

**Local disk (~100 GB/node ceiling).** Each TPU worker node has only ~100 GB.
- **Stream the model from GCS** (`--load-format runai_streamer`, gs:// URIs) —
  do NOT download a full checkpoint (122B-FP8 alone is ~122 GB).
- **Write `jobs_dir` to `gs://`, never local.**
- On memory-heavy/repo-based datasets, bound harbor RSS with
  `release_trial_payloads_in_memory: true` (`ctx32k_verified.yaml`) — else the
  orchestrator accumulates completed-trial payloads and OOMs the container
  (~256 GB host RAM, distinct from the 100 GB disk).
- `--disk` defaults to 5 GB ephemeral; raising it does not change the node ceiling.

**Node shape — get chip/host counts from the codebase, not arithmetic.**
"Chips ÷ 4 = hosts" is wrong (v5p counts *cores not chips*; v6e single-host packs
up to 8 chips). Authoritative sources:
- **Host/process count:** `iris.cli.job.get_tpu_topology("<variant>").vm_count`.
  Known good: `v5p-8 → 1`, `v5p-16 → 2`, `v5p-32 → 4`, `v6e-8 → 1`,
  `v6e-16 → 4`. The launcher uses this to auto-set `--replicas`.
- **v5p naming is CORES, not chips:** `v5p-N` = N cores = **N/2 chips**. So
  `v5p-8` = 4 chips (1 host), `v5p-32` = 16 chips (4 hosts). **Tensor-parallel
  degree must be ≤ chip count, not core count** — TP=8 won't fit v5p-8 (4 chips).
- **Live capacity + real chip counts:** query the cluster's `workers` table —
  ```bash
  $IRIS --cluster=marin query "SELECT device_variant, count(*) workers, sum(total_tpu_count) chips FROM workers WHERE device_type='tpu' GROUP BY device_variant ORDER BY device_variant" -f csv
  ```
- **Pools / variants / zones:** `marin:lib/iris/config/marin.yaml`. Per-chip HBM
  + slice totals in `iris_google_tpu_cloud_hardware.md`.
- 122B-FP8 fits **v5p** (95 GB HBM/chip) but **not v6e-8** (32 GB/chip, 256
  GB/slice) — weights + MoE footprint + compile peak exceed it.

**Cold-compile budget:** 122B-FP8 first-serve compile can take ~60 min. Pass
`--health_max_attempts 600`; the default (~50 min) kills the job before it serves.

---

## 2. Monitor

```bash
# Iris-side state (1=PENDING 2=starting 3=RUNNING 4=SUCCEEDED 5=FAILED 6=KILLED)
$IRIS --cluster=marin query "SELECT job_id, state FROM jobs WHERE state IN (1,2,3) AND job_id LIKE '/benjaminfeuer/%' ORDER BY job_id DESC" -f csv

# Full-history analyzer (paginates the WHOLE log via time windows; do NOT eyeball --tail)
/Users/benjaminfeuer/miniconda3/envs/otagent/bin/python scripts/iris/analyze_job_history.py /benjaminfeuer/<job> --refresh
#   sidecar JSON: total_runtime_s, iris_preemption_count, cycles[], serving_summary.{gen_tps,running}, non_empty_trials/total_trial_dirs, harbor_exception_stats

# Fetch daemon (mirrors GCS outputs → ~/.ot-agent/runs/<job>/ and captures .iris-job.log)
python -m hpc.iris_fetch_daemon status      # heartbeat should be ALIVE
$IRIS --cluster=marin job logs -f /benjaminfeuer/<job>    # live workload logs (controller tunnel)
```

> **⚠️ Never trust `iris job logs <job> --tail --max-lines N` for stats or
> debugging.** It truncates from the tail only — verbose Ray state-dumps crowd
> out the lines you care about, **under-sampling throughput by 10–100×**. The
> rollout/Harbor warning lines (e.g. a per-episode `AttributeError` in a Ray
> generator-worker actor → `generate/errors/*`) and throughput emissions live
> deep in the body. **Use `analyze_job_history.py`** — it paginates the entire
> log via fixed `--since-ms` + `--no-tail` windows and filters in python; even
> `--max-lines 200000` misses body lines the windowed walk recovers.

Outputs land in `~/.ot-agent/runs/<job>/` (daemon rsync) + `.iris-job.log`.
Health signal = productive-trial rate (`non_empty/total`) +
`harbor_exception_stats`; gen tok/s varies by dataset (short-task sets run lower).
Jobs on `:tpu` image `ae085bc8`+ **auto-upload** their HF repo on state-4 success —
verify the repo exists before any manual rescue.

---

## 3. Pitfalls & recovery

### Preemption (`--preemptible` workers)
- Normal and frequent; a slice can take 10+ preempts in a few hours. Each preempt
  → fresh worker → **cold XLA recompile** (~60 min v5p-8 cold; ~13–20 min warm).
- **XLA persistent cache** makes warm restarts fast. Namespaced per CPU-microarch
  and per model under `OT_AGENT_XLA_CACHE_BASE` (region-matched bucket, auto-set
  on iris) — do not point two different host CPUs at the same cache subdir
  (cross-host poison → wrong execution).
- **harbor resumes from the gs:// `jobs_dir`** — completed trials persist across
  preempts; only the recompile time is lost.
- `IRIS_TASK_ID` gains a `:N` suffix on retried/preempted attempts
  (e.g. `/user/job/0:2`) — rank-parsing must strip it
  (`.rsplit('/',1)[-1].split(':',1)[0]`) or it crashes on retry.
- **Stuck PENDING** = no capacity for that TPU type in the pinned region
  (preemptible pool can scale to zero). A finished job does NOT free its snapshot
  or instantly free capacity. For a fresh DATAGEN launch that won't place, the
  documented remedy is to relaunch **unpinned** with
  `--gcs-output-dir gs://marin-models-us/ot-agent` (iris places on any free US
  worker; note this override reverts outputs to the pricier multi-region bucket).
  Kill the stuck submission only with user permission.
- **⚑ PREEMPTIBLE JOBS STAY ON PREEMPTIBLE — a capacity-pending wait is NOT an
  escalation (operator, 2026-07-09).** When a `--preemptible` job (datagen OR a
  training child respawned by its coordinator) sits PENDING for hours on
  "no workers match constraints" = pure preemptible-pool scarcity, that is the
  EXPECTED steady state, not a fault. **Do NOT** propose/repin to a
  non-preemptible / on-demand slice, **do NOT** probe other zones to "rescue" it,
  and **do NOT** surface it to the user as a decision — the operator's standing
  answer is "jobs on preemptible stay there." Just report the pending status and
  let it self-place; a durable parent (`--max-retries`) or coordinator guarantees
  resume the moment a slice frees. (This overrides the earlier "escalate a long
  capacity stall" reflex. The unpinned-relaunch remedy above is still fine for a
  *fresh* datagen launch that never placed — that's a region-pin fix, not an
  on-demand upsell.)
- **⚑ DON'T GATE keep-N REFILLS ON A CAPACITY GUESS — let iris's scheduler place
  them (operator, 2026-07-09).** When keep-N is below target, SUBMIT the refill(s)
  and let the iris queue manager decide placement; a submitted job that sits PENDING
  behind a full pool is fine (it places when a slice frees). **Do NOT withhold or
  defer a refill because you eyeballed "0 free TPUs" / a long pending queue** — that
  is the scheduler's job, not yours, and guessing just starves the campaign. Submit
  to keep-N every tick; only skip on a HARD blocker you can act on (Daytona snapshot
  cap with nothing reclaimable → note + move on). A pending refill is not an
  escalation (see the preemptible rule above).

### How the pools work — monotonic tier ladder + crash-vs-preempt reservations (2026-07-11)

Two mechanics explain almost every long `v5p…`/`v6e…` PENDING, including multi-hour ones. Both are
**capacity behavior, not config faults** — HOLD FAST applies; do NOT mis-escalate them as quota/config blocks.

- **Monotonic tier ladder (per pool).** Each preemptible pool (e.g. `v5p-preemptible/us-east5-a`) has a
  **tier ladder by slice size** — `v5p-8` = tier 1, `v5p-16` = tier 2, `v5p-32` = tier 3, `v5p-64` = tier 4,
  … up to 2048 (static in `marin:lib/iris/config/marin.yaml` tpu_pools). **The autoscaler will NOT scale up a
  higher tier while a LOWER tier in the SAME pool has unsatisfied demand** — the pending reason is literally
  `Autoscaler: tier_blocked: quota-pool tier monotonicity`. So a `v5p-64` (tier-4) request can sit PENDING for
  hours *even when the raw chips exist*, purely because tier-1 `v5p-8` demand (often OTHER users' jobs) is
  backlogged in that pool/zone. It self-resolves the instant the lower-tier backlog drains. Diagnose by
  enumerating same-pool lower-tier demand (the `workers` table + autoscaler snapshot: `peak_demand`,
  `slice_failed … no more capacity in zone`), NOT by touching your job.

- **A crash tears down the reservation; a preempt holds it.** While a training child is RUNNING it *holds* its
  slice, so the tier gate above is moot — which is why days of **preempt→resume** work fine (a preempt keeps the
  slice reserved and re-attaches). But a **hard crash (e.g. SIGSEGV exit 139) destroys the slice entirely**; the
  coordinator's `--max-retries` respawn must then **re-acquire the tier-N slice from scratch**, and *that* is
  when it hits the monotonicity gate at whatever the current contention is. So a crash-resume can be
  dramatically slower to place than a preempt-resume, even for the identical job — expected, not a wedge.

Worked example (midtrain `1e23_p33m67_k0p20`, 2026-07-10→11): child SIGSEGV'd @18:42Z → respawn sat **13h12m**
PENDING with zero task-attempts on `tier_blocked` behind a last-24h surge of others' `v5p-8` demand
(`calvinxu/dm-delphi` sweeps, `tonyhlee/eval-chimera`, GCP zone `us-east5-a` capacity exhausted) → **self-placed
@07:55Z** the moment that tier-1 backlog drained, on a freshly-formed v5p-64 slice, resuming from its last
checkpoint. No config change, no quota grant, no intervention — the HOLD-FAST wait was correct.

### Wedged / stalled TRAINING run (coordinator + child) — checkpoint-resume
> **⚠ The Executor + `ExecutorStep` are RETIRED (marin PR #6649, 2026-07-09) → lazy `ArtifactStep`
> (`marin.execution.lazy`, `remote(fn,…)`, `name@version`). See `.claude/projects/marin-executor/`.** The
> coordinator/child + `.executor_info` model below still describes OLD executor-launched runs (Delphi
> midtrains) — read them the same way — but NEW Levanter training uses `ArtifactStep`. `StepRunner` + its
> per-step distributed lock survive and **DEADLOCK an SPMD (srun N-rank GPU) launch** (one rank wins the lock,
> the rest spin, the JAX mesh never forms — issue #7080); the workaround is to call the Levanter entrypoint
> directly in every rank (bypass `StepRunner`). Full detail in the marin-executor project doc.

For executor-dispatched training (a CPU **coordinator** submits a v5p **child**
training job, e.g. a Levanter midtrain), recovery differs from datagen:
- **`--max-retries` auto-resumes ORGANIC failures only.** Child FAILED(5)/preempted
  → coordinator relaunches a fresh child resuming from the latest checkpoint. It
  does NOT catch a WEDGE (child `state 3` but frozen — never FAILED, no retry).
  It also **reuses the launch-time bundle**, so a CODE fix reaches the worker only
  on a FRESH relaunch.
- **Stopping the COORDINATOR is TERMINAL** — `iris job stop <coordinator>` kills
  children AND does NOT relaunch (it's the *abandon* path). Do NOT expect
  `--max-retries` to bring it back.
- **Liveness/wedge = ADVANCEMENT, not presence.** A training run can be `state 3
  RUNNING` with `iris job logs` showing a recent step yet be dead — the CLI/IAP
  log window freezes at a stale timestamp. Judge by **SAVED-CHECKPOINT step AND
  its GCS mtime** advancing — but **use the RIGHT checkpoint path:**
  - **⚠ The PERMANENT path is COARSE (retains only sparse N×1500-step
    checkpoints) — do NOT use it for wedge detection.** For delphi/levanter it
    keeps only every-1500-steps (…3000, 4500, 6000, 7500…), so at ~76 s/step its
    newest checkpoint can be **>1 DAY old on a perfectly healthy run** → a FALSE
    "frozen/wedged" reading. (2026-07-08: permanent path stuck at step-6000 from
    2 days prior while training was live at step ~7256.)
    `gs://<bucket>/checkpoints/<run>/checkpoints/step-*/metadata.json`
  - **✅ Use the fine-grained TEMP (TTL) rolling-checkpoint path — that is the
    real liveness signal** (a fresh checkpoint every N steps, ~10-20 steps behind
    live):
    `gsutil ls -l 'gs://<bucket>/tmp/ttl=14d/checkpoints-temp/<bucket>/checkpoints/<run>/checkpoints/step-*/metadata.json' | sort -k2 | tail`
    (e.g. delphi `b6607e`: temp step-7235 @ 17:33Z, ~20 steps behind live 7256 —
    advancing = healthy.)
  - Cross-check with `iris job logs` live-step advancement too. Frozen TEMP-path
    checkpoint mtime for hours + frozen logs + healthy cgroup = progress WEDGE
    (not OOM/leak). NEVER declare a wedge off the permanent path alone.
- **Recover a confirmed wedge:**
  - **Option A (PRIMARY): stop the wedged CHILD only, NOT the coordinator.**
    `iris job stop <child_task_id>` forces the child terminal (KILLED) → the
    coordinator's executor respawn loop spawns a FRESH child that auto-resumes
    from the latest checkpoint on the pinned output dir. Proven recovery; preserves
    the healthy coordinator, `--max-retries` durability, and W&B run identity.
    Confirm the new child loads step-N (tqdm `Nk/29.9k` / cgroup callback), not step 0.
  - **Option B (FALLBACK): stop-coordinator → relaunch-FRESH on the SAME output
    dir.** Use ONLY if the coordinator is dead/won't respawn, OR to deploy a CODE
    fix (child-only bounce reuses the old bundle). Verify the relaunch targets the
    SAME output-dir tag (step-0 start = wrong dir = lost progress).
- **Standing authority (user, 2026-07-08):** a CONFIRMED-wedged training run
  (frozen checkpoint mtime + frozen logs for hours, cgroup healthy) may be
  auto-bounced (stop-coordinator + relaunch-fresh) autonomously.

### Daytona snapshot cap
Launches build a per-env Daytona snapshot; the shared `cli` org caps at 60. On
`SnapshotCapExceeded`, delete **only `MISSING`-state `harbor__*` snapshots**
(broken builds, safe). NEVER broad-prune (`cleanup_unused_snapshots`) on the shared
org — it removes ACTIVE snapshots other jobs depend on. Snippet in the
`run-datagen-iris` skill.

### Local-storage growth on the launch host
The daemon mirror under `~/.ot-agent/runs/` (and `.iris-job.log`, 10s of MB)
accumulates across jobs. `python -m hpc.local_paths inventory` lists sizes;
`... clean --older-than 30d --apply` purges old runs. (Launch *host*; distinct
from the 100 GB worker-node ceiling in §1.)

### Empty GCS prefix after a "successful" job
The workload didn't route through UPath. Confirm
`--harbor_extra_arg=--jobs-dir=<gcs>` is in the submitted command
(`iris job bug-report <id>`) and that the harbor pin is the UPath-aware build.

### TPU agentic eval `--upload_to_database` is a NO-OP (GCS-only results)
An `eval/cloud/launch_eval_iris.py` eval with `--upload_to_database` does NOT
push traces to HF and does NOT register the score to Supabase — post-eval upload
keys off a local Harbor job dir (`/app/jobs/<job>`) that doesn't exist on the TPU
runtime (trials stream to GCS). Log tell:
`[upload] Expected Harbor job directory /app/jobs/<job> does not exist; upload skipped.`
GPU/SLURM has the local dir so upload works there; TPU is the broken path.
Results land in **GCS only**, under the job's recorded output prefix
`<job_output_dir>/<job>/` (single- or multi-region — resolve it with
`python -m hpc.iris.job_output_resolver <job> --cluster …/marin.yaml`, don't
hardcode `gs://marin-models-us`).
- **Harvest scores from GCS**, not Supabase: `result.json` →
  `stats.evals.<id>.reward_stats` (+ `exception_stats`).
- **Traces to the Hub:** `gsutil rsync` the GCS job dir, then
  `scripts/harbor/make_and_upload_trace_dataset.py --episodes last --filter none --skip_register
  --chunk_size 300` (one row per trial; `--episodes all` explodes to per-step rows).

---

## 4. Teardown

```bash
# Kill a job (ONLY with explicit user permission for a RUNNING/placed job)
$IRIS --cluster=marin job kill /benjaminfeuer/<job>
```
- **Rescue banked traces** before/after a kill if the repo didn't auto-create
  (resolve the recorded output prefix — never hardcode a bucket):
  `OUT=$(python -m hpc.iris.job_output_resolver <job> --cluster …/marin.yaml)` then
  `gsutil -m rsync -r "$OUT/<job>/" /tmp/<job>/` then
  `scripts/harbor/make_and_upload_trace_dataset.py --job_dir /tmp/<job> --repo_id penfever/<slug>-... --episodes last --filter none --skip_register`.
- **NEVER** `iris cluster restart` / stop / bounce the cluster without explicit
  user approval — it kills every running job. `job kill` is job-scoped (safe with
  permission); cluster ops are not. Killing the job frees its workers; there is no
  separate teardown step for the TPU slice (iris reclaims preemptible workers).

> **CoreWeave GPU cluster (a DIFFERENT cluster) —** stop a job with
> `iris --cluster coreweave job stop /<user>/<job>` (full binary
> `/Users/benjaminfeuer/miniconda3/envs/otagent/bin/iris`; `which iris` fails).
> **`--cluster` goes BEFORE the subcommand** — bare `job stop <jid>` errors "No
> controller specified"; `job stop <jid> --cluster …` errors "No such option".
> Export `KUBECONFIG=~/.kube/coreweave-iris-gpu` first or it falls back to a stale
> lambdaconfig and errors. `stop`/`kill` are aliases → prints `Terminated jobs:`.
> A hard-kill ORPHANS in-flight Daytona sandboxes — reap them. Full GPU-RL ops:
> `coreweave_gpu_ops.md`.

---

## Authoritative references
- `marin:lib/iris/OPS.md` — cluster lifecycle, controller, SQL, GCP ops (read first).
- `marin:lib/iris/config/marin.yaml` — pools, variants, zones.
- `iris.cli.job.get_tpu_topology(variant)` — vm_count / chip topology (don't guess).
- `iris_google_tpu_cloud_hardware.md` (this dir) — per-chip + slice hardware specs.
- `notes/marin/tech.md` — the OT-Agent↔iris fetch-daemon architecture + flag cheat sheet.
