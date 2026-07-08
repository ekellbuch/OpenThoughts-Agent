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

**Region (cross-region egress is the #1 cost footgun).** Model weight buckets are
regional: `gs://marin-models-us/...` and `gs://marin-models-eu/...`. Cross-continent
reads are a major cost driver and project policy forbids them (`AGENTS.md`).
- Keep **model bucket, `jobs_dir`/output bucket, and worker region in the same
  multi-region** (all US or all EU).
- The launcher auto-pins the job to the region with most capacity for the TPU
  type (`hpc/iris_launch_utils.py:discover_region_for_tpu`) and routes output to
  the matching multi-region bucket. **Static default `DEFAULT_GCS_OUTPUT_ROOT` is
  `gs://marin-eu-west4/...`** — if you neither set `--gcs-output-dir` nor let the
  pin run, a US placement reads EU = egress.
- `--gcs-output-dir gs://marin-models-us/ot-agent` **opts out of region pin**
  (places on first free worker in any US region — the fix for a collapsed
  single-region pool, §3 stuck-PENDING) while keeping output in US multi-region.
  Only with a US model bucket.

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
  or instantly free capacity. Fix: relaunch **unpinned** with
  `--gcs-output-dir gs://marin-models-us/ot-agent` (iris places on any free US
  worker). Kill the stuck submission only with user permission.

### Wedged / stalled TRAINING run (coordinator + child) — checkpoint-resume
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
  its GCS mtime** advancing:
  `gsutil ls -l <output_dir>/checkpoints/step-*/metadata.json | sort -k2 | tail`
  — is a NEW step being written recently? Plus cgroup mem still moving. Frozen
  checkpoint mtime for hours + healthy cgroup = progress WEDGE (not OOM/leak).
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
Results land in **GCS only**: `gs://marin-models-us/ot-agent/<job>/<job>/`.
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
- **Rescue banked traces** before/after a kill if the repo didn't auto-create:
  `gsutil -m rsync -r gs://marin-models-us/ot-agent/<job>/<job>/ /tmp/<job>/` then
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
