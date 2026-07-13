---
name: datagen-launch-iris
description: Launch, monitor, and manually clean up a trajectory-generation (datagen) job on Marin's Iris TPU cluster via the OpenThoughts-Agent entrypoint. Use when asked to start, watch, rescue, or kill a datagen/tracegen run on Iris.
---

# datagen-launch-iris

> **📍 Iris orientation — read first.** Before acting on anything in this skill, read the Iris **tools
> catalog** (`.claude/ops/iris/iris_tools.md`) and the Iris **ops directory** (`.claude/ops/iris/` — the
> CoreWeave GPU particulars in `coreweave_gpu_ops.md`, the TPU `marin` particulars in `iris_job_lifecycle.md`).
> They carry the binding access/preamble/gotchas and the helper-script inventory the steps below rely on.

End-to-end operation of a datagen (trajectory-generation) job through
`data/cloud/launch_tracegen_iris.py`. Covers launch → monitor → manual cleanup.
For **eval** jobs use the **eval-agentic-launch-iris** skill instead.

## Required info (ask if missing)

1. `tasks` — the task source for `--tasks_input_path`: an **HF dataset id** (e.g.
   `DCAgent/exp_rpt_e2egit-v2`; the launcher `snapshot_download`s it and
   auto-explodes a `task_binary` parquet into task dirs) **or** a local tasks
   directory. Not both.
2. `slug` — short dataset name used for the job name and HF repo (e.g.
   `e2egit-v2`).
3. Operating point — default is **S1** (Qwen3.5-122B-A10B-FP8, 32k, v5p-8,
   single-host). Don't change configs unless asked.

## Prerequisites

- **Launch from a Python 3.12 env** — the iris client writes the launcher's
  `sys.version_info` into the worker's `uv sync --python`. Use the otagent conda
  env: `source /Users/benjaminfeuer/miniconda3/etc/profile.d/conda.sh && conda activate otagent`
  (or call `/Users/benjaminfeuer/miniconda3/envs/otagent/bin/python` directly).
- **Secrets**: `source "$DC_AGENT_SECRET_ENV"` (provides
  `DAYTONA_API_KEY` for the host-side snapshot pre-build, `HF_TOKEN` for upload,
  `MARIN_HMAC_*` for runai_streamer). Also pass `--secrets-env <path>` so they
  reach the worker. Do not echo secret values.
- If a launch fails with `marin-iris client is too old`, run
  `git -C /Users/benjaminfeuer/Documents/marin pull --ff-only origin main` (iris
  is an editable install from that checkout) — **not** `uv sync`.

## Launch

```bash
cd /Users/benjaminfeuer/Documents/OpenThoughts-Agent
source /Users/benjaminfeuer/miniconda3/etc/profile.d/conda.sh && conda activate otagent
source "${DC_AGENT_SECRET_ENV:?set DC_AGENT_SECRET_ENV to the secrets file first}"
TS=$(date +%Y%m%d-%H%M%S)
python data/cloud/launch_tracegen_iris.py \
  --harbor_config hpc/harbor_yaml/datagen/ctx32k_verified.yaml \
  --datagen_config hpc/datagen_yaml/qwen3_5_122b_a10b_fp8_runai_v5p8_s1.yaml \
  --tasks_input_path <HF-dataset-id | /abs/tasks/dir> \
  --tpu v5p-8 --preemptible \
  --n_concurrent 64 --n_attempts 1 --health_max_attempts 600 \
  --job_name "qwen3.5-122b-32k-<slug>-${TS}" \
  --secrets-env "$DC_AGENT_SECRET_ENV" \
  --gcs-output-dir gs://marin-models-us/ot-agent \
  --upload_hf_repo penfever/<slug>-qwen3.5-122b-32k-traces \
  --no-wait
```

Flag notes:
- **Per-model serve config now resolves from `model_config/`** (commit `e792bfbb`, 2026-07-02): the
  launcher looks up the served model (from `--datagen_config`'s `engine.model` / `--model`) in
  `model_config/<org>/<slug>.yaml` via the shared resolver and **merges/forwards its `agent_kwargs`** +
  applies serve intrinsics (`max_model_len` / `limit_mm` / `extra_args`) on the worker — so datagen no
  longer diverges from the source of truth. `tp_size` + `harbor_config` are **ignored** on TPU (tp from
  chip count; harbor_config CLI-required). **Precedence: explicit CLI / `--datagen_config` values always
  win** over `model_config/`; a model with no entry launches byte-unchanged (logged). Edit the source
  file `model_config/<org>/<slug>.yaml`, never the generated `eval/configs/model_configs.yaml`.
- **⚠ PREFER building `--gcs-output-dir` (and the model-mirror path) off `marin_prefix()`
  (`rigging.filesystem` — auto-resolves the active cluster's storage root; don't hardcode the region
  bucket); the literal below is a fallback. See `.claude/ops/iris/coreweave_gpu_ops.md` §rendezvous.**
- **`--gcs-output-dir gs://marin-models-us/ot-agent` opts OUT of the region
  pin** (the launcher would otherwise auto-pin to the region with the most v5p-8
  capacity). Use it to avoid the stuck-PENDING trap when a single region's v5p-8
  pool has collapsed — iris then places on the first free v5p-8 in any US region.
  Keep output in the **US** multi-region bucket (matches the model bucket; never
  read/write GCS cross-region).
- `ctx32k_verified.yaml` = verifier ON + the `release_trial_payloads_in_memory`
  flag (bounds worker host-RAM so heavy/repo-based datasets don't OOM the
  container). Use the 32k config with the 32k S1 engine.
- `--health_max_attempts 600` is mandatory (122B-FP8 cold compile can take ~60
  min; the default 100 ≈ 50 min kills the job before first serve).
- `--n_concurrent 64` = `max_num_seqs(64) × DP(1)`.
- Image `:tpu` at/after digest `ae085bc8` (commit c2073e0e) **auto-uploads** the
  HF repo on a clean (state-4) completion — no manual rescue needed for those.

After submit, confirm placement:
```bash
/Users/benjaminfeuer/Documents/marin/.venv/bin/iris --cluster=marin query \
  "SELECT job_id, state FROM jobs WHERE job_id='/benjaminfeuer/<job>'" -f csv
```
state 1=PENDING, 2=starting, 3=RUNNING, 4=SUCCEEDED, 5=FAILED, 6=KILLED.

## Monitor

```bash
/Users/benjaminfeuer/miniconda3/envs/otagent/bin/python \
  /Users/benjaminfeuer/Documents/OpenThoughts-Agent/scripts/iris/analyze_job_history.py \
  /benjaminfeuer/<job> --output /tmp/<job>_history.md --refresh
```
Read the `.json` sidecar (it paginates the full history — don't eyeball
`--tail`): `total_runtime_s`, `iris_preemption_count`, `cycles[]` (each with
`did_serve`/`time_to_first_serve_s`), `serving_summary.gen_tps`/`.running`
(n/mean/max), `non_empty_trials` / `total_trial_dirs` (productive rate),
`harbor_exception_stats`. S1 baseline ≈ 400 mean / 1115 peak gen tok/s; short-task
datasets (nl2bash, e2egit) run lower gen tok/s by nature — judge by productive
trial rate, not tok/s alone.

**Did it auto-upload?** On a state-4 job from image `ae085bc8`+, check the repo
exists before rescuing:
```bash
/Users/benjaminfeuer/miniconda3/envs/otagent/bin/python -c \
 "from huggingface_hub import HfApi; print(HfApi().dataset_info('penfever/<slug>-qwen3.5-122b-32k-traces').lastModified)"
```

## Manual cleanup

**Kill a job** (only with explicit user permission for a RUNNING/placed job):
```bash
/Users/benjaminfeuer/Documents/marin/.venv/bin/iris --cluster=marin job kill /benjaminfeuer/<job>
```

**Rescue banked traces** (any terminal job whose repo did NOT auto-create — e.g.
killed, OOM, or pre-`ae085bc8` image). Rsync the GCS job dir local, then push:
```bash
source "${DC_AGENT_SECRET_ENV:?set DC_AGENT_SECRET_ENV to the secrets file first}"
gsutil -m rsync -r gs://marin-models-us/ot-agent/<job>/<job>/ /tmp/<job>_traces/
/Users/benjaminfeuer/miniconda3/envs/otagent/bin/python \
  /Users/benjaminfeuer/Documents/OpenThoughts-Agent/scripts/harbor/make_and_upload_trace_dataset.py \
  --job_dir /tmp/<job>_traces \
  --repo_id penfever/<slug>-qwen3.5-122b-32k-traces \
  --episodes last --filter none --skip_register
```
(`--skip_register` = upload only, no Supabase row. Repo is public.)

**Daytona snapshot cap** — if a launch fails with `SnapshotCapExceeded` on the
shared `cli` org, delete ONLY broken (`MISSING`-state) `harbor__*` snapshots
(never `ACTIVE` ones — those may belong to running jobs, yours or a teammate's):
```bash
source "${DC_AGENT_SECRET_ENV:?set DC_AGENT_SECRET_ENV to the secrets file first}"
/Users/benjaminfeuer/miniconda3/envs/otagent/bin/python - <<'PY'
import os
from hpc.snapshot_manager import _parse_org_arg, _SnapshotManager, list_snapshots
org=_parse_org_arg(f"cli={os.environ['DAYTONA_API_KEY']}")
mgr=_SnapshotManager([org]); client=mgr._client(org)
for snaps in list_snapshots([org]).values():
    for s in snaps:
        if s.name.startswith("harbor__") and s.state=="MISSING":
            client.snapshot.delete(client.snapshot.get(s.name)); print("deleted", s.name)
PY
```
Do NOT run the broad `cleanup_unused_snapshots` against the shared `cli` org — it
deletes by your task set and would remove teammates' ACTIVE snapshots.

When the org is at cap with **0 MISSING** (all ACTIVE) but many `harbor__` are idle
leftovers of completed jobs, use the idle-gated reclaim (safe; spares live/teammate +
base snapshots) via the **utils-reclaim-stale-snapshots** skill:
`daytona_snapshot_manager.py --name-prefix harbor__ --delete-stale`
— at the stale threshold defined in `.claude/projects/daytona/daytona.md` § "How to clean stale
snapshots" (GT — don't restate the value); the 3-hourly cron routinely reclaims 15–34/tick on the shared
`cli` org.

**Stuck PENDING** (no v5p-8 capacity): report it and relaunch UNPINNED (the
`--gcs-output-dir gs://marin-models-us/ot-agent` flag above). Kill the stuck
submission first only with user permission.

## Current campaign: 131k A2 opencode (2026-07) — deltas from the 32k S1 default above

The live campaign (`qwen3.5-122b-131k-opencode`) runs a different operating point + hard-won
guards. Campaign specifics (dataset order, keep-3 state, per-arm status) live in the tracker:
`/Users/benjaminfeuer/Documents/experiments/active/qwen3.5-122b-131k-datagen-opencode-iris/tracker.md`.
Dated chronology (bringup + every ops tick, 2026-07-02→08):
`~/Documents/agent_logs/2026-07-08_qwen3.5-122b-131k-datagen-opencode-iris_history.md`.
Literal-trace decode/rescue reference: `.claude/projects/harbor/harbor.md` (§ Literal-token trace datasets).
Ingress topology (native `/proxy/t/*` capability-URL, token TTL, pinggy rollback): `.claude/ops/iris/iris_ingress.md`.

**Recipe deltas from the Launch block above:**
- `--datagen_config hpc/datagen_yaml/extra/qwen3_5_122b_a10b_fp8_runai_v5p8_131k_dp1_tp4_ep1_s32.yaml` (A2: TP4/DP1/EP-on/seqs32/131k), `--n_concurrent 32`.
- `--harbor_config hpc/harbor_yaml/datagen/opencode_ctx131k.yaml` — **FULL path required** (a bare `opencode_ctx131k.yaml` `FileNotFoundError`s on the worker — no config-dir fallback).
- `--agent opencode --record_literal` (literal-token capture; per-serve-unique log filenames since `c4728060`, so a preempt-resume can't clobber attempt-0's log).
- `--max-retries 200` — auto-resume across v5p preempts; the resume-fix (`a25a6125`) bakes a stable `--job_name` → deterministic `jobs_dir`/served-model-id, so a resume **continues the same GCS serve dir** (one serve dir, rising task count) instead of restarting from task 1.
- repo naming `penfever/<slug>-qwen3.5-122b-131k-opencode-traces`.

**⚠️ MODEL PATH GUARD (the #1 launch pitfall):** `--model gs://marin-models-us/ot-agent/models/Qwen/Qwen3.5-122B-A10B-FP8/` — the mirror, which streams via runai-streamer **direct-to-device (~0 local disk)**. **NEVER the bare HF id `Qwen/Qwen3.5-122B-A10B-FP8`** — that string is ONLY the uploader's `--served_model` provenance stamp; passing it as `--model` defeats streaming → vLLM `snapshot_download`s the full ~120GB → OOMs the 100GB worker disk → **deterministic startup hang (0 trials for hours, never self-heals)**. After launch, grep the new job's log to confirm the baked command carries `--model gs://…`.

**Native ingress (pinggy retired 2026-07-06):** add `--ingress_mode controller --ingress_host https://iris.oa.dev`. The Daytona sandbox reaches the co-located vLLM via the capability URL `https://iris.oa.dev/proxy/t/<token>/otagent-<slug>/v1` (access=LINK, 24h token re-minted per serve-spawn). Route health: `curl -sk -w "%{http_code}" https://iris.oa.dev/proxy/t/badtoken/serve.nope/v1/models` → **401**. If a job fails specifically on `/proxy/t`, do NOT redeploy pinggy — flag it (rollback = redeploy the sidecar via `scripts/inference/deploy_ingress_sidecar.py`).

**Liveness / wedge check — a RUNNING job can be silently DEAD (verify FRESHNESS, not just "activity present").** A datagen job can sit `state 3 RUNNING` with vLLM logs full of `running agent` lines yet be wedged — the engine cliff-died and every "activity" line is stamped at the SAME frozen timestamp (the §4b frozen-spinner trap). Do NOT judge liveness by the presence of `running agent`/serving lines; judge by whether things are ADVANCING over a short live window: (1) OUTER-dir GCS **trial-dir count grows** (sample, wait ~4–5 min, re-count); (2) harbor **`<done>/<total>` completed advances** / `result.json updated_at` is fresh (not hours stale); (3) vLLM emits **fresh** `chat/completions` `200 OK` + nonzero-and-moving `Running: N` (check the newest log TIMESTAMP is recent, not frozen). Frozen on all three with an engine that served-then-stopped = a confirmed wedge → kill+rescue+relaunch (standing kill authority). Carve-out: a job mid harbor GCS-`jobs_dir` **resume scan** after a preempt can be legitimately progress-frozen for up to ~6h WITH a *recent* engine-ready marker and NO prior serving on this attempt (monitor-restore-iris §4c) — that's NOT a wedge; don't kill. (History 2026-07-07: #18b `135552` sat "healthy" for ~6h across 3 ticks on frozen `19:59:40` spinner logs before a freshness check caught it dead.)

**Rescue vs RESUME policy (RESUME by default for sub-60% jobs):** when a job goes terminal
(FAILED/KILLED) short of completion, the default is to **RESUME** (relaunch the same arm → same
baked `--job_name`/gs:// job dir → harbor continues the remaining tasks), NOT to harvest a partial
banked slice into HF. **Rescue (banked-GCS → HF) ONLY when ≥60% complete** (`<done>/<total>` from
the harbor `Mean:` line, or productive-trial dirs / total); below 60%, resume — a rescue that far
short just publishes a stub and burns the arm's identity.

**⚠️ The harbor resume/export-push fix is NOT deployed in any current `:tpu` image.** `c49064a8`
(the `_maybe_init_existing_job` config-drift fix) did NOT bump the harbor version (still `0.8.0`),
so the image build's `uv pip install ".[datagen-tpu]"` resolved a **cached pre-fix `harbor==0.8.0`
wheel** — both `:tpu-de98374e` and the prior `:tpu-7951edcd` lack the fix (validated live 2026-07-08:
#18c died at `job.py:263` on preempt-resume). Consequence: **every preemptible datagen job still
dies on its first preempt-resume export-push → banked-GCS rescue is the reliable harvester.** Real
remedy (⚑ user decision): a corrected image rebuild that busts the harbor wheel cache (bump the
harbor version / `HARBOR_COMMIT` + `--force-reinstall`, or a fresh-from-GitHub source install),
then VERIFY by grepping the installed `harbor/job.py` INSIDE the built image — see the
**`build-tpu-image-iris`** skill (this exact stale-cached-wheel lesson is its reason to exist).
Rolling `:tpu` back to `879ebaba` is a no-op (it also predates the fix). **Fixing harbor code
requires busting the cached wheel to take effect in an image rebuild** — the general lesson.

**Terminal-job triage — rescue vs BLOCKED (precheck before rescuing):**
- Worker in-job auto-upload lands **TEXT-ONLY** (the pinned `:tpu` predates the schema-pin fix) OR fails at **export-push** (harbor `FileExistsError` on preempt-resume — see caveat above) — both leave banked GCS trials + `logs/*_literal.jsonl` intact → **RESCUE** from GCS (see Manual cleanup) with `--served_model Qwen/Qwen3.5-122B-A10B-FP8`, verify `count_populated_literal_rows` ≈ correlation yield. "SUCCEEDED"/"repo exists" is NEVER proof of trainable literals — always check the true count.
- A job with **0 valid traces** (100% `steps:0` / `NonZeroAgentExitCodeError exit 127` = agent binary absent in the sandbox) is GARBAGE → mark **BLOCKED**, NOT rescuable, needs a full re-run after the sandbox-install fix. Spot-check a few trials' `result.json`/`exception.txt` before rescuing.

**Harbor editable guard:** the uploader imports harbor from `/Users/benjaminfeuer/Documents/harbor` — it MUST be on `penfever/working` (verify + `git checkout penfever/working` if drifted) before any rescue, or the export crashes.

## Guardrails

- **Kill authority:** a HEALTHY (progressing) running job → NEVER kill without explicit
  user permission. BUT a CLEARLY-DOOMED / mis-launched job that makes 0 progress for hours
  and won't self-heal — bare-HF-id disk-full hang, exit-127 garbage, or a severed-Daytona
  zombie (frozen samples AND frozen harbor exception counters) — MAY be killed + relaunched
  correctly WITHOUT asking (user standing authorization, 2026-07-07); log it to the tracker.
- NEVER stop/restart/bounce a HEALTHY RUNNING job or the Iris cluster without explicit
  user permission in the current thread.
- NEVER read/write GCS across regions (cost). Keep everything in the
  bucket-matched US region.
- Rescue + snapshot-MISSING cleanup are safe maintenance; killing a running job
  and broad snapshot pruning are not — confirm first.
