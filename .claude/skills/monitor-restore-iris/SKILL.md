---
name: monitor-restore-iris
description: Re-register the every-3-hours Iris job-monitor cron (status check + datagen auto-rescue/keep-2-in-flight) if it has been lost. Primarily the marin TPU datagen/eval jobs ("iris" = the marin TPU cluster); also queries CoreWeave (cw-us-east-02a) GPU-RL as monitor-only. The cron is session-only and recurring crons auto-expire after 7 days, so it's routinely lost on a session restart. Use at the start of a new session, after a restart, or when the user asks to restore/check the iris monitoring cron. The sweep PROCEDURE the cron runs lives in monitor-cron-sweep-iris.
---

# monitor-restore-iris

> **📍 Iris orientation — read first.** Before acting on anything in this skill, read the Iris **tools
> catalog** (`.claude/ops/iris/iris_tools.md`) and the Iris **ops directory** (`.claude/ops/iris/` — the
> CoreWeave GPU particulars in `coreweave_gpu_ops.md`, the TPU `marin` particulars in `iris_job_lifecycle.md`).
> They carry the binding access/preamble/gotchas and the helper-script inventory the steps below rely on.

The recurring cron that watches all of `benjaminfeuer`'s Iris jobs is
**session-only** (it lives in the Claude session, not on disk) and recurring
crons **auto-expire after 7 days** — so it is routinely lost on a session
restart. This skill is the durable source of truth for re-creating it: the
**canonical cron prompt below is what gets (re-)installed — copy it verbatim
into `CronCreate`.** The per-tick sweep *methodology* the prompt executes is
**monitor-cron-sweep-iris**.

(This is the lightweight marin-TPU-centric monitor. The separate broader
tri-cluster monitor — Leonardo + CoreWeave + TACC — is restored via
**monitor-restore** / **monitor-cron-sweep**.)

## When to run
- Start of a new session where Iris jobs are in flight.
- The user says the monitor/cron is gone, down, or "not firing."
- After ~7 days (expiry).

## Steps
1. **Check if it already exists** — call `CronList`. If a recurring job whose
   prompt mentions "status check on ALL Iris jobs for user benjaminfeuer" is
   present, do nothing (a duplicate causes redundant SQL/tunnel load). If a stale
   **datagen-only** variant exists (prompt mentions only `qwen3.5-122b-32k-%`),
   `CronDelete` it and recreate with the all-jobs prompt below.
2. **If absent, call `CronCreate`** with:
   - `cron`: `23 */3 * * *`  (every 3 h at :23 — off the :00/:30 marks)
   - `recurring`: `true`
   - `prompt`: the exact text in the fenced block below.
3. Tell the user the new job id + the two caveats: **session-only** (dies when
   this Claude session exits — re-run this skill next session) and **7-day
   auto-expiry**.

## Notes
- `durable: true` is NOT honored in this harness (it still creates a
  session-only job), so don't rely on it — this skill IS the persistence layer.
- The cron only fires while the REPL is idle (not mid-task). If it reliably
  misses, the fallback is the user pasting the prompt manually, or an external
  launchd monitor (out of scope here).
- It tracks ALL `/benjaminfeuer/%` jobs but the autonomous write actions
  (auto-rescue, keep-2-in-flight) are **datagen-only**; eval jobs are
  monitor-only (they self-sync to Supabase+HF). See **datagen-launch-iris** (launch/refill),
  **datagen-job-cleanup** (the canonical idempotent post-run cleanup for a TERMINAL datagen
  arm — realness `avg_turns` check → HF upload with literals + `--served_model` → non-empty +
  literal-yield verify → safe disk cleanup; dispatch a subagent armed with it on completion), and
  **eval-agentic-launch-iris**.
- **Two clusters.** The cron queries both the **marin** TPU cluster and the
  **`cw-us-east-02a`** CoreWeave GPU cluster. The marin `.venv` iris carries the
  `[controller]` deps so it drives CoreWeave too — but the CoreWeave query MUST be
  prefixed `KUBECONFIG=~/.kube/coreweave-iris-gpu`, else iris falls back to the
  shell-default kubeconfig (`~/.kube/lambdaconfig`) and errors with
  `Invalid kube-config file … Expected object with name`. GPU-RL jobs on CoreWeave
  are **monitor-only** (no rescue, no keep-2); their pods GC on terminal, so logs
  come from the persistent finelog server. Other CoreWeave GPU configs exist
  (`coreweave*` = US-WEST-04A, CI/smoke) but are NOT in scope unless the user runs
  jobs there.
- **The methodology each step encodes** (how to run the analyzer, classify, rescue,
  refill) is **monitor-cron-sweep-iris** — read it when actually executing a tick;
  this skill is just the (re)install wrapper + the canonical prompt.

## Canonical cron prompt (copy verbatim into CronCreate)

```
Every-3-hours status check on ALL Iris jobs for user benjaminfeuer (datagen + eval + GPU-RL + anything else), across BOTH the marin TPU cluster and the CoreWeave GPU cluster.

**⚠ NO EXPERIMENT-SPECIFICS IN THIS PROMPT (they go stale): the per-campaign values — in-flight TARGET, refill cluster/grouping/order, harvest gates, repo/image patterns, and current bugs — live in the EXPERIMENT TRACKERS under `~/Documents/experiments/active/` (and the `*-launch` / `*-cleanup` / `analyze-*` skills). READ the relevant tracker each tick and drive off IT; never rely on a number hardcoded here.**

**⛔ DATAGEN IS OUT OF SCOPE (2026-07-08 operator directive): a DIFFERENT agent manages ALL datagen (`tracegen-iris-%` / `qwen3.5-122b-%`). Do NOT analyze, rescue, keep-N, or take any action on datagen jobs — the datagen classes/actions below (§3A, §4, §5) are RETIRED for this monitor. This monitor now covers EVAL (§3B) + Levanter TRAINING (§3C) + CoreWeave GPU-RL (§3D) only.**

1. Active jobs (query BOTH clusters):
   1a. marin (TPU):
       /Users/benjaminfeuer/Documents/marin/.venv/bin/iris --cluster=marin query "SELECT job_id, state FROM jobs WHERE state IN (1,2,3) AND job_id LIKE '/benjaminfeuer/%' ORDER BY job_id DESC LIMIT 20" -f csv
   1b. cw-us-east-02a (CoreWeave GPU) — KUBECONFIG prefix is REQUIRED (else iris uses the wrong shell-default kubeconfig and errors):
       KUBECONFIG=~/.kube/coreweave-iris-gpu /Users/benjaminfeuer/Documents/marin/.venv/bin/iris --cluster=cw-us-east-02a query "SELECT job_id, state FROM jobs WHERE state IN (1,2,3) AND job_id LIKE '/benjaminfeuer/%' ORDER BY job_id DESC LIMIT 20" -f csv
   For EACH cluster also query state IN (4,5,6) LIMIT 8 to catch jobs that went terminal since the last tick. If the cw query errors (cluster down / creds), report that and continue with marin — do not fail the whole tick.

2. For each ACTIVE marin (TPU) datagen/eval job, run the harbor analyzer (TPU/harbor-shaped — does NOT apply to CoreWeave GPU-RL jobs; handle those per class D). Use the analyze-job-history-iris skill (the reliable foreground-and-wait recipe; dispatch a patient subagent for the big task-history jobs):
   /Users/benjaminfeuer/miniconda3/envs/otagent/bin/python /Users/benjaminfeuer/Documents/OpenThoughts-Agent/scripts/iris/analyze_job_history.py <job_id> --output /tmp/$(basename <job_id>)_history.md --refresh
   Report from the .json sidecar: runtime_h, iris_preemption_count, cycles total/served, samples (serving_summary.gen_tps.n), gen tok/s mean/peak, Running mean/peak, non_empty/total trials = rate, t_first_serve, top harbor_exception_stats. ALSO report mean reward + completed/total tasks from the harbor progress line (NOT in the sidecar): /Users/benjaminfeuer/Documents/marin/.venv/bin/iris --cluster=marin job logs <job_id> --max-lines 8000 | grep -aoE '[0-9]+/[0-9]+ Mean: [-0-9.]+' | tail -1

3. Print `## Iris jobs status — <ISO UTC>`: one line per job (name + state + CLOSED/PARTIAL/OPEN/DEAD), a compact metrics block, and a survival check (past cold compile? throughput sane? traces/results landing on HF?). Classify each job by job_id prefix and apply the right treatment:

   A. **Datagen** (`qwen3.5-122b-%` / `tracegen-iris-%`): **⛔ OUT OF SCOPE — a DIFFERENT agent manages ALL datagen (2026-07-08 operator directive).** Do NOT query, analyze (§2), rescue (§4), keep-N (§5), or take ANY action on datagen jobs in this sweep. If one appears in the state query, note its existence in ONE line at most and move on. §4 + §5 are BOTH retired for this monitor.

   B. **Eval** (`eval-%`): auto-sync to Supabase + HF on completion (`--upload_to_database`); build sandboxes at runtime (MAIN Daytona org). **ALWAYS report the leading metric (`<done>/<total> Mean: <X>`) per in-flight eval** (from `iris … job logs <job_id>`; not in the analyzer sidecar) + productive rate + exceptions; on terminal, whether results landed. A **one-off** eval is monitor-only (no rescue/relaunch). **⚠ EXCEPTION — an eval CAMPAIGN with a tracker in `active/` (e.g. `~/Documents/experiments/active/flawed_summ_evals/reeval_tracker.md`) DOES run an active harvest+refill loop: drive it PER THAT TRACKER each tick — its in-flight TARGET, refill cluster/grouping/order, harvest gate + discriminator, and gotchas ALL live in the tracker's TOP BLOCK (never hardcode them here). Route harvest via the `eval-agentic-cleanup` skill, refill via the `eval-*-launch` skill.**

   C. **Other** job types (e.g. Levanter training `iris-run-…` — health via `analyze-training-run-iris`; source of truth = its `active/` experiment dir): report state + a one-line health read; take no autonomous write action.

   D. **GPU-RL** (CoreWeave `cw-us-east-02a`, e.g. `rl-iris-%` / `rl-%` — MarinSkyRL GRPO on whole H100x8 nodes, possibly gang-scheduled multi-node `replicas>1`): **monitor-only — NO rescue, NO keep-2-in-flight, NO auto-relaunch** (those are TPU-datagen concepts and do not apply). The harbor analyzer in §2 does NOT apply (no harbor trial sidecars). For each in-flight GPU-RL job report state + the latest RL progress by reading the persistent finelog (pods GC on terminal): `KUBECONFIG=~/.kube/coreweave-iris-gpu /Users/benjaminfeuer/Documents/marin/.venv/bin/iris --cluster=cw-us-east-02a job logs <job_id> --max-lines 100000 --no-tail` then grep `WANDB_MIRROR kind=train step=` for the latest `trainer/global_step`, `loss/avg_raw_reward`, and `generate/num_failed_trajectories`/`generate/errors`. For multi-node confirm `All N Ray node(s) joined`. On a terminal job report exit state (4=SUCCEEDED). Note: finelog retention is finite — older runs' step lines may have aged out (report what survives). NEVER kill/relaunch GPU-RL jobs.

STANDING ACTIONS — DATAGEN JOBS ONLY (override read-only; see memories auto_rescue_banked_trials, datagen_keep_two_in_flight):
4. AUTO-RESCUE (datagen only) — covers two cases, both autonomous:
   4a. TERMINAL rescue: if a datagen job is terminal (4/5/6) with productive trials banked in GCS that did NOT auto-upload (HF repo missing/stale), rescue automatically — no need to ask.
   4b. ZOMBIE kill-then-rescue (datagen only — killing the zombie IS the precondition for the rescue, so it falls UNDER this datagen-rescue authority, NOT under the §6 no-kill rule): if a datagen job is **state 3 AND harbor progress frozen ≥3h (harbor_updated_at stale) AND the task log shows ONLY `[fd-monitor]` heartbeats in that window (no vLLM/harbor/trial activity)**, it is a confirmed zombie — `iris --cluster=marin job stop <job>` it, then rescue its banked trials as in 4a. The fd-monitor-ONLY clause is the safety gate: a healthy cold-compiling/preempt-recompiling job emits XLA/vLLM compile logs (NOT fd-monitor-only), so it will NOT match (see memory datagen_watchdog_kills_healthy_jobs — a loose stale-only detector once killed a HEALTHY recompiling job; the fd-monitor-only requirement + ≥3h prevents that). ALSO require no healthy vLLM engine (no recent "Application startup complete"/serving marker) before calling it a zombie — a healthy job can sit fd-monitor-only for up to ~6h in a harbor resume scan (see §4c). If unsure whether it's a wedge vs a slow long-task cycle, do NOT kill — report and ask.
   4c. RESUME-SCAN CHECK (report every tick, NEVER kill) — the benign twin of 4b. A datagen job can match 4b's signals (state 3, harbor progress frozen, task log fd-monitor-only) yet be healthy: it is mid harbor GCS-jobs_dir resume after a preempt. On PRE-fix :tpu images that scan is O(already-completed trials), grows with progress, and has been observed up to ~6.25h (if-v2) while the vLLM engine is already up. DISCRIMINATOR: a recent engine-ready marker (grep the task log for "Application startup complete" / "Starting vLLM API server") with NO serving/progress-advance ⇒ resume-scan, NOT a zombie. Detect per active datagen job: (1) engine-ready timestamp via that grep on `iris --cluster=marin job logs <job> --max-lines 20000`; (2) is the harbor <done>/<total> Mean: line advancing across the tick (or harbor_updated_at fresh)? If the engine is up but progress is frozen and the recent tail (`--max-lines 300`) is fd-monitor-only, it is in a resume scan. REPORT it explicitly, e.g. `RESUME-SCAN ~<Nh> (engine up @<T>, harbor idle, <done>/<total> trials)`, note it is EXPECTED on pre-fix builds and self-resolves (do NOT kill, do NOT count it as a stall). For a job that keeps paying it, surface "relaunch on the current :tpu image (harbor 7010e48c, rebuilt 2026-07-02) to eliminate the resume tax — that fix makes resume take minutes not hours". ESCALATE as an OUTLIER worth a look (still NO auto-kill) only if the resume exceeds ~8h with a still-healthy engine; if there is NO healthy engine it is the 4b zombie path.
   Rescue mechanics (both cases) — for a COMPLETED/TERMINAL arm, dispatch a subagent ARMED WITH the canonical **datagen-job-cleanup** skill and follow its idempotent steps (it wraps the essentials below PLUS the `avg_turns≈1.0` realness gate (don't upload a dead-engine run), the `--served_model <ref>` tokenizer stamp, the `[trace-export] Literal yield: X/Y` + `count_populated_literal_rows>0` verify, and safe disk cleanup): rsync the OUTER experiments dir → /tmp/<job>_traces where the OUTER prefix is the job's RECORDED output URI, resolved via `OUT=$(…/otagent/bin/python -m hpc.iris.job_output_resolver "$JOB" --cluster …/marin.yaml)` (registry-first, iris-fallback; returns single-region `gs://marin-<region>/ot-agent/<job>` for new jobs and multi-region `gs://marin-models-{us,eu}/…` for legacy jobs — NEVER hardcode a bucket) (the OUTER `<job>/` — it carries BOTH the trial dirs AND the sibling `logs/<slug>_literal.jsonl`; do NOT rsync only the inner `<job>/<job>/`, which drops `logs/` and silently loses the trainable literals for a `--record_literal` job), then /Users/benjaminfeuer/miniconda3/envs/otagent/bin/python /Users/benjaminfeuer/Documents/OpenThoughts-Agent/scripts/harbor/make_and_upload_trace_dataset.py --job_dir /tmp/<job>_traces --repo_id <the HF repo pattern the job's datagen tracker records> --episodes last --filter none --skip_register (source "${DC_AGENT_SECRET_ENV:?set DC_AGENT_SECRET_ENV first}" first). Literals are AUTO-INCLUDED when a `logs/*_literal.jsonl` is present (the uploader auto-discovers it + correlates → `prompt_token_ids`/`completion_token_ids`/`logprobs` columns; `--no_literal_tokens` forces text-only; it FAILS LOUD if a literal.jsonl is present but 0 trials bind). Report final row count AND the `[trace-export] Literal yield: X/Y` line (X>0 for a `--record_literal` job); update the tracker.
5. KEEP TWO DATAGEN IN-FLIGHT — **RETIRED 2026-07-07: the 32k `qwen3.5-122b-32k-%` datagen keep-2 campaign is CLOSED** (moved to `~/Documents/experiments/complete/qwen3.5-122b-tt/`; see its `CLOSED.md`). Do NOT auto-launch `qwen3.5-122b-32k-%` datagen — the tracker no longer lives in `active/` and there is no keep-2 target. (If a DIFFERENT datagen campaign is later designated keep-N, re-add it here EXPLICITLY with its own `active/…/tracker.md` path + launch template. The separate `qwen3.5-122b-131k-opencode` campaign is NOT keep-2-driven.)
   SNAPSHOT-CAP HYGIENE (every tick, datagen only): the cli Daytona org (DAYTONA_API_KEY) has a hard cap of 60 snapshots. If a datagen launch fails with SnapshotCapExceeded OR the org is >=~58/60, reclaim idle harbor__ env snapshots (at the stale threshold defined in `.claude/projects/daytona/daytona.md` § "How to clean stale snapshots" — GT, don't restate the value) then retry the launch: `source "${DC_AGENT_SECRET_ENV:?set DC_AGENT_SECRET_ENV first}" && /Users/benjaminfeuer/miniconda3/envs/otagent/bin/python /Users/benjaminfeuer/Documents/OpenThoughts-Agent/scripts/daytona/daytona_snapshot_manager.py --api-key-env DAYTONA_API_KEY --delete-stale --yes` (run from the OT-Agent dir; the executing agent reads daytona.md for the threshold value). This deletes ONLY idle harbor__ snapshots (rebuilt on demand by harbor auto_snapshot); the --name-prefix harbor__ default GUARDS the shared base images (daytonaio/sandbox:*, daytona-*, windows-*) which must never be deleted. This supersedes the old "delete only MISSING harbor__" rule (which stalls at 0 MISSING when all are ACTIVE). See skill utils-reclaim-stale-snapshots.

6. NEVER kill/restart/bounce a RUNNING job or the cluster without express user permission — with ONE exception: a **confirmed zombie DATAGEN job** per §4b (the kill is the precondition of a datagen rescue, which IS autonomously authorized). The autonomous write actions are: datagen rescue (incl. the §4b zombie kill-then-rescue), datagen refill-launch. This exception is **datagen-only** — GPU-RL and all other RUNNING jobs stay strictly no-touch (flag for the user, never kill). If a job is stuck PENDING (no capacity), report it and surface the unpinned-relaunch option — do not kill a running/placed job unprompted.
```

If you change the cadence or scope, update BOTH the `cron`/`prompt` above and the
live job (delete + recreate), and keep **monitor-cron-sweep-iris** (the procedure)
in sync — so this skill stays the canonical copy.
