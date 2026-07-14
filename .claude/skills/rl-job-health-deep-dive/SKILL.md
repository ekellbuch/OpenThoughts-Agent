---
name: rl-job-health-deep-dive
description: >-
  Deep single-RL-job health probe → a KILL / NO-KILL recommendation for the supervisor. Dispatched as a
  subagent on every monitor tick for RL jobs in NEW/UNTESTED settings (new config/geometry/model, "debug" or
  "smoke-test" flavor, first launches after a code/config change). Goes BEYOND state-poll + the table metrics:
  syncs the job's trace_jobs + stderr/stdout + Ray logs to ~/Documents/experiments/traces via the EXISTING
  capture tool (CoreWeave: scripts/iris/peek_rl_rollouts.sh `pull`), then runs four gates — (1) liveness
  (tail stdout/stderr: zombie/wedged/dead?), (2) resource utilization (live-poll GPUs: all inference engines
  alive + generating at a hardware/model-size-reasonable cadence per the serving LUT; training stage not
  VRAM/RAM-OOM), (3) rollout quality (trace_jobs: trials init/complete, non-zero rewards, turns completing,
  agent outputs sane, tasks hard, verifiers firing), and emits ONE verdict with evidence + next steps. The
  subagent NEVER kills — it recommends; the supervisor owns the kill (standing guardrail). Cluster-agnostic;
  defers access/hardware/log-path particulars to .claude/ops/<cluster>/ and dependency facts to
  .claude/projects/<dep>/. Reference: scripts/iris/peek_rl_rollouts.sh, scripts/iris/watch_job_state.py,
  .claude/ops/iris/coreweave_gpu_ops.md, .claude/skills/rl-agentic-launch-iris §8.
---

> ⚠ **Do not add comments to YAMLs. Report your recommendations directly to the supervisor.**

# rl-job-health-deep-dive

A **deep, single-RL-job** health probe that ends in **one recommendation to the supervisor: `KILL` or
`NO-KILL`**, backed by hard evidence and next steps. This is the heavier, per-job complement to the breadth
sweep: `monitor-cron-sweep` looks at *every* job briefly; **this skill looks at ONE RL job hard** and is the
right tool when a run is in a **new or untested setting** (new config / geometry / model / image, a "debug" or
"smoke-test" launch, or the first launch after a code/config change) where state-poll + table metrics are
**necessary but not sufficient** to tell "genuinely progressing" from "silently dead."

> **You are a SUBAGENT producing a recommendation — you do NOT execute the kill.** Standing guardrail
> (`supervisor-init`, every launch skill): **never kill a RUNNING job without explicit permission.** Your
> deliverable is the `KILL`/`NO-KILL` verdict + reasoning + recommended next steps. The supervisor decides and,
> if KILL, runs `iris job kill …` / `scancel …` themselves. The ONLY exception is the supervisor's own standing
> autonomy over **our own deterministically-doomed / wedged** jobs — and even then the *decision* is theirs, on
> your evidence. **When in doubt, recommend NO-KILL + escalate** (a wrongly-killed healthy run wastes a whole
> bring-up; a wrongly-kept dead one wastes one more sweep — asymmetric).

## 0. THE CONTRACT — evidence-or-ERROR, exact tools, exact keys (READ FIRST)

A KILL/NO-KILL verdict is only as good as the evidence under it. **A verdict you cannot back with named,
quoted evidence is worse than useless — it looks authoritative and gets a job killed (or kept) on a guess.**
So this skill has a THIRD verdict:

> ### VERDICT: **ERROR** — return this whenever you could not obtain the evidence.
> If a required tool fails (auth/PATH/`--config`/timeout), or a log can't be fetched or parsed, or you cannot
> separate policy-mesh from engine GPUs, or the state-poll and the pod count disagree and you can't reconcile
> them → **STOP and return `VERDICT: ERROR`** with (a) the exact command you ran, (b) its exact failure output,
> (c) what evidence is therefore missing, (d) what you *did* establish. **Do NOT emit KILL/NO-KILL, do NOT
> default to NO-KILL, and NEVER substitute a plausible-sounding guess for missing evidence.** An ERROR that says
> "I couldn't read the engine state because `analyze_job_history` failed with X" is a USEFUL result the
> supervisor can act on; a confident NO-KILL built on an unread log is a trap.

**No verdict without its named artifact.** Every gate verdict must quote the specific evidence, by key:

| Gate | KILL/NO-KILL requires you to have READ + QUOTED | If you can't get it → |
|---|---|---|
| A liveness | the state-poll line (`state=… pods=…`) **and** the newest phase-Timer/`global_step` line + its timestamp | ERROR |
| B resources | per-rank `nvidia-smi` util% (policy ranks vs engine ranks, **separated**) **and** the engine `Running:/Waiting:`+`Avg generation throughput` lines | ERROR |
| C rollouts | actual `result.json` reward values / trial `exception.txt` you opened (not a count you assumed) | ERROR |

### Iris tool contract (exact working invocations — the #1 source of bad verdicts is mis-driving these)

```bash
PY=/Users/benjaminfeuer/miniconda3/envs/otagent/bin/python      # otagent env (symlinks fail in sandbox)
IRIS=/Users/benjaminfeuer/miniconda3/envs/otagent/bin/iris      # NOT marin .venv iris (broken kubernetes import)
# KUBECONFIG is PER-CLUSTER and a HARD prereq in the SAME shell as every call:
#   East (cw-us-east-02a): export KUBECONFIG=~/.kube/coreweave-iris-gpu
#   rno2a (cw-rno2a):      export KUBECONFIG=~/.kube/coreweave-iris          (context marin-rn02a_RNO2A)
```
- **Liveness (authoritative, both clusters):** `$PY scripts/iris/watch_job_state.py /benjaminfeuer/<job> --once --json`. states: **3=running, 5=terminal/failed, 6=stopped**. "running but pods=0 / record absent" = TERMINAL.
- **`job summary` is FLAKY on cw-rno2a** (execute_unary blips) — do NOT rely on it there; use `iris query`:
  `$IRIS --cluster=<cw-rno2a|cw-us-east-02a> query "SELECT name, state FROM jobs WHERE name LIKE '%<slug>%' LIMIT 5"`.
  ⚠ the jobs table has **no `submitted_at`/`failure_count` columns on some schemas** — if a column errors, drop it and re-query `name, state` only (don't abandon the query).
- **Known tool bug — `analyze_job_history.py` region resolver:** it passes `--config <cluster-name>` to iris, which
  wants a PATH. Pass the **config file path** as `--cluster`, e.g.
  `--cluster /Users/benjaminfeuer/Documents/marin/lib/iris/config/cw-rno2a.yaml`, and put `$IRIS` on PATH
  (`export PATH=/Users/benjaminfeuer/miniconda3/envs/otagent/bin:$PATH`). **It ALSO cannot resolve a pure RL job's
  output dir** (no `--jobs-dir/--gcs-output-dir` in the baked command) → it will `LookupError`. For RL jobs, get
  metrics from the finelog via §1b `acquire_complete_log` instead, or `iris job logs --since-ms <t> --no-tail` with
  a BOUNDED window. If none work → `VERDICT: ERROR` (don't guess the metrics).
- **Log volume discipline (memory `iris-log-resource-discipline`):** `iris job logs` has **no server-side grep**;
  NEVER `iris job logs --max-lines >~200 | grep` (dumps the whole log into the Mac → OOM-drops the session). Use
  §1b's server-side-filtered finelog fetch, or tight `--since-ms <window> --max-lines ≤200` slices.

### Exact log keys to grep (know these before you read a single line)

| Signal | Key (grep this) | Means |
|---|---|---|
| step phase timeline | `Started: '<phase>'` / `Finished: '<phase>', time cost:` | which phase is live = "how far into the step" (THE progress truth) |
| step counter | `global_step` / `Resumed training from global_step` | a step banked only after `step` finishes |
| **gen-buffer fill** | `Generation Buffer Progress: N/M` | rollout supply filling toward the batch; **now heartbeats even when FROZEN** (MarinSkyRL `98875d1e`+) → a repeating `N/M [Xs]` with growing elapsed + fixed N = a STALL |
| engine serving | `Avg generation throughput: … Running: R reqs, Waiting: W` | per-engine liveness; `Running≥1/Waiting:0` at ~0 tok/s + 0% GPU = starved (see below) |
| MoE path | `[MoE-PATH] … {grouped_mm|for_loop}` | fused kernel vs silent for-loop fallback |
| host-RAM | `HOST_RAM_BREAKDOWN node=… cgroup=<used>/<cap>` | OOM danger window (optimizer Adam-alloc spike) |
| crash (real) | first `Traceback`/`INTERNAL ASSERT`/`CUDA error` **by timestamp** | the primary fault (precedes the NCCL-watchdog survivor cascade) |
| NCCL hang | `watchdog got stuck for <N> seconds` / `WorkNCCL(… ran for … before timing out` | policy-mesh collective hang (detectable ~30min via the 1800s watchdog) |

### ⛔ Engine starvation is NOT a Daytona error — diagnose it LIVE, with evidence (memory `engine-starvation-not-daytona`)

When engines read `Running:1/Waiting:0`, ~0 tok/s, 0% GPU and the gen-buffer stalls, **do NOT attribute it to
Daytona / hung sandboxes / tool-tails** — that is a repeatedly-wrong default, not a diagnosis. The cause is
almost always on the RL/serving/coordination side (rollout-coordinator dispatch not queuing work → `Waiting:0`;
fully-async staleness/backpressure gate; vLLM scheduler state; `n_concurrent_trials` cap; the ReAct loop). To
find WHICH, **you must diagnose while the job is ALIVE** — once it's killed, py-spy is gone and the finelog alone
is often insufficient. On a live starvation/stall, BEFORE any KILL recommendation, capture:
- `py-spy dump` on the **RolloutCoordinator + a policy rank + a vLLM engine** proc (`kubectl exec -n iris <pod> -c task -- /opt/openthoughts/.venv/bin/py-spy dump --pid <pid>`; find pids via `ps -eo pid,rss,comm`) → what is the coordinator/engine actually blocked ON.
- the finelog window around the stall for `staleness`/`discard`/`n_concurrent`/dispatch keys.
Return that evidence with the verdict. A KILL recommendation for a starvation you did not diagnose live = an ERROR-quality answer; say so rather than guessing.

## Resources you must use (don't re-derive)

- **`.claude/ops/<cluster>/`** — *machine/cluster particulars*: access (kubeconfig/ssh), log-path discovery,
  GPU-poll mechanics, gpu-mem ceilings, the binding gotchas. **CoreWeave** → `ops/iris/coreweave_gpu_ops.md`
  (+ `coreweave_h100_cloud_hardware.md` for the node shape); **Leonardo** → `ops/leonardo/ops.md`; **TACC** →
  `ops/tacc/ops.md`. **Read the relevant one FIRST** — it tells you how to reach the job and read its logs
  safely (GPFS `find`/`du` ban, login-node false-drain, the kubeconfig export, the otagent iris binary).
- **`.claude/projects/<dep>/`** — *what each codebase is + its facts/gotchas*: `marinskyrl/` (the SkyRL/GRPO
  trainer — log-line vocabulary, sel_rows/EPDIAG, weight-sync), `vllm/` (the serve engine — the fork's MoE/DCP/R3
  flags, throughput knobs, enforce_eager), `harbor/` (the rollout/trial layout + `passthrough_exceptions`),
  `daytona/` (sandbox/reward-0 failure modes). Use these to read the logs correctly, not by guesswork.
- **`scripts/iris/peek_rl_rollouts.sh`** — the EXISTING capture tool for CoreWeave (§1). Do NOT hand-roll an
  ad-hoc R2/kubectl pull.
- **`scripts/iris/watch_job_state.py`** — the authoritative CoreWeave lifecycle state-poll (§2).
- **`rl-agentic-launch-iris` §8** — the per-rung CoreWeave bring-up ladder this skill operationalizes; cite it
  rather than restating every milestone string.

---

## 0. Inputs + setup (gather these first)

You need, from the dispatching supervisor (ask only if genuinely missing — most are derivable):

| Input | How to get it | Used for |
|---|---|---|
| **Cluster** | from the dispatch | which `ops/<cluster>` to read + which capture/poll path |
| **Job id** | from the dispatch (`/benjaminfeuer/<job>` on CoreWeave; SLURM jobid on Leonardo/TACC) | state-poll, log path |
| **Pod-name substring** (CoreWeave) | the `--job-name` slug (pod name = `iris-benjaminfeuer-<slug>-<rank>-<hash>-<gen>`) | `peek_rl_rollouts.sh` + `kubectl exec` |
| **Model + size (B, dense vs MoE active-B)** | from the config / `--model_path` | the serving-throughput LUT (§3) |
| **Stage** (bring-up / inference / training) | from the logs you capture in §1–2 | which §3 check applies |
| **What's "new/untested"** | from the dispatch (e.g. "TP=2+EP=2 first run", "R3+DCP unvalidated", "post-weight-sync-fix") | what to scrutinize hardest in §4 |

**Environment (CoreWeave example — adapt per `ops/<cluster>`):**
```bash
source "${DC_AGENT_SECRET_ENV:?set DC_AGENT_SECRET_ENV to the secrets file first}"   # HF/WANDB/DAYTONA (+ R2 creds injected pod-side)
export KUBECONFIG=~/.kube/coreweave-iris-gpu                # HARD prereq — Mac default points at the WRONG cluster
PY=/Users/benjaminfeuer/miniconda3/envs/otagent/bin/python  # otagent env (symlinks fail in the sandbox)
IRIS=/Users/benjaminfeuer/miniconda3/envs/otagent/bin/iris  # NOT the marin .venv iris (broken kubernetes import)
```
All `iris`/`kubectl` calls are **SYNCHRONOUS — never background them** (`ops/iris/coreweave_gpu_ops.md`).

**Restart-burn check (do this EARLY — it sets the syncdown scope in §1 and feeds the verdict).** Before anything
else, find out whether this job has already **burned a restart/retry** — because (a) a run that has *already
failed once* is a sharply different risk than a clean first attempt, (b) the prior-attempt failure is
high-signal for the verdict, and (c) the **remaining** retry budget bounds urgency (a run on its last retry
that's misbehaving is closer to a KILL than one with headroom). Cheap signals (no heavy capture yet):

```bash
# CoreWeave: failure_count + per-task retry state (authoritative), and the pod GENERATION suffix.
$PY scripts/iris/watch_job_state.py /benjaminfeuer/<job> --once --json     # → failure_count (retries burned so far)
$IRIS --cluster=cw-us-east-02a job summary /benjaminfeuer/<job> --json     # per-task state + restart/generation
# pod name = iris-benjaminfeuer-<slug>-<rank>-<hash>-<GEN>; GEN -0 = first attempt, -1/-2 = after a re-bring-up.
kubectl get pods -n iris -o name | grep "<slug>" | sed -E 's/.*-([0-9]+)$/gen \1/' | sort -u   # max GEN > 0 ⇒ restarts burned
```
Compare burned-count to the launch's **`--max-retries K`** (we launch with `K=1`) → `remaining = K − burned`.
**SLURM (Leonardo/TACC):** the restarts are the **`afterany` chain-restart legs** — `sacct -u <user> --name <jobname>
-X --format=JobID,State,Start,End,ExitCode` lists every chain link; each prior **terminal (FAILED/TIMEOUT/CANCELLED)**
leg before the running one is a burned attempt (a clean TIMEOUT→successor is a *normal* chain restart, NOT a burn —
distinguish: a wall-clock TIMEOUT with a healthy successor ≠ a crash-and-retry). `scontrol show job <id>` `Restarts=`
also counts requeues.

Record **`restarts burned = B / max K` (remaining = K−B)** and, if `B>0`, the **terminal state + exit/error of each
prior attempt** (from the summary/sacct). If `B>0`, set the syncdown in §1 to **include the failed runs** (next
section), and carry `B`, the remaining budget, and "same failure each attempt?" into Gate A (§2) and the verdict
(§5): the same crash repeating across every restart = **deterministically doomed → KILL**; a restart burned on a
genuine transient (e.g. an HF-weight-resolution flake the retry-wrapper now catches) that is now healthy = benign.

---

## 1. Sync the artifacts (use the existing tool — never ad-hoc)

Pull **trace_jobs + stderr/stdout + (if available) the Ray logs** into the canonical capture dir
**`~/Documents/experiments/traces/`**. Use the dependency's own capture tool — do not hand-roll a kubectl/R2
sync (the tool already handles per-object size-verify, the latest-generation pod, the REMOTE-R2-vs-node-local
trials_dir split, and a provenance MANIFEST).

**CoreWeave:**
```bash
IRIS_BIN=$IRIS bash scripts/iris/peek_rl_rollouts.sh <pod-name-substring> pull
#   → ~/Documents/experiments/traces/<slug>_<UTC-stamp>/
#       logs/iris_finelog.log      (complete iris finelog, --no-tail — the full bring-up + driver log)
#       logs/pod_rank*.log         (per-rank container stdout; rank0 = Harbor coordinator + driver; 1..N-1 = Ray workers)
#       trace_jobs/<trial>/…       (ALL Harbor trials synced from R2: config/prompt/conversation + result.json reward)
#       MANIFEST.md                (provenance: job id, rank-0 pod, trials_dir, started/completed counts)
```
The per-rank `pod_rank*.log` files **are** the engine/Ray stderr+stdout (k8s merges container stdout/stderr).
For deeper Ray actor logs, `kubectl exec -n iris <pod> -c task -- bash -lc 'ls /tmp/ray/session_latest/logs'`
and pull any `worker-*.err` / `raylet.*` of interest (the finelog usually surfaces the real traceback first).

> **If the §0 restart-burn check found `B>0` burned restarts → the syncdown MUST include the FAILED runs, not
> just the live generation.** The default `pull` captures the *current* (latest-generation) pods' logs + the
> shared trials — but the prior attempts are exactly the evidence you need. Capture them:
> - **iris finelog spans ALL generations** — it is the durable record of every burned attempt's bring-up +
>   crash (the GC'd prior-generation pods' stdout is gone, but the finelog kept it). `peek … pull` already
>   pulls it `--no-tail` full-history; **confirm it covers the original submission** (if it was tailed, re-pull
>   with `$IRIS … job logs /benjaminfeuer/<job> --since-ms <ORIGINAL submitted_at_ms> --no-tail`, using the
>   *first* submission time, not the latest generation's). Then **grep it for each prior attempt's terminal
>   traceback** — the real reason each restart was burned (and whether it's the SAME failure every time).
> - **The REMOTE R2 trials_dir is shared across generations** (`s3://marin-us-east-02a/iris/<job>/trace_jobs` —
>   the storage root resolves via `marin_prefix()`, see `.claude/ops/iris/coreweave_gpu_ops.md` §rendezvous; don't hardcode the region bucket), so
>   `pull` already grabbed failed-generation trials too — segregate/label them by episode/timestamp so the
>   §4 rollout read doesn't conflate a dead generation's reward-0 trials with the live one's.
> - **SLURM:** the failed legs are *separate job IDs* in the `afterany` chain — `rsync` the **`.out`/`.err` of
>   each prior terminal leg** (its own jobid, located via `scontrol`/`sacct` per §0) into the capture dir,
>   alongside the running leg's. Don't sync only the live leg.

**SLURM (Leonardo/TACC):** there is no `peek` tool — the equivalent is: locate the `.out`/`.err` via
`scontrol show job <id> -o` (`StdOut=`/`StdErr=`/`%Z` workdir — **never `find`/`du` on GPFS**, per
`ops/<cluster>`), then `rsync`/`scp` the `.out`, `.err`, and the run's `trace_jobs/` tree into
`~/Documents/experiments/traces/<job>_<stamp>/`. (Most "new/untested" RL right now is CoreWeave; the four
gates below are cluster-agnostic once you have logs + trials + a GPU view.)

> **If `pull` returns 0 trials / no logs:** that is itself a signal, not a tool failure — usually the rank-0
> pod is gone (terminal job → node-local trials GC'd) or was just (re)started. Cross-check with §2's state-poll
> before concluding; a fresh launch legitimately has 0 completed trials for a while (long episodes).

### 1b. Science-grade finelog parse (CoreWeave) — the AUTHORITATIVE "how far into the step + is it healthy" read

`peek … pull` gives the raw finelog; for the actual **training-progress + host-RAM + MoE-path** verdict, pull the
COMPLETE log (live ∪ R2-archive, deduped, coverage-verified) with an **RL-tuned pattern filter** — the default
`FINELOG_CONTAINS_PATTERNS` is datagen/TPU-tuned and drops the RL signals. Reuse `analyze_job_history.py`'s
`acquire_complete_log()` with the patterns swapped (proven 2026-07-13 on the 80B):

```python
# marin .venv python, with R2 archive creds sourced (see below) + IRIS_BIN=<otagent iris>, KUBECONFIG=cw
import time, sys; from pathlib import Path
sys.path.insert(0, ".../OpenThoughts-Agent/scripts/iris"); import analyze_job_history as ajh
ajh.FINELOG_CONTAINS_PATTERNS = (         # <- RL signal set (swap per what you're chasing)
  "Started: '", "Finished: '",            # fully_async_trainer loguru Timers → EXACT per-phase durations
  "Resumed training from global_step",    # the step counter
  "HOST_RAM_BREAKDOWN", "fd-monitor",     # per-minute cgroup host-RAM (fd_monitor.py)
  "WORKER_FORWARD_ENTER", "WORKER_DRAIN_BARRIER", "R3_RESIDENT_SET",  # forward/dispatch markers
  "[MoE-PATH]", "grouped-GEMM swap active",  # grouped_mm-vs-for-loop path (MarinSkyRL 8d1ca716+)
  "Traceback", "INTERNAL ASSERT", "SavedTensorHooks", "ProcessGroupNCCL", "watchdog",
  "NumelIn=1", "OOMKill", "out of memory", "CUDA error",              # crash/wedge terminal signatures
)
acq = ajh.acquire_complete_log("/benjaminfeuer/<job>", "cw-us-east-02a", int(time.time()*1000),
        Path("/tmp/<job>.filtered.log"), Path("/tmp/<job>.cov.json"), refresh=True, max_gap_seconds=3600.0)
# acq.lines = deduped, coverage-verified; acq.logs_complete asserts every attempt window is covered.
```

- **R2 archive creds are MANDATORY** (else `FileNotFoundError: bucket does not exist`): borrow from the iris-ns
  secret `finelog-cw-use02a-env` via the base64 loop, run under the **marin `.venv` python** with
  `IRIS_BIN=<otagent iris>`. Full procedure: **`.claude/ops/iris/finelog_r2_archive_creds.md`**. Do NOT also
  source the LAION secrets in that shell (they clobber the R2 creds).
- **The phase Timers are THE progress truth — not the trace count, not tqdm.** Parse the `Started: '<phase>'` /
  `Finished: '<phase>', time cost: <s>s` pairs into a timeline: `wait_for_generation_buffer` (buffer full =
  batch ready) → `convert_to_training_input` → `fwd_logprobs_values_reward` → `compute_advantages_and_returns`
  → `policy_train` / `train_critic_and_policy` → `sync_weights_to_inference_engines` → `step`. "How far into the
  step" = which phase has `Started` but no `Finished` yet. `global_step` ticks only after `step` finishes. A
  trace-count proxy (`result.json` count) MISLEADS — 512 completed trajectories ≠ "1 step banked" if `policy_train`
  is still running (bit us 2026-07-13).
- **✅ tqdm bars ARE captured now** (MarinSkyRL `de40d31c`+ non-TTY newline fallback, gpu-rl `19bd8c5e`+): a `\n`
  per update, so `Generation Buffer Progress: N/M (P%) [Xs]` / `Training Step Progress` LAND in the finelog. **And
  a FROZEN bar now HEARTBEATS** (MarinSkyRL `98875d1e`+): the SAME `N/M` line re-emits every 15s
  (`SKYRL_PROGRESS_HEARTBEAT_S`) with growing elapsed — so a stalled gen-buffer shows as a repeating `31/64 [Xs↑]`,
  NOT silence (before `98875d1e` a frozen bar went silent and a wedge looked identical to a healthy idle run — this
  is the signal that unmasks a live starvation). Still cross-check the phase Timers (`Started:`/`Finished:`) — they
  remain the authoritative "how far into the step" truth; a runs on an OLDER image without these fixes still lacks
  the bars, so fall back to the Timers there.
- **Host-RAM (the 80B OOM-fix target):** parse `HOST_RAM_BREAKDOWN node=<h> cgroup=<used>/<cap> GiB` → report
  PEAK + current vs the cgroup cap (e.g. 1700 GiB). The optimizer-step Adam-alloc spike is the danger window.
- **MoE path:** `[MoE-PATH] experts forward via {grouped_mm|for_loop} (w1=<type>, use_grouped_mm=…)` tells you
  whether the fused kernel actually ran or a silent per-expert for-loop fallback fired (a `fwd_logprobs` perf
  cliff). `[MoE] grouped-GEMM swap active` only confirms the swap FLAG was read, not the forward path.
- **Reading the exact crash:** the PRIMARY error precedes the NCCL `mesh_fsdp` watchdog cascade (the watchdogs
  are survivors detecting a dead peer — "remote process exited or network error"). Grep for the FIRST
  `Traceback`/`INTERNAL ASSERT`/`CUDA error` by timestamp, not the loudest NCCL line. For per-trial harbor
  exceptions, read the R2 `trace_jobs/<trial>/exception.txt` via the pod's boto3 (`Config(s3={"addressing_style":
  "virtual"})`, endpoint `$AWS_ENDPOINT_URL`) — the Mac lacks the CW object-store creds.

---

## 2. Gate A — Liveness (is it alive, or zombie/wedged/dead?)

**Liveness = authoritative STATE-POLL + a log-freshness read — NEVER a single log-string grep** (a clean
kill/eviction/preempt emits no terminal string and reaps pods, so a content-watch sits idle while the job is
gone — the exact miss `rl-agentic-launch-iris` §8 warns about).

```bash
# CoreWeave: authoritative lifecycle (state + per-task + pod cross-check; "no record AND 0 pods" = TERMINAL absent)
$PY scripts/iris/watch_job_state.py /benjaminfeuer/<job> --once --json
# SLURM: squeue -u <user> -t RUNNING ; sacct -j <id> -X --format=State,ExitCode  (validate vs false-drain — ops/<cluster>)
```
Then read the captured logs (§1) for **freshness + wedge signatures**:
- **Log freshness.** Compare the newest meaningful log line's timestamp to "now" against the run's expected
  cadence (a generating engine prints `Avg generation throughput …` every few seconds; a training step lands
  on the order of minutes). **Materially-stale-beyond-cadence = suspected wedge** (engine deadlock, NCCL
  stall, generation-buffer starvation) — a job can hold its whole gang for hours while hung and still read
  `running, pods=N`.
- **Wedge / death signatures** (grep the finelog + `pod_rank*.log`): a real `Traceback`, `CUDA out of memory`
  / `OutOfMemoryError`, `RayActorError` / `Raylet … died` / worker SIGKILL/SIGABRT, an NCCL hang
  (`WorkNCCL(...Timeout(ms)=...) ran for N ms before timing out`, `ProcessGroupNCCL preparing to dump debug
  info`), an RPC / `sample_tokens` / `execute_model` timeout, `EngineDeadError`.
- **Benign noise — do NOT read as death** (per `marinskyrl`/`vllm` projects + `monitor-job-tables`):
  `shm_broadcast … No available shared memory broadcast block found in 60/600s` (engine idle-waiting — heartbeat,
  not a kill), a transient `ghcr.io` blob EOF → `ImagePullBackOff` self-heal, the `Unknown vLLM environment
  variable VLLM_ALLOW_ROUTED_EXPERTS_DCP` whitelist note, `… 32769 input tokens > 32768 max` /
  `ContextLengthExceededError` / `AgentTimeoutError` (harbor passthrough — appear in *healthy* runs),
  `rollout_train_prob_diff_mean` in the millions (outlier-dominated, normal). `opCount dead` is benign debug-token
  noise, not `EngineDeadError`.

> **⚠️ THE COLOCATED-ENGINE DECEPTION — a wedged POLICY mesh that falsely reads ALIVE (learned 2026-06-28, BOTH
> MoE-EP8 arms wedged ~1h apart, undetected for hours).** In a disaggregated/colocated RL job the vLLM
> **inference engines + RolloutCoordinator are SEPARATE Ray actors** from the **policy FSDP mesh**. When the
> POLICY mesh hangs in a NCCL collective (`ProcessGroupNCCL watchdog got stuck for 1800s` → SIGABRT on rank-0 →
> gang wedges), the engines KEEP GENERATING — so **every cheap "liveness" signal LIES**: (a) the **state-poll**
> still reads `running / pods=N / failure_count=0` (pods alive, just not computing); (b) the **wandb heartbeat
> stays FRESH** (the engine/coordinator actors keep emitting system metrics — *wandb liveness is NOT policy-mesh
> liveness*); (c) **GPU util reads HIGH (28–96%)** on the engine-colocated pods (engines busy-generating
> rollouts no training step will ever consume). A util-only read MIS-diagnoses this as ALIVE-SLOW. **The three
> signals that DON'T lie — check ALL of them every probe on any multi-mesh (FSDP×EP×CP) RL run, proactively, do
> NOT wait for util to look wrong:**
> 1. **Ray logs (authoritative for actor/worker death).** kubectl-exec rank-0 (`-c task`):
>    `grep -iE 'died|SYSTEM_ERROR|connection error code 2|End of file|SIGABRT|Aborted|raylet.*fail|ActorDied' /tmp/ray/session_latest/logs/{raylet.*,python-core-*,worker-*.err}`. A `worker died … SYSTEM_ERROR … connection error code 2 … End of file` = the policy worker is DEAD (Ray may even respawn it and it re-hangs → a SECOND watchdog-stuck).
> 2. **The NCCL watchdog in the finelog** — `ProcessGroupNCCL('s)? watchdog got stuck for <N> seconds` / `Fatal Python error: Aborted` / `WorkNCCL(SeqNum=…,OpType=…) ran for … before timing out` / the **`TCPStore … sendBytes failed … Broken pipe`** cascade across FSDP ranks. The **1800s watchdog makes a hang detectable within ~30 min of onset** — grepping this each probe is the FASTEST catch (we set `TORCH_NCCL_DUMP_ON_TIMEOUT` → a flight-recorder dump also lands; capture it before any kill).
> 3. **Is the TRAINER/DRIVER log ADVANCING** (not just engine generation)? `mesh_fsdp` / `run_training` / `fwd_logprobs` / the step counter must MOVE; a frozen trainer + spinning engines = wedge. **Separate POLICY-mesh GPUs from the colocated ENGINE GPUs** — a wedged policy reads **0% on the policy ranks** even while engine GPUs show util (identify policy vs engine pods/GPUs from the placement; don't average them).
> A `running / fresh-heartbeat / high-util` reading is **necessary-but-NOT-sufficient** for a multi-mesh RL job —
> clear all three above before calling Gate A PASS. NB: this is a **NCCL-collective hang, NOT an OOM** (memory.peak
> held ~35% when it bit) — rule OOM out, but never let a clean memory read *imply* liveness.

**Gate A verdict:**
- **DEAD / TERMINAL** (state-poll says failed/killed/absent, or 0 pods) → there's nothing to kill; report
  TERMINAL + the root-cause traceback + whether it's transient (relaunch-worthy) or deterministic.
- **WEDGED** (RUNNING/pods present but a real hang signature + stale-beyond-cadence logs, no benign explanation)
  → **lean KILL** (it's burning a multi-node gang doing nothing). Carry the evidence to §5.
- **ALIVE + fresh logs** → proceed to Gate B.

---

## 3. Gate B — Resource utilization (are the GPUs actually working?)

**Live-poll the GPUs** (don't infer from logs alone). CoreWeave — exec `nvidia-smi` on **every** rank:
```bash
for p in $(kubectl get pods -n iris -o name | grep "<job>" | sed 's#pod/##'); do
  echo "== $p =="
  kubectl exec -n iris "$p" -c task -- nvidia-smi \
    --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader
done
```
(SLURM: `srun --jobid=<id> --overlap -w <node> nvidia-smi …`, or read the `.out`'s periodic util/mem prints —
see `ops/<cluster>`.) Interpret by **stage** (which stage the §1–2 logs put you in):

### B-inference (rollout/generation stage)
**Every inference engine must be alive, fed, and generating at a cadence reasonable for the hardware + model
size.** Concretely:
1. **All engines present + busy.** Engine count = `num_inference_engines × TP` GPUs should show **non-trivial
   `utilization.gpu`** (decode is bursty, but a generating engine is not pinned at 0%). A GPU at **0% util with
   memory resident** across several polls = an engine that loaded weights but is **not generating** (starved /
   wedged / never received requests).
2. **vLLM actually emitting tokens.** In the logs: `Avg generation throughput: >0 tokens/s, Running: R reqs,
   Waiting: W` recurring on **each** engine. This is the literal "is vLLM firing" check.
3. **Throughput is hardware-reasonable** — compare the aggregate `tokens/s` (summed across an engine's `Running`
   reqs) to the **serving LUT below**. Below-floor throughput while `Running > 0` is a red flag (eager mode,
   bad geometry, oversubscription thrash, or a degraded weight-sync producing pathological outputs).
4. **Queue is draining, not thrashing.** `Waiting ≈ 0` (or falling) = healthy. **`Waiting` persistently ≫
   `Running` with flat/low throughput = the throughput-starvation WEDGE** (oversubscribed: lower
   `n_concurrent_trials` / raise `max_num_seqs`) — this was the original `rl-131k-cpdcp2r3` silent death.

> **Serving-throughput LUT — H100-80GB SXM, vLLM, batched continuous-decode (the RL rollout regime).** These
> are **order-of-magnitude health FLOORS, not benchmarks** — the diagnostic is the SHAPE (throughput > 0,
> scales with `Running`, `Waiting ≈ 0`), not hitting a number. Decode is HBM-bandwidth-bound (H100 ≈ 3.35 TB/s);
> single-stream ≈ `HBM_BW / (2 × active_param_bytes)`; aggregate rises with batch until compute-bound.

| Model (active params) | Single-stream decode | **Healthy aggregate / engine** (dozens of `Running`) | "Awful" — investigate |
|---|---|---|---|
| Dense 7–8B | ~90–140 tok/s | **~2,000–6,000 tok/s** | < ~500 tok/s w/ many Running |
| Dense 14B | ~60–100 tok/s | ~1,500–4,000 tok/s | < ~400 |
| Dense 32B | ~30–55 tok/s | ~800–2,000 tok/s | < ~200 |
| **MoE 30B-A3B / 35B-A3B (~3–3.5B active)** | ~80–160 tok/s | **~2,000–6,000 tok/s** (low active → high) | **< ~500 tok/s** |
| MoE ~235B-A22B (~22B active) | ~25–45 tok/s | ~700–1,800 tok/s | < ~200 |

> **`enforce_eager: true` (CUDA graphs OFF) divides decode by ~3–10×** (kernel-launch-bound at small decode
> batch) — this is exactly why the CoreWeave MoE arms read **15–75 tok/s** and the supervisor called it
> "awful": the fix was `generator.enforce_eager: false` (CUDA graphs) in the iris config, NOT a serve-geometry
> change. **So: if you measure floor-or-below throughput, FIRST check `enforce_eager` in the resolved config**
> before blaming weights/geometry. Long context (131k) lowers steady-state tok/s (KV pressure) but should not
> floor it. TP/EP geometry: more, smaller engines (e.g. TP=2+EP=2 ×16) should *raise* aggregate cluster
> throughput vs few big TP=8 engines — if it doesn't, suspect the EP/R3 path.

### B-training (optimizer/update stage)
- **Not VRAM-OOM:** no `CUDA out of memory` / `OutOfMemoryError` in logs; `memory.used` not pinned at
  `memory.total` (≈80 GB) **while progress is stalled**. (Transient near-ceiling at peak activation is normal
  *if steps advance*.)
- **Not RAM/host-OOM:** no `OOMKilled` in pod state
  (`kubectl get pod <p> -o jsonpath='{.status.containerStatuses[*].lastState.terminated.reason}'`), no
  oom-killer / `Killed` in the rank logs, no repeated pod restarts.
- **GPUs actually computing:** during a step the policy/ref ranks show **high `utilization.gpu` + power draw**
  (a forward/backward is compute-bound), not 0%. All-0% with no log advance during "training" = wedge.

**Gate B verdict:** an engine resident-but-0%/no-tokens, throughput floored-or-zero with `Running>0` and
`enforce_eager:false`, `Waiting≫Running` flat, or a training-stage OOM / all-ranks-0%-no-progress → **lean
KILL** (carry evidence to §5). All engines generating ≥ floor with `Waiting≈0`, or training steps advancing
without OOM → **healthy on Gate B**, proceed to Gate C.

---

## 4. Gate C — Rollout quality (read the actual trace_jobs; use judgment)

State + GPUs can be green while the run produces **garbage** (e.g. a degraded FSDP→vLLM weight-sync serving a
policy that emits token-salad → 100% reward-0 → no learning signal). So **read the literal rollouts** under
`~/Documents/experiments/traces/<slug>_<stamp>/trace_jobs/`. (`peek_rl_rollouts.sh <substr> cat <trial-dir>` /
`grep <regex>` also read R2 directly for spot checks.) Work through, **using your best judgment** — this gate is
qualitative:

1. **Are trials INITIALIZING?** trial dirs being created (config/prompt present). Zero new trial dirs while
   engines generate ⇒ the Harbor RolloutCoordinator isn't dispatching (look for `TerminalBenchGenerator
   initialized … Concurrent trials: K` in the log).
2. **Are trials COMPLETING?** count `result.json` (the completed-trial marker carrying the reward). At +15/30
   min a 131k arm legitimately has **ZERO** completed (long episodes) — report that as *"rollouts executing, 0
   completed yet,"* **never** as "healthy/done." Completing at a steady rate ⇒ the loop is closing.
3. **Are any rewards NON-ZERO?** `grep result.json` for `"reward"`. **All-zero / all-timeout** is the headline
   failure mode — then ask *why*:
   - **Agent output is incoherent** (token-salad, repetition loops, wrong-language, empty) ⇒ **serving/weight-sync
     or geometry fault** (the FSDP→vLLM sync or the vLLM-fork build for this model/geometry). This is a **KILL**
     signal on a new/untested geometry — the policy being served is not the trained policy. (Check the
     tokenizer is right first — but a *correct* tokenizer + salad output = sync/build, per the prior CoreWeave
     diagnosis.) **Known CoreWeave MoE cause to check first:** the FusedMoE `w13` gate/up swap not re-applied
     on the disaggregated RL update (H100/FlashInfer-CUTLASS) → confirm `SKYRL_W13_RELOAD_BRACKET` is on
     (default 1) and the engine log shows `finish_weight_reload` (fix MarinSkyRL `2bb70a88`; marinskyrl doc).
   - **Agent output is COHERENT but wrong / runs out of turns** ⇒ tasks genuinely hard or the harness/verifier
     mis-set — NOT necessarily a kill; this can be a real (if low) learning signal. Read several conversations.
   - **Every trial ends in an environment/infra exception** ⇒ infra, not model — but **name the exception from an
     `exception.txt` you actually opened**, don't assume it. `VerificationNotCompletedError` everywhere = Daytona
     sandbox never came up / `DAYTONA_*` not forwarded; `Bearer token invalid` = auth. Only THEN KILL + fix the
     infra. ⚠ **Do NOT confuse this with engine STARVATION** — `Running:1/Waiting:0` + stalled gen-buffer is a
     *dispatch/serving* problem (see §0), NOT "every trial threw a Daytona exception." Starvation ≠ Daytona; if you
     haven't opened trial exception files showing an infra error, you have not established an infra cause.
4. **Are TURNS completing?** in `conversation`, count `role=="assistant"` turns. `avg_turns ≈ 1` (agent makes
   one move then stops/errors) is the dead-engine / broken-loop signature; multi-turn = real agent behavior.
5. **Are the AGENT OUTPUTS REASONABLE?** actually read 3–5 trajectories: is the model issuing sensible tool
   calls / code, or looping / emitting garbage / ignoring the task? (The task framing for the active CoreWeave
   arms — `pymethods2test` etc. — is simple code-contract Python; it should NOT need massive context, so a
   context-overflow storm on those is a red flag, not task difficulty.)
6. **Are the TASKS too hard / the VERIFIERS failing?** sample `verifier_output`: is it scoring a genuine
   attempt as fail, or erroring/timing out itself (`VerifierTimeoutError`)? A verifier that never returns a
   real score ⇒ no learning signal even with good generations.

**Gate C verdict:** incoherent generations / all-reward-0 from a serving-or-sync fault on a new geometry, or a
verifier/infra path that yields **zero learning signal** with no transient explanation → **lean KILL** (it
cannot learn in this state). Trials completing with *some* non-zero rewards (or coherent multi-turn attempts on
genuinely-hard tasks even at low pass-rate) → **NO-KILL, healthy/learning**.

---

## 5. Deliver ONE recommendation (the whole point)

Emit exactly one verdict to the supervisor, in this shape:

```
RL-JOB-HEALTH — /benjaminfeuer/<job>  (<model>, <geometry>, <stage>)   captured: <traces dir>

VERDICT: KILL | NO-KILL | ERROR          confidence: high|medium|low
  (ERROR = could not obtain the evidence — see §0. Give the failed command + its output + what's missing;
   do NOT emit KILL/NO-KILL and do NOT default to NO-KILL.)
Evidence I actually read: <the state-poll line; the per-rank util split; the engine Running/Waiting line;
  the reward values / exception.txt — quote them. If a row is blank, the matching gate is ERROR, not PASS.>
Restarts: <B/K burned, remaining K−B> — <none | same failure each attempt: … | transient, recovered>

Gate A (liveness):   PASS|FAIL — <state-poll verdict + log-freshness + any wedge/death signature>
Gate B (resources):  PASS|FAIL — <per-engine util/mem; aggregate tok/s vs LUT floor; Waiting/Running; enforce_eager; OOM?>
Gate C (rollouts):   PASS|FAIL — <trials started/completed; reward distribution; turns; output coherence; verifier sanity>

REASONING: <2–4 sentences — the load-bearing evidence, esp. for whatever was "new/untested" in this run>
NEXT STEPS (if KILL): <root cause + the concrete fix — config knob / weight-sync / image rebuild / infra —
                      and whether to relaunch on the corrected setting or hold for the supervisor's call>
NEXT STEPS (if NO-KILL): <what to watch next sweep + the specific signal that would flip it to KILL>
```

**Verdict rules:**
- **KILL** if ANY gate is a hard FAIL with **no transient/benign explanation**: terminal/wedged (A); engines
  resident-but-not-generating or floored throughput w/ `enforce_eager:false` or training-OOM (B); incoherent
  generations / all-reward-0 from a serving/sync/verifier fault on a new geometry (C). Give the root cause +
  fix — a KILL recommendation without a "what to change before relaunch" is incomplete.
- **KILL (restart-burn corroboration, from §0)** if the job has burned restarts repeating the **SAME** failure
  each attempt — it is **deterministically doomed**, and the remaining retry budget will only burn more
  nodes-hours reproducing it. State `B/K burned, remaining K−B, same failure: <traceback>` and the fix that
  must land before any relaunch. (A restart burned on a genuine *transient* the run has since recovered from is
  NOT a kill — say so and weigh the other gates.)
- **NO-KILL** if all three gates pass, **OR** the only failures have a legitimate transient/early-bring-up
  explanation (e.g. 0 completed trials at +15 min on a long-episode arm; an HF-weight-resolution flake that the
  `--max-retries`/retry-wrapper is catching; gang still admitting). Say what you're waiting on and the signal
  that would change the call.
- **Default to NO-KILL + escalate on genuine ambiguity** (low confidence) — the asymmetry favors not killing a
  possibly-healthy bring-up. Hand the supervisor the evidence and let them decide.

**You never run the kill.** If KILL, the supervisor executes `iris job kill /benjaminfeuer/<job>` (CoreWeave) /
`scancel <id>` (SLURM) with permission, then relaunches per `rl-agentic-launch-iris` / `rl-*-launch-*` on the
corrected setting. Log the probe + verdict to `~/Documents/agent_logs/` (dated) so the diagnosis isn't lost.

---

## Operating notes
- **This skill is for the HARD per-job read; `monitor-cron-sweep` is the breadth pass.** Don't duplicate the
  full sweep here — probe the ONE job you were handed, deeply, and return a verdict.
- **Cluster-agnostic by design:** the four gates (liveness / resources / rollouts / verdict) hold on any
  cluster; only the *mechanics* (capture tool, state-poll, GPU-poll, log paths) differ — and those live in
  `.claude/ops/<cluster>/`. CoreWeave is the worked example because that's where the new/untested RL runs.
- **Read logs through the dependency docs.** `.claude/projects/{marinskyrl,vllm,harbor,daytona}/` define the
  log vocabulary, the benign-vs-fault line, and the known failure modes — use them so you don't misread a
  heartbeat as a hang or a passthrough exception as a crash.
- **Never patch/hand-edit on a cluster.** If the fix is a config/code change, it goes in the LOCAL clone →
  commit → (push / next-launch upload) per CLAUDE.md — your job here is diagnosis + recommendation, not a
  cluster-side edit.
