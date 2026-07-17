---
name: monitor-job-tables
description: >-
  Format HPC job-status reports as box-drawing tables, bucketed by job type (RL · SFT · Datagen · Eval ·
  Catch-all), with the right metric columns, signal thresholds, and red-flags per bucket. Use whenever
  reporting active/recently-terminated job status — during a cron sweep (driven by monitor-cron-sweep),
  an ad-hoc "how are my jobs doing", or a single-job progress update. Covers which metrics are mandatory
  (entropy + collapse signals for RL, not just step/reward/grad), where to pull live status (SFT .out vs
  trainer_log.jsonl), the RL collapse-warning rule, and which log lines are benign noise vs real faults
  (shm_broadcast 600s, rollout_train_prob_diff_mean millions). Cluster-agnostic — refer to .claude/ops
  for paths.
---

# monitor-job-tables

> **Read `.claude/ops/<cluster>/ops.md` first** every sweep: it dictates how to locate logs safely
> (`scontrol show job <id> -o` `StdOut=`/`%Z` — **never `find`/`du` on GPFS`), login-node caveats, and
> debug-token noise (`opCount dead` is benign).
>
> **Active clusters = Leonardo + CoreWeave(iris) + TACC(Vista).** Jupiter SKIPPED (MDC downtime until
> ~2026-07-12). Log-location + state-poll differ by cluster type:
> - **Leonardo / TACC (SLURM)** — log path via `scontrol show job <id> -o` `StdOut=`/`%Z` (`ssh Leonardo` /
>   `ssh TACCVista`). TACC: compute nodes have internet (no proxy), GPUs are whole-node, RealMemory misreported.
> - **CoreWeave (iris/k8s, NO ssh)** — no SLURM `.out`, no log path to `stat`. **Liveness = STATE-POLL the
>   iris lifecycle** (never a log-string grep): `export KUBECONFIG=~/.kube/coreweave-iris-gpu`, then the
>   otagent-env iris binary `/Users/benjaminfeuer/miniconda3/envs/otagent/bin/iris`:
>   `scripts/iris/watch_job_state.py /benjaminfeuer/<job> --once --json` and/or
>   `iris --cluster=cw-us-east-02a job summary --json` (authoritative). **"running-but-0-pods / record
>   disappeared" = TERMINAL** (silent-wedge signature). Pull metrics via
>   `iris … job logs --since-ms <submitted_at_ms> --no-tail` + `scripts/iris/analyze_job_history.py`.
>   All `iris`/`kubectl` calls SYNCHRONOUS.
>
> **Log-path trap (Leonardo agentic eval) — verify the StdOut path EXISTS before concluding "dead."**
> `scontrol`'s `StdOut=` may point at a name never created (`eval_<name>_<jobid>.out`) while the real live
> log is `data_<jobid>.out` in the same dir. If the scontrol path is absent, **`ls` the job's `%Z` workdir
> for `data_<jobid>.out` (or any `*_<jobid>.out`) and read THAT** — absence at the scontrol path is a path
> mismatch, not a death.

Report **every** active and recently-terminated job, **bucketed by type**, in the formats below.
**Unify cross-cluster runs of the same type into ONE table** (RL: Leonardo standard-GRPO + CoreWeave
agentic/MoE rows together; eval: Leonardo + TACC together). A separate table for jobs still filling their
generation buffer (no metrics yet). Five buckets: **RL · SFT · Datagen · Eval · Catch-all**.

Cross-cutting (every bucket):
- **Chain-restart TIMEOUTs are normal, NOT failures** — when a 12h/24h job TIMEOUTs and its `afterany`
  successor is RUNNING/PENDING, report it as a normal restart (note the successor).
- **Completion → matching cleanup skill**: RL by flavor — agentic (Harbor/Daytona)→`rl-agentic-job-cleanup`,
  standard non-agentic GRPO→`rl-standard-job-cleanup`; SFT→`sft-job-cleanup`, datagen→`datagen-job-cleanup`,
  eval→`eval-agentic-cleanup`. CoreWeave RL routes the same way but artifacts go to HF / R2 (no on-disk
  `trace_jobs/`/`tasks/` to reap). On GPFS clusters, cleanup is not done until the artifact's on-disk
  `trace_jobs/`/`tasks/` tree is `rm`'d + inode reclaim verified (leaving it is the #1 inode leak on
  `/e/data1`). Inode how-to → `ops/jupiter/ops.md` `#inode-allocations` (dormant while Jupiter is down).
- **Genuine FAILED (exit≠0, not a wall TIMEOUT) → diagnose + dated `agent_logs/` entry**; recurring
  identical failures ≠ transient.

---

## RL

```
┌─────────────────────────┬───────┬────────┬─────────────┬───────────┬─────────────────────────────────────────┐
│           Job           │ Step  │ Reward │ Policy Loss │ Grad Norm │                  Trend                  │
├─────────────────────────┼───────┼────────┼─────────────┼───────────┼─────────────────────────────────────────┤
│ SWE-rebench 8B (shaped) │ 15/80 │ 0.619  │ -0.0040     │ 0.006     │ Checkpoint saved. Slight dip from 0.652 │
│ Code-contests 8B (base) │ 26/80 │ 0.451  │ -0.0930     │ 0.021     │ Stable, gradients strong                │
└─────────────────────────┴───────┴────────┴─────────────┴───────────┴─────────────────────────────────────────┘
```
Box-drawing tables (┌─┬─┐), **not** markdown — hard user preference for RL. Columns: Job, Step (`cur/max`),
Reward, Policy Loss, Grad Norm, Trend. **Entropy + collapse signals are mandatory**: include `policy_entropy`
+ TIS `log_ratio` + `grad_norm` (in Trend or extra columns) — without entropy you can't apply the collapse
rule. Metric not emitted yet (step 0 filling cohort) → mark `—`. **Leonardo (SLURM) RL** step from the `.out`
(tqdm `Training Step Progress: N/M` or `trainer/global_step`). **CoreWeave (iris) RL** has NO SLURM `.out` —
lifecycle state from the state-poll, metrics from the finelog or WANDB. A fresh CoreWeave launch still in
bring-up (gang/leafgroup Kueue admission, `apply_ep`/mesh-load, `shm_broadcast …60s` + transient
ImagePullBackOff self-heal are BENIGN) → buffer-filling table with `—` until first step lands.

**New/untested RL run? → deep-probe it, don't trust the row.** A row can read "healthy" on a silently dead
run (weight-sync garbage, engine-starvation wedge, 0 trials completing). For any RL job in a new/untested
setting (new config/geometry/model/image, "debug"/"smoke-test", first launch after a code/config change),
dispatch a subagent armed with **`rl-job-health-deep-dive`** → it syncs trace_jobs + logs, live-polls the
GPUs against the serving-throughput LUT, reads the literal rollouts, returns a **KILL/NO-KILL recommendation**.

**Inspecting literal CoreWeave RL rollouts:** `scripts/iris/peek_rl_rollouts.sh <pod-name-substr>
[ls|cat|grep|cp]` exec's into the rank-0 pod and reads Harbor's per-trial `trace_jobs`. With the default local
`trials_dir` (`/app/experiments/<run>/trace_jobs`) these are pod-local + **EPHEMERAL** (lost on pod GC/replace;
no shared FS/PVC) — re-check during active generation. Durable path: `launch_rl_iris.py --trials-dir auto`
(default) → `s3://marin-us-east-02a/iris/<job>/trace_jobs` (R2, NOT gs://; resolves via `marin_prefix()` —
don't hardcode), inspect with `aws s3 --endpoint-url <R2>`. The helper forces the coreweave kubeconfig itself
(don't rely on `$KUBECONFIG`); `<pod-name-substr>` matches the POD name (can differ from iris job_id display name).

### Metrics to track per RL run (priority order)
**Core 5 (always):** `reward/avg_raw_reward` (primary), `reward/avg_pass_at_8` (less noisy), `policy/policy_loss`,
`policy/policy_entropy` (direction + magnitude matter — pre-collapse), `policy/raw_grad_norm` (most predictive;
healthy < 1.0; > 1.0 for ≥2 steps has predicted collapse 2–5 steps early). Under **seqnorm global-denom**,
grad/policy_loss/log_ratio are genuinely ~1e-5 — the regime, NOT vanishing-grad.
**Clip ratio (if wandb):** `policy/ppo_clip_ratio` ≈0 normally; >1% = LR↔eps_clip mismatch. Also
`policy/z_clip/triggered` for StaleClip/ZClip ablations.
**TIS:** `tis/imp_ratio_mean` (~0.84–1.56 healthy), `tis/imp_ratio_capped_fraction` (~0 healthy).
**Per-token log-ratio diag** (SkyRL ≥2026-05-06): `log_ratio_abs_{mean,p99,max}`, `n_tokens_dp_gt_{1,10,50}pct`,
`log_ratio_abs_pos00..pos90`. Healthy: `mean`~0.005–0.02, `max`<0.5, `gt_50pct`≈0, position buckets even.

### NOT a collapse signal — `rollout_train_prob_diff_mean`
`policy/rollout_train_prob_diff_mean` = `exp(rollout_lp − train_recompute_lp).abs().mean()` — the mean per-token
importance ratio, **dominated by outlier tokens** (one ~20-nat disagreement → `exp(20)≈5e8`). **Millions/billions
are NORMAL** on healthy DENSE arms (Qwen3-8B lrboost ~1e7, Qwen3-32B ~1e8). Reward is verifier-computed
(test-pass rate), independent of logprobs, so this can never "hit the reward." For a per-token-divergence read
use the **capped** `tis/imp_ratio_mean`/`imp_ratio_capped_fraction`, the median, or `log_ratio_abs_*` — not this mean.

### NOT a failure/hang cause — context-overflow + passthrough-exception lines
vLLM `... 32769 input tokens > 32768 max`, `ContextLengthExceededError`, and `AgentTimeoutError` are **benign
and expected** in agentic rollouts (harbor `passthrough_exceptions` → verifier still scores, rollout completes;
they appear in successful runs). **NEVER the reason a job hangs/fails.** Find the real terminal signal: a
`Traceback`, OOM / Raylet-died / SIGKILL, an RPC / `sample_tokens` timeout, a `RuntimeError`, or a hung Ray
actor / Daytona trial that never returns. (See `feedback_context_overflow_not_failure_cause`.)

### Collapse rule (≥2 fire same step → cancel+salvage)
`raw_grad_norm`>1.0 (or >2× window); `policy_entropy` off its 10-step trend >30%; `log_ratio_abs_mean`
>2× window while `max` bounded; trial pass-rate <10% over last 100. **Exception:** spike-mitigation
ablations (zclip/staleclip/maxgn09 etc.) are NEVER auto-cancelled on 2/4 — observing the recovery IS the experiment.

---

## SFT

```
┌──────────────────────────────┬─────────┬────────┬───────────┬───────────────────────────────────┐
│             Job              │  Step   │  Loss  │ Grad Norm │               Trend               │
├──────────────────────────────┼─────────┼────────┼───────────┼───────────────────────────────────┤
│ swesmith cold-start 2ep 8B   │ 320/916 │ 1.21   │ 0.84      │ Loss descending; healthy          │
└──────────────────────────────┴─────────┴────────┴───────────┴───────────────────────────────────┘
```
Columns: Job, Step (`cur/total`), Loss, Grad Norm, Trend. **No reward.**

**For multi-cell SFT grids (e.g. Delphi 54-SFT), ALSO give a grid-completion rollup each sweep:**
- Per RUNNING cell: **progress % = step/total** + rough ETA. Plus a one-line **running / pending-unique / done** tally.
- **Dedupe the PENDING count — it's ~3× inflated** (`afterany` restart-chain resume copies + RUNNING cells'
  own resume-backups in PD). True count: `squeue -u $USER -h -o '%j|%t' | grep '^sft__' | grep '|PD' | cut -d'|'
  -f1 | sort -u | wc -l`, then subtract running cells' backups. Report *distinct* cells remaining.
- **Long-pole:** for Delphi, the **`1e22` 9.7B cells** gate completion (4-node, ~34.7k steps, 24h wall → each
  needs 1–2 checkpoint-resume cycles, ≤8 concurrent). Small/medium (≤3e20) clear fast.
- **Gotcha — grep the TRAINING tqdm, not the packing bar.** `grep -aoE '[0-9]+%\|[^|]*\| [0-9]+/[0-9]+ '` can
  catch the dataset-tokenization/packing bar — verify the denominator matches the cell's total optimization
  steps (e.g. 26788 / 34720), not an example count.
- **Gotcha — a single tailed `s/it` is NOT the rate.** Checkpoint-save spikes (~30s serialize) inflate one line
  (e.g. `13.5 s/it` vs `4.5 s/it` baseline at save-interval boundaries). **Use a TRAILING-WINDOW rate** (average
  several step lines, or `Δwall/Δstep`); a single high `s/it` recurring at the save cadence is the spike, not a regression.

**Pull live status from the `.out`, NOT `trainer_log.jsonl`.** The `.out` carries LLaMA-Factory's per-step
dicts — richer (live grad_norm, per-rank loss spread, token coverage, epoch):
```
{'loss': 0.50, 'grad_norm': 0.42, 'learning_rate': 6.5e-06,
 'loss_rank_avg': 0.27, 'loss_nan_ranks': 0,
 'valid_targets_min': 5081, 'valid_targets_mean': 16083.6, 'epoch': 0.12}
```
`trainer_log.jsonl` is unreliable mid-run (sparse/empty/frozen) → false "stale/dead" readings. Find latest
`.out` (`ls -t experiments/<job>/logs/*.out | head -1`), grep the last `{'loss': ...}` lines + tail. Use the
JSONL only for the `"percentage": 100.0` completion check before consolidate/upload. Total steps from the
rendered config / trainer banner (`Total optimization steps = N`).

**Red flags:** `ChildFailedError` / `Exited with exit code 1` (read the FIRST real traceback above the
elastic summary — often masked), CUDA OOM at first fwd/bwd (eager attn at 32k → see env/attn), `SIGTERM`
(node fault OR masked rank crash — recurring ~Nmin death is NOT transient), loss→NaN, grad explosion.
**On completion → `sft-job-cleanup`** (recognize 8B root-safetensors vs 32B ZeRO-3-shards path first).

---

## Datagen

```
┌────────────────────────────────────┬──────────────┬─────────┬───────────┬──────┬──────┬──────────────────────────┐
│             Datagen run             │    Chunks    │ Trials  │ avg_turns │ Mean │ exc% │           Trend          │
├────────────────────────────────────┼──────────────┼─────────┼───────────┼──────┼──────┼──────────────────────────┤
│ codenet-python-v2 (MiniMax Row #34) │ 18/20 done   │ ~8.6k   │ 5.1       │ 0.53 │ 19%  │ 2 chunks running         │
└────────────────────────────────────┴──────────────┴─────────┴───────────┴──────┴──────┴──────────────────────────┘
```
Columns: run (+ tracker row), Chunks (`done/total`), Trials (`result.json` count), avg_turns, **Mean** (mean
reward — the harbor `<done>/<total> Mean: <X>` line; `iris … job logs | grep -aoE '[0-9]+/[0-9]+ Mean:
[-0-9.]+' | tail -1`; mark `—` if no verifier/no Mean line), exc%, Trend. **avg_turns is the realness gate** —
`>1` = real multi-step; **`≈1.0` = dead-engine run, do NOT consolidate**. exc% ~20–25% AgentTimeout is normal
for hard sets.
**Red flags:** `TIMEOUT` **strands the traces** (Harbor's terminal upload killed — traces on disk, NOT
uploaded → consolidate manually); a chunk **hung** (`.out` silent for hours + `result.json` count stalled while
RUNNING); avg_turns≈1.0.
**On ALL chunks complete → `datagen-job-cleanup`**.

---

## Eval

```
┌──────────────────────────────┬───────────┬───────────┬───────────┬────────────────────────────────┐
│   Eval (model × benchmark)   │  Trials   │ pass-rate │  top exc  │         Infra / Trend          │
├──────────────────────────────┼───────────┼───────────┼───────────┼────────────────────────────────┤
│ laion/<model> × tb2          │ 142/300   │ 0.21      │ AgentTO   │ pinggy✓ vLLM✓ ; healthy        │
└──────────────────────────────┴───────────┴───────────┴───────────┴────────────────────────────────┘
```
Columns: model×benchmark, Trials (`result.json`/total), pass-rate (fraction with
`verifier_result.rewards.reward`>0), top exception type, Infra/Trend. **Infra column = the 4 launch-checks**
(pinggy auth+traffic, Daytona `api_base` = public pinggy URL not internal IP, vLLM POSTs growing + 200-OK,
trial progression) — see `eval-agentic-launch` §4.
**Red flags:** no `result.json` in 60+min while RUNNING → stall; vLLM `Running:0` reqs 10+min → agents not
generating; **all trials done but job RUNNING → zombie (cancel)**; instant-fail (`n_output_tokens: None`,
`finished_at`≈`started_at`) → tunnel not carrying traffic; repeated `Bearer token invalid` → Daytona auth degradation.
**Before calling an eval "dead," confirm the RIGHT log + a CURRENT window.** Read the actual live log
(`data_<jobid>.out` if scontrol StdOut is absent), count `result.json` over the WHOLE run (not just the tail).
A burst of `litellm.Timeout` / `AgentTimeoutError` in the last window is usually the hard-trial tail of a
nearly-done run — NOT "0 productive / unreachable vLLM." Verify vLLM is actually down (no recent `200 OK`
POSTs) before blaming the engine.

### NOT a reliability problem — a high `AgentTimeoutError` fraction
A large timeout share — **even a majority of trials** — is EXPECTED on hard / long-horizon benchmarks and does
**NOT** make the eval unreliable or warrant re-flagging the delta. `AgentTimeoutError` is a harbor
`passthrough_exception` → the trial is **still scored** (unfinished task scores as not-solved), reflecting
genuine model capability. If the baseline ran the same harness, the score + delta **stand** — do not call a
leg "untrustworthy"/"timeout-inflated" on timeout rate alone. The ONLY timeout red flag is the infra case:
essentially *every* trial failing with **zero completions / no `result.json`** is a stall, not a score.

**On completion → `eval-agentic-cleanup`** IF auto-upload/register failed. **EXEMPT:** `DCAgent2/*`
grid/throughput/OOM **measurement** runs — report as calibration, not production.

---

## Catch-all / other (ad-hoc)

Anything that isn't one of the four majors — consolidate, pretokenize, `hf_upload` (tmux/sbatch), SIF builds,
DCP/CP/feature smoke + GPU-CI tests, measurement/grid probes. **Don't force a metric table** — one line each:

| Job | Type | State | Elapsed | Note |
|---|---|---|---|---|
| `consol34` (tmux) | datagen-consolidate | running | 12m | pushing 9407 rows → penfever/… |
| `861267` | gpu-ci (loop-reward Stage D) | COMPLETED | 6m | 2 passed — think-mask loss-finite |
| `hf_upload_lr80` (tmux) | RL upload | running | 3m | laion/lrboost-80-8B |

State + elapsed + a human note (what it is, the one signal that matters, any follow-up). Flag terminal
COMPLETED/FAILED + whether it needs action (a stuck `hf_upload`, a FAILED build → diagnose).

---

## Benign log-noise (do NOT chase as faults)

- **`shm_broadcast.py:737` "No available shared memory broadcast block found in 600 seconds"** is `logger.info`
  (heartbeat), NOT a kill signal. It re-fires at 10/20/30-min multiples while the engine waits with nothing to
  schedule. Fault-indicative **only when co-firing with** a real NCCL hang (`WorkNCCL(...Timeout(ms)=...) ran
  for N ms before timing out`, or `ProcessGroupNCCL preparing to dump debug info`, or a SIGABRT). Alone → look
  upstream for the engine-idle cause (Daytona auth errors, agent timeouts, no pending requests); do NOT relaunch
  or patch the ring buffer.
- **`rollout_train_prob_diff_mean` in the millions/billions** — outlier-dominated, normal (see RL §).
