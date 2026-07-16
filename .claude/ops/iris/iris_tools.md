# Iris tooling catalog — `scripts/iris/`

Inventory + when-to-reach-for-which index for **Iris** jobs (CoreWeave **GPU** cluster `cw-us-east-02a` and Google **TPU** `marin` cluster). Per-cluster access/hardware particulars live in `coreweave_gpu_ops.md` (GPU) and `iris_job_lifecycle.md` + `iris_google_tpu_cloud_hardware.md` (TPU); launch procedure in the **`rl-agentic-launch-iris`** skill.

> **Preamble for every script below** (full rationale in `coreweave_gpu_ops.md`):
> - **Python / CLI = the otagent env, full path:** `/Users/benjaminfeuer/miniconda3/envs/otagent/bin/python` and `/Users/benjaminfeuer/miniconda3/envs/otagent/bin/iris` (the marin `.venv` iris has a broken `kubernetes` import → CANNOT drive CoreWeave).
> - **`export KUBECONFIG=~/.kube/coreweave-iris-gpu`** before ANY CoreWeave call (the Mac default points at a different context → misleading "0 pods / not found").
> - **`source "$DC_AGENT_SECRET_ENV"`** for submit/mirror scripts (HF/WANDB/DAYTONA/AWS passthrough; see `.claude/secret.md`).
> - **All `iris`/`kubectl` calls SYNCHRONOUS** — never background them.
> - `--cluster=cw-us-east-02a` is a **top-level** flag on the `iris` CLI (BEFORE the subcommand: `iris --cluster=cw-us-east-02a job logs …`), not a per-subcommand option.

---

## Tier 1 — everyday operational tools (monitoring + rollout inspection)

### `watch_job_state.py` — authoritative job-state watcher *(the liveness primitive)*
Polls the **authoritative iris job lifecycle state** (`iris job summary --json` → SQL `query` fallback → `kubectl` pod cross-check), NOT log-string content — so it catches clean kills/evictions/preemptions/early crashes that emit no terminal log line.
- **Use:** every liveness/terminal check; importable as the watch primitive (`get_job_state()` returns a `JobStateSnapshot`; `watch()` runs the loop, returns the terminal snapshot).
- **CLI:** `watch_job_state.py <job_id> [--cluster cw-us-east-02a] [--interval 60] [--once] [--no-pods] [--max-polls N] [--json]`.
- **Exit codes:** `0` succeeded · `1` failed/killed/worker_failed/unschedulable · `2` absent from controller AND 0 pods (disappeared) · `3` watch error.
- **Parse gotcha:** with `--json --once` it prints a human `[HH:MM:SS] … state=running …` line *before* the JSON object — a naive `json.load(stdin)` chokes on both. Parse the human line (`grep -oE "state=[a-z]+"`) or strip to the JSON braces.
- **JobState enum:** `0` UNSPECIFIED `1` PENDING `2` BUILDING `3` RUNNING `4` SUCCEEDED `5` FAILED `6` KILLED `7` WORKER_FAILED `8` UNSCHEDULABLE.

### `analyze_job_history.py` — full-log pull + throughput/preemption stats *(the science tool)*
Paginates the ENTIRE job log by fixed time windows (`--since-ms` + `--no-tail`, the only way past `--tail`'s line cap) and filters at the python level to retain just the signal (cycle boundaries, vLLM throughput emissions), caching the filtered stream to `/tmp/iris_history_<job>.filtered.log`. Emits a markdown report + JSON sidecar with **§1** preemption count + time-to-preempt, **§2** trace progress per cycle (from harbor GCS output), **§3** serving throughput (full + warmup-excluded).
- **Use:** sel_rows / EPDIAG / throughput **science only** — *never* for liveness/terminal detection (that's `watch_job_state.py`). Also the way to recover a dead run's root cause from the full log.
- **CLI:** `analyze_job_history.py <job_id> --output <report.md> [--refresh] [--warmup-seconds 180] [--cluster …] [--iris-bin …] [--gsutil-sample …]`. Auto-resolves the cw-capable iris binary (`resolve_iris_bin()`: `$IRIS_BIN` → PATH → otagent env → marin `.venv` last).
- **⚠ CoreWeave (`--cluster cw-us-east-02a`) needs R2 archive creds.** The archive half reads `s3://marin-na/finelog/cw-us-east-02a` (R2) — the Mac lacks the creds, so the run crashes `FileNotFoundError: The specified bucket does not exist` unless you first source them from the `iris`-ns secret. **Full procedure: `.claude/ops/iris/finelog_r2_archive_creds.md`.** (The live half on cw uses a k8s tunnel, NOT IAP — so the `analyze-job-history-iris` skill's `marin-login` step is marin/TPU-only.)

### `peek_rl_rollouts.sh` — inspect / capture a running RL job's Harbor rollout artifacts
Reaches the **rank-0 pod** of a running agentic-RL job and reads its `trace_jobs` (per-trial trajectory + prompts/responses + `verifier_output` + `result.json` reward). The jobs use a **remote object-store `trials_dir`** (`s3://marin-us-east-02a/iris/<job>/trace_jobs`, durable) whose creds live only in the pod (the launch-host Mac lacks cluster creds), so **all object-store ops run INSIDE the pod via boto3**. (`finelog` archive `s3://marin-na/finelog/…` is a SEPARATE marin-controlled location.) `result.json` is the COMPLETED-trial marker (carries the reward) → its count is the real "how many trials finished".
- **Use:** "is the rollout buffer actually filling / what rewards are coming back / pull the full trace bundle for analysis".
- **Subcommands:** `<pod-substr>` (summary: started + completed + breakdown) · `ls [glob]` · `cat <trial-dir>` (dump a trial's json) · `grep <pattern>` · `cp <trial-dir> [dest]` · `pull [out-base]` (FULL CAPTURE → date-stamped dir: finelog + per-rank pod logs + all `trace_jobs` synced from the CW object store `s3://marin-us-east-02a` + `MANIFEST.md`).
- **`<substr>` matches the POD name** (`iris-benjaminfeuer-<name>-<rank>-<hash>-0`), which can differ from the iris job_id display name; no match → lists candidate RL pods.
- **Env:** `PEEK_KUBECONFIG` (default `~/.kube/coreweave-iris-gpu`), `NS`/`CONTAINER`, `PEEK_CLUSTER`, `IRIS_BIN`, `PEEK_OUT`, `PEEK_TRIALS_S3`, `PEEK_MAX_OBJECT_BYTES` (pull skip-size, default 20 MB; `0` = fetch everything). Forces the cw kubeconfig — ignores an inherited `$KUBECONFIG`.

---

## Tier 2 — RL runtime (load-bearing; you don't invoke it by hand)

### `start_rl_iris_controller.py` — the per-node multi-node RL bootstrap
Canonical copy lives in **MarinSkyRL `cloud/iris/start_rl_iris_controller.py`** (invoked by `python -m cloud.iris.launch_rl_iris`). iris runs **this same entrypoint on every node** of a gang (injecting `IRIS_TASK_ID`/`IRIS_NUM_TASKS`/`IRIS_ADVERTISE_HOST` per task). It bootstraps ONE cross-node Ray cluster: **rank 0** `ray start --head` → publishes head IP to the rendezvous file → waits for all nodes to join → `exec`s the MarinSkyRL driver with `RAY_ADDRESS` set; **ranks 1..N-1** read the head IP from the rendezvous, `ray start --address=…`, contribute their 8 H100s, and block until rank 0 writes the `done` marker.
- **Rendezvous:** `ray_head.json` / `ray_head.done` under `--rendezvous-dir` (`OT_AGENT_IRIS_RENDEZVOUS_DIR`); opened via `fsspec` so `gs://` / `s3://` (CoreWeave CW object store `marin-us-east-02a`) / NFS all work. Pins ALL Ray agent ports outside the worker range (fixes the nondeterministic port-collision).
- **Invoked by** the RL launcher — you never type it directly; edit it in MarinSkyRL locally (rides the `/app` upload, no image rebuild). (The old OT-Agent copy `scripts/iris/start_rl_iris_controller.py` and its one-shot MoE/EP bring-up probes have been REMOVED — the MarinSkyRL port is the sole home; author new bring-up probes against `cloud/iris/` in MarinSkyRL.)

---

## Tier 3 — TPU-cluster data plumbing (the `marin` cluster, NOT CoreWeave)

Weight-mirroring helpers for the **Google TPU** Iris (`marin`) — staging model weights between HF, GCS, and the LAION/Jülich S3 so vLLM's `runai_streamer` (needs real S3 + GCS HMAC keys it doesn't have) can read them. Each is a `mirror_*` worker + a `launch_*` iris-job submitter.

| Script | Direction | Notes |
|---|---|---|
| `mirror_hf_to_gcs.py` | HF repo → `gs://marin-eu-west4/ot-agent/models/` | One shard at a time (download→upload→delete), so it doesn't need the full model on disk; idempotent (size-skip), resumable. |
| `launch_hf_mirror.py` | submits `mirror_hf_to_gcs.py` as an iris job | Marin has no CPU-only pool → runs on the smallest TPU slice (v6e-4), TPU idle; one-shot ~30–60 min, don't queue against a busy cluster. |
| `mirror_gcs_to_s3.py` | GCS prefix → S3 (e.g. LAION `mmlaion` @ Jülich) | Streaming gcsfs→boto3, one file at a time; endpoint from `$AWS_ENDPOINT_URL` or `--s3-endpoint`; idempotent. Workaround for missing GCS HMAC keys `runai_streamer` requires. |
| `launch_gcs_to_s3.py` | submits `mirror_gcs_to_s3.py` as an iris job | Companion to `launch_hf_mirror.py`, opposite direction. |

### `patch_tpu_inference.py` — runtime patches to the TPU worker's `tpu-inference`
Invoked from the TPU launcher's bash bootstrap **after `uv sync`, before the workload**. Each patch is idempotent + prints a one-line status. Currently: the `hbm_usage_bytes()` non-addressable-device skip (guards `device.memory_stats()` on multi-host slices >v6e-8 where non-local chips raise `INVALID_ARGUMENT`).

---

## Cross-reference
- **Access / hardware / scheduling (GPU):** `coreweave_gpu_ops.md` (incl. the full-log pagination recipe + the liveness=state-poll rule these tools implement).
- **Launch procedure (GPU RL):** the `rl-agentic-launch-iris` skill; canonical launcher **MarinSkyRL `cloud/iris/launch_rl_iris.py`** (`python -m cloud.iris.launch_rl_iris`, run from `~/Documents/MarinSkyRL`). The OT-Agent `rl/cloud/launch_rl_iris.py` copy has been removed.
- **TPU job lifecycle / hardware:** `iris_job_lifecycle.md`, `iris_google_tpu_cloud_hardware.md`, `iris_eval_fixed_snapshot_template_scoping.md`.
