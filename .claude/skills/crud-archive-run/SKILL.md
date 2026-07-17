---
name: crud-archive-run
description: >-
  Durably ARCHIVE everything informative from a finished run / experiment before it's cleaned up or its
  cluster artifacts age out — ALL Harbor trace_jobs (raw per-trial traces), ALL ray logs, ALL stdout/stderr
  (incl. vLLM/serving logs), and wandb. Pack-rat by design: if it's potentially informative, keep it. Only
  skip the non-informative-or-huge (model weights/checkpoints, core/memory dumps, massive raw tmux-pane /
  terminal-recording bytes). Many-tiny-files → tar THEN rsync (never rsync thousands of small files raw). Use
  when concluding/archiving an experiment, before a cleanup skill `rm`s an on-disk tree, or before CoreWeave
  R2/pod artifacts get GC'd. Per-run-type component maps live below; WHERE each artifact lives per cluster is a
  pointer into `.claude/ops/<cluster>/` and `.claude/projects/{harbor,marinskyrl,ot-agent}/`.
---

# crud-archive-run

**Principle (operator is a pack rat):** *if it is potentially informative, ARCHIVE it.* Bias to keeping
too much. Only skip files that are both **non-informative AND large**. When in doubt, keep it.

## ✅ ARCHIVE (always — every run type)
- **All raw Harbor traces** — every per-trial dir under `trace_jobs/` (RL) / `eval_jobs/<name>/` (eval) /
  the datagen trace dir: `result.json`, `config.json`, `manifest.json`, `lock.json`, `agent/trajectory.json`
  (the raw agent transcript — INFORMATIVE, keep even at multi-MB), `verifier/` (`reward.txt`, `ctrf.json`,
  `test-stdout.txt`), `step_results`, `trajectory.summarization-*.json`.
- **All ray logs** — the ray session dir (raylet, gcs_server, per-worker `*.out`/`*.err`, `python-core-*`).
- **All stdout / stderr** — SLURM `.out`/`.err`, the CoreWeave finelog (the WHOLE log, not a tail), the
  serving log **`vllm.log`** (yes, keep it even at 100s of MB — a log is informative), `job.log`, per-trial
  `trial.log`.
- **wandb** — the local `wandb/` run dir if present; else record the run URL/id in the archive's `MANIFEST`.
- **Configs / launch command / rendered YAML / metric CSVs / `trainer_log.jsonl`.**

## ❌ SKIP (non-informative AND large)
- **Model weights / checkpoints** — `*.safetensors`, `*.pt`, `*.bin`, `global_step_*/`, consolidated shards
  (they live on HF / R2; not useful for post-hoc debugging).
- **Core / memory dumps** — `core.*`, `*.hprof`, coredump trees.
- **Massive raw terminal-pane bytes** — `*.pane` (raw tmux pane dumps) and `agent/recording.cast` (asciinema)
  **only when large** and redundant with `trajectory.json`; keep small casts.
- Conda/uv/pip caches, extracted wheel trees, `__pycache__`, `.venv`.

## Mechanic — tar many-small-files THEN rsync
rsync of thousands of tiny trace files is dominated by per-file round-trips + inode overhead. So on the
**source** (cluster/pod), `tar` the small-file tree first, then rsync the single tarball:
```
# on the cluster (SLURM) — one tarball per run, excluding the SKIP set
tar --exclude='*.safetensors' --exclude='*.pt' --exclude='*.bin' --exclude='global_step_*' \
    --exclude='core.*' --exclude='*.pane' \
    -czf /tmp/<run>_archive.tgz -C <run_dir> trace_jobs logs *.log config* wandb  # adjust to what exists
rsync -aP <cluster>:/tmp/<run>_archive.tgz  <dest>/         # then rm the /tmp tarball
```
Keep genuinely-large single logs (`vllm.log`) either inside the tarball or rsync'd alongside — do not drop
them. Verify the tarball is non-empty + lists the expected trees (`tar tzf … | head`) before deleting the source.

## Per-run-type components — WHAT + WHERE (pointers, they drift — read the ops/projects doc)
- **CoreWeave agentic RL (SkyRL/MarinSkyRL)** — traces at the durable R2 path
  `s3://marin-us-east-02a/iris/<job>/trace_jobs` (`--trials-dir auto`; pull with `aws s3 --endpoint-url <R2>`)
  OR pod-local `/app/experiments/<run>/trace_jobs` (EPHEMERAL → grab before pod GC via
  `scripts/iris/peek_rl_rollouts.sh <pod> cp`). Full log = `iris … job logs --since-ms <submit> --no-tail`
  (finelog keeps init→crash). ray logs + `vllm.log` are pod-local (exec/peek). wandb online (record URL).
  Details: `.claude/ops/iris/coreweave_gpu_ops.md`, `.claude/projects/marinskyrl/`.
- **SFT (LLaMA-Factory / axolotl, SLURM)** — `.out` per-step logs at `experiments/<job>/logs/*.out`,
  `trainer_log.jsonl`, rendered config, wandb. Weights → HF (SKIP). Log path via `scontrol show job <id> -o`
  `StdOut=`/`%Z`. Details: `.claude/projects/{llama-factory,axolotl}/`, the cluster ops doc.
- **Datagen (Harbor traces)** — one-level `trace_jobs/<trial>/result.json` + the harbor run log; the artifact
  is the trace set (→ HF), but archive the trace_jobs + logs. Details: `.claude/projects/harbor/`.
- **Eval (agentic Harbor)** — `eval_jobs/<name>/<trial>/{result.json,config.json,agent/trajectory.json,
  verifier/,manifest.json,lock.json}` + top-level `vllm.log`, `job.log`, per-trial `trial.log`. Skip the big
  `recording.cast`/`*.pane` when large. Details: `.claude/projects/harbor/`, `eval-agentic-cleanup`.
- **Cluster WHERE-do-logs-live** is per-cluster: `.claude/ops/iris/` (CoreWeave, no ssh — iris SDK + R2),
  `.claude/ops/tacc/` (SLURM `.out` + `/scratch/…/eval_jobs`), `.claude/ops/empireai/` (SLURM `~/logs/*.out`,
  socket-flaky → sbatch-detach). READ the relevant one first.

## Destination
Default: into the experiment's tracker dir — `~/Documents/experiments/<active|complete>/<exp>/run_archive/<run-id>/`
(`experiments/` is not a git repo → large local archives are fine here). Write a one-line `MANIFEST` per run
(run-id, cluster, job-id, dates, what was kept/skipped, wandb URL). For truly-durable off-Mac storage, also push
the tarball to HF (`penfever/…-archive`, PUBLIC default `laion/`) or R2 — but local-under-the-experiment is the
baseline. Do this BEFORE any `*-cleanup` skill `rm`s the on-disk tree (archive → verify → then cleanup reclaims).
