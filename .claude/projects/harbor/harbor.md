# Harbor — dependency overview

The agent framework OT-Agent uses for **trace generation** (RL/SFT data) and **agentic eval**. Written
2026-06-14 from notes + the live `penfever/working` tree. Operational eval-job file layout + resume-debug
specifics live alongside this in `ops.md`; this is the architecture/facts overview.

- **Repo:** local `/Users/benjaminfeuer/Documents/harbor`, branch **`penfever/working`**, remote `marin` = `marin-community/harbor` (v0.7.0). Editable-installed on every cluster; synced via git (commit→push→`git pull`), never patched on the cluster.
- **CLI:** Typer app `harbor.cli.main:app` — `harbor run`, `harbor jobs start`, `harbor view`, `harbor trials start`.
- **Two OT-Agent uses:** (1) **datagen trace-gen** — run an agent over a task set, record rollout trajectories → HF dataset; (2) **agentic eval** — run an agent over a benchmark, verify, compute metrics. Both go through `hpc/launch.py` (`--job_type datagen`/`eval`) or the unified eval listener.

---

## Environment backends — `src/harbor/environments/`

Selected per-run (`--trace_env`/`--harbor_env`, or the YAML `environment.type`); lazy-loaded via
`environments/factory.py:EnvironmentFactory`. Backends: **daytona** (`daytona.py` + `daytona_utils.py` —
the production cloud-sandbox path), **docker** (`docker/`), **modal**, plus apptainer/enroot (HPC), e2b,
runloop, gke, tensorlake, apple_container.

**Daytona sandbox model** (the one we run): async REST SDK. Per trial: fetch the task Dockerfile →
`create_sandbox_from_image()` → `run_command()` in the live sandbox → `delete_sandbox()` (or snapshot).
**Snapshots** are named `harbor__<env_hash>` and are **shared across every run using those tasks** (keyed by
the sandbox-environment hash, not per-dataset) — see `.claude/projects/daytona/daytona.md` for the hard
org/per-launch caps and the prebuild flow (`hpc/snapshot_manager.py`). `daytona_utils.py` carries
exponential-backoff retry callbacks for transient errors; concurrency is gated by a semaphore in
`trial/queue.py`.

**Docker / Podman backend (local, no cloud).** For running trace-gen on a local/SLURM box without Daytona,
use `--trace-env docker` — `hpc/docker_runtime.py` auto-detects Docker vs Podman and sets `DOCKER_HOST`
(supports SSH tunnels to a remote daemon). Configs: `hpc/harbor_yaml/trace_docker_*.yaml`. Run via
`python -m data.local.run_tracegen --harbor-config hpc/harbor_yaml/trace_docker_<...>.yaml --tasks-input-path <dir> --trace-env docker`;
for SLURM+Podman `source docker/setup_docker_runtime.sh` first. Modal is the third backend (`--trace-env modal`).

---

## Terminus-2 agent — `src/harbor/agents/terminus_2/`

The production agent (`terminus_2.py`, ~1800 lines; tmux session mgmt + JSON/XML tool-call parsers +
asciinema recorder). Load-bearing behaviors:

- **Summarization** (`enable_summarize`, `proactive_summarization_threshold` default ~8000 free tokens): when context nears the limit, a subagent compresses history and the main chat is reset (trajectory file continues). **When `enable_summarize=false` (our RL/trace default for fidelity), NOTHING truncates the growing prompt** → context overflow (`VLLMValidationError: 32769 input tokens`); `max_input_tokens` is inert in that mode (see `.claude/skills/datagen-launch`). Notes: summarization fires on ~19–29% of trials and those trials solve 2–9× lower (confounded by harder/longer tasks); ~half the conversation text can be summarization bookkeeping.
- **Tool calling:** per-turn `debug.json`/`prompt.txt`/`response.txt`; parser (`json` or `xml`) → tmux exec → observation.
- **PRM hook (`turn_callback`):** invoked each episode; returning a **string** injects it (e.g. SkyRL's `prm/teacher_hint.py` → the literal `[HINT FROM TEACHER]:` marker prepended to the next observation), returning **True** requests early stop. See `.claude/skills/analyze-rl-behavior` for grepping hints.
- `store_all_messages`, `trajectory_config.{raw_content, linear_history}` control how much is persisted.

---

## Trial / trace data model

- **`RolloutDetail`** (`models/agent/rollout_detail.py`): per-turn `prompt_token_ids`, `completion_token_ids`, `logprobs`, and `extra: dict[str, list]` (provider fields like vLLM's `routed_experts`). Populated by **`Chat._accumulate_rollout_details()`** (`llms/chat.py`) after each LLM turn.
- **TIS length-parity guard** (commit `8737426c`): SkyRL zips `logprobs` onto `completion_token_ids` by position; if a turn's lengths mismatch, Harbor records an **empty logprob list** (not a silent mis-pair) so index alignment survives — downstream surfaces it via `tis/alignment_fail_count`. This is the harbor half of the TIS exact-alignment hardening (the SkyRL half is in `.claude/projects/marinskyrl/marinskyrl.md`).
- **`trajectory.json`** (per episode): ATIF `steps[]` with `source`/`message`/`tool_calls`; subagent (summarization) trajectories are separate files. `raw_content` dumps the raw LLM response instead of parsed tool_calls.
- **Per-trial footprint is large** — terminus-2 writes ~70–120 files/trial (3 per episode + subagent dirs); a 30k-trial job ≈ 2–3M FS entries. This is why trace export must prune (see below) and why GPFS hygiene matters.

---

## Resume + cross-cluster port

- **Resume** (`job.py`): if `<job_dir>/config.json` exists, `Job.create()` resumes — each existing trial dir is matched by **strict Pydantic equality** against the planned `TrialConfig`; any config drift → `FileExistsError`/`ValueError` (delete the stale run dir only after confirming no useful trials — see `ops.md`). **Cancelled trials are treated as complete and NOT re-attempted** by default; `--filter-error-type=CancelledError` deletes those dirs so they re-run (`notes/harbor/resume.md`).
- **Port checklist** (`notes/harbor/port_checklist.md`): tiers what to port first when syncing harbor changes (config/deps → core trace/LLM utils → HPC backends → architecture → deletions). The OT-Agent orchestrator/TrialQueue path is kept; upstream's separate orchestrator system + Terminus-3 are NOT ported.

---

## Fork facts / load-bearing commits (on `penfever/working`)

- **`94379963`** — `iter_trial_dirs` prunes the `os.walk` at the trial-dir level → trace export no longer GPFS-stat-storms on 30k-trial runs (the Step-8 cleanup fix; see `.claude/skills/rl-job-cleanup`).
- **`8737426c`** — the TIS per-turn logprob/token-id length-parity guard (above).
- **`ec508562`** (+ throttle follow-up `e05d569d`) — reap orphaned LiteLLM logging tasks on between-turns timeout/cancel → fixes the Ray ObjectRef-leak SIGABRT on AgentTimeout-heavy datasets (a separate bug class from the uvloop fix).
- Install: `pip install -e .` with extras `[daytona]`/`[modal]`/`[cloud]`/`[all]`.

---

## Key config surfaces (the `hpc/harbor_yaml/` YAMLs in OT-Agent)

`n_concurrent_trials` (concurrency / sandbox count — per-job Daytona ceiling ~128), `n_attempts`
(samples/task), agent `enable_summarize` / `max_input_tokens` / `store_all_messages` /
`trajectory_config.{raw_content,linear_history}`, `environment.type` (backend). Datagen configs under
`hpc/harbor_yaml/datagen/` (match the model's `max_model_len`: `ctx32k.yaml`/`ctx131k.yaml`), eval configs
under `hpc/harbor_yaml/eval/` (`eval_ctx32k_non_it.yaml` etc.). Recompute metrics offline:
`scripts/harbor/recompute_result_json.py <run_dir> --metrics-config <yaml>`.

> Eval-run file layout (`$EVAL_JOBS_DIR/<run_tag>/`), `exception.txt` debugging, and the config-mismatch
> resume fix live in `ops.md` in this directory.
