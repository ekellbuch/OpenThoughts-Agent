# Harbor — dependency overview

The agent framework OT-Agent uses for **trace generation** (RL/SFT data) and **agentic eval**.

- **Repo:** local `/Users/benjaminfeuer/Documents/harbor`, branch **`penfever/working`**. Editable-installed on every cluster; synced via git (commit→push→`git pull`), never patched on the cluster.
- **⚠️ CANONICAL UPSTREAM = `marin-community/harbor`** (v0.7.0). `penfever/working` tracks `marin/penfever/working` — **always push here**. Other remotes (`laude`, `charlie`, `marianna`) are forks/mirrors — do not push to or pull from them. If a cluster `git pull` reports "Already up to date" but the fix isn't there, check `git remote get-url origin` points at `marin-community/harbor`.
- **CLI:** Typer app `harbor.cli.main:app` — `harbor run`, `harbor jobs start`, `harbor view`, `harbor trials start`.
- **Two OT-Agent uses:** (1) **datagen trace-gen** — run an agent over a task set, record rollout trajectories → HF dataset; (2) **agentic eval** — run an agent over a benchmark, verify, compute metrics. Both go through `hpc/launch.py` (`--job_type datagen`/`eval`) or the unified eval listener.

---

## Contributing to this fork (marin-community fork workflow)

Harbor is a **marin-community shared fork** — code contributions follow the marin-community fork workflow (shared with MarinSkyRL, evalchemy):

1. **Read `AGENTS.md` first.** Obey env/dev setup, test + lint entry points, PR norms.
2. **Follow the marin style conventions** (`uv run infra/pre-commit.py --all-files --fix`, `ty check`).
3. **Dev in an isolated worktree off a fresh branch from `main` → PR into `main`.** The `penfever/working` mega-consolidation branch is **RETIRED** — never commit to it or `main` directly.
4. **Iterate until ALL CI checks pass.** Diagnose + fix, never force past failures.
5. **Do NOT self-merge — return for approval.** No `Co-Authored-By` trailers; use the `agent-generated` PR label.

**Scope:** this applies ONLY to marin-community shared forks (harbor, MarinSkyRL, evalchemy). OT-Agent and the vllm fork keep their current norms (self-merge + trailers allowed).

---

## OT-Agent's harbor pin (`@main`, not a branch)

Two OT-Agent files pin harbor and must stay in sync: `pyproject.toml` (`harbor[daytona] @ git+…marin-community/harbor.git@main`) and `docker/Dockerfile.tpu` (`ENV HARBOR_COMMIT=<sha>` + `--force-reinstall`). The gpu-rl image's harbor pin lives in **MarinSkyRL** `docker/Dockerfile.gpu-rl`. `uv.lock` is the RUNTIME gate on iris workers (`uv sync --frozen`), so a harbor bump is a `uv lock --upgrade-package harbor` + commit, not only an image rebuild.

- **Pin is `@main`** (changed 2026-07-15 from `@penfever/working`). Harbor's repo has **auto-delete-head-branch ON**, so a squash-merge auto-deletes the merged branch — `@main` cannot be auto-deleted. Do not re-pin to a mergeable/deletable branch.
- The `pyproject.toml` pin resolves to main's floating HEAD; the Dockerfiles pin the concrete `HARBOR_COMMIT` sha (bump to main's HEAD on each rebuild).

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

## Per-trial `TimingInfo` duty-cycle recipe (result.json — reusable across clusters)

Every harbor trial writes a **`result.json`** (`TrialResult`, `models/trial/result.py`) with per-phase `TimingInfo` blocks. This is the clean source for a per-trial **duty-cycle breakdown** (LLM-gen vs tool-exec vs sandbox-lifecycle vs verifier). It is a harbor artifact, so the recipe is cluster-agnostic; only the per-cluster access to the trials bucket differs (`.claude/ops/<cluster>/`).

**Discipline — aggregates only, never pull raw trials to the Mac.** Read a bounded sample (newest ~200 by `LastModified` for steady-state; ~500 for error tails) and print only computed medians/percentiles. When the bucket is in-cluster-only (CoreWeave LOTA), aggregate in-pod.

**`result.json` field → phase map** (each phase is a `TimingInfo {started_at, finished_at}`, UTC ISO):

| Field (path) | Phase | Duration |
|---|---|---|
| `started_at` / `finished_at` | **Trial total wall** | `finished_at − started_at` |
| `environment_setup` | **Sandbox create** | `TimingInfo` dur |
| `agent_setup` | agent setup | `TimingInfo` dur |
| `agent_execution` | **LLM-gen + tool-exec (combined)** | `TimingInfo` dur |
| `verifier` | verifier | `TimingInfo` dur |
| `agent_result.metadata.api_request_times_msec` | **LLM-gen ONLY** (list of per-call msec) | `Σ list ÷ 1000` → s |
| `exception_info.exception_type` | error class | — |

Derived: **Tool-exec** = `agent_execution − LLM-gen`. **Teardown/gap** = `finished_at − max(phase.finished_at)` (sub-second on healthy runs). **Frac LLM-gen** = LLM-gen / total; **frac sandbox** = (`environment_setup` + teardown-gap) / total.

**Interpretation:** LLM-gen ≫ sandbox (e.g. ~89% vs <1%) ⇒ LLM-turn-bound, not sandbox-churn — lever is `n_concurrent_trials` / buffer depth. A material sandbox fraction (>10%, or `environment_setup` >10 s tail) = real re-provision churn. Count trials with `verifier.finished_at > finished_at` — expected 0 (non-zero = release-race).

---

## Literal-token trace datasets (opencode datagen)

Canonical reference for the `--record_literal` opencode trace datasets and their downstream SFT use. Applies to the opencode-131k campaign (`penfever/<task>-qwen3.5-122b-131k-opencode-traces`).

### What the literal columns are

`make_and_upload_trace_dataset` on a `--record_literal` job emits three parallel columns alongside the text `conversations`:

- `prompt_token_ids`, `completion_token_ids` — **list-of-lists**, one inner list per agent step (turn). Verbatim tokens the serving engine emitted (RecordProxy capture, correlated by `literal_correlator.py`).
- `logprobs` — list-of-lists of floats, same shape.

A row is "literal-populated" when `prompt_token_ids` is non-empty.

### Capture + correlation

For installed agents like `opencode`, LLM calls happen INSIDE the Daytona sandbox, so Harbor's `RolloutDetail` never sees the token IDs. A co-located `RecordProxy` captures the real tokens into one job-global `literal.jsonl` (interleaving all concurrent trials). There is no join key (ai-sdk strips the upstream completion id); `literal_correlator.py` reconstructs attribution from content via message-prefix extension + count-sequence matching (verify-or-skip: omit rather than mis-attribute).

Low "literal yield" is usually a **capture ceiling**, not corruption: short/fast or early-failing interactions leave few records; long multi-turn debugging loops capture richly.

### Decoding the token IDs

The IDs only decode with the **EXACT tokenizer the engine served with** — for the 131k campaign: **`Qwen/Qwen3.5-122B-A10B-FP8`** (`Qwen2Tokenizer`, vocab 248044). A stock Qwen3 tokenizer (~152k vocab) decodes word tokens to garbage. GCS mirror: `gs://marin-models-us/ot-agent/models/Qwen/Qwen3.5-122B-A10B-FP8/` (pull `tokenizer.json`/`tokenizer_config.json`/`vocab.json`/`merges.txt`).

Always pass `--served_model Qwen/Qwen3.5-122B-A10B-FP8` on any literal upload/rescue — it stamps `tokenizer_provenance.json`. Decode per-turn with `skip_special_tokens=False` to keep `<|im_end|>`/`<tool_call>`/`<think>` markers.

### Rescue from GCS

The worker's end-of-job HF upload cannot be trusted for preemptible jobs. Every terminal job is rescued from banked GCS by the ops cron. Before rescuing a FAILED job, spot-check trials' `result.json`: if 100% errored with `steps: 0` (e.g. `NonZeroAgentExitCodeError` exit 127 = opencode binary absent), there is nothing to rescue — it needs a full re-run.

**Rescue:** rsync the outer `gs://marin-models-{us,eu}/ot-agent/<job>/` (so `logs/*_literal.jsonl` rides along), clear the target repo's partial `data/`, re-run the uploader with `--served_model`. Verify with `count_populated_literal_rows`.

### Literal traces → SFT

`scripts/harbor/literal_traces_to_sft.py` converts a literal trace dataset into SFT data whose assistant turns are decoded verbatim from the literal completion tokens. Auto-resolves the tokenizer from `tokenizer_provenance.json`. Rows without literals, or whose assistant-turn count ≠ literal step count, are dropped. **LAION upload gotcha:** the `laion` org's PRIVATE storage quota is full — push SFT datasets PUBLIC.

---

## Resume + cross-cluster port

Two layers, easy to conflate:

- **Auto-resume** (`job.py` `Job.create()`): if `<job_dir>/config.json` exists, the run resumes — each existing trial dir is matched by strict Pydantic equality against the planned `TrialConfig` (any config drift → `FileExistsError`). **An errored trial still has a dir (with `exception_info` + no reward), so auto-resume treats it as complete and will NOT re-run it.** Re-launching the same run-tag with the same config is auto-resume.
- **`harbor jobs resume` (`cli/jobs.py:1375`) — the path that actually re-runs errored trials.** Walks trial dirs, reads each `result.json`, and **deletes any whose `exception_info.exception_type` is in `--filter-error-type`** so they re-run. `-f`/`--filter-error-type` is repeatable and **defaults to `["CancelledError"]`** — for non-cancelled errors you MUST pass the actual type(s) (bare Python class name: `DaytonaError`, `EnvironmentStartTimeoutError`, `DaytonaRateLimitError`). `AgentTimeoutError` is passthrough (verifier still scores it → has reward → counts VALID). Discover which types your errored trials carry via the aggregate `result.json`'s `exception_stats` map.
  - **Resuming an EVAL needs the served model live**, so it runs inside the eval sbatch (which brings vLLM up). The canonical eval sbatchs already wire `harbor jobs resume` with the standard error-type filters — just re-submit the same eval.
- **Port checklist** (`notes/harbor/port_checklist.md`): tiers what to port first when syncing harbor changes.

---

## Fork facts / load-bearing commits (on `penfever/working`)

- **`94379963`** — `iter_trial_dirs` prunes the `os.walk` at the trial-dir level → trace export no longer GPFS-stat-storms on 30k-trial runs (the Step-8 cleanup fix; see `.claude/skills/rl-agentic-job-cleanup`).
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
