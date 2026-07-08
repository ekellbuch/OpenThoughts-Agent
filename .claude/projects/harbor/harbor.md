# Harbor — dependency overview

The agent framework OT-Agent uses for **trace generation** (RL/SFT data) and **agentic eval**. Written
2026-06-14 from notes + the live `penfever/working` tree. Operational eval-job file layout + resume-debug
specifics live alongside this in `ops.md`; this is the architecture/facts overview.

- **Repo:** local `/Users/benjaminfeuer/Documents/harbor`, branch **`penfever/working`**. Editable-installed on every cluster; synced via git (commit→push→`git pull`), never patched on the cluster.
- **⚠️ CANONICAL UPSTREAM = `marin-community/harbor`** (v0.7.0). The laptop's `origin`/`marin`/`upstream` all point here and `penfever/working` tracks `marin/penfever/working` — **always push here**. Other remotes on the laptop (`laude` = `laude-institute/harbor`, `charlie`, `marianna`) are forks/mirrors and are NOT kept in sync — do not push to or pull from them. **Gotcha (fixed 2026-06-17):** both cluster clones had a STALE `origin` = `laude-institute/harbor` (frozen at an old commit), so `git pull` silently no-op'd and never saw new laptop pushes. Both were repointed to `marin-community/harbor`. If a cluster `git pull` reports "Already up to date" but the fix isn't there, **check `git remote get-url origin` points at `marin-community/harbor`** first.
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

## Literal-token trace datasets (opencode datagen)

Canonical reference for the `--record_literal` opencode trace datasets and their downstream SFT use. Applies to the opencode-131k campaign (`penfever/<task>-qwen3.5-122b-131k-opencode-traces`) and any future literal datagen.

### What the literal columns are

`make_and_upload_trace_dataset` on a `--record_literal` job emits three parallel columns alongside the text `conversations`:

- `prompt_token_ids`, `completion_token_ids` — **list-of-lists**, one inner list per agent step (turn). The verbatim tokens the serving engine emitted (RecordProxy capture, correlated by `literal_correlator.py`).
- `logprobs` — list-of-lists of floats, same shape as `completion_token_ids`.

A row is "literal-populated" when `prompt_token_ids` is non-empty (`count_populated_literal_rows` checks exactly this).

### Capture + correlation — and why "literal yield" varies by arm

For **installed agents like `opencode`** the LLM calls happen INSIDE the Daytona sandbox (via ai-sdk), so Harbor's `RolloutDetail` never sees the token IDs — the trajectory steps carry only token *counts*. A co-located `harbor.literal.proxy.RecordProxy` on the iris worker captures the real `prompt_token_ids`/`completion_token_ids`/`logprobs` into one job-global `literal.jsonl` (interleaving all ~N concurrent trials). ai-sdk strips the upstream vLLM completion id, so there is **no join key**; OT-Agent's `scripts/harbor/literal_correlator.py` reconstructs attribution from content, verify-or-skip (omit rather than mis-attribute). A trial's tokens must clear **two gates** to land in the exported dataset:

- **Gate 1 — capture.** The proxy only logs `status==200` responses carrying a `completion_token_ids` block. A trial that makes few successful LLM calls contributes few/zero records. This is the dominant yield driver and it is **task-shaped**: short/fast or early-failing interactions (e.g. `llm-verifier`, `methods2test`) leave almost no records, while long multi-turn debugging loops (code/bug-fix arms) capture richly.
- **Gate 2 — unique binding.** Records are grouped into per-trial chains by exact message-**prefix** extension (opencode replays full history each turn), then a chain binds to a trajectory only if their per-step `(prompt_tokens, completion_tokens)` **count sequences match exactly and uniquely**. Fails when: duplicate/templated task prompts make a record extend two chains (**ambiguous → skipped**), or a **short trajectory** yields a non-distinctive count signature that collides across trials (long trajectories are effectively fingerprints).

**So low "literal yield" (populated-rows / total-trials) is usually a Gate-1 capture ceiling, not corruption or a binding bug.** Measured on the qwen3.5-122b-131k-opencode campaign: code/bug arms ~77–95% (nl2bash 90%, nemotron 95%, stack-junit 77%); verifier/test arms ~6–10% (llm-verifier 510/8965=5.7%, methods2test 108/1039=10.4%). For llm-verifier the log held only ~678 chains for 8965 trials (capture ceiling ~7.6%), and binding then succeeded on ~75% of *captured* chains — i.e. the low headline is few-records-captured, not can't-bind. The literals that *are* captured are correct (verify-or-skip guarantees omission over mis-join).

### Decoding the token IDs → text

The IDs only decode with the **EXACT tokenizer the engine served with**. For the 131k campaign that is **`Qwen/Qwen3.5-122B-A10B-FP8`** (a `qwen3_5_moe`, `Qwen2Tokenizer`, **vocab 248044**, specials at 248044–248076). A stock Qwen3 tokenizer (~152k vocab) shares only the low-id digit/whitespace/structure tokens, so word tokens decode to **garbage**.

- GCS mirror (guaranteed source; the HF repo may be gated): `gs://marin-models-us/ot-agent/models/Qwen/Qwen3.5-122B-A10B-FP8/` — pull just `tokenizer.json`/`tokenizer_config.json`/`vocab.json`/`merges.txt` (~22 MB, no weights).
- Which tokenizer produced a dataset is recorded in its **`tokenizer_provenance.json`** (+ a README decode recipe), written by `make_and_upload_trace_dataset` when you pass `--served_model`. **Always pass `--served_model Qwen/Qwen3.5-122B-A10B-FP8` on any literal upload/rescue** — omitting it still uploads the columns but stamps only the engine-reported served-name.
- Decode per-turn (list-of-lists) with `skip_special_tokens=False` to keep `<|im_end|>`/`<tool_call>`/`<think>` markers.

### Every terminal job must be rescued from GCS

The worker's end-of-job HF upload cannot be trusted for these preemptible (`v5p + --max-retries`) jobs — "SUCCEEDED" or "repo exists" is never proof of trainable data. In-job export lands text-only (the pinned `:tpu` worker image predates the literal-column fix), and preempt-resumed runs fail export outright. So **every terminal job is rescued from banked GCS by the 3-hourly ops cron**. (`FAILED at export-push` and `landed text-only` both mean "rescue from GCS.")

**Precheck before rescuing — is there anything to rescue?** A third terminal mode is a job that produced **zero valid traces**: every trial errored with `steps: 0` / no LLM calls (e.g. `NonZeroAgentExitCodeError` exit **127**: the `opencode` binary/nvm/node absent in the Daytona sandbox — a task-env provisioning failure, dataset-specific). These have no literal.jsonl AND no usable text — NOT rescuable; they need a **full RE-RUN after the sandbox issue is fixed**, not a GCS rescue. Before rescuing a FAILED job, spot-check a few trials' `result.json`/`exception.txt`: if 100% errored with 0 steps, mark BLOCKED + flag for re-run.

**Rescue procedure:** rsync the OUTER `gs://marin-models-{us,eu}/ot-agent/<job>/` (so sibling `logs/*_literal.jsonl` rides along), clear the target repo's partial `data/` (keep README + `tokenizer_provenance.json`), then re-run the uploader with `--served_model` from the otagent env. Verify with `count_populated_literal_rows` that the literal count ≈ the correlation yield.

### Literal traces → SFT

`scripts/harbor/literal_traces_to_sft.py` converts a literal trace dataset into an SFT dataset whose **assistant turns are decoded verbatim from the literal completion tokens** (real `<think>` + native tool calls). It emits `conversations` (ShareGPT) + a reasoning-preserving `text` string, and **auto-resolves the tokenizer from the source's `tokenizer_provenance.json`** (override with `--tokenizer`). Rows without literals, or whose assistant-turn count ≠ literal step count, are dropped. `--validate N` dry-runs.

**LAION upload gotcha:** the `laion` org's PRIVATE storage quota is full — push SFT datasets there **public** (public storage isn't quota-capped). A private push succeeds but 403s on readback (`Private repository storage limit reached`).

---

## Resume + cross-cluster port

Two layers, easy to conflate:

- **Auto-resume** (`job.py` `Job.create()`): if `<job_dir>/config.json` exists, the run resumes — each existing trial dir is matched by **strict Pydantic equality** against the planned `TrialConfig` (any config drift → `FileExistsError`/`ValueError`; delete the stale run dir only after confirming no useful trials — see `ops.md`). **It keeps EVERY existing trial dir and runs only the truly-missing ones. An errored trial still has a dir (with `exception_info` + no reward), so auto-resume treats it as complete and will NOT re-run it.** Re-launching the same run-tag with the same config is auto-resume.
- **`harbor jobs resume` (`cli/jobs.py:1375`) — the partial-resume path that actually re-runs errored trials.** Before resuming it walks the trial dirs, reads each `result.json` → `TrialResult`, and **deletes any whose `exception_info.exception_type` is in `--filter-error-type` so they re-run.** `-f`/`--filter-error-type` is **repeatable** and **defaults to `["CancelledError"]`** — so for non-cancelled errors you MUST pass the actual type(s). The type string is the bare Python class name (`type(e).__name__` in `models/trial/result.py:31`) — e.g. `DaytonaError`, `EnvironmentStartTimeoutError`, `DaytonaRateLimitError`. **`AgentTimeoutError` is passthrough (the verifier still scores it → it has a reward → counts VALID) and is deliberately NOT filtered.** `--upload` after is an idempotent fill-in-missing-trials sweep.
  - **Discover which types your errored trials carry** before choosing `-f`: the aggregate `result.json` has an `exception_stats` map keyed by exception_type (`models/job/result.py`), or parse per-trial `result.json` → `exception_info.exception_type`.
  - **Resuming an EVAL re-runs trials → needs the served model live**, so it runs inside the eval sbatch (which brings vLLM up), not as a bare CLI call. **The canonical eval sbatch already wires this** (Jupiter `eval/jupiter/eval_harbor.sbatch` resume block; the live Leonardo variant `eval/leonardo/eval_harbor.sbatch:625` mirrors it): if `$RUN_DIR/config.json` exists it calls `harbor jobs resume -p $RUN_DIR --filter-error-type EnvironmentStartTimeoutError --filter-error-type DaytonaError --filter-error-type DaytonaRateLimitError`. **So the clean way to resume a partial eval is just to re-submit the same eval (same run-tag) — valid trials are kept, those three error classes re-run.** If an eval's errored trials are a type NOT in that list (check `exception_stats` first; the Jupiter `eval/jupiter/eval_harbor.sbatch` also filters the Daytona auth/authorization/notfound variants), widen the sbatch's filter list (commit→push→pull) or run `jobs resume` by hand with the right `-f` flags.
- **Port checklist** (`notes/harbor/port_checklist.md`): tiers what to port first when syncing harbor changes (config/deps → core trace/LLM utils → HPC backends → architecture → deletions). The OT-Agent orchestrator/TrialQueue path is kept; upstream's separate orchestrator system + Terminus-3 are NOT ported.

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
