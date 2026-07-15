# Local (this Mac) — ops

The control machine: launches, uploads, Supabase queries, code edits, and analysis run from here; clusters do the GPU work. Sync discipline is commit→push→`git pull` on the cluster (see `.claude/projects/marinskyrl/marinskyrl.md`). Re-probe specs if acting on this months later.

## System
- **macOS 26.5.1** (25F80), **Apple M4 Max** (`Mac16,5`), **arm64**.
- **14 CPU cores** (10P + 4E), **36 GB** unified RAM.
- **No local NVIDIA GPU.** All CUDA/training/vLLM-serving runs on clusters; MPS is for tiny smoke tests only. Treat the Mac as CPU-only for ML.

## Disk
- Single 926 GB APFS container. **Check free space with `df -h /System/Volumes/Data` (the `Avail` column).**
- **⚠ Do NOT use `df -h /`'s `Size`.** `/` is the sealed **System** volume; its `Size` is the whole shared APFS container, so computing "free = Size − Used" over-estimates massively. The true free is the Data volume's `Avail`; the 20-GB-prune threshold is measured against THAT.
- **Reducible hogs (du -sh):** `~/Documents/experiments/traces` (RL rollout traces — prune per-experiment ONLY after HF upload is confirmed, never blanket `rm`), `~/.cache/huggingface` (safe `rm -rf`, re-downloads on demand), `~/Library/Caches`. Prune the HF cache first if free drops toward ~50 GB.
- **Do NOT pull large model weights / trace datasets to the Mac** — stage on cluster scratch. Local is code + notes + logs only.

## Homebrew
- **Homebrew 5.1.14**, prefix `/opt/homebrew` (arm64).
- **Leaf formulae:** `agg asciinema cmake codex coreutils gh git-lfs go helm helm@3 k3d kubelogin kubernetes-cli kustomize llama.cpp marp-cli netcat pandoc pango poppler rpm rustup socat step tectonic wget`.
- **Casks:** `docker-desktop`, `gcloud-cli`, `font-lato`, `font-raleway`.
- Load-bearing: `gh`, `git-lfs`, `step` (CINECA/Leonardo step-ca SSH certs), `gcloud-cli` (Iris/TPU + GCS), `kubernetes-cli`/`helm`/`k3d` (k8s), `tectonic` (LaTeX), `coreutils` (GNU `g*`).

## Conda
- **conda 25.5.1**, base at **`/Users/benjaminfeuer/miniconda3`** (Python 3.13.5, torch 2.8.0 — don't train in base).
- **`otagent` is the canonical env for all OT-Agent / Harbor / SkyRL work** (Python 3.12.12, torch 2.9.0). Use the full interpreter path — symlinks don't work in the sandbox:
  ```
  /Users/benjaminfeuer/miniconda3/envs/otagent/bin/python
  ```
- Other envs (`conda env list`): `harbor` (3.13.9), `llama-factory` (3.12.12), `marin` (3.13.13), `oumi` (3.12.11), `ajudge`, `marvis`, `abb`, `openreview`, `sweagent`, `tokviz-rt`. `otagent` is default unless a task is env-specific (e.g. `ajudge` — see `.claude/projects/ajudge/`).
- Syntax/lint: use the IDE MCP `mcp__ide__getDiagnostics`, not `python -m py_compile`/`flake8` (bash output capture is unreliable).

## Local codebases (`/Users/benjaminfeuer/Documents/`)
All editable-installed into the relevant conda env; **edit here, commit, push, then `git pull` on the cluster** — never patch on the cluster. Branch SoT per repo:
- **`OpenThoughts-Agent/`** — launcher/orchestration. Branch `penfever/working` (origin `open-thoughts/OpenThoughts-Agent`). See `.claude/projects/ot-agent/`.
- **`harbor/`** — agent framework. Branch `penfever/working`; remote `marin` = `marin-community/harbor`. See `.claude/projects/harbor/`.
- **`MarinSkyRL/`** — RL framework. Branch `penfever/working` (= `marin-community/MarinSkyRL`). See `.claude/projects/marinskyrl/`.
- **`vllm/`** — inference-engine fork (`mlfoundations/vllm`). Currently on `feuer/dcp-gqa-lse-fix`. Local clone is ground truth; **built from source on each cluster (per-arch) from the committed fork** (or baked into a SIF) — **never** rsync working-tree edits or hand-patch. Some envs run vanilla vLLM. See `.claude/projects/vllm/`.

## Other local paths
> **All paths here live under `/Users/benjaminfeuer/Documents/`, NOT inside the `OpenThoughts-Agent` repo.** Always reference them by ABSOLUTE path — a bare relative `agent_logs/` / `experiments/` / `notes/` resolves against the current working directory (usually the repo checkout for a subagent or launcher) and silently writes the wrong place. Hand any of these to a subagent by full path.
- **`/Users/benjaminfeuer/Documents/agent_logs/`** — dated investigation/post-mortem logs, `YYYY-MM-DD_<topic>.md`. Write a dated entry when diagnosing a genuine FAILED job (per the cron-sweep skill). Absolute path, always — NOT the repo's `OpenThoughts-Agent/agent_logs/`.
- **`experiments/`** — per-experiment working state, one subdir per series with its own tracker(s). See `.claude/ops/experiments/ops.md`.
- **`notes/`** — private knowledge base (subdirs: `RL/`, `marin/`, `harbor/`, `llama-factory/`, `vllm/`, `ot-agent/`, `jsc/`, `nvidia/`, `scaling_laws_papers/`, …). SoT for several things the `.claude/` docs mirror (MiniMax datagen tracker, pinggy bank). Per-cluster notes: `notes/leonardo.md`, `notes/jsc/`, `notes/perlmutter` (in `ot-agent/`).
- **Runtime secrets** — path, key inventory, and load snippet live in `.claude/secret.md` (set `$DC_AGENT_SECRET_ENV` to point at it; `~/secrets.env` on clusters). Load so a subprocess inherits the vars:
  ```bash
  set -a; source "$DC_AGENT_SECRET_ENV"; set +a
  ```
- **`.claude/secret.md`** (untracked, gitignored) — privileged values pulled out of the committable skills/ops docs (pinggy URL+token bank, Leonardo HedgeDoc URL). Referenced by name from those docs.

### Secrets
Credentials provided: HuggingFace (`HF_TOKEN`), **Daytona** (`DAYTONA_API_KEY`, `DAYTONA_B_KEY`, `DAYTONA_DATA_API_KEY` [used by datagen], `DAYTONA_RL_API_KEY`), **Supabase** (`SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`, `SUPABASE_ANON_KEY`), W&B (`WANDB_API_KEY`), OpenAI (`OPENAI_API_KEY` — see the LLM-judge datagen note re: the OUMI key), **LAION S3** (`LAION_ENDPOINT`, `LAION_BUCKET_NAME`, `LAION_ACCESS_KEY`, `LAION_SECRET_KEY`), **Marin GCS HMAC** (`MARIN_HMAC_ACCESS_ID`, `MARIN_HMAC_SECRET`, `MARIN_PREFIX`), AWS (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`), Docker (`DOCKER_USER_ID`, `DOCKER_TOKEN`), **Pinggy** (`PINGGY_API_KEY`, `PINGGY_IDENTITY_FILE`), Modal (`MODAL_PROFILE`), Together (`TOGETHER_API_KEY`), Vast (`VAST_API_KEY`), OpenReview (`OPENREVIEW_USERNAME`, `OPENREVIEW_PASSWORD`), `SSH_KEY`. **Reference by name only** in committable docs — never paste a value.

## Cluster SSH aliases (`~/.ssh/config`)
`Jupiter`, `Leonardo`, `perlmutter`, `torch` (NYU), `TACCVista`, `ALCFPolaris`, `OLCFFrontier`, `TUDCapella`, `EmpireAI_Alpha1`, `OumiLambdaSLURM`, `OracleCloud`, NYU `hegde-lambda-{1,2}`. Active cron scope: **Jupiter + Leonardo** (Perlmutter dropped). Per-cluster access/paths/preamble live in `.claude/ops/<cluster>/`. Leonardo uses a `step ssh certificate` cert expiring ~12h (refresh from here — see `.claude/ops/leonardo/ops.md`).
