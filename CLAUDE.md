# CLAUDE.md

Guidance for Claude Code when working in this repository.

**ot-agent** is a distributed LLM training + evaluation system for HPC clusters, with four subsystems —
**datagen** (Harbor/Daytona traces + standard vLLM generation), **SFT** (LLaMA-Factory), **RL** (SkyRL/GRPO),
**eval** (terminal-bench / agentic). One unified launcher: `python -m hpc.launch --job_type <type>`.

## Source of truth = `.claude/`

This file is a thin index. The real, maintained documentation lives under **`.claude/`** — read the relevant
piece for the task at hand (skills are also invocable by name via the Skill tool):

- **`.claude/skills/<name>/SKILL.md`** — operational how-tos, one per task. By prefix:
  - **launch:** `rl-agentic-launch-jupiter`, `rl-agentic-launch-iris` (CoreWeave H100), `rl-standard-launch-leonardo` (non-agentic GRPO), `sft-launch` (LLaMA-Factory + axolotl · Delphi), `datagen-launch` (agentic Harbor trace-gen), `datagen-standard-launch` (Curator + `generate.py`), `eval-agentic-launch`, `eval-standard-launch` (+ `*-iris` variants).
  - **cleanup:** `rl-agentic-job-cleanup` (agentic RL + traces), `rl-standard-job-cleanup` (standard GRPO — model + metrics only, no traces), `sft-job-cleanup`, `datagen-job-cleanup`, `eval-agentic-cleanup`, `eval-standard-cleanup`.
  - **monitor:** `monitor-cron-sweep`, `monitor-job-tables`, `rl-job-health-deep-dive` (per-RL-job probe → KILL/NO-KILL), `monitor-restore` (3-hourly sweep loop), `monitor-restore-iris-cron`.
  - **analysis / data / db:** `analyze-rl-behavior`, `analyze-dataset-token-length`, `analyze-id-eval-ranking` (z-score ranking), `datagen-reduce-dataset-snapshots`, `crud-otagent-supabase`, `crud-purge-stale-eval-placeholders`, `crud-purge-below-gate-evals`.
  - **code:** `code-create-staged-plan` → `notes/<codebase>/`, `code-execute-staged-plan` → `agent_logs/`.
  - **build:** `build-gpu-rl-image-iris` (kaniko in-cluster; Mac can't build it).
  - **role:** `supervisor-init` — session bootstrap.
- **`.claude/projects/<dep>/`** — facts & gotchas per codebase: `ot-agent/` (branches + launcher map), `marin/`, `marinskyrl/`, `harbor/`, `vllm/`, `llama-factory/`, `axolotl/`, `daytona/`, `ajudge/`.
- **`.claude/ops/<target>/`** — machine/cluster particulars (access, paths, env/SIF map, gotchas): `jupiter/`, `leonardo/`, `torch/`, `iris/`, `local/` (this Mac), `all/` (cross-cluster HF/tmux), `experiments/` (the per-experiment tracker workspace `~/Documents/experiments`), `data/` (dataset trackers — e.g. `tasktrove.md`, the full TaskTrove inventory).
- **`.claude/secret.md`** — untracked, gitignored; holds privileged values (pinggy bank, etc.) pulled out of the committable docs. Referenced by name from skills/ops.

## Always (apply before any skill loads)

- **Run Python via the otagent env's full interpreter path** (symlinks don't work in the sandbox): `/Users/benjaminfeuer/miniconda3/envs/otagent/bin/python`. (`curator` env only for Curator datagen.)
- **Syntax/lint check** with the IDE MCP tool `mcp__ide__getDiagnostics`, NOT `python -m py_compile`/`flake8` (bash output capture is unreliable here).
- **Local clones are ground truth; clusters never diverge.** All edits go in local checkouts → push → `git pull` on the cluster. Branch flow differs by repo:
  - **marin-community forks** (`harbor`, `MarinSkyRL`, `evalchemy`): worktree → PR → `main`. `penfever/working` is RETIRED. A subagent never self-merges a marin PR.
  - **`OpenThoughts-Agent` + `vllm` fork**: stay on `penfever/working` (commit → push; may self-merge).
  - Clusters `git pull` the tracked branch (Python repos are editable installs, live after pull). An unmerged marin-fork fix rides `--harbor-ref`/`--skyrl-ref`, never a shared-branch mutation. **No untracked/divergent changes on a cluster; no patch-by-rsync; no hand-editing.** vLLM (compiled) is built from source on each cluster from the committed fork.
- **Standing ML-ops guardrails** (full statements in `monitor-restore` / the cleanup skills): `enable_db_registration: false` in YAMLs (manual DB register only); ≤6 RUNNING RL jobs per cluster (Daytona); a3 series CONCLUDED; Daytona snapshot caps are HARD (clean stale, never raise); cross-user FK safety pre-check before any Supabase delete/mutate; HF uploads default PUBLIC to `laion/`; never kill a RUNNING job without explicit permission.
