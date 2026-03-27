# Harbor Evaluation — Sherlock HPC (Stanford)

End-to-end guide for running Harbor agent evaluations on Sherlock.

## Architecture

```
┌──────────────┐     ┌──────────────────┐     ┌───────────┐     ┌──────────┐
│   Harbor     │────>│    Apptainer     │────>│  Verifier  │────>│ Results  │
│   Agent      │     │    Sandbox       │     │  (tests/)  │     │  (JSON)  │
│  (terminus-2)│     │  (task image)    │     │            │     │          │
└──────────────┘     └──────────────────┘     └───────────┘     └──────────┘
       │
  vLLM Server
  (GPU node)
```

Key difference from other clusters: Sherlock uses **Apptainer** (not Docker/Daytona)
as the sandbox backend. Harbor runs inside `harbor_runner.sif` which launches
nested Apptainer containers for each task.

## Prerequisites

- Harbor venv at `$GROUP_HOME/$USER/Projects/coding_agent/scripts/harbor/.venv`
- Harbor runner image at `$GROUP_SCRATCH/$USER/.cache/harbor/harbor_runner.sif`
- Task images cached at `$GROUP_SCRATCH/$USER/.cache/harbor/` (auto-pulled on first use)
- uv/uvx binaries at `$GROUP_HOME/$USER/Projects/coding_agent/scripts/harbor/tools/bin/`

## Quick Start

### 1. Validate infrastructure (oracle)

The oracle agent runs each task's gold `solution.sh`. Use it to verify the
eval pipeline works before spending GPU time on model evals.

```bash
cd $GROUP_HOME/$USER/Projects/coding_agent/modules/OpenThoughts-Agent

# Single task smoke test
sbatch eval/sherlock/oracle.sbatch 1 largest-eigenval

# All tasks
sbatch eval/sherlock/oracle.sbatch
```

Oracle should achieve ~100% on tasks whose images are cached.

### 2. Run model evaluation

```bash
# Submit eval job (starts vLLM + Harbor)
sbatch eval/sherlock/unified_eval_harbor.sbatch <MODEL> <DATASET>

# Example:
sbatch eval/sherlock/unified_eval_harbor.sbatch \
  nvidia/Nemotron-Terminal-8B \
  DCAgent2/terminal_bench_2
```

### 3. Monitor

```bash
# Job status
squeue -u $USER

# Oracle logs
tail -f eval/sherlock/logs/oracle_<JOBID>.out

# Model eval logs (vLLM + Harbor)
tail -f eval/sherlock/logs/eval_<JOBID>.out
tail -f eval/sherlock/logs/vllm_<JOBID>.log
```

## Files

| File | Purpose |
|------|---------|
| `oracle.sbatch` | Oracle eval (no GPU, validates infrastructure) |
| `oracle.yaml` | Harbor config for oracle agent + Apptainer |
| `unified_eval_harbor.sbatch` | Model eval (GPU, vLLM + Harbor) |
| `dcagent_eval_config.yaml` | Harbor config for model agent + Apptainer |
| `secret.env.template` | API keys template |
| `snapshot_download.py` | HuggingFace dataset downloader |

## Environment Variables

Sherlock provides these automatically:

| Variable | Example | Used for |
|----------|---------|----------|
| `$GROUP_HOME` | `/home/groups/swl1` | Project files, venvs |
| `$GROUP_SCRATCH` | `/scratch/groups/swl1` | Caches, images, temp data |
| `$USER` | `ekb` | User-specific paths |

## Experiment Configs

Cluster-generic scripts live here (`eval/sherlock/`). Experiment-specific configs
(model, dataset, concurrency, etc.) go in:

```
$GROUP_HOME/$USER/Projects/coding_agent/experiments/<experiment-name>/
```

## Troubleshooting

**Job pending on Priority**: Try adding more partitions (`--partition=owners,normal,swl1`)
or use `dev` for quick tests (`--partition=dev --time=1:00:00 --cpus-per-task=2 --mem=16G`).

**ModuleNotFoundError in Harbor**: The harbor venv needs all dependencies installed.
Check with: `uv pip list --python scripts/harbor/.venv/bin/python | grep <module>`

**Image pull timeout**: Pre-pull large images with `scripts/harbor/oracle/prepull_images.sbatch`.

**Config not found**: SLURM copies scripts to its spool dir, so `$(dirname "$0")` doesn't work.
All paths in sbatch scripts must be absolute (derived from `$GROUP_HOME/$USER`).
