# eval/marlowe

Harbor eval on Marlowe DGX H100 SuperPod. Adapted from eval/sherlock/.

## Quick start

### 1. Validate infrastructure (oracle)

Oracle runs each task's gold `solution.sh` — no model, no GPU needed.

```bash
cd /users/ekb/Projects/tbench_sherlock/modules/OpenThoughts-Agent

# Single task smoke test
sbatch eval/marlowe/oracle.sbatch 1 largest-eigenval

# All tasks
sbatch eval/marlowe/oracle.sbatch

# With Daytona (set EVAL_ORACLE_CONFIG)
EVAL_ORACLE_CONFIG=path/to/oracle_daytona.yaml sbatch eval/marlowe/oracle.sbatch
```

### 2. Run model evaluation (YAML-driven)

The YAML config drives everything — model, agent, environment, concurrency.

```bash
cd /users/ekb/Projects/tbench_sherlock

# Via run_eval.sh (reads YAML, sets vLLM env, calls sbatch)
./scripts/oa/eval/run_eval.sh docs/marlowe/nemotront_eval8b_daytona/dcagent_eval_config_daytona.yaml

# Or directly (YAML-driven path)
EVAL_HARBOR_CONFIG=path/to/config.yaml \
EVAL_VLLM_MAX_MODEL_LEN=40960 \
EVAL_VLLM_TOOL_CALL_PARSER=hermes \
  sbatch modules/OpenThoughts-Agent/eval/marlowe/unified_eval_harbor.sbatch
```

### 3. Monitor

```bash
tail -f /scratch/m000120-pm05/ekb/data/coding_agent/outputs/logs/ot-eval-*.out
tail -f /scratch/m000120-pm05/ekb/data/coding_agent/outputs/logs/ot-vllm_*.log
```

## Differences from Sherlock

| | Sherlock | Marlowe |
|---|---|---|
| OS | CentOS 7, kernel 3.10 | Ubuntu 22.04, kernel 5.15 |
| GPU | Mixed | H100 80GB |
| Harbor | Runs inside `harbor_runner.sif` (glibc 2.17 too old) | Runs natively from `scripts/harbor/.venv` |
| Apptainer fakeroot | Works | Broken (not in `/etc/subuid`) |
| Daytona | Supported | Supported (primary env for evals) |
| SLURM | `--partition=swl1,gpu,owners` | `--partition=preempt --account=marlowe-m000120` |

## Files

| File | Purpose |
|------|---------|
| `oracle.yaml` | Harbor config for oracle agent (Apptainer) |
| `oracle_daytona.yaml` | Harbor config for oracle agent (Daytona) |
| `oracle.sbatch` | SLURM job for oracle evaluation |
| `dcagent_eval_config.yaml` | Harbor config for model eval (Apptainer) |
| `dcagent_eval_config_daytona.yaml` | Harbor config for model eval (Daytona) |
| `unified_eval_harbor.sbatch` | SLURM job for model eval (vLLM + Harbor) |
| `unified_eval_listener.sh` | Marlowe wrapper for shared eval listener daemon |
| `snapshot_download.py` | HuggingFace dataset downloader |
| `secret.env` | API keys (not committed, see `secret.env.template`) |

## How configs drive evals

When `EVAL_HARBOR_CONFIG` is set, `unified_eval_harbor.sbatch` uses `harbor run -c` and lets the YAML control agent, model, environment, and concurrency. The model name for vLLM is also read from the YAML (`model_name:` field). Without `EVAL_HARBOR_CONFIG`, it falls back to CLI args for backward compatibility.
