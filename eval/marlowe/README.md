# eval/marlowe

Harbor eval on Marlowe DGX H100 SuperPod. Adapted from eval/sherlock/.

## Quick start

```bash
# 1. Source secrets
source ~/.env

# 2. Submit eval (Nemotron-8B on Terminal-Bench 2.0)
sbatch eval/marlowe/unified_eval_harbor.sbatch \
    nvidia/Nemotron-Terminal-8B \
    /scratch/m000120-pm05/ekb/sky_harbor/terminal-bench-2

# 3. Monitor
tail -f eval/marlowe/logs/eval_<JOB_ID>.out
```

## Differences from Sherlock

| | Sherlock | Marlowe |
|---|---|---|
| GPU | Mixed | H100 80GB |
| vLLM | `module load py-vllm` | `uv run` from sky_harbor venv |
| Harbor | Runs inside `harbor_runner.sif` | Runs directly from sky_harbor venv |
| Apptainer cache | `/scratch/groups/swl1/ekb/.cache/singularity` | `/scratch/m000120-pm05/ekb/.apptainer-cache` |
| SLURM | `--partition=swl1,gpu,owners` | `--partition=preempt --account=marlowe-m000120` |
