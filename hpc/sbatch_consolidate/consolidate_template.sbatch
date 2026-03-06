#!/bin/bash
{partition_directive}
{account_directive}
#SBATCH --time={time_limit}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={cpus_per_task}
{gpu_directive}
{mem_directive}
#SBATCH --job-name={job_name}
#SBATCH --output={output_path}
#SBATCH --mail-type=END,TIME_LIMIT,FAIL

set -euo pipefail

# --- Environment setup (auto-generated from HPC config) ---
{environment_preamble}

# --- Source dotenv and secrets ---
if [ -n "${{DCFT:-}}" ] && [ -f "$DCFT/hpc/dotenv/{cluster_env_file}" ]; then
  source "$DCFT/hpc/dotenv/{cluster_env_file}"
elif [ -n "${{DC_AGENT:-}}" ] && [ -f "$DC_AGENT/hpc/dotenv/{cluster_env_file}" ]; then
  source "$DC_AGENT/hpc/dotenv/{cluster_env_file}"
fi

if [ -n "${{DCFT_ACTIVATE_ENV:-}}" ]; then
  set +u
  eval "$DCFT_ACTIVATE_ENV"
  set -u
fi

SECRET_FILE="${{DC_AGENT_SECRET_ENV:-${{KEYS:-}}}}"
if [[ -n "${{SECRET_FILE}}" && -f "${{SECRET_FILE}}" ]]; then
  set -a
  source "${{SECRET_FILE}}"
  set +a
fi

export PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=1

if [ -z "${{DCFT:-}}" ]; then
  if [ -n "${{DC_AGENT:-}}" ]; then
    export DCFT="$DC_AGENT"
  else
    export DCFT="$PWD"
  fi
fi

mkdir -p "{experiments_dir}"
mkdir -p "{experiments_dir}/logs"

export CONSOLIDATE_INPUT="{consolidate_input}"
export CONSOLIDATE_INPUT_KIND="{consolidate_input_kind}"
export CONSOLIDATE_OUTPUT_REPO="{consolidate_output_repo}"
export CONSOLIDATE_BASE_REPO="{consolidate_base_repo}"
export CONSOLIDATE_WORKDIR="{consolidate_workdir}"
export CONSOLIDATE_COMMIT_MESSAGE="{consolidate_commit_message}"
export CONSOLIDATE_PUSH_TO_HUB="{consolidate_push_to_hub}"

python3 {python_script} \
  --input "${{CONSOLIDATE_INPUT}}" \
  --input-kind "${{CONSOLIDATE_INPUT_KIND}}" \
  --output-repo "${{CONSOLIDATE_OUTPUT_REPO}}" \
  --base-repo "${{CONSOLIDATE_BASE_REPO}}" \
  --workdir "${{CONSOLIDATE_WORKDIR}}" \
  --commit-message "${{CONSOLIDATE_COMMIT_MESSAGE}}" \
  --push-to-hub "${{CONSOLIDATE_PUSH_TO_HUB}}" \
  --project-root "{project_root}"
