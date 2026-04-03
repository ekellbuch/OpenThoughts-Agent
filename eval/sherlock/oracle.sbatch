#!/bin/bash
#SBATCH --job-name=tb-oracle
#SBATCH --partition=owners,normal,swl1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/groups/swl1/ekb/logs/eval_sherlock/oracle_%j.out
#SBATCH --error=/scratch/groups/swl1/ekb/logs/eval_sherlock/oracle_%j.err

# ==============================================================================
# Oracle Eval — Sherlock (Stanford Research Computing)
#
# Runs gold reference solution.sh for each Terminal-Bench 2.0 task to validate
# the Apptainer eval infrastructure. Oracle should achieve ~100% on tasks whose
# images are cached. No model or GPU needed.
#
# Usage:
#   sbatch eval/sherlock/oracle.sbatch                    # all tasks
#   sbatch eval/sherlock/oracle.sbatch 3                  # smoke test (3 tasks)
#   sbatch eval/sherlock/oracle.sbatch 1 largest-eigenval # single task
# ==============================================================================

set -euo pipefail

N_TASKS="${1:-}"
TASK_NAME="${2:-}"

CODING_AGENT_DIR="$GROUP_HOME/$USER/Projects/coding_agent"
REPO_DIR="$CODING_AGENT_DIR/modules/OpenThoughts-Agent"
SCRIPT_DIR="$REPO_DIR/eval/sherlock"
HARBOR_VENV="$CODING_AGENT_DIR/scripts/harbor/.venv"
HARBOR_CLI="$HARBOR_VENV/bin/harbor"
HARBOR_DIR="$CODING_AGENT_DIR/modules/harbor"
CONFIG="$SCRIPT_DIR/oracle.yaml"

TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
OUTPUT_DIR="$GROUP_SCRATCH/$USER/data/coding_agent/outputs/harbor/tb-oracle-${TIMESTAMP}"

export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-$GROUP_SCRATCH/$USER/.cache/singularity}"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# --- Preflight checks ---
if [ ! -f "$HARBOR_CLI" ]; then
    log "ERROR: Harbor venv not found at $HARBOR_VENV"
    exit 1
fi

if [ ! -f "$CONFIG" ]; then
    log "ERROR: Config not found: $CONFIG"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
mkdir -p "$GROUP_SCRATCH/$USER/logs/eval_sherlock"

log "Starting Terminal-Bench 2.0 oracle evaluation on $(hostname)"
log "  Config:  $CONFIG"
log "  Output:  $OUTPUT_DIR"

# --- Build harbor args ---
HARBOR_ARGS="run -c $CONFIG --jobs-dir $OUTPUT_DIR"
if [ -n "${N_TASKS:-}" ] || [ -n "${TASK_NAME:-}" ]; then
    HARBOR_ARGS="$HARBOR_ARGS --dataset terminal-bench@2.0"
    [ -n "${N_TASKS:-}" ] && HARBOR_ARGS="$HARBOR_ARGS -l $N_TASKS"
    [ -n "${TASK_NAME:-}" ] && HARBOR_ARGS="$HARBOR_ARGS -t $TASK_NAME"
fi

# --- Run Harbor via Singularity (Apptainer-in-Apptainer) ---
HARBOR_RUNNER_IMAGE="${HARBOR_RUNNER_IMAGE:-$GROUP_SCRATCH/$USER/.cache/harbor/harbor_runner.sif}"
HARBOR_SITE_PACKAGES="$HARBOR_VENV/lib/python3.13/site-packages"
HARBOR_SRC="$HARBOR_DIR/src"

log "Running: harbor $HARBOR_ARGS"

singularity exec \
    --bind "$GROUP_HOME":"$GROUP_HOME" \
    --bind /home/users:/home/users \
    --bind /scratch:/scratch \
    --bind /usr/bin/apptainer:/usr/bin/apptainer \
    --bind "$CODING_AGENT_DIR/scripts/harbor/etc_apptainer":/etc/apptainer \
    --bind /usr/libexec/apptainer:/usr/libexec/apptainer \
    --bind /var/lib/apptainer:/var/lib/apptainer \
    --env "PYTHONPATH=$HARBOR_SITE_PACKAGES:$HARBOR_SRC" \
    "$HARBOR_RUNNER_IMAGE" \
    bash -c 'unset APPTAINER_BIND SINGULARITY_BIND APPTAINER_CONTAINER SINGULARITY_CONTAINER APPTAINER_NAME SINGULARITY_NAME APPTAINER_COMMAND SINGULARITY_COMMAND APPTAINER_ENVIRONMENT SINGULARITY_ENVIRONMENT; exec /usr/local/bin/python3 -m harbor.cli.main '"$HARBOR_ARGS"

log "Done. Results: $OUTPUT_DIR"
