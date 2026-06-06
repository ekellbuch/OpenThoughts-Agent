#!/bin/bash
# Launch a batch of OPD grid cells on Leonardo from opd_grid_cells.txt.
# Each cell -> one sbatch of sbatch_opd_qwen3.sh (2 nodes) with --job-name=opd_<cell>,
# 8h wall, the cell's MAX_STEPS, and the cell's overrides appended.
#
# Usage:
#   launch_opd_grid.sh <cell1> [<cell2> ...]    # launch named cells
#   launch_opd_grid.sh --batch1                 # the 8 Batch-1 cells
#   launch_opd_grid.sh --batch2                 # the 6 Batch-2 base cells
# Appends "cell=<name> jobid=<id>" to opd_grid_manifest.txt for harvest.
set -uo pipefail
CFGDIR=/leonardo_work/AIFAC_5C0_290/bfeuer00/code/OpenThoughts-Agent/hpc/skyrl_yaml/leonardo
SPEC=$CFGDIR/opd_grid_cells.txt
SBATCH=$CFGDIR/sbatch_opd_qwen3.sh
MANIFEST=$CFGDIR/opd_grid_manifest.txt
WALL=08:00:00

# The sbatch's hardcoded defaults are SMOKE sizing. Always prepend the TRUE BASE
# config (matches live full run 44708545); per-cell overrides come AFTER and win.
BASE_KNOBS="TRAIN_BATCH_SIZE=64 MINI_BATCH_SIZE=64 N_SAMPLES=8 MAX_GEN_LEN=1024 TOPK=128"

BATCH1=(base topk32 topk64 topk256 teacherTP4 tbs128 gen512 cudagraph)
BATCH2=(lr3e6 lr3e5 lr1e4 n4 n16 topk16)

declare -a WANT
if [[ "${1:-}" == "--batch1" ]]; then WANT=("${BATCH1[@]}")
elif [[ "${1:-}" == "--batch2" ]]; then WANT=("${BATCH2[@]}")
else WANT=("$@"); fi
[[ ${#WANT[@]} -eq 0 ]] && { echo "no cells given"; exit 1; }

get_line() { grep -E "^\s*$1\s*\|" "$SPEC" | head -1; }

for cell in "${WANT[@]}"; do
    line=$(get_line "$cell")
    if [[ -z "$line" ]]; then echo "!! no spec for cell '$cell' in $SPEC"; continue; fi
    # split on |
    IFS='|' read -r c steps overrides <<< "$line"
    c=$(echo "$c" | xargs); steps=$(echo "$steps" | xargs); overrides=$(echo "$overrides" | xargs)
    echo ">> launching cell=$c MAX_STEPS=$steps base=[$BASE_KNOBS] overrides=[$overrides]"
    jid=$(sbatch --parsable --job-name="opd_${c}" --time="$WALL" "$SBATCH" \
            $BASE_KNOBS MAX_STEPS="$steps" $overrides)
    rc=$?
    if [[ $rc -ne 0 || -z "$jid" ]]; then echo "!! sbatch FAILED for $c (rc=$rc)"; continue; fi
    echo "   submitted jobid=$jid"
    echo "cell=$c jobid=$jid steps=$steps overrides=[$overrides] ts=$(date -u +%FT%TZ)" >> "$MANIFEST"
done
echo "== manifest: $MANIFEST =="
tail -20 "$MANIFEST"
