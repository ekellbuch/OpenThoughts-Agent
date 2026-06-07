#!/bin/bash
# Launch the OPD THROUGHPUT grid on Leonardo from throughput_grid_cells.txt.
# Each cell -> one sbatch of sbatch_opd_qwen3.sh (2 nodes, 8h wall) with
# --job-name=opdtp_<cell>, the cell's MAX_STEPS, and its overrides appended.
#
# Concurrency: LANES (default 8) jobs run at once. Cells beyond LANES are
# chained onto the least-loaded lane via --dependency=afterany:<prev_jobid>,
# so a lane's next cell starts when its previous cell ENDS (any exit).
#
# BASE = combo_opt (the accuracy winner). BASE_KNOBS is prepended to every
# cell; per-cell overrides come AFTER and win (env/hydra last-wins).
#
# Usage:
#   launch_throughput_grid.sh --batch1            # the launchable Batch-1 cells
#   launch_throughput_grid.sh <cell1> [<cell2>...] # named cells
#   LANES=8 launch_throughput_grid.sh --batch1
# Appends "cell=<name> jobid=<id> dep=<prev>" to throughput_grid_manifest.txt.
set -uo pipefail
CFGDIR=/leonardo_work/AIFAC_5C0_290/bfeuer00/code/OpenThoughts-Agent/hpc/skyrl_yaml/leonardo
SPEC=$CFGDIR/throughput_grid_cells.txt
SBATCH=$CFGDIR/sbatch_opd_qwen3.sh
MANIFEST=$CFGDIR/throughput_grid_manifest.txt
WALL=08:00:00
LANES=${LANES:-8}

# combo_opt base: lr3e-5 . topk64 . n8 . gen1024 . teacher 32B TP2 . cudagraph off.
BASE_KNOBS="TRAIN_BATCH_SIZE=64 MINI_BATCH_SIZE=64 N_SAMPLES=8 MAX_GEN_LEN=1024 TOPK=64 trainer.policy.optimizer_config.lr=3e-5"

# Launchable Batch-1 cells (existing knobs; teacher14b held pending Qwen3-14B fetch).
BATCH1=(base teacherTP4 topk16 topk32 topk128 gen512 n4 student_gmu085 teacher8b teacher_fp8 chunk_score)

declare -a WANT
if [[ "${1:-}" == "--batch1" ]]; then WANT=("${BATCH1[@]}")
else WANT=("$@"); fi
[[ ${#WANT[@]} -eq 0 ]] && { echo "no cells given"; exit 1; }

get_line() { grep -E "^\s*$1\s*\|" "$SPEC" | head -1; }

# Per-lane last jobid (for afterany chaining).
declare -a LANE_LAST
for ((l=0; l<LANES; l++)); do LANE_LAST[$l]=""; done

idx=0
for cell in "${WANT[@]}"; do
    line=$(get_line "$cell")
    if [[ -z "$line" ]]; then echo "!! no spec for cell '$cell' in $SPEC"; continue; fi
    IFS='|' read -r c steps overrides <<< "$line"
    c=$(echo "$c" | xargs); steps=$(echo "$steps" | xargs); overrides=$(echo "$overrides" | xargs)
    lane=$(( idx % LANES ))
    dep="${LANE_LAST[$lane]}"
    DEPARG=()
    if [[ -n "$dep" ]]; then DEPARG=(--dependency=afterany:"$dep"); fi
    echo ">> cell=$c lane=$lane dep=${dep:-none} MAX_STEPS=$steps overrides=[$overrides]"
    jid=$(sbatch --parsable "${DEPARG[@]}" --job-name="opdtp_${c}" --time="$WALL" "$SBATCH" \
            $BASE_KNOBS MAX_STEPS="$steps" $overrides)
    rc=$?
    if [[ $rc -ne 0 || -z "$jid" ]]; then echo "!! sbatch FAILED for $c (rc=$rc)"; continue; fi
    echo "   submitted jobid=$jid"
    echo "cell=$c jobid=$jid lane=$lane dep=${dep:-none} steps=$steps overrides=[$overrides] ts=$(date -u +%FT%TZ)" >> "$MANIFEST"
    LANE_LAST[$lane]="$jid"
    idx=$(( idx + 1 ))
done
echo "== manifest: $MANIFEST =="
tail -20 "$MANIFEST"
