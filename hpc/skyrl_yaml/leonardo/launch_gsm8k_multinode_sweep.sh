#!/bin/bash
# Launch the gsm8k multi-node generator-scaling sweep on Leonardo.
# Holds the per-step batch FIXED (combo_D: tbs256 x n8 = 2048 seqs) and scales
# only the number of inference engines via NODES (4 engines/node, TP1):
#   nd2 -> 2 nodes / 8 GPUs / 8 engines
#   nd4 -> 4 nodes / 16 GPUs / 16 engines
#   nd8 -> 8 nodes / 32 GPUs / 32 engines
# (nd1 baseline reuses combo_D from the 1-node grid; not re-run here.)
#
# Usage (on Leonardo login node):
#   bash launch_gsm8k_multinode_sweep.sh 2 4 8     # launch nd2, nd4, nd8
set -euo pipefail
CFG=/leonardo_work/AIFAC_5C0_290/bfeuer00/code/OpenThoughts-Agent/hpc/skyrl_yaml/leonardo
SBATCH="$CFG/sbatch_gsm8k_grid_multinode.sh"

# combo_D fixed batch + engine knobs (identical across cells).
COMBO_D="trainer.train_batch_size=256 trainer.policy_mini_batch_size=256 \
generator.n_samples_per_prompt=8 generator.gpu_memory_utilization=0.85 \
generator.enforce_eager=false trainer.micro_train_batch_size_per_gpu=16 \
trainer.micro_forward_batch_size_per_gpu=16 trainer.max_steps=25 \
generator.inference_engine_tensor_parallel_size=1 \
trainer.placement.colocate_all=true \
trainer.placement.policy_num_gpus_per_node=4 \
trainer.placement.ref_num_gpus_per_node=4"

for N in "$@"; do
    ENGINES=$((4 * N))
    JOBNAME="grid_mn_nd${N}"
    echo "Submitting $JOBNAME: nodes=$N engines=$ENGINES"
    sbatch --nodes="$N" --job-name="$JOBNAME" "$SBATCH" \
        $COMBO_D \
        generator.num_inference_engines="$ENGINES" \
        trainer.placement.policy_num_nodes="$N" \
        trainer.placement.ref_num_nodes="$N"
done
