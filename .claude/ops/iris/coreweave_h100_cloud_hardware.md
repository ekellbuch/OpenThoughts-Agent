# CoreWeave H100 node configurations (`cw-us-east-02a`)

Hardware datasheet for the CoreWeave GPU cluster reached via iris (the x86/H100 analogue of
`iris_google_tpu_cloud_hardware.md`). **Ops/access/scheduling live in `coreweave_gpu_ops.md`** — this
file is the chip/node hardware reference. Cluster-observed facts are marked `[obs]`; the rest is the NVIDIA H100-SXM5 datasheet.

## H100x8 (the only node shape here)

`cw-us-east-02a` is homogeneous: **one node shape, whole-node-exclusive** (`H100x8`, one iris
task per node, no co-tenants). `[obs]`

Node composition `[obs]`:
- **8× NVIDIA H100-80GB SXM5** per node, **x86_64** host (NOT aarch64/GH200).
- **~128 CPU cores/node**; **~64–68 are system/daemonset overhead** → ~48–60 free (why
  `--cpu 48` admits a multi-node gang and `--cpu 64` does not).
- **~2 TB host DRAM/node** allocatable.
- **NVLink (NVSwitch) intra-node + InfiniBand inter-node.** A TP=8 vLLM engine places
  intra-node on one 8-GPU node (no cross-node TP).
- **~36 H100 nodes total** `[obs]` → ceiling ~4 simultaneous 8-node gangs (minus other tenants).

Per-chip **H100-80GB SXM5** spec (NVIDIA datasheet; dense regime — we don't run 2:4 sparsity, so
sparsity-doubled figures don't apply):
- **80 GiB HBM3, 3.35 TB/s** bandwidth
- **bf16 / fp16 tensor: ~989 TFLOPs/s** dense (1979 w/ sparsity)
- **FP8 tensor: ~1979 TFLOPs/s** dense (3958 w/ sparsity) — native FP8 (Transformer Engine)
- **int8 tensor: ~1979 TOPS** dense
- **TF32 tensor: ~495 TFLOPs/s** dense · **FP64 tensor: ~67 TFLOPs/s**
- **NVLink: 900 GB/s** per GPU (4th-gen, full all-to-all via NVSwitch on-node)
- 700 W TDP · **sm_90** (Hopper → `TORCH_CUDA_ARCH_LIST="9.0"` for from-source builds)

H100x8 node totals:

┌───────────────┬─────────────┬──────────────────────────────┐
│    metric     │  per GPU    │          × 8 GPUs            │
├───────────────┼─────────────┼──────────────────────────────┤
│ HBM           │ 80 GiB      │ 640 GiB                      │
├───────────────┼─────────────┼──────────────────────────────┤
│ HBM bandwidth │ 3.35 TB/s   │ 26.8 TB/s aggregate          │
├───────────────┼─────────────┼──────────────────────────────┤
│ bf16 FLOPs/s  │ ~989 TFLOPS │ ~7.9 PFLOPs/s (dense)        │
├───────────────┼─────────────┼──────────────────────────────┤
│ FP8 FLOPs/s   │ ~1979 TFLOPS│ ~15.8 PFLOPs/s (dense)       │
├───────────────┼─────────────┼──────────────────────────────┤
│ int8 OPs/s    │ ~1979 TOPS  │ ~15.8 POPS (dense)           │
├───────────────┼─────────────┼──────────────────────────────┤
│ NVLink BW     │ 900 GB/s    │ on-node all-to-all (NVSwitch)│
├───────────────┼─────────────┼──────────────────────────────┤
│ host CPU      │ —           │ ~128 cores (~48–60 free)     │
├───────────────┼─────────────┼──────────────────────────────┤
│ host DRAM     │ —           │ ~2 TB/node                   │
└───────────────┴─────────────┴──────────────────────────────┘

## Interconnect (what shapes the parallelism)

- **Intra-node = NVLink/NVSwitch, 900 GB/s/GPU, full all-to-all.** TP=8 (and TP=8 + DCP=2)
  belongs ON ONE NODE: decode/EP all-reduce + all-to-all ride NVLink, not the slower fabric.
  **Use NCCL defaults** — the GH200/SIF disables (`NCCL_P2P_DISABLE` / `NVLS=0` / `COLLNET=0`)
  would cripple this on-node path.
- **Inter-node = InfiniBand**, gang-scheduled in one IB leaf fabric (Kueue `topology 'infiniband'`,
  all-or-nothing — see `coreweave_gpu_ops.md`). Typical CoreWeave H100 config is **8× 400 Gb/s
  NDR** (one NIC/GPU, GPUDirect RDMA) ≈ 3.2 Tb/s/node — *datasheet-typical, not re-measured on
  `cw-us-east-02a`.* Cross-node collectives (FSDP all-gather/reduce-scatter on the policy mesh,
  inter-engine) go over IB → keep TP intra-node, shard the slower axes (FSDP/CP) across nodes.

## How this informs RL geometry (the practical upshot)

- **640 GiB HBM/node** hosts a TP=8 vLLM engine for a 30B–35B-class MoE at long context: the
  131k MoE arm runs **4 engines × TP=8 / DCP=2** (32 inference GPU = 4 nodes) — each engine one
  node, KV + weights under 640 GiB at `gpu_memory_utilization 0.80`. (Contrast Jupiter's
  4-GPU/96 GiB GH200 nodes, which forced DCP=1 and made TP=8 unplaceable.)
- **H100-80GB < GH200-96GB HBM** → porting a GH200 config, the per-GPU budget tightens: drop
  `gpu_memory_utilization` (0.80→0.75) / `max_num_seqs` first on a KV-bind OOM (config-authoring
  detail in the launch skill §4).
- **~2 TB host DRAM** is generous for `cpu_offload`, but host-RAM OOM at FSDP weight-load on the
  policy nodes is still possible at 131k + EP8 + offload (observed on a 30B run); reduce
  `n_concurrent` / the rollout-worker count if it recurs.
- **sm_90** everywhere → from-source builds (vLLM fork, flash-attn) target
  `TORCH_CUDA_ARCH_LIST="9.0"` (baked in the gpu-rl image; see `build-gpu-rl-image-iris`).

## Cross-reference
- **Access / scheduling / KUBECONFIG / build / monitoring** → `coreweave_gpu_ops.md`.
- **Launch procedure + config-authoring (geometry, NCCL, extra_env)** → the `rl-agentic-launch-iris` skill.
- **TPU (Google) node shapes** → `iris_google_tpu_cloud_hardware.md` (a DIFFERENT cluster on the same iris SDK).
