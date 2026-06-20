# 2026-06-20 — 131k CP+DCP+R3 config: placement-group fix (smoke 934104)

## Failed run
- Job **934104**, `16node_qwen3_30b_a3b_131k_cp_dcp_r3_SMOKE.yaml`, 16 nodes (jpbo-117-[01-16]).
- FAILED at Ray placement-group formation, BEFORE any CP/DCP/R3 code ran.
- Log: `/e/data1/datasets/playground/ot-baf/rl_30b_a3b_131k_cpdcpr3_SMOKE/logs/rl_30b_a3b_131k_cpdcpr3_SMOKE_934104.out`

```
RuntimeError: Failed to create placement group with 8 bundles (requiring 8.0 GPUs, 8.0 CPUs total) in 180 seconds.
(autoscaler) Error: No available node types can fulfill resource request {'GPU': 8.0, 'CPU': 8.0}
  ...create_ray_wrapped_inference_engines... ray_wrapped_inference_engine.py:334
```

## Root cause (REFINED — NOT the supervisor's ref-budget hypothesis)

The supervisor's hypothesis was "policy(8 nodes/32 GPU) + ref(8 nodes/32 GPU) = all 64 GPU,
0 left for inference -> 96 > 64 oversubscription." **That mechanism is wrong: the ref model is
INERT.** `use_ref_model = use_kl_loss or use_kl_in_reward` (skyrl_train/utils/utils.py:372,
main_base eligibility :39). This config has `use_kl_loss=false`, so **no RefWorker is built and
`ref_num_nodes=8` allocates 0 GPUs**. The actual GPU budget is policy 32 + ref 0 + inference 32
= 64 = 16 nodes, which FITS. The log confirms: `get_policy_pg` reserved "8 node(s) x 4 GPU"
(32 GPU, PACK, per-GPU bundles) and SUCCEEDED, claiming 8 of 16 nodes; the failure was the
*next* step — the inference-engine PG.

**The real bug is intra-engine node-atomicity, not total budget.** With
`inference_engine_mp_backend=false` (RAY executor, mandatory for the R3+DCP path) and TP=8 > 1,
`create_ray_wrapped_inference_engines` takes the `use_per_engine_strict_pack` branch
(ray_wrapped_inference_engine.py:322): it creates one **STRICT_PACK** placement group per engine,
each demanding `per_engine_gpu_count = TP = 8` × {GPU:1} bundles that MUST co-locate on **one
node**. Jupiter nodes have only **4 GH200 each** (.claude/ops/jupiter/ops.md:6), so an 8-GPU
STRICT_PACK PG is UNPLACEABLE -> the exact `{'GPU': 8.0}` request that no node type can fulfill.

This is a HARD geometric impossibility, independent of node count:
- DCP ceiling: `dcp <= tp // num_kv_heads` (utils.py:909, vLLM init). Model has **4 KV heads**,
  so **dcp=2 REQUIRES tp >= 8**.
- A TP=8 engine needs 8 GPUs on one node. RAY executor -> 8×{GPU:1} STRICT_PACK on one node
  (impossible on 4-GPU nodes). mp executor -> one atomic {GPU:8} bundle per engine (equally
  impossible on a 4-GPU node).
- Therefore **DCP=2 / TP=8 cannot be placed on Jupiter for this 4-KV-head model at ANY node
  count.** No budget/node-count tweak fixes it.

This is exactly the conclusion the 64GPU 131k parent
(`64GPU_qwen3_30b_a3b_longctx131k_cp_dcp.yaml`) reached on 2026-06-17 (USER-APPROVED): "Option A"
(4 engines × TP8, DCP=2) died on cross-node-TP decode; "Option B" (8 engines × TP4, DCP=1) is the
proven-placeable geometry. The new r3 config had silently reverted to the dead Option-A geometry.

## Fix — adopt Option-B geometry (mirror the validated 64GPU parent)

| knob | before (dead) | after (fixed) |
|------|---------------|---------------|
| `inference_engine_tensor_parallel_size` | 8 | **4** |
| `num_inference_engines` | 4 | **8** |
| `inference_engine_decode_context_parallel_size` | 2 | **1** |

Everything else UNCHANGED: policy mesh EP8×FSDP2×CP2=32 GPU/8 nodes, `mp_backend=false` (RAY),
`async_scheduling=false`, R3 ON, 131k, CP=2, SIF, extra_env, batch sizes, node count = 16.

Applied to BOTH `16node_..._131k_cp_dcp_r3.yaml` and `..._SMOKE.yaml` (+ header rewrites).
**Filename node count unchanged (still 16 nodes) -> no rename.**

## Mesh arithmetic (post-fix)
- Policy: EP=8 × FSDP=2 × CP=2 = 32 GPU = 8 nodes.
- Ref: 0 GPU (inert — use_kl_loss=false).
- Inference: 8 engines × TP=4 (DCP=1) = 32 GPU = 8 nodes. Each TP=4 engine = exactly one
  4-GPU node -> per-engine STRICT_PACK places cleanly, on-node NVLink decode all-reduce.
- **Total: 32 + 0 + 32 = 64 GPU = 16 nodes. <= 64. FITS.**

## Divisibility re-checks (all hold)
- MoE dim-0 guard (128 experts // EP8) % FSDP2 = 16 % 2 = 8 even -> VALID.
- EP×FSDP×CP = 8×2×2 = 32 = policy GPU -> VALID.
- CP G4: 131072 % (2·2=4) = 0 -> OK.
- vLLM TP=4 divides 32 attn heads (8/GPU), 4 KV heads (1/GPU, valid GQA), 128 experts (32/GPU).
- DCP ceiling: dcp=1 <= tp//num_kv_heads = 4//4 = 1 -> LEGAL.
- Batch (unchanged, fixed in b5ce5812/53e6e6bc): policy_dp_size=32; SMOKE train==mini==32,
  32>=32, 32%32==0, mini_per_gpu = 32·2//32 = 2 (>0, %micro_train(1)==0). Production
  train==mini==64, 64>=32, 64%64==0. ref inert so lcm_dp_size=policy_dp_size=32.

## Axes status
- 131k context: KEPT (max_model_len 131072, no YaRN — Coder-Instruct native 262144).
- CP=2 ring-SDPA: KEPT (policy/ref fsdp_config unchanged).
- R3 routed-experts capture: KEPT (enable_return_routed_experts=true, moe_router_replay=true).
- DCP: RESHAPED 2 -> 1. **DCP=2 is geometrically impossible on Jupiter's 4-GPU nodes for a
  4-KV-head model** (needs an 8-GPU node). DCP=2 KV-sharding can only be validated on a
  >=8-GPU-node cluster or a <=2-KV-head model. Flagged to supervisor as the one unresolvable knob.

## Relaunch (supervisor — after push + `git pull` on Jupiter)
```
python -m hpc.launch --job_type rl \
  --rl_config ./hpc/skyrl_yaml/jupiter/extra/16node_qwen3_30b_a3b_131k_cp_dcp_r3_SMOKE.yaml \
  --model_path Qwen/Qwen3-Coder-30B-A3B-Instruct \
  --train_data '["DCAgent/exp_rpt_pymethods2test-large"]' \
  --num_nodes 16 --reservation reformo --time_limit 02:00:00 \
  --job_name rl_30b_a3b_131k_cpdcpr3_SMOKE
```
