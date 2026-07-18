# Jupiter runtime / environment map

**6 distinct runtimes** on Jupiter run SkyRL/vLLM code, across **two vLLM lines** (0.16-era vs 0.20.2rc0) and **two torch lines** (2.9 vs 2.11). `vllm.__version__` is **unreliable** (reports `dev`, `0.1.dev16577+g…`, or `0.20.2rc0` for what are different builds), so runtimes are routinely mis-identified. **`torch.__version__` is the reliable discriminator** (§0); verify with §4 before trusting.

Last verified: **2026-06-13** (versions probed live).

---

## 0. TL;DR — the discriminator

> - **torch 2.9.0+cu130 → OLD stack → vLLM 0.16-era** (RL venv, `*_r3baked.sif`)
> - **torch 2.11.0+cu130 → NEW stack → vLLM 0.20.2rc0** (otagent conda, `*vllm0202rc0_r3*.sif`)
>
> DCP (`decode_context_parallel_size`/`get_dcp_group`), torch-native CP (`context_parallel`), and anything needing torch≥2.10 ONLY exist in the **NEW stack**. If a parity/feature test "fails" on a torch-2.9 runtime, you tested the wrong vLLM.

---

## 1. Decision table — which runtime for which workstream

| Workstream | Runtime | vLLM / torch |
|---|---|---|
| **Local** Harbor/SkyRL code, launching, uploads, eval/datagen orchestration, count tooling, Supabase | **`otagent` conda (local Mac AND Jupiter)** | local: no GPU / Jupiter otagent: vLLM `0.1.dev…041cfa68e`, **torch 2.11** |
| **Standard dense RL** (8B/32B FSDP2: a3, seqnorm/TIS/shaped/symclip/lrboost/loopshape) | **RL venv** `$WORKDIR/envs/rl` | vLLM `dev` (**0.16-era**), **torch 2.9** |
| **MoE / Megatron RL** (Qwen3-Coder-30B-A3B; prod 80B Qwen3-Next-80B-A3B R3+TIS) | **SIF `skyrl_megatron_vllm_r3baked.sif`** | vLLM **0.16.0**, **torch 2.9** (NGC) |
| **NEW torch2.11 / vLLM-0.20.2rc0 work** (FSDP2-CP, DCP, Mixtral multi-node, anything needing torch≥2.10) | **SIF `skyrl_megatron_vllm0202rc0_r3.sif`** | vLLM **0.20.2rc0**, **torch 2.11** |
| Datagen / eval (Harbor + Daytona, no training) | **`otagent` conda** | torch 2.11 |
| **SFT (Qwen3.5 — 9B/27B)** | **`sft-qwen35` conda** (§2f) | **transformers 5.3.0 / torch 2.9.1+cu130** |
| Axolotl SFT (Sera/CoderForge) | `sera-axolotl` conda | torch 2.9.1+cu130 |
| Curator datagen | `curator` conda | — |

> **The launcher chooses venv vs SIF for RL.** `hpc/sbatch_rl/universal_rl.sbatch` resolves `RL_ENV_DIR="${RL_ENV_DIR:-$WORKDIR/envs/rl}"` and a container var (`RL_CONTAINER` / `RL_CONTAINER_OVERLAYS`). Dense FSDP2 → venv; Megatron/MoE/80B → SIF. **To know which a job used, read its rendered sbatch** (`experiments/<job>/sbatch/*.sbatch`) for `apptainer exec … <sif>` vs the venv python — don't assume.

---

## 2. The runtimes in detail

### 2a. `otagent` conda — orchestration + local dev
- **Local Mac:** `/Users/benjaminfeuer/miniconda3/envs/otagent/bin/python` (symlinks don't work in the sandbox — use the full path).
- **Jupiter:** `/e/scratch/jureap59/feuer1/miniforge3/envs/otagent/` (torch 2.11.0+cu130; vLLM `0.1.dev…041cfa68e` build).
- **Use for:** Harbor/SkyRL code, `hpc.launch`, HF uploads, eval/datagen listeners, `count_snapshots_from_tasks.py`, Supabase, trace upload + `parse_skyrl_metrics.py` (needs `google.cloud.storage` + matplotlib — the RL venv lacks these).
- **Load (Jupiter, non-interactive):** `/e/scratch/jureap59/feuer1/miniforge3/envs/otagent/bin/python …` (`$DCFT_ACTIVATE_ENV` doesn't work over non-interactive ssh).
- **Local-dev gotcha:** Harbor's `__init__.py` fails without the package installed; to test modules in isolation locally, mock the package first: `import sys,types; m=types.ModuleType('harbor'); m.__path__=['/Users/benjaminfeuer/Documents/harbor/src/harbor']; m.__version__='0.0.0'; sys.modules['harbor']=m; sys.path.insert(0,'/Users/benjaminfeuer/Documents/harbor/src')`. System python + conda base lack loguru/pydantic — always use the otagent env.
- **`flash_attn` (FA2) INSTALLED (2026-06-14):** `2.8.3+cu130torch2.11` (prebuilt wheel `flash_attn-2.8.3+cu130torch2.11-cp312-cp312-manylinux_2_34_aarch64.whl`, from mjun0812/flash-attention-prebuild-wheels release **v0.9.22**). Installed `--no-deps` so torch/vLLM untouched; exact match to otagent's torch 2.11.0+cu130 / cp312 / aarch64 / glibc 2.34. `import flash_attn; import flash_attn_2_cuda; from flash_attn import flash_attn_func` all clean → `attn: fa2` SFT configs no longer fall back to eager.

### 2b. RL venv `$WORKDIR/envs/rl` — standard dense RL training
- **Path:** `/e/scratch/jureap59/feuer1/OpenThoughts-Agent/envs/rl` (resolved via `RL_ENV_DIR` in `hpc/sbatch_rl/universal_rl.sbatch`).
- **Versions:** vLLM `dev` (**0.16-era**), **torch 2.9.0+cu130**.
- **Use for:** the 8B/32B FSDP2 RL ablations (a3, seqnorm/TIS/shaped/symclip/lrboost/loopshape). Default RL rollout+train runtime — DCP/CP do NOT live here.
- **Load:** `$WORKDIR/envs/rl/bin/python …` (venv, NOT conda — `conda activate` is wrong; see memory `reference_jupiter_rl_venv_path`).

### 2c. `skyrl_megatron_vllm_r3baked.sif` — MoE + prod 80B RL (OLD stack)
- **Path:** `/e/scratch/jureap59/feuer1/containers/skyrl_megatron_vllm_r3baked.sif` (9.7 GB, 2026-06-08).
- **Versions:** vLLM **0.16.0**, **torch 2.9** (NGC base), transformers 5.10.1. **Overlays baked in** (`vllm_http_overlay` + `fla_tilelang_overlay`) — no `--overlay` needed (see memory `reference_80b_baked_sif_overlay_fix`).
- **Use for:** Qwen3-Coder-30B-A3B MoE RL, and the prod 80B Qwen3-Next-80B-A3B R3+TIS run (validated job 662928).
- **Load:** `apptainer exec --nv <sif> python …` (the `--nv` + baked `LD_LIBRARY_PATH`/`TRITON_LIBCUDA_PATH` are required or TransformerEngine import fails on `libcuda.so` — see `reference_skyrl_megatron_container`).
- **Predecessor SIFs `skyrl_megatron_vllm.sif` and `skyrl_megatron.sif` were DELETED 2026-06-13** — don't look for them; their active references were stale comments only.

### 2d. `skyrl_megatron_vllm0202rc0_r3.sif` — NEW torch2.11 / vLLM 0.20.2rc0 ⭐
- **Path:** `/e/scratch/jureap59/feuer1/containers/skyrl_megatron_vllm0202rc0_r3.sif` (11.6 GB, 2026-06-12). **NEWEST base SIF.**
- **Versions (verified 2026-06-13):** vLLM **0.20.2rc0**, **torch 2.11.0+cu130**. `decode_context_parallel_size` field present; `get_dcp_group` imports. Source = `mlfoundations/vllm` `v0.20.2rc0-306-g3e3a1c45d` (local tree `/Users/benjaminfeuer/Documents/vllm`, branch `v2-migration`).
- **Use for:** FSDP2 torch-native **CP** (Stage 1+ pins torch≥2.10 here), vLLM **DCP**, Mixtral-8x7B multi-node. Any DCP/CP/torch≥2.10 test MUST use this — NOT the venv or r3baked SIF.
- **Load:** `apptainer exec --nv <sif> python …`. vLLM lives at `/opt/vllm_build/vllm/` (NOT `dist-packages/vllm`), which shadows R3 single-file binds — remove the binds when R3 is OFF. Multi-node here surfaces 3 torch-2.11 fixes the OLD SIF doesn't need (vLLM-bind-removal when R3 off, `NCCL_P2P_DISABLE=1`, `pg_options→backend_options` — see `reference_new_sif_torch211_multinode_fixes`).
- **GOTCHA — set `VLLM_USE_FLASHINFER_SAMPLER=0` for any direct `vllm.LLM`/engine call.** No `flashinfer` shipped; 0.20.2rc0 defaults the flag True and unconditionally imports `FlashInferBackend` → `ModuleNotFoundError: No module named 'flashinfer'` before its graceful fallback. The PyTorch/Triton sampler is irrelevant to greedy parity.
- **GOTCHA — multi-GPU (tp>1) spawned workers fail Triton linking** (`ld: cannot find -l:libcuda.so.1` on GH200/aarch64). Fix: `export LIBRARY_PATH=/.singularity.d/libs:${LIBRARY_PATH:-}`.
- **`VLLM_ATTENTION_BACKEND` is a dead env var on 0.20.2rc0** (logs "Unknown vLLM environment variable"); the engine auto-selects FlashAttention-3. Don't rely on it to pin a backend.

### 2e. CP-variant SIF chain — `skyrl_megatron_vllm0202rc0_r3_cp*.sif` (#232 FSDP2-CP) ⭐
Built off §2d (torch 2.11 / vLLM 0.20.2rc0) for the #232 FSDP2 context-parallel (ring-SDPA) + R3 work. Path prefix `/e/scratch/jureap59/feuer1/containers/`. Load like §2d (same flashinfer/libcuda/attention-backend gotchas apply).
- **`skyrl_megatron_vllm0202rc0_r3_cp_fixb3.sif`** (11.6 GB, 2026-06-19) — ⭐ **CANONICAL CP+R3 SIF.** `.vllm_commit = 4d167a4af` (`penfever/working` with merged **#237** rank-symmetric R3-capture fix baked in). **Use for ALL new CP and/or R3 runs** (#232 cp2 / cp2_r3 rungs point here).
- Non-CP R3 variants (`_r3_fixb.sif` / `_r3_v2migration.sif`) are NOT rebuilt with the #237 fix as of 2026-06-19 — separate rebake needed if a non-CP R3 run must carry it.
- **KNOWN CP BUG (SkyRL-side, 2026-06-19 — NOT the SIF):** Qwen3-**MoE** crashes in CP≥2 policy forward (`model_wrapper.py:668` passes a dict attention_mask MoE's `create_causal_mask` can't consume → `AttributeError: 'dict' has no attribute 'ndim'`). Dense-Qwen3 & CP1 unaffected. Fix is in the SkyRL host clone (`OpenThoughts-Agent/SkyRL/skyrl-train/skyrl_train/model_wrapper.py`) — editable install, no SIF rebuild. See `agent_logs/2026-06-19_cp2_forward_dict_ndim_bug.md`.

### 2f. `sft-qwen35` conda — Qwen3.5 SFT (and other SFT/datagen condas)
- **Path (Jupiter):** `/e/scratch/jureap59/feuer1/miniforge3/envs/sft-qwen35/`
- **Versions (Jupiter-verified 2026-06-13):** **torch 2.9.1+cu130, transformers 5.3.0** (deepspeed `zero3_bf16` for sharding).
- **Why separate:** Qwen3.5 (9B/27B) uses a hybrid GatedDeltaNet + Attention architecture that needs **transformers ≥ 5.3** — the default LLaMA-Factory/`otagent` SFT stack (transformers 4.x) cannot load or train it. Use for any `sft/lf_configs/qwen3_5/*` run.
- **Load:** `/e/scratch/jureap59/feuer1/miniforge3/envs/sft-qwen35/bin/python …` (full path over non-interactive ssh). `hpc.launch --train_config_path sft/lf_configs/qwen3_5/...` wires conda activation on Jupiter; on **Leonardo** hand-patch the sbatch (conda activate + WORKDIR) per CLAUDE.md "SFT Launch on Leonardo".
- **Launch + cleanup specifics:**
  - Harbor role/content tags: `--role_tag role --user_tag user --assistant_tag assistant --content_tag content` (or the thinking preprocessor finds 0 assistant messages → garbage).
  - Qwen3.5 writes **full safetensors at the checkpoint root on completion → SKIP the `consolidate` step** (memory `feedback_qwen35_9b_no_consolidate`); follow the **8B SFT cleanup checklist**.
  - Before HF upload, copy `preprocessor_config.json` from the base model into the checkpoint (LLaMA-Factory doesn't emit it; vLLM needs it).
  - HF-only / DB-register per the SFT checklist; uploads default PUBLIC to `laion/`.

**Other condas on Jupiter:**
- `sera-axolotl` — Sera/CoderForge axolotl SFT (torch 2.9.1+cu130; see CLAUDE.md "Axolotl SFT on Jupiter" for install recipe + mandatory env patches).
- `curator` — Curator sharded datagen (`run_curator_datagen_sharded.sbatch`, `--account=reformo`).

---

## 3. Overlay images (`containers/*.img`)

Overlays stack onto a SIF at `apptainer exec --overlay <img>` (or are baked in). On GPFS they can FUSE-mount-timeout on the Ray head — the prod 80B SIF bakes them in to avoid that.

| Overlay | Size | Provides | Status |
|---|---|---|---|
| `vllm_http_overlay.img` | 0.5 G | vLLM HTTP `routed_experts` serialization (R3 routing capture) | baked into both `*r3*.sif` |
| `fla_tilelang_overlay.img` | 4 G | tilelang 0.1.8 + FlashQLA fused GatedDeltaNet kernels (FSDP2-EP Stage 8) | baked into `*r3*.sif` |
| `skyrl_titan_overlay.img` | 4 G (2026-06-13) | torchtitan (CP **+EP** / MoE expert-parallel — CP Stage-6 TEST3) | overlay (stack when CP+EP) |

**R3 routing-capture works on stock-0.16 (no SIF rebuild):** `vllm_http_overlay` serializes `routed_experts` over `/chat/completions`; the only blocker was `enable_return_routed_experts=False` (now true). For **Qwen3-Next**, the vLLM **Ray Compiled-DAG** backend deadlocks on the hybrid arch when capture is on → run with **mp executor backend** (`generator.inference_engine_mp_backend: true`, ran clean 12/12 rounds TP=4), plus the hybrid-kv-buffer fix + defensive clip (`gmr_fix`/`scheduler_fix`/`capturer_fix` single-file binds; SkyRL flag on branch `r3-mp-backend-qwen3next-20260608`). The FSDP2 router-replay hook EXISTS and ran a full GRPO backprop step on the 80B (do NOT repeat the "Megatron-only / no FSDP2 replay" claim).

---

## 4. VERIFY before you trust — copy-paste probes

**SIF:**
```bash
C=/e/scratch/jureap59/feuer1/containers
apptainer exec --nv $C/<sif>.sif python -c "import vllm,torch; print('vllm',vllm.__version__,'torch',torch.__version__); \
from vllm.engine.arg_utils import EngineArgs; print('DCP field', 'decode_context_parallel_size' in EngineArgs.__dataclass_fields__)"
```
**venv / conda:**
```bash
/e/scratch/jureap59/feuer1/OpenThoughts-Agent/envs/rl/bin/python -c "import vllm,torch; print(vllm.__version__, torch.__version__)"   # → dev 2.9.0  (OLD)
/e/scratch/jureap59/feuer1/miniforge3/envs/otagent/bin/python   -c "import torch; print(torch.__version__)"                          # → 2.11.0     (NEW)
```
torch 2.9 ⇒ 0.16-era (no DCP/CP); torch 2.11 ⇒ 0.20.2rc0 (has DCP/CP). If a feature requiring torch≥2.10 "isn't there" or "fails parity," confirm you're on a torch-2.11 runtime before concluding it's a real defect — and pin the parity/feature smoke to the SIF that carries the feature (§2d/§2e).
