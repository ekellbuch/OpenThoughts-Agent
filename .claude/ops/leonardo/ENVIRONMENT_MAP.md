# Leonardo runtime / environment map

**Purpose:** which conda env / venv / container to use for which workstream on **CINECA Leonardo**, and the
version facts that bite. Leonardo differs from Jupiter on every axis that matters for binaries:
**x86_64** (not aarch64), **A100-64GB** (not GH200), **CUDA 12.x / cu128 wheels** (not cu130), and
**SingularityPRO** (`/usr/bin/singularity`, **no `apptainer`, no podman**). So Jupiter SIFs/wheels do NOT
transfer — Leonardo builds its own. Access/preamble/paths/HF-upload live in `ops.md`; this is the runtime map.

Last verified: **2026-06-14** (from `notes/leonardo.md` + the MarinSkyRL canary). Re-confirm with §3 if acting on this months later.

---

## 0. TL;DR — the discriminators

> - **otagent conda** = the default runtime for orchestration, eval, datagen, AND agentic work — **torch 2.9.1+cu128, vLLM 0.16.0 (built from source), flash_attn 2.8.3+cu128torch2.9**.
> - **sft-qwen35 conda** = Qwen3.5 (9B/27B) SFT only — **torch 2.9.1+cu128, transformers ≥5.3, deepspeed ≥0.18**.
> - **MarinSkyRL uv venv inside a singularity SANDBOX dir** = SkyRL RL (validated non-agentic gsm8k GRPO) — **torch 2.8.0+cu128, vLLM 0.11.0, flash-attn 2.8.3** (a DIFFERENT, SkyRL-native stack from otagent).
> - Compute nodes have **no internet** → pre-download models on the login node; jobs use proxychains over an SSH SOCKS tunnel.

---

## 1. Decision table — which runtime for which workstream

| Workstream | Runtime | Stack |
|---|---|---|
| Orchestration, `hpc.launch`, eval listener, datagen, HF uploads, Supabase | **`otagent` conda** | torch 2.9.1+cu128, vLLM 0.16.0 (src), FA2 2.8.3 |
| Agentic eval / datagen (Harbor + Daytona over proxychains) | **`otagent` conda** | same |
| **SkyRL RL** (gsm8k GRPO canary; SkyRL-native examples) | **MarinSkyRL uv venv in a singularity sandbox dir** | torch 2.8.0+cu128, vLLM 0.11.0, FA 2.8.3 |
| **SFT (Qwen3.5 9B/27B)** | **`sft-qwen35` conda** | torch 2.9.1+cu128, transformers 5.3+, deepspeed 0.18+ |
| Compilers (for any from-source build) | **conda/mamba**, NOT system modules | gcc/gxx 14 + conda nvcc; `module load cuda/12.2` only for llama.cpp |

---

## 2. The runtimes in detail

### 2a. `otagent` conda — orchestration + eval + datagen + agentic (the default)
- **Path:** `/leonardo_work/AIFAC_5C0_290/bfeuer00/miniforge3/envs/otagent/` (activate via the `ops.md` preamble).
- **Stack:** **torch 2.9.1+cu128**, **vLLM 0.16.0 built FROM SOURCE** against that torch (the from-source vLLM build recipe + env scrubbing is in `notes/leonardo.md` "Install"), **flash_attn 2.8.3+cu128torch2.9** (prebuilt x86 wheel from mjun0812 release **v0.9.0**, NOT the aarch64 wheel Jupiter uses).
- **Use for:** everything that isn't SkyRL-RL or Qwen3.5-SFT — `hpc.launch`, the unified eval listener (`eval/leonardo/unified_eval_harbor.sbatch`, TP=4, 1 node, 24h), datagen, uploads. `proxychains` is provided by this env (compute-node internet).
- **Build gotchas (from-source vLLM):** set `CUDA_HOME=$CONDA_PREFIX`, `CUDACXX/CMAKE_CUDA_COMPILER=$CONDA_PREFIX/bin/nvcc`, `LD_LIBRARY_PATH=$CONDA_PREFIX/lib:…`; `mamba install cmake ninja`; `VLLM_TARGET_DEVICE=cuda … uv pip install vllm==0.16.0 --no-build-isolation`. `-DCMAKE_CUDA_FLAGS=--allow-unsupported-compiler` if the host compiler is rejected.

### 2b. `sft-qwen35` conda — Qwen3.5 (9B/27B) SFT
- **Create:** `conda create -n sft-qwen35 python=3.12 && pip install uv`; `uv pip install --index-url https://download.pytorch.org/whl/cu128 "torch==2.9.1" …`; then in `sft/llamafactory`: `uv pip install "transformers>=5.3.0" "accelerate>=1.12" "deepspeed>=0.18" "datasets>=4.4" "peft>=0.18" "trl>=0.29" liger-kernel hf-kernels`.
- **Why separate:** Qwen3.5's hybrid GatedDeltaNet+Attention arch needs **transformers ≥5.3**, which the default LLaMA-Factory/otagent stack (transformers 4.x) can't load. Use for any `sft/lf_configs/qwen3_5/*` run.
- **Leonardo-specific:** the launcher doesn't wire conda activation here the way it does on Jupiter — **hand-patch the sbatch** (conda activate + WORKDIR) per CLAUDE.md "SFT Launch on Leonardo". Cleanup follows the 8B path (Qwen3.5 writes root safetensors → skip consolidate; copy `preprocessor_config.json` from the base before upload). HF upload uses the `ops.md` sbatch-tunnel (or the login-node nohup fallback when the cert is expired).

### 2c. MarinSkyRL — uv venv inside a singularity SANDBOX (RL runtime)
First successful non-agentic SkyRL RL on Leonardo (2026-06-05): a full gsm8k GRPO epoch on
Qwen2.5-1.5B-Instruct, job 44478923 COMPLETED 0:0 (reward 0.14→0.64, pass@4 0.78, weight-sync working).
NO Harbor/Daytona/agentic yet (proxyserver deferred).
- **Repo:** `marin-community/MarinSkyRL` `penfever/working` at `/leonardo_work/AIFAC_5C0_290/bfeuer00/code/MarinSkyRL` (distinct from the removed old `/code/SkyRL`). Non-agentic recipe: `skyrl-train/examples/gsm8k/` via `python -m skyrl_train.entrypoints.main_base` (NOT main_tbench).
- **Image = a writable singularity SANDBOX DIR, NOT a `.sif`:** `singularity build --sandbox $SCRATCH_FAST/marinskyrl_sandbox docker://anyscale/ray:2.51.1-slim-py312-cu128`. **`mksquashfs` OOM-kills on the Lustre login node + hits fatal `lustre.lov` xattr errors → `.sif` packaging is deferred** (needs a non-login, xattr-clean build host). A sandbox dir execs fine via `singularity exec --nv`. Binary is `/usr/bin/singularity` (SingularityPRO 4.3.1).
- **Env = uv, not conda** (SkyRL-native): `uv 0.9.4` + `uv sync --extra vllm` against the committed `uv.lock` → venv with **torch 2.8.0+cu128, vLLM 0.11.0, flash-attn 2.8.3**. conda can't satisfy the pinned cu128/flashinfer/torch/vllm/flash-attn graph.
- **Gotchas:** Triton JIT needs a C compiler the ray base image lacks → bind the host miniforge **gcc 14.3.0** onto PATH + set `CC`/`CXX` inside the container; also `RAY_USAGE_STATS_ENABLED=0`. **`HOME`/`trainer.export_path` default to `${HOME}`** which is read-only `/leonardo` inside the container → point both at writable `$SCRATCH_FAST`.
- **Configs:** `hpc/skyrl_yaml/leonardo/{run_gsm8k_canary.sh, sbatch_gsm8k_canary.sh, note.txt}` (commit `44afed58`). sbatch = `--account=AIFAC_5C0_290 --partition=boost_usr_prod`, 1 node `--gres=gpu:a100:4` (debug QOS `boost_qos_dbg` ≤30min). Prestage on the LOGIN node (compute has no internet): model → `$WORK/data/hub`, gsm8k parquet → `$WORK/data/gsm8k`. `WANDB_MODE=offline`.

---

## 3. VERIFY before you trust

```bash
# otagent
/leonardo_work/AIFAC_5C0_290/bfeuer00/miniforge3/envs/otagent/bin/python -c "import torch,vllm; print('otagent', torch.__version__, vllm.__version__)"   # → 2.9.1+cu128 / 0.16.0
# sft-qwen35
/leonardo_work/AIFAC_5C0_290/bfeuer00/miniforge3/envs/sft-qwen35/bin/python -c "import torch,transformers; print('sft', torch.__version__, transformers.__version__)"  # → 2.9.1+cu128 / 5.3+
# SkyRL sandbox venv
singularity exec --nv $SCRATCH_FAST/marinskyrl_sandbox python -c "import torch,vllm; print('skyrl', torch.__version__, vllm.__version__)"   # → 2.8.0+cu128 / 0.11.0
```

---

## 4. Hardware / filesystems (the constraints that shape the above)

- **Booster nodes:** 32-core Ice Lake (x86_64) @ 2.6GHz, **4× A100-64GB**, 512 GB RAM, 3456 nodes. CUDA 12.2/12.3/12.6 modules; gcc 12.2 system / conda gcc 14.
- **Filesystems:** HOME `/leonardo/home/userexternal/bfeuer00` (50 GB, NFS, persistent) — too small for envs; **WORK** `/leonardo_work/AIFAC_5C0_290` (~1.465 PB shared GPFS, persistent — code/envs/HF-cache/experiments live here); **SCRATCH/fast** `/leonardo_scratch/fast/AIFAC_5C0_290` (1 TB shared Lustre, **auto-purged**, fastest — vLLM/Triton/FlashInfer caches + job tmp; often OVER quota, may need project cleanup). Env vars in `hpc/dotenv/leonardo.env`: `HF_HOME`/`HF_HUB_CACHE=$WORK/data/hub`, `VLLM_CONFIG_ROOT`/`TRITON_CACHE_DIR`/`FLASHINFER_WORKSPACE_BASE=$SCRATCH_FAST/vllm_cache`.
- **Account:** `AIFAC_5C0_290` (the expired `EUHPC_E03_068` / `CMPNS_E03_068` are dead — do NOT use). Partition `boost_usr_prod`, max wall 24h (`--time 23:59:00`), debug QOS `boost_qos_dbg` ≤30min. Budget: `saldo -b`; storage: `cinQuota`/`cindata`.
