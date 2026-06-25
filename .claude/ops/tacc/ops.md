# TACC Vista Access

**SSH**: `ssh TACCVista` (alias in `~/.ssh/config`). Direct: `ssh penfever@login2.vista.tacc.utexas.edu`.
User `penfever`, account `CCR24067`. Login requires password (in 1Password) + TOTP (authenticator app).

**Cluster**: TACC Vista — **Grace Hopper GH200 96GB** GPUs, **aarch64** (ARM64), SLURM scheduler.
Vista user docs: <https://docs.tacc.utexas.edu/hpc/vista/>.
Common pitfalls: [TACC Vista Google Doc](https://docs.google.com/document/d/1URcWe8mLQF8HMNre7vZwgk6TiRNxFxVBwK_HCcRXQdM/edit).

**Pre-launch preamble** (run before launching any job):
```bash
source $SCRATCH/OpenThoughts-Agent/hpc/dotenv/tacc.env && \
source $SCRATCH/keys.env && \
cd $SCRATCH/OpenThoughts-Agent && git pull && \
git submodule update --init --remote sft/llamafactory && \
cd $SCRATCH/harbor && git pull && \
cd $SCRATCH/OpenThoughts-Agent && \
source $SCRATCH/miniconda3/bin/activate otagent
```

**Key paths**:
- Code (`$DCFT`): `$SCRATCH/OpenThoughts-Agent` = `/scratch/10635/penfever/OpenThoughts-Agent`
- Harbor: `$SCRATCH/harbor` = `/scratch/10635/penfever/harbor`
- Conda: `$SCRATCH/miniconda3` = `/scratch/10635/penfever/miniconda3`
- Secrets (`$DC_AGENT_SECRET_ENV`): `$SCRATCH/keys.env` = `/scratch/10635/penfever/keys.env`
- HF cache (`$HF_HUB_CACHE`): `$SCRATCH/hub` = `/scratch/10635/penfever/hub`
- Checkpoints (`$CHECKPOINTS_DIR`): `$SCRATCH/checkpoints`
- Tokenized datasets: `$SCRATCH/tokenized_datasets`
- Evalchemy (standard / pass@k evals): `$SCRATCH/evalchemy` — conda env `evalchemy`
- DCFT shared (read-only): `/scratch/08002/gsmyrnis/dcft_shared/`
- Negin/Richard eval traces: `/scratch/08134/negin/dc-agent-shared/dc-agent/eval/tacc/jobs/`

> **⚠ Allocation**: `CCR24067` — **107,765 SUs** available, **expires 2025-12-31**. Monitor periodically
> (SUs are consumed per GPU-hour). Check balance: `bbalance`.
>
> **⚠ Use `gg` for builds/installs.** The login node is shared and under heavy load — `uv` and other
> Rust-based tools crash with OOM. Reserve a compute node via `srun -p gg` for any install or build.

**Disk quotas** (as of 2026-06):
| Disk | Usage | Limit | Notes |
|---|---|---|---|
| `/scratch` | ~3,780 GB | unlimited (no quota) | 1,248,064 inodes — primary code/data/envs |
| `/home1` | 1.9 GB | 23.3 GB | login dotfiles only — do NOT write large data here |
| `/work` | 0 GB | 1,024 GB | project work — currently unused |

## Partitions & QOS

| Partition | QOS | MaxWall | MaxJobs (running) | MaxSubmit (total) | Use |
|---|---|---|---|---|---|
| `gh` | `qgh` | 48:00:00 | 20 | 40 | Grace Hopper production |
| `gg` | `qgg` | 48:00:00 | 20 | 40 | Grace Grace production |
| `gh-dev` | `qdevelopment` | **02:00:00** | **1** | **3** | Interactive dev (2h cap!) |
| — | `qnormal` | 48:00:00 | 20 | 40 | General |

> **Use `gg` or `gh` for production.** `gh-dev` is development-only with a **2-hour wall cap** and
> **1 running job** — requesting `>120 min` on `gh-dev` triggers `QOSMaxWallDurationPerJobLimit`
> even though the partition shows `MaxTime=UNLIMITED`.

**Interactive session** (for debugging / manual runs):
```bash
idev -A CCR24067 -p gg -m 1400     # 23h20m on gg
idev -A CCR24067 -p gh-dev -m 120  # 2h on gh-dev (max allowed)
# or srun:
srun -p gh -N 1 -n 1 -t 12:00:00 --pty bash -l
```

**SLURM submit**: standard `sbatch`. Watch your jobs: `watch squeue -u penfever`.

## Environment setup

> **⚠ uv crashes on the login node** with a Rust memory allocation failure
> (`memory allocation of 1520 bytes failed`). All `uv pip install` / build-heavy
> operations MUST run on a **compute node** via `srun` (see Partitions & QOS).
> The login node is fine for `git pull`, `squeue`, and lightweight commands.

### otagent env (primary — datagen / eval / RL / SFT)

**Current state** (as of 2026-06-25):
- Python 3.12.13
- torch 2.11.0+cu128, torchvision 0.26.0+cu128, torchaudio 2.11.0+cu128
- vllm 0.16.0
- transformers 4.57.3
- harbor 0.8.0 (editable, `penfever/working` branch)
- flashinfer-python 0.6.3
- **SFT extras**: deepspeed 0.18.0, liger-kernel 0.8.0, peft 0.19.1, trl 1.6.0,
  llamafactory 0.9.4.dev0 (editable), gradio 6.17.3, torchao 0.17.0
- **No flash_attn** — using SDPA attention instead (no prebuilt aarch64 wheel exists
  for torch 2.11+cu128; see flash_attn note below)

**Fresh setup** (run each `uv` step on a compute node — see `srun` recipe below):
```bash
# 1. Create env + harbor (on compute node)
conda create -y -n otagent python=3.12 && conda activate otagent
cd $SCRATCH/harbor && pip install uv && uv pip install -e .

# 2. OT-Agent datagen extras (on compute node)
cd $SCRATCH/OpenThoughts-Agent
git submodule update --init --remote sft/llamafactory
uv pip install -e ".[datagen]"

# 3. SFT extras (on compute node)
cd sft/llamafactory
git checkout penfever/working_branch   # has relaxed version ceilings (see below)
uv pip install -e ".[liger-kernel,deepspeed,hf-kernels]"

# 4. Env vars
source $SCRATCH/keys.env && source ./hpc/dotenv/tacc.env
```

**`srun` build wrapper** (use this pattern for any install):
```bash
# Write a script, then srun it on a compute node
ssh TACCVista 'cat > $SCRATCH/build.sh << '\''EOF'\''
#!/bin/bash
source $SCRATCH/miniconda3/etc/profile.d/conda.sh && conda activate otagent
cd $SCRATCH/OpenThoughts-Agent
# ... your uv pip install commands here ...
EOF
chmod +x $SCRATCH/build.sh'

ssh TACCVista "srun -p gg -N 1 -n 1 -t 01:00:00 --account=CCR24067 bash /scratch/10635/penfever/build.sh"
```

### LLaMA-Factory version ceilings (relaxed)

The upstream `requirements.txt` pins hard ceilings that downgrade core packages
(datasets<=4.4.1, safetensors<=0.5.3, pydantic<=2.11.9, gradio<=5.45.0).
Branch **`penfever/working_branch`** on `mlfoundations/llama-factory` removes
these ceilings so the SFT install does NOT clobber the datagen env:
- `datasets>=2.16.0` (was `<=4.4.1`)
- `safetensors` (was `<=0.5.3`)
- `pydantic` (was `<=2.11.9`)
- `gradio>=4.38.0` (was `<=5.45.0`)

Always `git checkout penfever/working_branch` in the submodule before installing.

### flash_attn on aarch64 — status

**No prebuilt wheel exists for torch 2.11+cu128 on aarch64.** Checked
`mjun0812/flash-attention-prebuild-wheels` (2026-06-25):
- v0.9.41: torch 2.12, cu126/130/132 — no cu128
- v0.9.40: torch 2.10/2.9, cu126/130 — no cu128
- v0.9.39: torch 2.9, cu126 — no cu128
- v0.6.4 (old): `flash_attn-2.8.3+cu128torch2.9-cp312` — requires torch 2.9.0 downgrade

**Decision: use SDPA** (PyTorch native scaled dot-product attention). No torch
downgrade needed. If flash_attn becomes necessary later, downgrade torch to 2.9.0
and install the v0.6.4 wheel.

### Building the vLLM wheel from source (otagent env, aarch64 + Hopper sm_90)

**Proven recipe (2026-06-25, job 787093, node i614-011, fork `mlfoundations/vllm` @ `76259c63a`).**

> **⚠ Builds MUST run in a CPU SLURM allocation, NEVER on the login node.** The vLLM CUDA
> compile is `nvcc` host-side work that does NOT need a GPU to BUILD — request a **Grace-Grace
> CPU node (`-p gg`)**, not a GPU node. The login node is shared/under load and `uv`/Rust tooling
> OOMs there.
>
> **`salloc` is BLOCKED on Vista** (`salloc job submission is not allowed`) — interactive sessions
> must use `idev` (2h cap on `gh-dev`, too short for the ~hour build). So the build is submitted as
> a **CPU-only `sbatch` batch job** that runs the whole clone → build → install → verify script.
> (An `idev -p gg -m 700` interactive session inside tmux is the alternative, but sbatch is the
> proven, persistent path.)

**Toolchain** — must match the otagent env's torch `cu128`: load **nvcc 12.8 + gcc 13.2** (NOT the
default `nvidia/24.7`, which only ships cuda/12.5; and `cuda/12.8` won't load until a real gcc is
loaded first):
```bash
module purge && module load gcc/13.2.0 cuda/12.8   # -> nvcc 12.8.93, gcc 13.2.0, $TACC_CUDA_DIR set
```

**Build flags**: `TORCH_CUDA_ARCH_LIST="9.0"` (Hopper GH200), `CUDA_HOME=$TACC_CUDA_DIR`,
`VLLM_TARGET_DEVICE=cuda`, `MAX_JOBS=32`, `NVCC_THREADS=4`. Build with **`--no-build-isolation`** so it
uses the env's `torch==2.11.0` as the base; first `pip install` the build-system requires
(`setuptools<81 setuptools-scm packaging wheel ninja jinja2 cmake` — the env ships ninja but NOT
setuptools_scm). **Install these WITH deps (no `--no-deps`)** — they are pure-Python and don't touch
torch, and setuptools-scm 10.x needs its `vcs-versioning` helper (a `--no-deps` install of just
setuptools-scm fails the build with `ModuleNotFoundError: No module named 'vcs_versioning'`).
Wheel command: `python -m pip wheel . --no-build-isolation --no-deps -w dist/` (the `--no-deps` HERE,
on the vLLM wheel itself, is correct — it stops pip resolving vLLM's runtime deps).

The whole thing is captured in `$SCRATCH/build_vllm.sbatch` (clone → toolchain → build → install →
import-verify). Submit + watch:
```bash
sbatch $SCRATCH/build_vllm.sbatch          # -> job NNN on a gg node; log: $SCRATCH/vllm_build.NNN.log
squeue -u penfever ; tail -f $SCRATCH/vllm_build.NNN.log
```
Install (done by the script, or manually after): `pip install --no-deps --force-reinstall $SCRATCH/vllm/dist/vllm-*.whl`
then verify `python -c "import vllm; print(vllm.__version__, vllm.__file__)"`.

### vLLM standalone (vllm_sandboxes env)
```bash
source /scratch/10635/penfever/miniconda3/bin/activate vllm_sandboxes
module load gcc/15.1.0
export TRITON_CC=$(which gcc)
export LD_LIBRARY_PATH=/home1/apps/gcc/15.1.0/lib64:/scratch/10635/penfever/miniconda3/lib/python3.12/site-packages/torch/lib:$LD_LIBRARY_PATH
vllm serve meta-llama/Llama-3.1-8B-Instruct
```

### Evalchemy (standard evals)
```bash
cd $SCRATCH/evalchemy && conda activate evalchemy
```

### Source conda (generic)
```bash
source /scratch/10635/penfever/miniconda3/etc/profile.d/conda.sh
```

## aarch64 specifics

- Miniconda installer: `Miniconda3-latest-Linux-aarch64.sh` (NOT x86_64).
- System modules via Lmod: `module load gcc/15.1.0`, `module load cuda/12.6`, `module load nvidia/24.7`.
- Default loaded modules: `TACC`, `cmake/4.1.1`, `gcc/15.1.0`, `nvidia/24.7`, `openmpi/5.0.5`, `nvpl/24.5`, `ucx/1.18.1`.
- GH200 unified memory: GPU memory is used as filesystem cache and **cannot always be reclaimed** after
  an application exits → occasional hung nodes requiring reboot. Avoid relying on clean memory reclamation.

## Syncing eval traces to local (via AWS S3)

TACC eval traces (especially from Negin/Richard's shared dir) can be synced to local through an AWS S3
bucket:

```bash
# TACC → S3 (from TACC login node)
aws s3 sync /scratch/08134/negin/dc-agent-shared/dc-agent/eval/tacc/jobs/to_upload/<run_dir> \
  s3://oumi-science-donotdelete/benjamin/<run_dir> --only-show-errors

# S3 → local Mac (from Mac)
aws s3 sync s3://oumi-science-donotdelete/benjamin/<run_dir> ../evaltraces/<run_dir> --only-show-errors
```

## VS Code remote tunnel

```bash
~/apps/vscode-cli/code tunnel --name vista-idev --verbose
```

## Shared directory ownership

To preserve executability while locking out non-group members:
```bash
chmod -R u+rwX,g+rwX,o-rwx /path/to/dir
find /path/to/dir -type d -exec chmod g+s {} +
umask 007
setfacl -R -d -m g::rwx /path/to/dir
setfacl -R -d -m o::--- /path/to/dir
```

## Gotchas

- **`uv` crashes on the login node** (Rust memory allocation failure). Use `srun -p gg`
  to get a compute node for any `uv pip install` / build operation. See Environment setup.
- **`/home1/10635/penfever/.cache` is a symlink** to `$SCRATCH/.cache` (fixed 2026-06-25;
  was previously a broken symlink to `$SCRATCH/hf_cache/.cache`). If `uv` fails with
  `failed to create directory ... File exists`, check this symlink.
- **`gh-dev` has a 2-hour wall cap.** Use `gh`/`gg` for anything longer.
- **`idev -p gh-dev -m >120` fails** with `QOSMaxWallDurationPerJobLimit` regardless of partition's
  `MaxTime=UNLIMITED`.
- **GH200 memory is not always reclaimable.** A crashed job can leave a node unusable until reboot;
  if `srun`/`idev` lands on such a node, exit and request a fresh allocation.
- **No `eval/clusters/tacc.yaml` or `eval/tacc/` sbatch template exists yet.** TACC eval currently uses
  the legacy `eval/tacc/*_eval_listener.py` scripts (if present) or manual `harbor jobs start`. A cluster
  config + unified_eval_harbor.sbatch port is pending.
- **SkyRL TACC setup**: <https://github.com/NovaSky-AI/SkyRL/blob/arm/skyrl-train/scripts/tacc_setup.sh>
  (the `arm` branch has TACC-specific setup).
- **SUs expire 2025-12-31.** Check `bbalance` periodically; request renewal if running low.
