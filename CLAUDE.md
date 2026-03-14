# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Claude Code Environment Notes

**Conda Environment**: Use the otagent Python directly (symlinks don't work in the sandbox):
```bash
/Users/benjaminfeuer/miniconda3/envs/otagent/bin/python your_script.py
```

**Syntax Checking**: Use the IDE MCP tool `mcp__ide__getDiagnostics` for checking Python syntax errors and linting issues. Do NOT use bash commands like `python -m py_compile` or `flake8` as the bash environment may have issues with output capture.

```
# Example: Check a file for errors
mcp__ide__getDiagnostics(uri="file:///path/to/file.py")
```

## Repository Overview

ot-agent is a distributed training and evaluation system for large language models on HPC clusters. It consists of four main subsystems:

1. **Data Generation**: Task and trace generation pipelines using Harbor/Daytona
2. **SFT Training (DCFT)**: Supervised fine-tuning using LLaMA-Factory
3. **RL Training**: Reinforcement learning training with SkyRL (using GRPO algorithm)
4. **Evaluation**: Terminal-bench based evaluation system

## Architecture

### Directory Structure

- **`data/`**: Data generation pipelines - each subdirectory is a named pipeline
- **`hpc/`**: DCFT SFT training launcher (uses LLaMA-Factory)
- **`train/hpc/`**: OT-Agent RL training launcher (uses SkyRL framework)
- **`eval/`**: Evaluation systems for both TACC and JSC clusters
- **`database/unified_db/`**: Supabase registry for datasets, models, and agents
- **`scripts/`**: Utility scripts for database, datagen, harbor, vllm, etc.

Both HPC launchers share similar architecture:
- `launch.py`: Main entry point for job submission
- `hpc.py`: Cluster detection and configuration (Pydantic models)
- `arguments.py`: CLI argument parsing
- `sbatch/`: SLURM job templates (Jinja2 for RL, plain for SFT)
- `dotenv/`: Environment variable files per cluster
- `scripts/common.sh`: Shared bash utilities and aliases

### Key Distinction: Internet Access

The codebase handles two types of HPC clusters:

**Internet-enabled clusters** (TACC: Vista, Lonestar; ZIH: Capella, Alpha):
- Compute nodes can directly access HuggingFace Hub
- Standard dataset/model loading works

**No-internet clusters** (JSC: Jureca, Jupiter, Juwels; Leonardo):
- Compute nodes have NO internet access
- `pre_download_dataset()` function pre-downloads datasets/models on login nodes
- Downloads stored in `HF_HUB_CACHE` before job submission
- Training uses SSH tunnels and Ray for coordination

### Supported HPC Clusters

**TACC (Texas Advanced Computing Center)**:
- Vista: GH200 96GB GPUs, 552 nodes, internet access
- Lonestar (ls6): A100 40GB GPUs, 73 nodes, internet access

**JSC (Jülich Supercomputing Centre)**:
- Jureca: H100 94GB GPUs, 16 nodes, no internet
- Jupiter: GH200 96GB GPUs, 48 nodes, no internet
- Juwels: A100 40GB GPUs, 936 nodes, no internet

**ZIH (TU Dresden)**:
- Capella: H100 94GB GPUs, 146 nodes, internet access
- Alpha: A100 40GB GPUs, 37 nodes, internet access

**Leonardo** (CINECA):
- A100 64GB GPUs, 3456 nodes, no internet

### Data Generation System

`data/` contains named pipeline directories. Two approaches are supported:

**Declarative scripts (`generate.py`)**: Self-contained Python scripts for local/one-off runs
```bash
python data/<dataset>/generate.py --optional-flags
```

**Class-based generators (`generate_abstract.py`)**: Subclass `BaseDataGenerator` for HPC runs with managed vLLM endpoints
```bash
python -m hpc.launch \
  --job_type datagen \
  --datagen_script data/<dataset>/generate_abstract.py \
  --datagen_target_repo <org/repo> \
  --datagen_extra_args "--stage both --limit 2000"
```

Key modules in `data/generation/`: `base.py` (BaseDataGenerator), `schemas.py` (GenerationRequest/Result), `engines.py` (InferenceEngine implementations for OpenAI/Anthropic/vLLM)

### Harbor Environment Backends

Harbor supports three environment backends for running sandbox containers:

- **`daytona`** (default): Cloud-managed containers via Daytona API
- **`docker`**: Local Docker/Podman runtime
- **`modal`**: Modal's cloud container platform

**Docker Backend Setup**:
```bash
# Auto-detect Docker/Podman (sets DOCKER_HOST automatically)
python -m data.local.run_tracegen \
    --harbor-config hpc/harbor_yaml/trace_docker_32concurrency_ctx32k.yaml \
    --tasks-input-path ./my-tasks \
    --trace-env docker

# For SLURM with Podman, source the helper first
source docker/setup_docker_runtime.sh
python -m data.local.run_tracegen --trace-env docker ...
```

Docker backend configs are in `hpc/harbor_yaml/trace_docker_*.yaml`.

**Runtime Detection** (`hpc/docker_runtime.py`):
- Auto-detects Docker vs Podman
- Sets `DOCKER_HOST` environment variable
- Supports SSH tunnels to remote Docker daemons

### Database System

`database/unified_db/`: Supabase-backed registry for datasets, models, and agents
- Auto-fills 9-12 metadata fields per entry
- Supports HuggingFace and local file registration
- Python API: `register_hf_dataset()`, `register_local_parquet()`, `register_hf_model()`, `register_agent()`

## Common Commands

### DCFT SFT Training (hpc/)

Setup:
```bash
# Initialize LLaMA-Factory submodule
git submodule update --init --remote dcft/train/llamafactory

# Install dependencies
uv pip install -r hpc_requirements.txt

# Source environment (cluster-specific)
source /PATH/TO/ot-agent/hpc/dotenv/tacc.env  # or jureca.env, etc.
cd $DCFT
$DCFT_ACTIVATE_ENV
```

Launch training:
```bash
python3 -m hpc.launch \
  --train_config_path dcft/train/hp_settings/paper/reasoning_medium.yaml \
  --time_limit 24:00:00 \
  --num_nodes 16 \
  --dataset mlfoundations-dev/your-dataset

# Dry run (preview without submitting)
python3 -m hpc.launch --dry_run [other args]
```

Helper commands (defined in `hpc/scripts/common.sh`):
```bash
gotrain <name>   # Standard (medium) hyperparameters
gosmall <name>   # Small scale training
golarge <name>   # Large scale training
gofast <name>    # More GPUs for faster training
goeval <name>    # Eval on pipeline evals
fulleval <name>  # Full reasoning evals including held-out
```

### OT-Agent RL Training (train/hpc/)

Setup:
```bash
cd train
source hpc/setup.sh  # Auto-detects cluster and loads environment
```

Launch training:
```bash
python3 -m hpc.launch \
  --job_type rl \
  --rl_config ./hpc/skyrl_yaml/jupiter/48GPU_base_32b.yaml \
  --model_path Qwen/Qwen3-32B \
  --time_limit 11:59:00 \
  --num_nodes 12 \
  --train_data '["DCAgent/dataset-name"]' \
  --max_restarts 8 \
  --experiments_dir /e/data1/datasets/playground/ot-baf

# Dry run
python3 -m hpc.launch --dry_run [other args]
```

**RL config files** (`hpc/skyrl_yaml/jupiter/`):
- `24GPU_base.yaml` — 8B model, 6 nodes (8 TP=1 inference engines)
- `48GPU_base_32b.yaml` — 32B model, 12 nodes (16 TP=2 inference engines, thinking)
- `48GPU_base_32b_nothink.yaml` — 32B nothink variant
- `_131k` variants — 131k context length

**Key RL defaults** (from YAML, override via CLI `trainer.X=Y` or `generator.X=Y`):
- `trainer.epochs`: 2, `trainer.max_steps`: 60
- `trainer.train_batch_size`: 64, `generator.n_samples_per_prompt`: 8
- `trainer.policy.optimizer_config.lr`: 5e-6
- `generator.sampling_params.temperature`: 0.7
- `trainer.strategy`: fsdp2, `trainer.algorithm.advantage_estimator`: rloo_n
- `--experiments_dir`: defaults to `experiments/` in repo; use `/e/data1/datasets/playground/ot-baf` for personal runs on Jupiter

Helper commands (same as DCFT):
```bash
gotrain <name>   # Standard training
gosmall <name>   # Small scale
golarge <name>   # Large scale
gofast <name>    # Fast training
```

### Job Monitoring (both systems)

```bash
sqme              # Show your queued jobs
status [lines]    # Show job status and recent logs
sfail [hours]     # Show failed jobs in last N hours
swin [hours]      # Show completed jobs in last N hours
soops [hours]     # Show cancelled jobs in last N hours
sinf              # Show formatted cluster information

# Tail logs
tail $DCFT/experiments/logs/<job_name>_<job_id>.out
tail $CHECKPOINTS_DIR/<job_name>/trainer_log.jsonl

# Cancel jobs
scancel <job_id>
scancel -u $USER -t PENDING  # Cancel all pending jobs
scancelall                    # Cancel all your jobs

# Cleanup
rmlogs [threshold]  # Remove old log files
```

### Database Commands

```bash
# Install database CLI
cd database/unified_db
pip install -r requirements.txt

# Setup environment
export SUPABASE_URL=your_url
export SUPABASE_ANON_KEY=your_key
export SUPABASE_SERVICE_ROLE_KEY=your_service_key

# Apply schema
psql $DATABASE_URL -f complete_schema.sql
```

## Key Implementation Details

### HPC Auto-Detection

Both launchers use `hpc.py:detect_hpc()` to automatically detect the cluster from hostname:
- Matches hostname against regex patterns for each cluster
- Returns HPC configuration object with cluster-specific settings
- If not recognized, raises ValueError

### Pre-Download System (JSC/No-Internet Clusters)

For JSC clusters, `train/hpc/launch.py:pre_download_dataset()`:
1. Runs on login node (which has internet access)
2. Uses `huggingface_hub.snapshot_download()` for datasets and models
3. Downloads to `HF_HUB_CACHE` directory
4. Compute nodes then use cached files during training

### SLURM Job Templates

Templates use Jinja2 for dynamic generation:
- `hpc/sbatch/*.sbatch`: DCFT training templates
- `train/hpc/sbatch/*.sbatch`: RL training templates
- Variables substituted from CLI arguments and HPC config

### Environment Variables

Critical environment variables (set in `dotenv/*.env`):
- `HF_TOKEN`: HuggingFace API token
- `WANDB_TOKEN`: Weights & Biases token
- `HF_HUB_CACHE`: Cache directory for HF datasets/models
- `DCFT`: Base directory for DCFT system
- `DC_AGENT_TRAIN`: Base directory for RL training system
- `CHECKPOINTS_DIR`: Output directory for checkpoints

### Batch Job Submission

To launch multiple jobs at once:
```bash
cat << 'EOF' | while read -r model; do [[ -z "$model" ]] || gotrain "$model"; done
Qwen/Qwen2.5-7B-Instruct
microsoft/phi-2
meta-llama/Llama-3-8b
EOF
```

## Testing

```bash
# Test HPC detection
cd train
python3 hpc/test_hpc.py

# Test database
cd database/unified_db/test_sft_dataset_register
python test_dataset_upload.py
python test_model_upload.py
python test_agent_upload.py

# Example terminal-bench run
python example_tbench.py
```

## Important Notes

### Dependencies

This repository does NOT have managed dependencies. Each subsystem has its own:
- Data: `data/data_requirements.txt`
- HPC: `hpc/hpc_requirements.txt`
- SFT: `dcft/train/llamafactory/pyproject.toml` (install with liger-kernel, deepspeed, hf-kernels extras)
- RL: SkyRL requirements (coming soon)

### Git Submodules

LLaMA-Factory is included as a submodule:
```bash
git submodule update --init --remote dcft/train/llamafactory
```

### Configuration Files

**SFT configs**: `dcft/train/hp_settings/` and `dcft/train/lf_configs/qwen2_5/`
- Paper configs: `paper/reasoning_{small,medium,large}.yaml`

**RL configs**: Arguments passed via CLI to `train/hpc/launch.py`
- Algorithm: GRPO (default), Strategy: FSDP2 (default), Backend: vLLM (default)

### Common Pitfalls

1. **JSC pre-download**: Always ensure datasets/models are pre-downloaded on login node before job submission
2. **Node exclusions**: Some clusters have exclusion lists for faulty nodes (see `hpc.py`)
3. **Internet access**: Know whether your cluster has internet on compute nodes. JSC compute nodes use proxychains for HF access (W&B doesn't work through proxychains)
4. **LLaMA-Factory submodule**: Must be initialized before SFT training
5. **Environment sourcing**: Must source correct `dotenv/*.env` for your cluster
6. **ShareGPT role tags**: Harbor/DCAgent datasets use `role`/`content` keys with `user`/`assistant` values. LLaMA-Factory defaults to `from`/`value` with `human`/`gpt`. Always pass explicit tags for Harbor datasets:
   ```
   --role_tag role --user_tag user --assistant_tag assistant --content_tag content
   ```
   Without these, the thinking preprocessor finds 0 assistant messages and training produces garbage.
7. **push_to_hub on JSC**: Defaults to `false` (no-internet cluster). Override with `--push_to_hub true` since proxychains provides HF Hub access on compute nodes.

## JSC Jupiter Access

**SSH**: Connect with IPv4-only (`-4` required):
```bash
ssh -i ~/.ssh/id_ed25519_jsc feuer1@login01.jupiter.fz-juelich.de -4
```

**Tmux**: Sessions persist across SSH disconnects. Key sessions:
```bash
tmux ls                    # List sessions
tmux attach -t 2           # Attach to session "2" (main work session)
```

**Pre-launch preamble** (run before launching any new job — pulls latest code):
```bash
source ~/.bashrc; source ~/secrets.env; \
cd /e/scratch/jureap59/feuer1/harbor && git stash && git pull; \
cd /e/scratch/jureap59/feuer1/OpenThoughts-Agent/SkyRL && git stash && git pull; \
conda activate otagent; \
cd /e/scratch/jureap59/feuer1/OpenThoughts-Agent && GIT_TERMINAL_PROMPT=0 git pull && \
git submodule update --init --remote sft/llamafactory; \
source hpc/dotenv/jupiter.env
```
Note: `GIT_TERMINAL_PROMPT=0` prevents interactive auth prompts from blocking the shell.

**Key paths**:
- Code: `/e/scratch/jureap59/feuer1/OpenThoughts-Agent`
- Shared data: `/e/data1/datasets/playground/ot`
- Personal data: `/e/data1/datasets/playground/ot-baf`
- SFT checkpoints: `/e/data1/datasets/playground/ot/checkpoints/`
- Harbor: `/e/scratch/jureap59/feuer1/harbor`

**Job management** (SLURM):
```bash
sqme                       # Show your queued/running jobs
squeue -u feuer1           # Detailed job queue
scancel <job_id>           # Cancel a job
```

**Rsync files to local** (from Mac):
```bash
rsync -avz --progress -e "ssh -i ~/.ssh/id_ed25519_jsc -4" \
  feuer1@login01.jupiter.fz-juelich.de:/remote/path /local/path
```

**Cluster details**: GH200 96GB GPUs, 48 nodes, no internet on compute nodes. Pre-download datasets/models on login node before submitting jobs.

## NERSC Perlmutter Access

**SSH**: Uses ControlMaster multiplexing (2FA required on first connect):
```bash
ssh perlmutter    # Complete 2FA once; socket persists 8h
```

**Pre-launch preamble** (run before launching any new job):
```bash
conda activate dcagent; cd $SCRATCH/OpenThoughts-Agent; git pull; \
source hpc/dotenv/perlmutter.env; source ~/secrets.env; \
git submodule update --init --remote sft/llamafactory; \
cd $SCRATCH/SkyRL; git pull; \
cd $SCRATCH/harbor; git pull; \
cd $SCRATCH/OpenThoughts-Agent;
```

**Key paths**:
- Code: `$SCRATCH/OpenThoughts-Agent`
- SkyRL: `$SCRATCH/SkyRL`
- Harbor: `$SCRATCH/harbor`

**Cluster details**: A100 80GB GPUs, internet access on compute nodes. User: `penfever`.

## ALCF Polaris Access

**SSH**: Uses ControlMaster multiplexing (2FA required on first connect):
```bash
ssh ALCFPolaris    # Complete 2FA once; socket persists 8h
```

**Pre-launch preamble** (run before launching any new job):
```bash
source ~/.bashrc && conda activate otagent && \
cd /lus/eagle/projects/CausalAlign/penfever42/code/OpenThoughts-Agent && git pull && \
cd /lus/eagle/projects/CausalAlign/penfever42/code/harbor && git pull && \
source hpc/dotenv/polaris.env && source ~/secrets.env && \
cd /lus/eagle/projects/CausalAlign/penfever42/code/OpenThoughts-Agent
```

**Key paths**:
- Code: `/lus/eagle/projects/CausalAlign/penfever42/code/OpenThoughts-Agent`
- Harbor: `/lus/eagle/projects/CausalAlign/penfever42/code/harbor`
- Data/HF cache: `/lus/eagle/projects/CausalAlign/penfever42/data/hub`
- Experiments: `/lus/eagle/projects/CausalAlign/penfever42/experiments`

**Cluster details**: A100 40GB GPUs, 4/node, 560 nodes, PBS Pro scheduler (not SLURM). Internet via proxy (`proxy.alcf.anl.gov:3128`). User: `penfever42`.

**Important**: The OT-Agent repo is `open-thoughts/OpenThoughts-Agent` (NOT `laude-institute`). Harbor is `laude-institute/harbor`.

**Package management**: Use `uv pip install` (not bare `pip`) for all installs on Polaris.

## Eval Job Submission Defaults

When submitting eval jobs via `unified_eval_listener.py`, always use these flags unless explicitly told otherwise:
- `--require-priority-list` — only eval models in the priority file
- `--n-concurrent 64` on Jupiter (48 times out with fewer concurrent trials)
- `--n-concurrent 48` on other clusters
- `--harbor-config hpc/harbor_yaml/eval/eval_ctx32k_non_it.yaml` for 32k context models
- `--harbor-config hpc/harbor_yaml/eval/eval_ctx131k_non_it.yaml` for 131k context models

Model lists live in `eval/jupiter/lists/` (`models_32b.txt`, `models_131k.txt`).

## CINECA Leonardo Access

**SSH**: Uses ControlMaster multiplexing + step-ca certificate auth:
```bash
ssh Leonardo    # Complete 2FA once; socket persists 8h
```

**Pre-launch preamble** (run before launching any new job):
```bash
source /leonardo_work/EUHPC_E03_068/bfeuer00/miniforge3/etc/profile.d/conda.sh && \
conda activate otagent && module load gcc/12.2.0 cuda/12.6 && \
cd /leonardo_work/EUHPC_E03_068/bfeuer00/code/OpenThoughts-Agent && git pull && \
cd /leonardo_work/EUHPC_E03_068/bfeuer00/code/harbor && git pull && \
source hpc/dotenv/leonardo.env && source ~/secrets.env && \
cd /leonardo_work/EUHPC_E03_068/bfeuer00/code/OpenThoughts-Agent
```

**Key paths**:
- Code: `/leonardo_work/EUHPC_E03_068/bfeuer00/code/OpenThoughts-Agent`
- Harbor: `/leonardo_work/EUHPC_E03_068/bfeuer00/code/harbor`
- Data/HF cache: `/leonardo_work/EUHPC_E03_068/bfeuer00/data/hub`
- Experiments: `/leonardo_work/EUHPC_E03_068/bfeuer00/experiments`

**Cluster details**: A100 64GB GPUs, 4/node, 3456 nodes, SLURM scheduler. No internet on compute nodes (use proxychains/SSH tunnel). User: `bfeuer00`. Account: `AIFAC_5C0_290`.

**Important**: Must load `gcc/12.2.0` and `cuda/12.6` modules before building or running vLLM. Default GCC (8.5) is too old.

## Experiment Launch Command References

- **SFT experiments**: `/Users/benjaminfeuer/Documents/notes/ot-agent/sft_experiments.md`
- **RL experiments**: `/Users/benjaminfeuer/Documents/notes/ot-agent/rl_experiments.md`

When resubmitting cancelled jobs, look up the original launch command in these files first.

## RL Job Cleanup Checklist

After an RL job terminates (early or completed), follow these steps to preserve and publish the checkpoint:

1. **Locate the last checkpoint** in the exports folder:
   ```bash
   ls -lt $EXPERIMENTS_DIR/<job_name>/exports/ | head -10
   ```
   The most recent `global_step_*` or `episode_*` directory is the final checkpoint.

2. **Locate the W&B run**: Check the job logs or `trainer_log.jsonl` for the wandb run URL. Format: `https://wandb.ai/dogml/OpenThoughts-Agent/runs/<run_id>`

3. **Flatten model files**: Move all files out of nested subdirectories into the checkpoint root. Remove empty dirs:
   ```bash
   cd $CHECKPOINT_DIR
   # Move nested files up (e.g., from global_step_*/policy/ or actor/)
   find . -mindepth 2 -type f -exec mv {} . \;
   find . -mindepth 1 -type d -empty -delete
   ```

4. **Copy the launch config**: Copy the RL YAML config used to launch the job into the model folder for reproducibility:
   ```bash
   cp hpc/skyrl_yaml/<config_used>.yaml $CHECKPOINT_DIR/rl_config.yaml
   ```

5. **Upload to HuggingFace**: Use `huggingface-cli upload-large-folder` to push to `laion/<job_name>`:
   ```bash
   huggingface-cli upload-large-folder laion/<job_name> $CHECKPOINT_DIR
   ```

6. **Register in DB**: Manual push of the RL model to the unified DB:
   ```bash
   python -m database.unified_db.register_model --hf-repo laion/<job_name> --base-model <base_model_hf>
   ```

7. **Upload RL traces**: Upload the training traces from the job:
   ```bash
   python -m scripts.harbor.make_and_upload_trace_dataset \
     --job_dir "$EXPERIMENTS_DIR/<job_name>/<job_name>" \
     --repo_id penfever/<job_name> \
     --episodes last
   ```

8. **Clean up experiments dir**: Only after all above steps succeed, remove the local job directory to free disk space.

## Code Ownership (DRIs)

- Data: Etash (`EtashGuha`)
- RL: Tyler and Charlie (`tyler-griggs`, `CharlieFRuan`)
- SFT: Ben (`penfever`)
- Eval: Negin (`neginraoof`)
