# SERA v4 Baseline — Jupiter Reproduction

Faithful re-creation of the SERA SFT training pipeline for Qwen3-8B on the JSC Jupiter cluster (GH200, aarch64, CUDA 13), matching Ai2's methodology end-to-end including the Hermes/Qwen3 `<tool_call>` wire-format pre-render that axolotl's stock chatml preset drops.

- **Upstream paper / blog**: https://allenai.org/papers/opencodingagents
- **Upstream code**: https://github.com/allenai/SERA (data-prep) + https://github.com/allenai/SERA-SWE-agent (eval harness)
- **Upstream model**: https://huggingface.co/allenai/SERA-8B
- **Sibling baseline**: `/baselines/coderforge` (same axolotl env, CoderForge-Preview data source)

---

## Iteration history — why v4?

| Version | Data source | Tool-call rendering | Outcome |
|---|---|---|---|
| **v1** (deprecated) | `allenai/Sera-4.5A-Full-T1` | — | Prototype only |
| **v2** (deprecated, deleted) | `allenai/Sera-4.5A-Full-T1` → shareGPT via custom converter | Converter silently dropped `tool_calls` field | SWE-bench eval: 3/295 (1%), 100% harness-use failures |
| **v3** (deprecated, deleted) | `allenai/Sera-4.5A-Full-T1`, raw JSONL passed to axolotl `chat_template: chatml` | chatml preset ignores the OpenAI-format `tool_calls` field → zero `<tool_call>` tokens in training loss | SWE-bench eval: 0/297 (0%). Root-caused 2026-04-22 via preprocessed-cache decode: `<tool_call>` count = 0 in input_ids |
| **v4** (2026-04-22, deprecated) | `allenai/Sera-4.6-Lite-T2` pre-rendered by `subset_sera_v4.py` via `transform_traj_hermes` | `<tool_call>\n{"name":…, "arguments":…}\n</tool_call>` into `content`, tool responses → `role: user` + `<tool_response>…</tool_response>` | SWE-bench eval: 0/89 (0%). Root-caused 2026-04-23: model emitted whitespace collapse after ~2 turns. `chat_template: chatml` (bare) at train mismatched stock Qwen3 template at inference (which has `<think>`-stripping for non-last assistant turns) → multi-turn OOD drift |
| **v4-v2** (2026-04-23, SLURM 389486, `laion/Sera-4.6-Lite-T2-v4-316-axolotl__Qwen3-8B-v2`) | `laion/Sera-4.6-Lite-T2-v4-316` (same as v4) | same as v4 | **Structural recovery** — model stops collapsing, emits coherent `<tool_call>{JSON}</tool_call>` output. But: invalid JSON (`"arguments": {...}}}` with 3 closing braces) + wrong tool names (`"view"` instead of `"str_replace_editor"`). SWE-bench eval: 0/67 (0%). Fix applied: `chat_template: tokenizer_default` so train-time render matches inference stock-template render byte-identically |
| **v4-v3** (2026-04-24, SLURM 391109, `laion/Sera-4.6-Lite-T2-v4-316-axolotl__Qwen3-8B-v3`, **current**) | same as v4-v2 (training data probe confirmed clean: valid JSON, correct tool names, matching SWE-agent harness schema) | same as v4-v2 | `num_epochs: 3 → 6`. Hypothesis: at size=316 × 3 epochs / 8 grad-accum ≈ 120 gradient updates, not enough signal to lock the outer tool-name slot distinct from `arguments.command` or keep brace-stack correct. Doubling training doubles signal while preserving v4-v2 structural gains. Eval pending post-retrain |

### Additional gotcha (2026-04-23)

After v4 training completed and models were uploaded to HF, the first eval cycle still showed 0/89 pass. Root cause: **axolotl saves a stripped-down `tokenizer_config.json` + a bare 4-line `chat_template.jinja` that don't handle `tool_calls` or `role:tool` at serve time**, so vLLM silently dropped every `tool_calls` field in the incoming OpenAI-format messages and passed raw tool observations to the model. The SFT'd model had learned the correct wire format but the served prompt was malformed. Ai2 sidesteps this by shipping SERA-8B with tokenizer files **byte-identical** to stock `Qwen/Qwen3-8B` — they overwrote axolotl's bare config at publish time.

**Always apply the tokenizer-restore step** (§ Post-training, step 4) after any axolotl SFT on a Qwen3/Qwen2.5 base.

---

## Dataset

Uploaded to `laion/Sera-4.6-Lite-T2-v4(-<SIZE>)` as raw JSONL after pre-rendering via `transform_traj_hermes`:

| Dataset | Rows |
|---|---|
| `laion/Sera-4.6-Lite-T2-v4` | 36,083 (full — not yet uploaded) |
| `laion/Sera-4.6-Lite-T2-v4-316` | 316 |
| `laion/Sera-4.6-Lite-T2-v4-1000` | 1,000 |
| `laion/Sera-4.6-Lite-T2-v4-3160` | 3,160 (pending rollout) |
| `laion/Sera-4.6-Lite-T2-v4-10000` | 10,000 (pending rollout) |
| `laion/Sera-4.6-Lite-T2-v4-31600` | 31,600 (pending rollout) |

Each row carries `messages: list[{role, content, train}]`, where:
- Roles are `system | user | assistant` (upstream `role: tool` observations are rewritten to `role: user` with `<tool_response>…</tool_response>` wrapping).
- `content` for assistant turns contains the original `<think>…</think>` block + visible prose + one or more `<tool_call>\n{JSON}\n</tool_call>` blocks (appended per upstream `transform_traj_hermes`).
- `train: bool` on each message is the per-message loss mask consumed by axolotl's `message_field_training: train`.
- Plus `instance_id`, `source="allenai/Sera-4.6-Lite-T2"`, and `docker_image`/`problem_statement` passthroughs.

Sampling: deterministic random, seed=42, row-indexed into the full 36,083-row source. Row subsets are nested.

### Why `Sera-4.6-Lite-T2` (not `Sera-4.5A-Full-T1`)?

Discovered 2026-04-23 while investigating the v3 eval failure: the Together AI blog references `Sera-4.6-Lite-T2` as the SFT training set (file: `sera-4.6-lite-t2_36083_string_enriched.jsonl`). The 4.5A-Full-T1 dataset (72k rows, used by v2/v3) is the raw pre-filter trajectory pool, not the filtered+deduplicated set used in the paper.

---

## Environment (Jupiter, aarch64, CUDA 13)

`sera-axolotl` conda env on Jupiter. Core pins after install:

| Package | Version |
|---|---|
| python | 3.12.13 |
| torch | 2.9.1+cu130 |
| axolotl | 0.16.0.dev0 (from v0.16.1 tag) |
| transformers | 5.5.0 |
| accelerate | 1.13.0 |
| deepspeed | 0.18.9 |
| datasets | 4.5.0 |
| flash-attn | 2.8.3+cu130torch2.9 (prebuilt wheel) |
| triton | 3.5.1 |

Notable absences (excluded by axolotl's aarch64 filter in `setup.py`): `torchao`, `fla-core`, `flash-linear-attention`.

### Install recipe

Axolotl has two issues to work around on aarch64 + torch 2.9:

1. **`torchao==0.17.0` hard-pins `torch==2.8.0`** → would clobber our Jupiter-safe `torch==2.9.1+cu130`. Axolotl's `setup.py` already filters torchao on aarch64, but uv's prebuilt-wheel metadata resolution doesn't run that code. Fix: force a source build with `--no-build-isolation` so the filter runs.
2. **flash-attn** has no prebuilt aarch64 wheel for torch 2.9. Use [mjun0812/flash-attention-prebuild-wheels](https://github.com/mjun0812/flash-attention-prebuild-wheels) release `v0.7.16`.

```bash
conda activate sera-axolotl  # already has torch 2.9.1+cu130

# Axolotl v0.16.1
mkdir -p /e/scratch/jureap59/feuer1/code && cd /e/scratch/jureap59/feuer1/code
git clone https://github.com/axolotl-ai-cloud/axolotl.git && cd axolotl
git checkout v0.16.1

uv pip install "setuptools>=64" wheel "setuptools_scm>=8" "packaging==26.0"
uv pip install -e . --no-build-isolation

# Deepspeed (excluded from axolotl deps)
export CUDA_HOME=/e/software/default/stages/2026/software/CUDA/13
export PATH=$CUDA_HOME/bin:$PATH
uv pip install "deepspeed>=0.18.6,<0.19.0" --no-build-isolation

# flash-attn prebuilt wheel
cd /tmp
WHL=flash_attn-2.8.3+cu130torch2.9-cp312-cp312-manylinux_2_34_aarch64.whl
wget -q "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/$WHL"
uv pip install --no-deps "./$WHL"

# axolotl-ai-cloud's cut-cross-entropy fork (required by import chain even though we disable the plugin in config)
uv pip install "cut-cross-entropy[transformers] @ git+https://github.com/axolotl-ai-cloud/ml-cross-entropy.git@63b15e6" --no-build-isolation
```

### Mandatory axolotl source patches

1. **`src/axolotl/utils/callbacks/qat.py`** — guard torchao imports in a `try/except ImportError:` block; assign unreachable stub classes so `isinstance()` checks in `toggle_fake_quant` still return False. Without this, `from axolotl.core.builders import …` fails on aarch64.

2. **`convert_axolotl_checkpoint.py`** — kept at `/e/scratch/jureap59/feuer1/code/axolotl/convert_axolotl_checkpoint.py` on Jupiter (also tracked locally as `baselines/sera/convert_axolotl_checkpoint.py`). Strips `_checkpoint_wrapped_module.` prefixes from state_dict keys so vLLM / sglang can load the checkpoint.

---

## Data subsetter — faithful `transform_traj_hermes` port

`subset_sera_v4.py` (in `scripts_dataset_build/`) reads the upstream `allenai/Sera-4.6-Lite-T2` JSONL and applies the same render logic as Ai2's `sera/datagen/data/postprocess/utils.py::transform_traj_hermes`:

- Assistant turns → `content` = original `<think>…</think>\n{prose}\n\n<tool_call>\n{"name": "…", "arguments": {…}}\n</tool_call>` (one `<tool_call>` block per OpenAI tool_call, concatenated with `\n`).
- Tool turns → `role: "user"` with content wrapped as `<tool_response>\n{text}\n</tool_response>`.
- System / user turns → pass through `content` unchanged (flatten list-of-dicts `[{"type":"text","text":…}]` to string if needed).

Adds `train: bool` per message (True only for assistant), drops upstream metadata fields (`thought`, `action`, `agent`, `message_type`, `tool_call_ids`, `cache_control`) that axolotl doesn't read.

Run locally (reads `HF_TOKEN` from env):
```bash
source /Users/benjaminfeuer/Documents/secrets.env
cd /Users/benjaminfeuer/Documents/scripts_dataset_build
python subset_sera_v4.py
```

The default `SUBSET_SIZES = [316, 1000]` — edit the constant to upload more sizes on the ladder.

---

## Axolotl config

- `configs/template_qwen3_8b_sera_v4.yaml` — axolotl config template (sed-substitute `__SIZE__` for a specific subset).

Key lines:
```yaml
base_model: Qwen/Qwen3-8B
chat_template: chatml   # passes <tool_call>/<tool_response> wire tokens through verbatim because the subsetter already rendered them into content
datasets:
  - path: laion/Sera-4.6-Lite-T2-v4-__SIZE__
    data_files:
      - sera-4.6-lite-t2_v4___SIZE__.jsonl
    type: chat_template
    field_messages: messages
    ds_type: json
    message_field_training: train
sequence_len: 32768
num_epochs: 3
learning_rate: 1e-5
lr_scheduler: cosine
warmup_ratio: 0.1875
gradient_accumulation_steps: 8
micro_batch_size: 1
weight_decay: 0.01
max_grad_norm: 1.0
save_strategy: epoch
deepspeed: /e/scratch/jureap59/feuer1/code/axolotl/deepspeed_configs/zero3_bf16.json
# hub_model_id and hub_strategy are intentionally absent — with HF_HUB_OFFLINE=1
# set on Jupiter compute nodes, init_hf_repo would crash on job start.
```

### Config gotchas

- **Omit `hub_model_id` / `hub_strategy`** — transformers' `init_hf_repo` runs at job start and calls `create_repo` → HF API → `OfflineModeIsEnabled` crash under `HF_HUB_OFFLINE=1`. Push manually after training.
- **Disable `CutCrossEntropyPlugin`** — on aarch64+torch2.9+FA2, CCE causes bf16 grad explosion (grad_norm 9.8e+11) → loss → NaN → masked as 0 after 3-7 steps. Comment out under `plugins:`.
- **Set `max_grad_norm: 1.0` explicitly** as belt-and-suspenders.
- **Use `zero3_bf16.json`** — not `zero1` (OOM without CCE), not `zero2/zero3` defaults (offload Adam to CPU → `DeepSpeedCPUAdam` JIT compile → GCC 14.3 rejects `-march=armv9-a+…+nossbs+nopauth`).

---

## sbatch env (Jupiter compute)

`sbatch/axolotl_sera_v4.sbatch` — SLURM template. Sed-substitute `__SIZE__` and `__NODES__` before submitting. Key env block:

```bash
export CUDA_HOME=/e/software/default/stages/2026/software/CUDA/13
export GCC_HOME=/e/software/default/stages/2026/software/GCCcore/14.3.0
export CC=$GCC_HOME/bin/gcc
export CXX=$GCC_HOME/bin/g++         # Triton JIT kernels need these
export PATH=$CUDA_HOME/bin:$GCC_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$GCC_HOME/lib64:${LD_LIBRARY_PATH:-}

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_MODE=offline
export AXOLOTL_DO_NOT_TRACK=1 DO_NOT_TRACK=1 HF_HUB_DISABLE_TELEMETRY=1

# Optional for multi-node: force IB-interconnect FQDN to avoid localhost:29500 / IPv6 failures on some nodes
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)-interconnect-1.jupiter.internal
export NCCL_SOCKET_IFNAME=ib0 NCCL_IB_TIMEOUT=60
```

Node ladder (empirical, 3-epoch walls on Jupiter GH200):

| Subset | Nodes | Approx total wall |
|---|---|---|
| 316 | 1 | ~28 min |
| 1000 | 1 | ~1h17m |
| 3160 | 2 | ~1h |
| 10000 | 4 | ~2h |
| 31600 | 8 | ~3h |

---

## Pre-download (required on no-internet clusters)

Jupiter compute nodes have no internet. Run `axolotl preprocess` once on the login node for each subset so the tokenized dataset cache lives under `dataset_prepared_path`:

```bash
conda activate sera-axolotl
source /e/scratch/jureap59/feuer1/OpenThoughts-Agent/hpc/dotenv/jupiter.env
source ~/secrets.env
export CUDA_HOME=/e/software/default/stages/2026/software/CUDA/13
export PATH=$CUDA_HOME/bin:$PATH
cd /e/scratch/jureap59/feuer1/code/axolotl

SIZE=316
CFG=/e/scratch/jureap59/feuer1/code/axolotl_configs/qwen3_8b_sera_v4_${SIZE}.yaml
sed "s/__SIZE__/${SIZE}/g" \
  /e/scratch/jureap59/feuer1/code/axolotl_configs/template_qwen3_8b_sera_v4.yaml > $CFG
axolotl preprocess $CFG
```

### Training-cache sanity check (verify `<tool_call>` is in loss)

This is the single most useful guardrail. If it fails, the subsetter or template is broken:

```python
from datasets import load_from_disk
import glob, numpy as np
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
p = glob.glob(f"/e/data1/datasets/playground/ot-baf/axolotl_dataset_cache/sera-v4-316/*")[0]
ds = load_from_disk(p)
ids = np.array(ds[0]["input_ids"])
labels = np.array(ds[0]["labels"])
tc = tok.convert_tokens_to_ids("<tool_call>")  # 151657
in_loss = int(((ids == tc) & (labels != -100)).sum())
print(f"<tool_call> tokens in loss (row 0): {in_loss}")  # must be > 0
```

Expected: 52 for row 0 of sera-v4-316; 23 for row 0 of sera-v4-1000 (varies per row; floor is ~10 assistant turns × 1 tool_call each).

---

## Training launch

```bash
SIZE=316 ; NODES=1
cd /e/scratch/jureap59/feuer1/code/axolotl_sbatch
sed -e "s/__SIZE__/$SIZE/g" -e "s/__NODES__/$NODES/g" \
  template_sera_v4.sbatch > sera_v4_${SIZE}.sbatch
sbatch sera_v4_${SIZE}.sbatch
```

Multi-node launch uses `srun --ntasks-per-node=1` + `accelerate launch --num_machines=$SLURM_JOB_NUM_NODES` with rendezvous on the first node of the allocation and DeepSpeed ZeRO-3-bf16. Chain retries with `--dependency=afterany:$J1` if you expect timeouts or transient NCCL failures.

---

## Post-training (axolotl → vLLM-servable)

Four steps. All required; step 4 is the newest and most commonly forgotten.

### 1. Strip FSDP prefixes

Axolotl with gradient checkpointing writes state_dict keys prefixed with `_checkpoint_wrapped_module.` — vLLM / sglang won't load these.

```bash
SRC=$CHECKPOINTS_DIR/sera-v4-${SIZE}-axolotl__Qwen3-8B
DST=$CHECKPOINTS_DIR/sera-v4-${SIZE}-axolotl__Qwen3-8B-converted

# First remove intermediate checkpoint-N dirs so we don't upload cruft
rm -rf $SRC/checkpoint-* $SRC/.cache

python /e/scratch/jureap59/feuer1/code/axolotl/convert_axolotl_checkpoint.py $SRC $DST
# Converted keys should have no `_checkpoint_wrapped_module.` and total ~399 on Qwen3-8B
```

### 2. Secret scan

```bash
grep -rIE '(sk-[a-zA-Z0-9]{20,}|AKIA[0-9A-Z]{16}|ghp_[a-zA-Z0-9]{36}|hf_[a-zA-Z0-9]{34})' $DST
```

### 3. Upload weights to HF

```bash
source ~/secrets.env
huggingface-cli upload-large-folder \
  laion/Sera-4.6-Lite-T2-v4-${SIZE}-axolotl__Qwen3-8B \
  $DST --repo-type=model
```

### 4. ★ Restore stock Qwen3-8B tokenizer files (CRITICAL) ★

Axolotl saves a stripped-down `tokenizer_config.json` (~665 bytes) and a bare 4-line `chat_template.jinja`. The bare template:
- Does NOT loop `message.tool_calls` → drops tool-call structured fields from incoming OpenAI messages at serve time
- Does NOT wrap `role: "tool"` content in `<tool_response>…</tool_response>`
- Does NOT declare `<tool_call>`/`</tool_call>`/`<think>` in `added_tokens_decoder`

vLLM loads these on server startup and renders every SWE-agent prompt with the broken template → the model sees a malformed prompt with no tool_call wire tokens → goes out-of-distribution → emits degenerate loops → 0% pass rate on SWE-bench harness evals, even though training was healthy. Confirmed 2026-04-23.

Ai2 sidesteps this: SERA-8B's `tokenizer_config.json` is **byte-identical** to stock `Qwen/Qwen3-8B`. Do the same after every axolotl upload:

```bash
source ~/secrets.env
python - <<PY
import os
from huggingface_hub import HfApi, hf_hub_download
api = HfApi(token=os.environ['HF_TOKEN'])
REPO = "laion/Sera-4.6-Lite-T2-v4-${SIZE}-axolotl__Qwen3-8B"
for f in ['tokenizer_config.json', 'tokenizer.json', 'vocab.json', 'merges.txt']:
    local = hf_hub_download('Qwen/Qwen3-8B', f, token=os.environ['HF_TOKEN'])
    api.upload_file(path_or_fileobj=local, path_in_repo=f,
                    repo_id=REPO, repo_type='model',
                    commit_message='Restore stock Qwen3-8B tokenizer (axolotl bare template fix)')
    print(f"  ✓ {f}")
PY
```

Verify: `api.model_info(REPO).siblings` → `tokenizer_config.json` should be ~9700 bytes (not ~665), and the embedded `chat_template` string should contain both `"tool_calls"` and `"tool_response"`.

### 5. Register in the unified DB

```bash
python scripts/database/manual_db_push.py \
  --hf-model-id laion/Sera-4.6-Lite-T2-v4-${SIZE}-axolotl__Qwen3-8B \
  --base-model Qwen/Qwen3-8B \
  --dataset-name laion/Sera-4.6-Lite-T2-v4-${SIZE}
```

### 6. Clean up local checkpoint dirs

```bash
rm -rf $SRC   # pre-convert checkpoint; DST is what we uploaded. Frees ~16 GB per model.
```

---

## Evaluation

SERA was trained on **SWE-agent JSON tool-calling format**:
`<think>...</think>...<tool_call>{"name":"<tool>","arguments":{...}}</tool_call>`. Pair it with the matching harbor harness (`swe_agent_ctx32k_eval_.yaml`) and the SERA team's SWE-agent scaffold config — DO NOT use the OpenHands harness for SERA.

### Pinggy and Daytona

Both clusters route the vLLM endpoint through pinggy. URL+token pairs are catalogued in `/Users/benjaminfeuer/Documents/notes/ot-agent/pinggy_bank.md`. Pairs 1–7 are reserved for sibling experiments; 8–10 are the rotating slots for ad-hoc evals (8 may be in use by a labmate — confirm first).

Daytona key MUST be `$DAYTONA_BASE_API_KEY` — `DAYTONA_API_KEY` is the RL-org key and blocks declarative builds (failure mode: `DaytonaValidationError`).

### Critical: bump `agent.max_requeries`

SWE-agent's default is `agent.max_requeries: 3`. After 3 consecutive `FormatError` / `_BlockedActionError` / `BashIncorrectSyntaxError`, the agent emits **"Exit due to repeated format/blocklist/bash syntax errors"** and forfeits — even if it was making real progress. Sera-trained 8B models hit this floor often, so we override the SWE-agent config via the `SWEAGENT_CONFIG` env var pointing to a patched gist of SERA's [e2e.yaml](https://github.com/allenai/SERA/blob/main/sera/configs/sweagent/e2e.yaml) with `max_requeries: 50`:

```
SWEAGENT_CONFIG=https://gist.githubusercontent.com/penfever/f327eabde26934630ee2aea1a59bb511/raw/sera_e2e_patched.yaml
```

Harbor's `swe_agent.py` reads `os.environ["SWEAGENT_CONFIG"]` — if it's a URL, harbor curls it into the daytona container. If it's a path, it's read directly inside the container. The URL form is portable across clusters.

To inject the env var into a launcher-generated sbatch:
```bash
SB=experiments/<run_dir>/sbatch/<job_name>__eval.sbatch
sed -i '/^set -eo pipefail$/a export SWEAGENT_CONFIG="https://gist.githubusercontent.com/penfever/f327eabde26934630ee2aea1a59bb511/raw/sera_e2e_patched.yaml"' "$SB"
sbatch "$SB"
```

If you need a different override, edit the gist (or fork it) and update the URL.

### Launch — Perlmutter (A100, has internet)

```bash
ssh perlmutter
source ~/.bashrc; source ~/secrets.env; module load conda; conda activate dcagent
cd $SCRATCH/OpenThoughts-Agent && git pull
source hpc/dotenv/perlmutter.env
git submodule update --init --remote sft/llamafactory
cd $SCRATCH/SkyRL && git pull && cd $SCRATCH/harbor && git pull
cd $SCRATCH/OpenThoughts-Agent

python -m hpc.launch \
  --job_type eval \
  --model_path laion/<sera_v7_hub_id> \
  --tasks_input_path DCAgent2/swebench-verified-random-100-folders \
  --trace_harbor_config hpc/harbor_yaml/eval/swe_agent_ctx32k_eval_.yaml \
  --datagen_config hpc/datagen_yaml/qwen3_8b_vllm_serve_32k_4xA100.yaml \
  --trace_agent_name swe-agent \
  --daytona_api_key "$DAYTONA_BASE_API_KEY" \
  --pinggy_persistent_url <pair-N URL> --pinggy_token <pair-N token> \
  --time_limit 11:59:00 \
  --num_nodes 1 --gpus_per_node 4 \
  --trace_n_concurrent 16

# After hpc.launch prints the SLURM job id, patch the sbatch with SWEAGENT_CONFIG
# (see above) and re-sbatch — the launcher submits without the override by default.
```

`--datagen_config` MUST be the A100 variant on Perlmutter; the GH200 yaml uses `--all2all-backend pplx` which crashes on A100.

### Launch — Jupiter (GH200, no internet on compute)

Replace `qwen3_8b_vllm_serve_32k_4xA100.yaml` → `qwen3_8b_vllm_serve_32k_4xGH200.yaml`. Otherwise identical. Login node has direct HF Hub access; the launcher pre-downloads the model into `$HF_HUB_CACHE` before submitting.

### Health check (15 min after submit)

Per `feedback_eval_15min_infra_check.md`, every active eval gets a 15-minute infra cadence:
- vLLM endpoint responds (check `vllm_endpoint.json` → `endpoint_url`).
- Pinggy tunnel alive (`curl -sSL https://<pair>.a.pinggy.link/health`).
- Daytona reachable (look for `DaytonaValidationError` or `Bearer token is invalid` in trial logs).
- Trial throughput accumulating (`find $TRIALS -maxdepth 2 -name result.json | wc -l` rising).

### Aggregating results

```bash
P=experiments/<run_dir>/trace_jobs/<run_tag>
find $P -maxdepth 1 -mindepth 1 -type d | wc -l        # total trials launched
find $P -maxdepth 2 -name result.json | wc -l           # trials completed
python3 -c "
import json, glob
n = ok = 0
for r in glob.glob('$P/*/result.json'):
    d = json.load(open(r))
    n += 1
    if (d.get('verifier_result') or {}).get('rewards', {}).get('reward', 0) > 0: ok += 1
print(f'pass: {ok}/{n} = {100*ok/max(n,1):.1f}%')
"
```

### Failure-mode notes (from prior iterations)

- **Wrong harness (OpenHands instead of SWE-agent)** → 0% pass; the model emits `<tool_call>{...}</tool_call>` JSON, but the OpenHands harness expects `<function=NAME>...</function>` XML and parses nothing.
- **Default `max_requeries: 3`** → trials forfeit prematurely with the format/blocklist/bash exit even though the agent was producing valid actions on most turns.
- **`DaytonaValidationError` flood** → wrong daytona key; switch to `DAYTONA_BASE_API_KEY`.
- **`SummarizationTimeoutError` on every trial** → wrong harness (`terminus-2` instead of `swe-agent`).
- **Pinggy bank conflict** → if the same persistent URL is in use by another concurrent eval, vLLM endpoint binds but the agent can't reach it. Use a free pair (verify with `find experiments -name vllm_endpoint.json | xargs grep pinggy`).

---

## Files in this directory

- `README.md` — this doc.
- `configs/template_qwen3_8b_sera_v3.yaml` — legacy v3 axolotl template (kept for reference only; deprecated).
- `configs/template_qwen3_8b_sera_v4.yaml` — current v4 axolotl training config template.
- `sbatch/axolotl_sera_v3.sbatch` — legacy v3 SLURM template.
- `sbatch/axolotl_sera_v4.sbatch` — current v4 SLURM template.
- `subset_sera_v3.py` — legacy v3 subsetter (deprecated).
- `convert_axolotl_checkpoint.py` — FSDP-prefix stripper (local copy; canonical lives at `/e/scratch/jureap59/feuer1/code/axolotl/` on Jupiter).
- `zero2_no_offload.json` — custom DeepSpeed config exploring zero2 without CPU offload (superseded by `zero3_bf16.json`, kept for reference).

Subsetter (v4) lives at `/Users/benjaminfeuer/Documents/scripts_dataset_build/subset_sera_v4.py` alongside other data-build scripts.
