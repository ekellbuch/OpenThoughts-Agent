---
name: sft-launch
description: >-
  Launch SFT via `python -m hpc.launch --job_type sft` on any cluster (JSC Jupiter GH200, CINECA
  Leonardo A100, TACC Vista GH200), with EITHER backend — LLaMA-Factory (default) or axolotl
  (`--sft_backend axolotl`) — including Delphi tool-calling models (delphi template, tokenizer prep,
  jinja-as-ground-truth masking). This skill is the cluster-AGNOSTIC core (backend choice, Delphi
  handling, config maps, node-scaling, dataset mixing, cleanup recognition, common traps). Per-cluster
  particulars (preamble, paths, QOS/wall, sbatch patches, no-internet handling, HF-upload mechanics)
  live in `.claude/ops/<cluster>/ops.md §SFT`. Use when asked to SFT / launch a finetune / train a
  model on Jupiter, Leonardo, or TACC. Reference: notes/ot-agent/sft_experiments.md, CLAUDE.md.
---

# sft-launch

> **⚠ Local clone = ground truth (CLAUDE.md §Always).** ALL code/config/sbatch edits
> go in the local Mac checkout (`~/Documents/OpenThoughts-Agent`) → commit → push →
> `git pull` on the cluster. **NEVER** hand-edit, `git commit`, or leave divergent/
> untracked changes on a cluster; no patch-by-rsync. New/changed configs are authored
> locally + synced, never on the cluster. Bake this into every subagent you dispatch.

SFT runs through **`python -m hpc.launch --job_type sft`** — one entrypoint, both
backends, all clusters. What changes per cluster is the *preamble* (ssh, submodule
pulls, conda env, dotenv, WORKDIR guard), paths, QOS/wall, and upload/cleanup
mechanics — all in **`.claude/ops/<cluster>/ops.md §SFT`** (read it before
launching). This doc is the cluster-agnostic decision layer.

## 1. Pick backend, cluster, env — then launch

| | LLaMA-Factory (default) | axolotl (`--sft_backend axolotl`) |
|---|---|---|
| When | Everything today; the validated production path | Delphi jinja-as-ground-truth SFT; when you want axolotl's template/plugin stack |
| Launcher runs | `accelerate` + DeepSpeed ZeRO-3 (multi-node) / torchrun | `-m axolotl.cli.train` |
| Conda env | `otagent` (or `sft-qwen35` for Qwen3.5 hybrid arch) | `sft-axolotl` (`--conda_env sft-axolotl`) |
| Flag-off contract | — | `--sft_backend llamafactory` (default) is **byte-identical** to before the backend existed |

Universal launch shape (fill per cluster from `ops/<cluster>/ops.md §SFT`):
```bash
python -m hpc.launch --job_type sft [--sft_backend axolotl] \
  --train_config_path sft/<lf_configs|axolotl_configs>/<cfg>.yaml \
  --num_nodes N --gpus_per_node <4|1> --time_limit <cluster max> \
  --dataset <hf-dataset> --role_tag role --user_tag user --assistant_tag assistant --content_tag content \
  --hub_model_id laion/<name> [--conda_env <env>]
```
**Always `--dry_run` the first cell** and eyeball the rendered
`<exp>/configs/*_train_config.yaml` (model, template, epochs, LR, role tags,
`push_to_hub`, `output_dir`) before the real submit. A wrong template silently
trains garbage (§5); cluster WORKDIR/DCFT guards hard-fail fast if the preamble was
skipped.

## 2. Backend: axolotl specifics

- **aarch64 clusters (TACC Vista / Jupiter GH200): use SDPA.** No flash-attn-2
  wheel for torch 2.11+cu128 on aarch64 → the config must set
  `attn_implementation: sdpa` (translator defaults to fa2). Also
  `pip install --no-deps torchao==0.17.0` into `sft-axolotl` (axolotl excludes
  torchao on aarch64 but `qat.py` imports it → `builders` import fails; the
  pure-python wheel fixes it).
- **⚠ The launcher REBUILDS the dataset block from flags.** `hpc.launch` does NOT
  honor a config's hand-authored `datasets:` block — `_build_datasets`
  (`hpc/axolotl_config_utils.py`) rebuilds it from `--dataset` +
  `--messages/--role_tag/--content_tag` (defaults to sharegpt
  `conversations`/`from`/`value`). **Pass those flags at launch;** the in-config
  `datasets:` block is honored ONLY by direct `axolotl.cli.preprocess` (the
  mask-dump path).
- **On internet-node clusters, set `WANDB_MODE=disabled`** — the launcher sets
  `report_to=wandb`; wandb 0.28.x crashes on the compute-node service socket
  (`WANDB_MODE=disabled` makes it a no-op; loss still logs to `trainer_state.json`).
- **Precision:** `pure_bf16: true` (fp32-master OOMs an 8B on 96 GiB).
- Validated on TACC Vista (Stage 3 smoke, Stage 4 footgun-through-launcher, delphi
  masking canary). Full backend gotcha list → `.claude/projects/axolotl/axolotl.md`.

## 3. Delphi model SFT (both backends)

Delphi checkpoints are the **Llama-3 tokenizer** (vocab 128256, NO chat_template,
NO ChatML) PLUS reasoning/tool tokens. Two hard prerequisites, same for LF and
axolotl:
1. **Prep the tokenizer FIRST** (single-token delphi specials):
   `python sft/delphi/prepare_delphi_tokenizer.py --model <ckpt> --output <dir>`
   (reserved-slot rename + mean-init → `<|start_think|>`/`<|end_think|>`/
   `<|tool_call|>`/`<|tool_result|>` become single tokens). Launch with
   `--model_path <dir>`.
2. **Use a Llama-3-family template, NEVER `qwen3`.** The `delphi` template =
   Llama-3 header/turn format (`<|start_header_id|>…<|eot_id|>`, EOS `<|eot_id|>`)
   + the reasoning/tool tokens. `qwen3` (ChatML `<|im_start|>`) would shred every
   example. **This template×tokenizer mismatch is the #1 silent ruin** — `--dry_run`
   + eyeball the first rendered example of EACH source (an instruction turn AND a
   `<think>` warmup example) before launching.

- **Datasets** registered in `sft/delphi/dataset_info.json` (per-dataset schema
  tags — the instruction sets use heterogeneous ShareGPT schemas). Launch with
  `--dataset_dir sft/delphi` and the 90/10 mix:
  `--dataset <instr>,delphi_warmup --mix_strategy interleave_under --interleave_probs 0.9,0.1`.
- **Axolotl delphi path = jinja-as-ground-truth (train == serve).** Config:
  `chat_template: delphi` + `tokenizer_save_jinja_files: false` + the
  `template_integrity` plugin (embeds the chat_template into `tokenizer_config.json`,
  covering the per-checkpoint dirs the flag ignores). **Validate the loss mask** with
  `axolotl.cli.preprocess <cfg> --debug` (assistant + `<|start_think|>…<|end_think|>`
  trained, user/system masked, 0 `Last turn is not trainable` skips). Canary:
  `sft/axolotl_configs/delphi_canary.yaml` (validated, TACC job 802053).
- **The Delphi RL-scaling-laws grid (#6279) is HF-upload ONLY** —
  `enable_db_registration: false`; do NOT run `manual_db_push.py`. LR = shared
  conventional SFT LR (2e-5) across all cells for comparability.

## 4. Config maps (cluster-agnostic)

- **Qwen3-8B** (`sft/lf_configs/qwen3/` + `…/extra/`): `32k_base.yaml` (default 32k
  thinking), `32k_base_nothink.yaml`, `131k_base.yaml`; `extra/32k_base_bs96.yaml`
  (node-scaling, §4b), `extra/32k_base_bs96_opt1k.yaml` (small <1k-row: 7ep/lr4e-5),
  `extra/32k_base_bs96_opt100k.yaml` (large ≈11k+: 5ep/lr4e-5), plus other
  sizes/coder.
- **Qwen3-32B** (`…/32k_base_32b*.yaml`): DeepSpeed ZeRO-3, writes **sharded
  `global_stepN/`, NOT root safetensors** → **launch WITHOUT `--hub_model_id`**,
  then consolidate → upload (§6 + `ops/<cluster>/ops.md §SFT`).
- **Qwen3.5 hybrid (9B/27B)** (`sft/lf_configs/qwen3_5/*.yaml`): GDN+Attention arch
  not in transformers 4.x → needs the **`sft-qwen35`** env (transformers ≥5.3) +
  `DISABLE_VERSION_CHECK=1`. 9B → root safetensors (SKIP consolidate, like 8B); 27B
  → 32B consolidate flow. Copy `preprocessor_config.json` from base into the ckpt
  before upload (LF doesn't emit it; vLLM needs it).
- **axolotl** (`sft/axolotl_configs/`): `smoke.yaml`, `parity_llama3.yaml`,
  `delphi_canary.yaml`, `marin/delphi_all3.yaml` (all-3-plugins). aarch64 → SDPA.

### 4b. Node-scaling — the `bs96` configs
`bs96` configs fix `global_batch_size: 96` and auto-derive
`gradient_accumulation_steps = 96 / (num_nodes*gpus)`. **Adding nodes shrinks
grad-accum → faster wall-clock at a fixed effective batch.** `--num_nodes 4`(16
GPU)→accum 6; `--num_nodes 8`(32)→accum 3. Keep `96 % (num_nodes*gpus_per_node) == 0`.

## 5. Dataset mixing & the parse-tags rule

- `--dataset` is **repeatable**. Concatenate: `--dataset A --dataset B
  --mix_strategy concat`. Interleave: `--mix_strategy interleave_under|interleave_over
  --interleave_probs 0.7,0.3` (weights in dataset order).
- **Role tags are MANDATORY for Harbor/DCAgent datasets** (`role`/`content` +
  `user`/`assistant`): `--role_tag role --user_tag user --assistant_tag assistant
  --content_tag content`. LF defaults to `from`/`value`+`human`/`gpt`; without the
  right tags the thinking preprocessor finds **0 assistant messages → garbage
  training.** Older ShareGPT sets use `--role_tag from --user_tag human
  --assistant_tag gpt --content_tag value`.
- **Mixed-schema MIXES:** a single global `--role_tag` silently yields 0 assistant
  turns on the mismatched source. Register per-dataset `columns`/`tags` in a
  `dataset_info.json` and launch with `--dataset_dir <registry>` (how the Delphi mix
  works — §3).

## 6. Cleanup — recognize the path, then follow the cluster's §SFT

After training, check the checkpoint root:
`ls $CHECKPOINTS_DIR/<job>/ | grep -E 'safetensors|global_step'`:
- `model-*.safetensors` at root → **8B path** (also Qwen3.5-9B): drop intermediate
  `checkpoint-*` + `.cache`, upload, DB-register.
- `global_stepN/` + `zero_to_fp32.py`, no root safetensors → **32B path** (ZeRO-3
  shards): **consolidate first** (`--job_type consolidate`), then upload from
  `final_repo/`.

**DB registration is a MANUAL cleanup step** via
`scripts/database/manual_db_push.py` (the `--upload_to_database` launch flag is
EVAL-only — a no-op for SFT). HF uploads default **PUBLIC** to `laion/`.
**Per-series no-DB exception:** HF-upload-only series (e.g. Delphi #6279,
`enable_db_registration: false`) SKIP `manual_db_push.py`. The *mechanics* (which
node uploads, tunnels, cert) are cluster-specific → `ops/<cluster>/ops.md §SFT`.
Live status: tail the `.out` for `{'loss':…, 'grad_norm':…}` step lines
(`trainer_log.jsonl` is unreliable mid-run).

## 7. Common traps (all clusters)

- **`AF_UNIX path too long` at dataset tokenization** — the HF-datasets
  `SyncManager` binds a socket under `$TMPDIR` (108-byte `sun_path` cap). The
  launcher redirects TMPDIR to a short `/tmp/sft_<job>` for BOTH backends; if you
  still see it: confirm the rendered sbatch's `_TMPROOT`/`TMPDIR` is short, or
  `export SFT_KEEP_TMPDIR_LOCAL=1` before launch.
- **`overwrite_output_dir` rejected by HfArgumentParser (transformers v5)** — the
  launcher strips this launcher-only key from the LF config before write.
  `grep -c overwrite_output_dir <exp>/configs/*_train_config.yaml` must print `0`.
  The `--overwrite_output_dir true` CLI flag still works (⊥ `--max_restarts`).
- **Multi-node "24h timeout" that never checkpointed** — usually the per-node
  HF-datasets cache RACE, not slow tokenization (~65s). Fix: a config with
  `data_shared_file_system: true` (global barrier; same tokens/loss). Diagnose in
  order: dsfs → schema-key `KeyError` → only then suspect genuinely-slow
  tokenization (`--pretokenize`). Details in `ops/leonardo/ops.md §SFT`.
- **axolotl multi-node bring-up dies with `OSError [Errno 37] No locks available`
  (ENOLCK) / `[Errno 116] Stale file handle` (ESTALE) during dataset load**
  (masquerades as a `C10d RendezvousConnectionError` in the log tail — that's
  teardown noise; the real error is upstream, `datasets/builder.py:821` FileLock
  and/or the axolotl `FileLockLoader` at `utils/data/lock.py`). `data_shared_file_system:true`
  does NOT save axolotl (axolotl's lock.py always locks). Fixes:
  - **Interim (big-node-local-/tmp clusters, e.g. TACC Vista gh=261G): route ALL
    per-rank WRITE caches node-local** — `export SFT_KEEP_TMPDIR_LOCAL=1` (the
    sbatch write-cache guard points `HF_DATASETS_CACHE`/`TRITON_CACHE_DIR`/
    `TORCHINDUCTOR`/`RAY`/`TMPDIR`/`XDG` at `/tmp/otsft_$JOBID`, per-node) +
    `dataset_prepared_path: /tmp/...` in the axolotl config; keep `HF_HUB_CACHE`
    shared+populated (pre-download once) so node-local arrow builds read cached
    parquet (each rank builds uncontended).
  - **Durable (portable, incl. small-/tmp clusters): pretokenize-once into a SHARED
    `dataset_prepared_path` + a lock-free persistent-sentinel fast-path in axolotl
    `lock.py`**. Full saga: `agent_logs/2026-07-08_sft-815251-c10d-rendezvous-fail.md`.
- **Template × tokenizer mismatch** — §3; the top silent ruin for
  delphi/Llama-3-family models.

## 8. Per-cluster particulars — READ before launching

| Cluster | Env | Wall | ops §SFT |
|---|---|---|---|
| **JSC Jupiter** (GH200, 4/node, aarch64) | `otagent` / `sft-qwen35` / `sft-axolotl` | 12h booster (`11:59:00`) | `.claude/ops/jupiter/ops.md §SFT` |
| **CINECA Leonardo** (A100-64GB, 4/node, no-internet-compute) | `otagent` / `sft-qwen35` | 24h (`23:59:00`) | `.claude/ops/leonardo/ops.md §SFT` |
| **TACC Vista** (GH200, aarch64) — axolotl path | `sft-axolotl` / `otagent` | per-partition | `.claude/ops/tacc/ops.md` |

Each `ops §SFT` has the exact preamble (ssh, submodule updates incl. `sft/axolotl`
pinned WITHOUT `--remote`, conda env, dotenv, WORKDIR/DCFT guard), the
checkpoint/scratch write paths, QOS/account rules, the sbatch post-patch (if any),
no-internet handling, and the HF-upload mechanics. **Never launch without reading it.**
