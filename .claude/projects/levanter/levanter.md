# Levanter — facts & how experiments are specced

JAX foundation-model training library (the engine under marin's pretrain/midtrain/SFT steps).
Learned 2026-06-29 while pulling Delphi midtraining configs. Source of truth = the marin monorepo
checkout `/Users/benjaminfeuer/Documents/marin/lib/levanter` (published to PyPI as `marin-levanter`).
See also `marin-executor` (how it's launched) and `mum` (how to find run configs). Grug is **not** a
Levanter frontend — see `[grug]` notes / `marin/.agents/projects/grugformer.md`.

---

## What it is (correct the "SFT engine" misconception)

- A **full** LLM/foundation-model training library — **pretraining, midtraining (CPT), SFT, eval,
  export** — not specifically an SFT engine. SFT is a *mode* of `train_lm`, not a separate engine.
- Stack: **Haliax** named tensors (`NamedArray` + explicit `Axis`) over **Equinox** modules;
  **draccus** dataclass configs; `AsyncDataset` data pipeline. Runs **TPU + GPU**.
- **Bitwise deterministic** (same config → same result across preempt/resume).

## How experiments are specced — a draccus dataclass tree, one per entry point

- Entry points live in `lib/levanter/src/levanter/main/`: `train_lm.py`, `train_dpo.py`, `lora_lm.py`,
  `eval_lm.py`, `export_lm_to_hf.py`, `sample_lm.py`, `train_asr.py`, … Each has a root config dataclass.
- `train_lm` → **`TrainLmConfig`** (`src/levanter/main/train_lm.py:42`), composed of nested sub-configs:
  - `data: LmDataConfig` — tokenizer, `cache_dir`, `components` (per-component `source` = url|hf + `format`
    = chat for SFT), **`train_weights`** (the mixture map — this is the `pNNmNN` Delphi mix), `shuffle`.
  - `trainer: TrainerConfig` — `train_batch_size`, `num_train_steps`, `mp` (precision e.g. `p=f32,c=bfloat16`),
    `per_device_parallelism`, `tracker` (wandb).
  - `model: LmConfig` (default `LlamaConfig`) — `type: llama`, dims, heads, `rope`, etc.
  - `optimizer: OptimizerConfig` (default `AdamConfig`) — `learning_rate`, `weight_decay`, `warmup`, schedule.
  - `train_seq_len`, `initialize_from_hf` / `initialize_from_checkpoint_path` (CPT/SFT/midtrain **init**),
    `hf_save_path`/`hf_upload`, `eval_harness`, `adapter` (LoRA).
- Polymorphism: a **`type:` field** selects the variant (model `type: llama`, optimizer/tracker/data-source
  types) — draccus discriminated unions.

## Authoring + running it (the raw-Levanter path)

- A **YAML** populates the tree (`lib/levanter/config/*.yaml` — e.g. `llama3_small_fast.yaml`,
  `train_lm_config` shapes; SFT example `train_lm_llama3_tulu_sft.yaml` uses `train_weights: {tulu: 1.0}` +
  `format: {type: chat}`).
- Run via the draccus entry (`levanter.config.main(main)()` at `train_lm.py:428`):
  `python -m levanter.main.train_lm --config_path config/<f>.yaml --trainer.num_train_steps 5000` (any field
  CLI-overridable with dotted keys).
- **SFT / midtraining are just `train_lm` with a different data block + init** — SFT = chat-format component +
  `initialize_from_hf`; midtrain/CPT = a `train_weights` mixture + `initialize_from_{hf,checkpoint_path}`.

## Where the RESOLVED config is persisted (this is how to recover what a run actually used)

A finished run writes its full resolved config to its GCS run dir — **but the filename depends on the launch path**:
- **Script/midtrain-launch** → `gs://…/checkpoints/<run>/train_lm_config.yaml` (+ a run manifest, e.g.
  `midtrain_manifest.json`).
- **marin-executor launch** → no `train_lm_config.yaml`; the config is inside `.executor_info` under
  `config.train_config` (`jq '.config.train_config'`). See the `marin-executor` doc.
This split is why the same experiment family can have two artifact shapes (and why `mum run` finds some configs
but not others — see `mum`).

## Worked example (Delphi K=0.20 midtraining = TrainLmConfig instances)
`data.train_weights` = the pNNmNN mixture (math `nemotron_cc_math_v1/4plus` + Nemotron-CC web replay);
`initialize_from_hf` = the matching base; optimizer = Muon dual-LR (`learning_rate` matrix + `adam_lr`),
`lr_schedule: linear`, `warmup: 0.1`, `min_lr_ratio: 0`; tokenizer `meta-llama/Meta-Llama-3.1-8B`, seq_len 4096,
`block_cross_document_attention: true`. Configs pulled to `~/Documents/experiments/active/midtrain-25B/configs/`.

## Checkpoint & resume gotchas (production — `delphi_1e22`, 2026-07-11)

- **⚠ RESUME-ONLY OOM = BFC-allocator fragmentation → fix with the `cuda_async` DEFRAG allocator.** A large-model
  run trains fine from step 0 but, on RESUME, semi-deterministically OOMs at (or a few thousand steps after) the
  first post-resume step: `RESOURCE_EXHAUSTED: … allocate <N>GiB [executable_name='jit__train_step']`, while the BFC
  free-map shows plenty of FREE HBM but **no contiguous hole**. Cause: on resume, tensorstore materializes params +
  Adam μ/ν on-device in a layout differing from the compiled step's, and the default **BFC allocator can't compact**
  → the step's large transient contiguous block can't be placed against the load-fragmented heap (steady-state
  in-place reuse never has to). Fragmentation, not a real shortage — marin issue **#7115** (proper upstream fix =
  buffer donation across the resume boundary). **FIX (allocator plumbing only, math-neutral): switch to `cuda_async`**
  — `JAX_PJRT_CLIENT_CREATE_OPTIONS=allocator:cuda_async` (env: `SINGULARITYENV_…`; + mirror in the executor's
  `trainer.jax_config` as `"jax_pjrt_client_create_options":"allocator:cuda_async"`, re-listing the DEFAULT_JAX_CONFIG
  keys since supplying `jax_config` REPLACES them). **⚠ ALLOCATOR-ENV TRAP (version-specific):** on jax/jaxlib
  **0.10.1 + `jax_cuda13`** the allocator comes from the PJRT create-options dict, populated ONLY by
  `JAX_PJRT_CLIENT_CREATE_OPTIONS` — so **`XLA_PYTHON_CLIENT_ALLOCATOR=platform` is INERT** (older-JAX flag), and
  `TF_GPU_ALLOCATOR=cuda_malloc_async` is inert too (a TF var). Verify via `device.memory_stats()['pool_bytes']` —
  **None = cuda_async (good), numeric = still BFC**. **Do NOT also set `XLA_PYTHON_CLIENT_MEM_FRACTION`** with
  cuda_async (0.90 shrinks out-of-pool headroom to ~6.5 GB — risky for NCCL/cublas; the 0.75 default leaves ~16 GB).

- **⚠ Wall-SIGKILL mid-save orphans a metadata-only staging checkpoint that POISONS resume discovery.** If SIGKILL'd
  at the wall *during* a save, Levanter leaves a ghost `step-<N>/` in the atomic-write staging mirror
  `<exp>/tmp/checkpoints-temp/…/checkpoints/step-<N>/` with **only `metadata.json` (~80 B)** — no `d/`, no
  `manifest.ocdbt`, no tensors (post-save cleanup + atomic rename never ran). Discovery scans INTO the staging
  mirror, sees the ghost `> ` the last complete step, picks it → **all ranks `FileNotFoundError` on every tensor
  leaf in <33s, 0 steps**, every afterany resume identical. The crash-teardown `grpc … UNAVAILABLE … Connection
  refused` is NOISE, not the cause (`jax.distributed.initialize` actually succeeded — ranks jointly log "Discovered
  latest checkpoint at step-<ghost>"; don't chase a coordinator red herring). **Fix:** `rm -rf
  <exp>/tmp/checkpoints-temp` (the real `checkpoints/step-*` tree is untouched) → discovery falls back to the last
  complete step (verify the ghost is metadata-only + the fallback has `d/`+`manifest.ocdbt` first). **Durable guard:**
  `rm -rf "$OUT/tmp/checkpoints-temp"` before `srun` so every relaunch self-cleans.

- **Save cadence vs mean-time-to-failure:** `keep: [{every: N}]` retains only every-N permanent keeps; the 30m
  rolling "latest" is NOT reliably rediscovered after an abnormal (wedge/OOM-hang) termination → resume lands on the
  last every-N keep. On a flaky cluster where the job dies BETWEEN keeps, it re-resumes the same keep = a
  no-progress loop. Make `keep` frequent enough to bank progress inside the mean-time-to-failure.

(Full write-ups: `agent_logs/2026-07-10_1e22_dod_resume_chain_jax_coordinator_death.md`.)
