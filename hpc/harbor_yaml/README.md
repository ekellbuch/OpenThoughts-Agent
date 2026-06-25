# `hpc/harbor_yaml/` — single source of truth for Harbor harness config YAMLs

This directory is **THE** authoritative home for every Harbor harness config YAML in the
repo — the configs consumed by the Harbor agentic harness via `harbor jobs start --config`
(eval) and the unified launcher / listener (datagen + eval). If you are looking for "which
config is authoritative," the answer is: **here.** No other location holds a *real* Harbor
config (the two documented exceptions below are not part of the central launch surface).

A **Harbor config** carries the harness schema (`orchestrator` / `environment` / `verifier` /
`agents` / `datasets` / `timeout_multiplier` …). It is NOT a vLLM serving config
(`baseline_model_configs*.yaml`, `api_model_configs.yaml`), a listener cluster-config
(`eval/clusters/*.yaml`), or a benchmark preset (`eval/presets/*.yaml`) — those are separate
layers and live elsewhere on purpose.

## Layout

```
hpc/harbor_yaml/
├── eval/                                   # agentic-EVAL Harbor configs
│   ├── dcagent_eval_defaults.yaml          # CANONICAL 8B-class default (timeout_multiplier 2.0)
│   ├── dcagent_eval_defaults_32b.yaml      # CANONICAL 32B-class default (timeout_multiplier 16.0)
│   ├── configs/                            # the SLURM listener/sbatch `--config-yaml` family
│   │   ├── dcagent_eval_config.yaml             # listener default (terminus-2)
│   │   ├── dcagent_eval_config_no_override.yaml # LIVE Leonardo + every preset's default
│   │   ├── dcagent_eval_config_swe_agent.yaml
│   │   ├── dcagent_eval_config_openhands{,_qwen3_coder,_toolcall}.yaml
│   │   ├── dcagent_eval_config_mini_swe_agent.yaml
│   │   └── dcagent_eval_config_aider_agent_nothink.yaml
│   ├── eval_ctx{32k,131k}.yaml             # named context-length eval harnesses
│   ├── eval_{mini_swe,openhands}_ctx*.yaml # installed-harness eval variants
│   ├── {openhands,swe_agent}_ctx32k_eval_.yaml
│   └── extra/                              # debug / API / 100-task / kira / yarn / modal variants
├── datagen/                                # non-agentic + API datagen Harbor configs
├── datagen_apptainer/  datagen_docker/  datagen_podman/   # runtime-specific trace-gen configs
```

## The size-selection rule (eval defaults)

The unified eval listener selects the canonical default **by model size**, so a normal
agentic eval needs no `--harbor-config` flag:

| model size (param count from HF name) | selected config                          | timeout multiplier |
|---|---|---|
| **8B-class** (≤ ~14B; 1.5B/7B/8B/14B)  | `eval/dcagent_eval_defaults.yaml`        | **2×** (in the file) |
| **32B-class** (~28–42B; incl. MoE 30b-a3b) | `eval/dcagent_eval_defaults_32b.yaml` | **16×** (in the file) |
| out-of-band / no size token in name    | base default `dcagent_eval_defaults.yaml` | 2× (+ logged note) |

The two files have an identical body; **only `timeout_multiplier` differs.** Keep them in
sync. The multiplier lives IN the file (not a CLI default). See
`.claude/skills/eval-agentic-launch/SKILL.md` §3b for the full policy.

## `eval/configs/` compat symlinks (the "two places" note)

`eval/configs/dcagent_eval_config*.yaml` are **relative symlinks** into
`hpc/harbor_yaml/eval/configs/` — committed as mode-`120000` git objects with relative
targets, so a cluster `git pull` reproduces them byte-for-byte. They exist because the live
SLURM path resolves `--config-yaml` basenames from `eval/configs/` first:
- `eval/leonardo/unified_eval_harbor.sbatch` (the LIVE Leonardo path) and the root
  `eval/unified_eval_harbor.sbatch` resolve `eval/configs/ → eval/<cluster>/ → eval/MBZ/`.
- All `eval/presets/*.yaml` forward `config_yaml: dcagent_eval_config_no_override.yaml`.
- The listener's `_resolve_agent_name_from_config_yaml()` checks `eval/configs/` too.

So `eval/configs/` is an **explicit compat shim, not a second source of truth** — every real
file lives here under `hpc/harbor_yaml/`, and the shim follows the symlink. Edit the config
**here**; never replace a symlink at `eval/configs/` with a real file.

> **Follow-up (needs a maintenance window):** removing the `eval/configs/` symlinks and
> repointing the two sbatch resolvers + the listener directly at
> `hpc/harbor_yaml/eval/configs/` is the cleaner end-state, but it is a code change to the
> live resolution path. A RUNNING Leonardo eval re-reads
> `eval/configs/dcagent_eval_config_no_override.yaml` on resume, so the symlinks must not be
> removed until no eval is in flight and the updated sbatch is on the cluster. Deferred.

## Documented exceptions (real Harbor configs that intentionally live elsewhere)

These are NOT part of the central launch surface and are kept in place on purpose:

- **`eval/tacc/dcagent_eval_config.yaml`** — a genuinely-distinct TACC per-cluster variant
  (`n_concurrent_trials: 4`, `max_retries: 100`, `force_build: true`, `jobs_dir: jobs`). It
  belongs to the self-contained `eval/tacc/` subsystem (its own listeners + sbatches) and is
  referenced by ~95 frozen experiment artifacts under `data/{ablation_experiments,sbatches,
  vllm_experiments}/` via the absolute `${DCAGENT_DIR}/eval/tacc/dcagent_eval_config.yaml`.
  Relocating it would churn those frozen records for no functional gain.
- **`data/nl2bash_sampled_verified/validation/job.yaml`** — a dataset-local validation
  fixture (a tiny `agents: [oracle]` local-docker config with relative `jobs_dir:
  ../local_runs/...`), used only by its sibling `validate.sh`. It travels with its dataset
  and is not a launch config.

## Deprecated configs — DO NOT resurrect

The `eval_ctx*_non_it*.yaml` / `ctx32k_non_it_16x_eval_.yaml` family was **deleted**
(commit `928698e0`). They carried the `penfever/temp-override`-era `mean-drop-ei` /
`accuracy-drop-ei` metrics that no Marin-branch Harbor `JobConfig` accepts — loading one
raises a `JobConfig` ValidationError. **Do not re-add `mean-drop-ei` / `accuracy-drop-ei`
metrics or recreate these configs.** Use the size-selected `dcagent_eval_defaults{,_32b}.yaml`.
