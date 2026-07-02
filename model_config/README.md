# model_config/ — unified per-model vLLM serve config

The **single source of truth** for "what vLLM parameters does model X require." Replaces
the two divergent sources (the eval-listener mono-file `eval/configs/model_configs.yaml`
and the datagen per-model files under `hpc/datagen_yaml/`) with one file-per-model home.

## Layout

```
model_config/
├── _patterns.yaml          # regex fallbacks (size-inference defaults) for models with no file
├── <org>/<model-slug>.yaml # one file per model (e.g. Qwen/Qwen3-32B.yaml)
└── resolver.py             # the resolution function all entrypoints call
```

## File schema (layered)

```yaml
model: Qwen/Qwen3-32B              # canonical HF id (for matching)
# --- base intrinsics (small: the constants, same regardless of who's serving) ---
trust_remote_code: true
max_model_len: 32768
tool_call_parser: hermes           # absent on Pattern-D models (deliberate)
reasoning_parser: qwen3
hf_overrides: ...
limit_mm_per_prompt: ...
# --- subsystem overlays (eval / datagen / iris): the workflow-specific stuff ---
subsystems:
  eval:                           # the eval listener / Iris
    tensor_parallel_size: 2       # (parity geometry)
    swap_space: 32
    agent_kwargs: [extra_body={...enable_thinking:true}]
    extra_args: --enable-prefix-caching
    variants:                     # hardware-geometry overrides (merged last, win)
      gh200: {tensor_parallel_size: 1}
      gh200-65k: {tensor_parallel_size: 1}
  datagen:                        # trace generation (later migration)
    extra_args: --dtype bfloat16 ...
    swap_space: 12
```

**Resolution merge order (later wins):** base intrinsics → subsystem(s) → hardware variant.

A field goes to **base** only if it's model-intrinsic (the same regardless of subsystem).
Intentional divergences (e.g. GLM-4.7 Pattern D omits `tool_call_parser` for eval; datagen
needs it) live under the relevant `subsystems:` overlay — not in base.

## API

```python
from model_config import resolve_model_config

# "What vLLM params does Qwen3-32B need for an eval run on a multi-GPU node?"
cfg = resolve_model_config("Qwen/Qwen3-32B", subsystem="eval", hardware="multi_gpu")
# -> {trust_remote_code: true, max_model_len: 32768, tool_call_parser: hermes,
#     tensor_parallel_size: 2, swap_space: 32, ...}

# Stack multiple subsystems (named overrides, later wins):
cfg = resolve_model_config("QuantTrio/GLM-4.7-AWQ", subsystem="eval",
                           subsystems=["eval", "pattern_d"])
```

## Migration status

- ✅ **Eval registry** (`eval/configs/model_configs.yaml`, 52 entries) → migrated to 44
  per-model files + `_patterns.yaml` (lossless; the mono-file is retained as the listener's
  input until the listener is wired to the resolver — a careful follow-up that needs the
  old profile-exclusion semantics handled).
- ⏳ **Datagen YAMLs** (`hpc/datagen_yaml/`, 167 files) → not yet migrated; stale files
  to be retired first. Later stage.
- ⏳ **Iris wiring** → `resolve_model_config` available; Iris integration is the next step.
- ⏳ **Listener wiring** → reads the mono-file for now; migrating to the resolver directly
  is a follow-up (the resolver's "always-apply intrinsics" behavior is an improvement over
  the old registry's profile-exclusion quirk, but it's a behavioral change to gate carefully).

## Adding a new model

Create `model_config/<org>/<slug>.yaml` with the base intrinsics + an `eval` subsystem
overlay. The slug is the model name with non-`[\w.\-]` chars replaced by `_`.
