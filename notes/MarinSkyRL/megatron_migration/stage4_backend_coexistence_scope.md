# Stage 4 — Backend coexistence wiring (flag-off byte-identical) ★ (critical path)

- **Date:** 2026-07-13 · **Status:** `scoped GO`
- **Companion:** `megatron_migration_plan.md`

## Why this is the right next step
Before any megatron GPU run, the `strategy=megatron` selector must be reachable from OUR launcher/config stack while `strategy=fsdp` (default) stays **byte-identical**. Upstream already has `trainer.strategy` + `MegatronConfig` — this stage is about wiring it through our `rl/cloud/launch_rl_iris.py` + `hpc/skyrl_yaml/**` without perturbing the FSDP2 arms.

## Change-set
- OT-Agent `rl/cloud/launch_rl_iris.py`: a `--strategy {fsdp,megatron}` selector (default `fsdp`) that also selects the megatron image (Stage 2) when `megatron`; all-defaults launch injects nothing new → pod env byte-identical.
- A new `hpc/skyrl_yaml/iris/megatron/*.yaml` family carrying `trainer.strategy: megatron` + `MegatronConfig` (TP/PP/EP/ETP/CP) — separate files, so no existing FSDP2 YAML is touched.
- MarinSkyRL side (per Stage 1's chosen tree): ensure the megatron worker/config are importable only on the megatron path (no megatron/TE import on the fsdp path).

## Validation gate (GO / NO-GO)
- **GO (flag-off FIRST):** with `--strategy fsdp` (default), a rendered config + injected pod env are **byte-identical** to today for a representative FSDP2 arm (dense-8B + 30B-EP) — diff-clean; no megatron/TE import on that path. THEN a `strategy=megatron` config renders + type-checks and passes a 1-GPU init smoke.
- **NO-GO:** the megatron path leaks into the fsdp default (import side-effects, config-field bleed) → fix before advancing.
- **Cost:** CPU + a 1-GPU flag-off/render smoke.

## Composes with / depends on
Depends on Stages 1 (tree), 2 (image), 3 (vLLM). Gates Stages 5–9. Enforces the plan's flag-off byte-identical invariant.
