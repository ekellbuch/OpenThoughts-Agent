# Scoping: fixed Daytona snapshot template (`--no-force-build`) for the Iris eval path

**Status:** scoping / not implemented. The SLURM eval path
(`eval/jupiter/eval_harbor.sbatch`) already supports this; the Iris eval launcher
(`eval/cloud/launch_eval_iris.py`) does not. This doc scopes porting it.

## The capability (what SLURM already does)

`eval/jupiter/eval_harbor.sbatch` (lines ~790-793): when `EVAL_SNAPSHOT_NAME` is set,
it appends to the harbor command:

```
--environment-kwarg snapshot_template_name=$EVAL_SNAPSHOT_NAME --no-force-build
```

i.e. **reuse one pre-existing named Daytona snapshot for every task, skip the per-task
build.** Both flags are real harbor CLI options (`harbor/src/harbor/cli/jobs.py`:
`--environment-kwarg` line 785, `--force-build/--no-force-build` line 707 → sets
`EnvironmentConfig.force_build`).

## Why it matters on Iris

Iris eval uses the harbor-config default `force_build: true` (`hpc/harbor_yaml/eval/*.yaml`)
→ harbor **force-builds a sandbox per task at runtime**. Correct for **per-task-env**
benchmarks (e.g. SWE-bench-verified: each task is a different repo image). **Wasteful for
single-shared-env benchmarks** (see eligibility table): force-build rebuilds the same image
N times — startup latency + Daytona-pool load for zero benefit (relevant to the SLURM path's
2000-active-sandbox ceiling; Iris has no such guard yet — see `iris_job_lifecycle.md`). A
fixed template builds it **once** and reuses it.

## Eligibility — only single-env benchmarks

| benchmark | per-task env? | fixed-template eligible? |
|---|---|---|
| swebench / swebench_full | yes (per-repo images) | **No** — needs force_build per task |
| terminal-bench (tb2), aider, bfcl, gaia, medagentbench, financeagent | no (one shared sandbox) | **Yes** |

This is a **per-benchmark** opt-in, not a global switch (see Tradeoffs/Footgun).

## Proposed Iris exposure

Two layers:

1. **Launcher flag** on `launch_eval_iris.py`: `--snapshot-template <name>`
   (mutually-informative with `--harbor_env daytona`). When set, the launcher
   injects into the harbor command (it already forwards `harbor_extra_args` →
   `build_harbor_command`):
   - `--environment-kwarg snapshot_template_name=<name>`
   - `--no-force-build`  (overrides the config's `force_build: true`)
   These ride the existing `--harbor_extra_arg` passthrough, same as the SLURM path.

2. **Preset field** (preferred, composes with the new `--preset` work):
   add an optional `snapshot_template: <name>` key to the eligible preset YAMLs
   in `eval/presets/`. Then `--preset tb2` (etc.) auto-applies it on BOTH the
   SLURM listener (via `EVAL_SNAPSHOT_NAME`) and Iris (via the flag above),
   keeping the two entrypoints' benchmark knowledge in one place. The
   `_apply_preset` mapping in `launch_eval_iris.py` would move `snapshot_template`
   from the "ignored" set into "applied". Leave it unset for swebench presets.

## Who builds the template + where

The named snapshot must exist in the **MAIN Daytona org (`DAYTONA_API_KEY`)** before
use (built once, reused). Options:
- A small one-time `harbor`/snapshot-build invocation per eligible benchmark, named
  by convention (e.g. `eval-tb2-base`, `eval-aider-base`).
- Document the build step + naming in `run-eval-iris` SKILL; treat a missing template
  as a hard error (fail fast, do not silently fall back to force-build).

## Tradeoffs / risks

- **Staleness:** the template is a snapshot in time. If the benchmark's base env
  changes (deps, harness version), the template must be rebuilt or evals run against
  a stale sandbox. Needs a rebuild/versioning discipline (bake a version into the
  template name).
- **Footgun:** applying a template to a per-task-env benchmark silently runs the
  wrong env. The preset-field approach contains this (only eligible presets set it);
  a raw `--snapshot-template` flag needs an operator who knows the benchmark is
  single-env.

## Open questions (confirm before implementing)

1. Exact harbor semantics of `snapshot_template_name` + `--no-force-build` together on the
   daytona backend at the current pin — does harbor pull the named snapshot and skip build
   entirely, or still verify/patch it? (Read the daytona environment impl, not just the CLI.)
2. Which "shared-env" benchmarks are *truly* single-env at the harbor task level (some agent
   benchmarks inject per-task files into a shared base — that's still fine; per-task *images*
   are not).
3. Naming + ownership of the templates in the MAIN org, and a rebuild trigger when the base
   env changes.
4. Whether to add a launch-time check that the named template exists in the org (fail fast) —
   recommended.

## Effort estimate

Small. The launcher flag + preset-field plumbing is ~1 day incl. the SKILL update and a dry-run
test; the larger work is the operational discipline (building/naming/versioning the templates)
and confirming Q1's harbor semantics.
