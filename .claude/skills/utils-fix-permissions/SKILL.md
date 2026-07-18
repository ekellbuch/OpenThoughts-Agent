---
name: utils-fix-permissions
description: >-
  Fix file permissions on a directory tree on an HPC cluster (Leonardo, Jupiter,
  TACC, etc.) so other users can read your shared data, conda envs, or work
  directories. Runs `scripts/permissions/fix_permissions.sh <dir>` over SSH on
  the target cluster — sets directories to 755, files to 644, then restores
  execute bits on bin/ entries, ELF binaries, and shebang scripts, and ensures
  all ancestor directories are traversable (o+x). Use when another user reports
  "Permission denied" reading your files, after creating a new conda env or
  shared data dir, or when a cross-user pipeline can't access your checkpoint/
  eval/trace outputs.
---

# utils-fix-permissions

Set safe shared-readable permissions on a directory tree on any HPC cluster.

## When to use

- Another user reports **"Permission denied"** reading your files / data / envs.
- After creating a new **conda env** or **shared data directory** that collaborators need to read.
- When a cross-user pipeline (eval, datagen, DB push) can't access your **checkpoint / eval / trace** outputs.
- After `rsync` / `scp` that preserved restrictive source permissions (600/700).

## What it does

`scripts/permissions/fix_permissions.sh <dir>` performs 5 passes:

| Pass | Action |
|---|---|
| 0 | Ensure all **ancestor directories** up to `/` are traversable (`o+x`) — only touches dirs you own |
| 1 | Set all **directories** to `755` (`rwxr-xr-x`) |
| 2 | Set all **files** to `644` (`rw-r--r--`) |
| 3 | Restore `755` on executables in **`bin/`** directories |
| 4 | Restore `755` on **ELF binaries** (detected by `\x7f ELF` magic header) |
| 5 | Restore `755` on **shell scripts with shebang** (`#!`) |

Files remain **owner-writable only** — collaborators get read + execute, not write.

## How to invoke

SSH to the target cluster and run the script from the repo root:

```bash
# Leonardo example
ssh Leonardo 'cd $WORK/OpenThoughts-Agent && bash scripts/permissions/fix_permissions.sh /leonardo_work/AIFAC_5C0_290/bfeuer00'

# TACC example (large tree — run on a compute node if login node is loaded)
ssh TACCVista 'cd $SCRATCH/OpenThoughts-Agent && bash scripts/permissions/fix_permissions.sh $SCRATCH/eval_jobs'

# Jupiter example
ssh Jupiter 'cd $DCFT && bash scripts/permissions/fix_permissions.sh /p/home/../../public'
```

### Large directories

For big trees (conda envs, HF cache, thousands of eval trial dirs), the `find` + `chmod` passes can take several minutes. On clusters with loaded login nodes (Leonardo, TACC), run via an **interactive compute node** instead:

```bash
# TACC
ssh TACCVista 'srun -p gh-dev -N 1 -n 1 -t 08:00:00 --account=CCR24067 bash -c "cd $SCRATCH/OpenThoughts-Agent && bash scripts/permissions/fix_permissions.sh $SCRATCH/miniconda3"'

# Leonardo (idev or sbatch)
ssh Leonardo 'srun -p boost --account=AIFAC_5C0_290 --time=08:00:00 -N 1 bash -c "cd $WORK/OpenThoughts-Agent && bash scripts/permissions/fix_permissions.sh $WORK"'
```

## Gotchas

- **Does NOT make files group-writable.** If collaborators need write access (shared editing), additionally run `chmod -R g+w <dir>` after.
- **Conda envs after `pip install`:** new files created by pip inherit the umask (often 700). Re-run the script after any install into a shared env.
- **NFS symlink chains:** the ancestor-traversal pass (pass 0) only fixes dirs **owned by you** — it won't touch system paths like `/leonardo_work` itself.
- **The script is safe to re-run** — it's idempotent (setting 755/644 on already-correct files is a no-op).
