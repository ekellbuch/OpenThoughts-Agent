"""Regression tests for ``hpc/launch_utils.py:setup_experiments_dir``.

Specifically covers the collision-rename path. The bug this guards
against is documented in
``notes/ot-agent/agent_logs/2026-05-26_launcher_trials_dir_collision_bug.md``:
when the experiments dir already exists with a different run's
configs, the launcher appends ``_N`` to land at a fresh dir, but
prior to this fix the renamed path was NOT propagated back into
``exp_args["experiments_dir"]``. Downstream consumers
(``hpc/rl_config_utils.py`` deriving trainer.trials_dir / ckpt_path
/ export_path; ``hpc/launch.py`` deriving wandb_dir) then used the
un-suffixed canonical path, causing concurrent renamed chains to
share the same inner trainer dirs → data corruption / write-race
hazards.

Run from the OT-Agent repo root with:
    .venv/bin/python -m pytest tests/hpc/test_launch_utils_collision.py -v
"""

from __future__ import annotations

import json
from pathlib import Path


from hpc.launch_utils import setup_experiments_dir


def _seed_existing_experiment(experiments_root: Path) -> None:
    """Seed ``experiments_root/configs/<file>.json`` so collision detection fires."""
    configs = experiments_root / "configs"
    configs.mkdir(parents=True, exist_ok=True)
    (configs / "prior_run_config.json").write_text(json.dumps({"job_name": "prior"}))


def test_no_collision_leaves_path_unchanged(tmp_path: Path) -> None:
    """When no prior dir exists, exp_args["experiments_dir"] stays at canonical form."""
    target = tmp_path / "ot-baf" / "myjob"
    exp_args = {"experiments_dir": str(target), "job_type": "rl"}

    paths = setup_experiments_dir(exp_args, job_name="myjob")

    assert paths.root == target
    assert exp_args["experiments_dir"] == str(target)
    assert paths.configs == target / "configs"
    assert paths.sbatch == target / "sbatch"
    assert paths.logs == target / "logs"


def test_collision_propagates_renamed_path_to_exp_args(tmp_path: Path) -> None:
    """A real collision must bump exp_args["experiments_dir"] to the _2 form.

    This is the core regression: prior to the fix, exp_args was NOT
    updated, so downstream derivations like trainer.trials_dir read
    the un-suffixed canonical path while sbatch/configs/logs lived
    at <name>_2.
    """
    canonical = tmp_path / "ot-baf" / "myjob"
    _seed_existing_experiment(canonical)

    exp_args = {"experiments_dir": str(canonical), "job_type": "rl"}
    paths = setup_experiments_dir(exp_args, job_name="myjob")

    expected_renamed = tmp_path / "ot-baf" / "myjob_2"
    assert paths.root == expected_renamed
    assert exp_args["experiments_dir"] == str(expected_renamed)
    # Inner artifact subdirs must also land at the renamed root.
    assert paths.configs == expected_renamed / "configs"
    assert paths.sbatch == expected_renamed / "sbatch"
    assert paths.logs == expected_renamed / "logs"


def test_collision_chain_increments_to_3(tmp_path: Path) -> None:
    """Two prior dirs (_canonical_ and ``_2``) force a bump to ``_3``."""
    canonical = tmp_path / "ot-baf" / "myjob"
    _seed_existing_experiment(canonical)
    _seed_existing_experiment(tmp_path / "ot-baf" / "myjob_2")

    exp_args = {"experiments_dir": str(canonical), "job_type": "rl"}
    paths = setup_experiments_dir(exp_args, job_name="myjob")

    expected_renamed = tmp_path / "ot-baf" / "myjob_3"
    assert paths.root == expected_renamed
    assert exp_args["experiments_dir"] == str(expected_renamed)


def test_renamed_path_feeds_downstream_derivations(tmp_path: Path) -> None:
    """End-to-end: after a collision rename, the rl_config_utils-style
    derivation ``f"{experiments_dir}/{job_name}/trace_jobs"`` MUST land
    inside the renamed dir, not the canonical one.

    This is the exact pattern in
    ``hpc/rl_config_utils.py:build_skyrl_hydra_args`` for
    trials_dir / ckpt_path / export_path. The bug report's evidence
    showed trials_dir pointing at the un-suffixed canonical dir
    (``<D>/<job_name>/<job_name>/trace_jobs``) while
    experiments_dir in the same config JSON was the ``_3`` form.
    """
    job_name = "myjob"
    canonical = tmp_path / "ot-baf" / job_name
    _seed_existing_experiment(canonical)

    exp_args = {"experiments_dir": str(canonical), "job_type": "rl"}
    setup_experiments_dir(exp_args, job_name=job_name)

    # Replay rl_config_utils.build_skyrl_hydra_args' path derivation.
    experiments_dir_after = exp_args["experiments_dir"]
    derived_trials = f"{experiments_dir_after}/{job_name}/trace_jobs"
    derived_ckpt = f"{experiments_dir_after}/{job_name}/checkpoints"
    derived_export = f"{experiments_dir_after}/{job_name}/exports"

    renamed_root = str(tmp_path / "ot-baf" / "myjob_2")
    assert derived_trials.startswith(renamed_root + "/")
    assert derived_ckpt.startswith(renamed_root + "/")
    assert derived_export.startswith(renamed_root + "/")


def test_disable_dedup_skips_rename_and_still_writes_back(tmp_path: Path) -> None:
    """When the resume manager disables dedup, no rename happens AND
    exp_args["experiments_dir"] is still set to the canonical absolute
    path (so callers can rely on the field shape unconditionally)."""
    canonical = tmp_path / "ot-baf" / "myjob"
    _seed_existing_experiment(canonical)

    exp_args = {"experiments_dir": str(canonical), "job_type": "datagen"}
    paths = setup_experiments_dir(exp_args, job_name="myjob", disable_dedup=True)

    assert paths.root == canonical
    assert exp_args["experiments_dir"] == str(canonical)
