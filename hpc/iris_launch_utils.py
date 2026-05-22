"""IrisLauncher — base class for submitting OT-Agent jobs to a Marin Iris cluster.

This is the Iris analog of ``hpc/cloud_launch_utils.CloudLauncher`` (which
targets SkyPilot). It exists in parallel rather than as a "provider" plugin
because Iris and SkyPilot disagree on key abstractions (workdir bind mount
vs file_mounts, in-cluster scheduling vs bring-up-a-VM, autostop vs job
timeout). Trying to share one interface created leaky bolts; two clean
modules is cheaper to reason about.

Backend-agnostic helpers under ``hpc/`` (e.g. ``arg_groups``,
``harbor_utils``, ``datagen_config_utils``) are reused as-is.

Output handling — GCS only. The workload writes directly to
``--gcs-output-dir/<job-name>/`` (default
``gs://marin-eu-west4/ot-agent/``; override with ``$OT_AGENT_GCS_OUTPUT_ROOT``
or the flag). A local fetch daemon (``hpc.iris_fetch_daemon``, planned)
polls the iris controller and pulls completed jobs into
``~/.ot-agent/runs/<job-name>/``. The previous "rsync from worker
workdir" mode was removed on 2026-05-22: the worker workdir is on
ephemeral tmpfs and iris GCs it at task end, so any laptop-side rsync
loop is fragile by construction; see
``notes/marin/flows/iris-outputs-redesign.md`` for the post-mortem.

Multi-host slices (TPU vm_count > 1) are scaffolded but only validated
on v6e-8. Confirm cross-host JAX init + coscheduling before relying on
larger slices.
"""

from __future__ import annotations

import argparse
import os
import shlex
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from hpc.local_paths import PATHS as LOCAL_PATHS, ensure as ensure_local_paths
from hpc.iris_job_registry import register_submission

# Default chips per TPU host VM on every family currently exposed by marin
# (ct5lp-hightpu-4t, ct6e-standard-4t, ct5p-hightpu-4t, ct4p-hightpu-4t).
# If marin ever provisions ``-8t`` host variants this needs revisiting.
CHIPS_PER_TPU_HOST = 4

DEFAULT_TASK_IMAGE = "ghcr.io/open-thoughts/openthoughts-agent:tpu"
DEFAULT_CLUSTER_CONFIG = "lib/iris/config/marin.yaml"
DEFAULT_PRIORITY = "interactive"

# Default GCS prefix for workload outputs. EU-region matches where most
# of our v6e-preemptible TPU slices land; us-region jobs incur small
# cross-region writes (eval outputs are ~MB-scale, so this is fine).
# Override with $OT_AGENT_GCS_OUTPUT_ROOT or the --gcs-output-dir flag.
DEFAULT_GCS_OUTPUT_ROOT = "gs://marin-eu-west4/ot-agent"


def parse_tpu_vm_count(tpu_spec: Optional[str]) -> int:
    """Return the host-VM count implied by a TPU variant like ``v6e-16``.

    Chips per host is 4 on every family currently configured in marin's
    cluster YAML, so ``vm_count = chips / 4``. Returns 1 when no TPU is
    requested or the spec doesn't end in ``-<int>``.
    """
    if not tpu_spec:
        return 1
    try:
        chips = int(tpu_spec.rsplit("-", 1)[-1])
    except ValueError:
        return 1
    return max(1, chips // CHIPS_PER_TPU_HOST)


class IrisLauncher:
    """Base class for OT-Agent launchers targeting Marin Iris.

    Subclasses override:
      - ``add_task_specific_args(parser)``
      - ``normalize_paths(args)``
      - ``build_task_command(args, remote_output_dir) -> list[str]``
      - ``build_env(args) -> dict[str, str]``  (optional override)
    """

    task_name: str = "ot-iris"
    job_name_prefix: str = "iris"
    default_n_concurrent: int = 16
    default_tpu: str = "v6e-4"

    # Daytona is the only sandbox backend that works without DinD on iris.
    # Users may still pass --harbor_env docker; iris workers don't mount
    # /var/run/docker.sock so the job will fail at runtime — by design.
    default_harbor_env: str = "daytona"

    def __init__(self, repo_root: Path):
        self.repo_root = Path(repo_root).resolve()

    # ------------------------------------------------------------------
    # Argument parsing
    # ------------------------------------------------------------------

    def create_argument_parser(self, description: str = "") -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description=description or self.task_name)
        self._add_iris_common_args(parser)
        self.add_task_specific_args(parser)
        return parser

    def _add_iris_common_args(self, parser: argparse.ArgumentParser) -> None:
        g = parser.add_argument_group("iris")
        g.add_argument("--cluster-config", "--cluster_config",
                       default=self._resolve_cluster_config_default(),
                       help="Path to the iris cluster YAML (default: marin via lib/iris/config/marin.yaml in the marin repo).")
        g.add_argument("--task-image", "--task_image",
                       default=DEFAULT_TASK_IMAGE,
                       help=f"Container image for the task (default: {DEFAULT_TASK_IMAGE}).")
        g.add_argument("--tpu", default=self.default_tpu,
                       help=f"TPU variant (default: {self.default_tpu}). For multi-host slices "
                            "(e.g. v6e-8, v6e-16) the launcher will gang-schedule replicas — "
                            "**UNTESTED in v1**, see module docstring.")
        g.add_argument("--cpu", type=float, default=8.0,
                       help="CPU cores for the entrypoint task (default 8).")
        g.add_argument("--memory", default="64GB",
                       help="Memory for the entrypoint task (default 64GB).")
        g.add_argument("--disk", default="200GB",
                       help="Ephemeral disk (default 200GB).")
        g.add_argument("--priority", default=DEFAULT_PRIORITY,
                       choices=["production", "interactive", "batch"],
                       help="Iris priority band (default interactive).")
        g.add_argument("--max-retries", "--max_retries", type=int, default=0,
                       help="Max retries on failure (does NOT cover preemption — iris retries "
                            "preemptions automatically up to its own limit).")
        g.add_argument("--timeout", type=int, default=0,
                       help="Job timeout in seconds (0 = no timeout).")
        g.add_argument("--preemptible", dest="preemptible", action="store_true", default=None,
                       help="Force scheduling on preemptible workers (overrides iris heuristic).")
        g.add_argument("--no-preemptible", dest="preemptible", action="store_false",
                       help="Force scheduling on non-preemptible workers.")
        g.add_argument("--no-wait", dest="no_wait", action="store_true", default=False,
                       help="Submit and detach instead of streaming logs.")
        g.add_argument("--extras", action="append", default=None,
                       help="OpenThoughts-Agent extras to install in the iris worker's "
                            "/app/.venv via `uv sync --extra <name>`. Repeatable. "
                            "Default: ['datagen-tpu'] (matches the :tpu task image's "
                            "intended dep set). Pass --extras '' to install no extras.")

        og = parser.add_argument_group("outputs")
        og.add_argument("--gcs-output-dir", "--gcs_output_dir",
                        default=os.environ.get("OT_AGENT_GCS_OUTPUT_ROOT", DEFAULT_GCS_OUTPUT_ROOT),
                        help=f"GCS prefix for workload outputs; workload writes to "
                             f"<this>/<job-name>/. Defaults to $OT_AGENT_GCS_OUTPUT_ROOT or "
                             f"{DEFAULT_GCS_OUTPUT_ROOT}. The fetch daemon "
                             f"(hpc.iris_fetch_daemon) pulls completed jobs from here into "
                             f"{LOCAL_PATHS.runs}/<job-name>/.")

        sg = parser.add_argument_group("secrets")
        sg.add_argument("--secrets-env", "--secrets_env", default=None,
                        help="Path to a KEY=VALUE env file (~/Documents/secrets.env style). "
                             "Every entry is loaded into the iris task's env_vars at submit "
                             "time. Pairs with the hardcoded launcher passthrough list "
                             "(DAYTONA_API_KEY, OPENAI_API_KEY, etc.) — file values win on "
                             "conflict, explicit `-e` iris-CLI flags can't override since we "
                             "use IrisClient.submit() directly. Lines starting with '#' and "
                             "blank lines are ignored; leading 'export ' is stripped.")
        # NOTE: --dry-run / --dry_run is provided by hpc.arg_groups.add_model_compute_args
        # which subclass launchers call from add_task_specific_args. We don't redeclare
        # it here to avoid argparse conflicts.

    def _resolve_cluster_config_default(self) -> str:
        """Find the marin repo's cluster config relative to common locations."""
        candidates = [
            Path.home() / "Documents/marin" / DEFAULT_CLUSTER_CONFIG,
            Path("/Users/benjaminfeuer/Documents/marin") / DEFAULT_CLUSTER_CONFIG,
            Path(os.environ.get("MARIN_ROOT", "")) / DEFAULT_CLUSTER_CONFIG,
        ]
        for c in candidates:
            if c.exists():
                return str(c)
        return DEFAULT_CLUSTER_CONFIG

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    def add_task_specific_args(self, parser: argparse.ArgumentParser) -> None:
        raise NotImplementedError

    def normalize_paths(self, args: argparse.Namespace) -> None:
        """Subclass hook: validate/normalize paths and infer defaults."""

    def build_task_command(self, args: argparse.Namespace, remote_output_dir: str) -> List[str]:
        """Subclass hook: build the ``python data/...py ...`` invocation."""
        raise NotImplementedError

    def build_env(self, args: argparse.Namespace) -> dict:
        """Subclass hook: env vars to inject into the iris task container.

        HF_TOKEN, WANDB_API_KEY, HF_DATASETS_TRUST_REMOTE_CODE, and
        TOKENIZERS_PARALLELISM are auto-injected by iris workers — no need
        to add them here.
        """
        return {}

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def _derive_job_name(self, args: argparse.Namespace) -> str:
        job_name = getattr(args, "job_name", None)
        if job_name:
            return job_name
        ts = time.strftime("%Y%m%d-%H%M%S")
        return f"{self.job_name_prefix}-{ts}"

    def run(self, args: argparse.Namespace) -> int:
        self.normalize_paths(args)

        if not args.gcs_output_dir:
            raise SystemExit(
                "--gcs-output-dir is required (set OT_AGENT_GCS_OUTPUT_ROOT or pass the flag)."
            )

        job_name = self._derive_job_name(args)
        user = os.environ.get("USER") or os.environ.get("USERNAME") or "user"

        # The workload writes outputs directly to GCS; the fetch daemon
        # pulls them back to LOCAL_PATHS.runs/<job-name>/ on completion.
        remote_output_dir = f"{args.gcs_output_dir.rstrip('/')}/{job_name}"

        # Make sure the local managed tree exists so the daemon (and any
        # downstream consumers) find LOCAL_PATHS.runs/ on first run.
        ensure_local_paths(
            LOCAL_PATHS.home, LOCAL_PATHS.state, LOCAL_PATHS.runs, LOCAL_PATHS.logs,
        )

        command = self.build_task_command(args, remote_output_dir)
        env_vars = self.build_env(args)

        # Default extras = ["datagen-tpu"]; allow override via repeated --extras
        # or --extras '' (single empty) to install nothing extra.
        if args.extras is None:
            extras = ["datagen-tpu"]
        else:
            extras = [e for e in args.extras if e]

        # OT-Agent's build_support.py syncs the sft/llamafactory git submodule
        # at every setuptools.build_meta call (i.e. every editable install),
        # even when no sft-* extra is being installed. Inside the iris worker
        # container there's no git remote configured for that submodule, so
        # the sync errors out with exit 128. The build_support helper already
        # supports an escape hatch — opt in when no sft-* extra is requested.
        if not any(e.startswith("sft-") for e in extras):
            env_vars.setdefault("OT_AGENT_SKIP_SFT_SYNC", "1")

        # OT-Agent uses setuptools with [tool.setuptools.packages.find]
        # listing several top-level dirs (hpc, eval, data, ...). When iris's
        # entrypoint runs `python eval/local/run_eval.py`, Python sets
        # sys.path[0] to /app/eval/local, not /app — so `from hpc.* import
        # ...` raises ModuleNotFoundError. Setting PYTHONPATH=/app at boot
        # exposes the top-level dirs but in iris workers ALSO triggers
        # "unknown location" namespace-package resolution for some real
        # wheels (e.g. pydantic), so we can't use that. Instead we rewrite
        # the user command into a tiny python -c bootstrap that appends
        # /app to sys.path AFTER the venv has been activated and the
        # interpreter has built its initial path — namespace package
        # machinery has already cached real packages, so appending /app at
        # the end is safe.

        # The :tpu image sets ENV VIRTUAL_ENV=/opt/openthoughts/.venv so its
        # own preinstalled wheels are visible at container start. iris's
        # entrypoint runs `uv sync ...` from /app, then `source .venv/bin/
        # activate`, expecting `.venv` to live under /app. uv honors the
        # existing VIRTUAL_ENV unless told otherwise, so without this it
        # installs deps into /opt/openthoughts/.venv and then activates an
        # empty /app/.venv at run time — every `import pydantic` fails.
        # Force uv to use /app/.venv via UV_PROJECT_ENVIRONMENT, which has
        # higher precedence than VIRTUAL_ENV.
        env_vars.setdefault("UV_PROJECT_ENVIRONMENT", "/app/.venv")
        # Also clear VIRTUAL_ENV so `uv pip install` (used by iris for
        # cloudpickle/py-spy/memray) lands in /app/.venv, not the image's
        # preinstalled venv at /opt/openthoughts/.venv.
        env_vars.setdefault("VIRTUAL_ENV", "/app/.venv")
        # Force uv to materialize wheel contents into the venv instead of
        # symlinking them from /root/.cache/uv/archive-v0/... . On iris
        # workers, the uv cache lives in a tmpfs / different mount than the
        # venv: `uv sync` builds the symlinks during sync, but when the user
        # command runs the cache target is unreadable, so Python sees e.g.
        # /app/.venv/.../pydantic/__init__.py as a broken symlink and falls
        # back to namespace-package resolution — `from pydantic import
        # BaseModel` then raises "cannot import name BaseModel from
        # 'pydantic' (unknown location)". Copy mode avoids the symlink path
        # entirely. Confirmed via _iris_diag.py: pydantic/__init__.py was a
        # symlink to /root/.cache/uv/archive-v0/BYLjs1LAJOgakDOL/... which
        # didn't exist at runtime.
        env_vars.setdefault("UV_LINK_MODE", "copy")
        # Forward Ray/vLLM subprocess stdout/stderr to the parent process so
        # they appear in ``iris job logs``. Without this, vLLM controller
        # crashes during init (e.g. before ``write_endpoint_json`` runs) leave
        # no diagnostic trail in iris — the workload exits with the generic
        # "vLLM controller exited before writing the endpoint JSON" symptom
        # and the actual stacktrace is only in the per-task workdir
        # ``logs/vllm_controller.log``, which rsync hasn't picked up yet.
        env_vars.setdefault("OT_AGENT_INHERIT_SUBPROC_LOGS", "1")
        # Skip the vLLM --help flag-discovery probe in start_vllm_ray_controller.
        # On vllm-tpu (0.20.0) the import path of vllm.entrypoints.openai
        # cold-bootstraps libtpu inside a subprocess.run, which can hang for
        # multi-minute stretches and deadlock the parent controller with no
        # diagnostic output. The launcher emits a stable known-good set of
        # flags so skipping discovery is safe.
        env_vars.setdefault("VLLM_SKIP_FLAG_DISCOVERY", "1")
        # Skip the pre-Popen Ray probe in start_vllm_ray_controller. The probe
        # calls ray.init/cluster_resources/shutdown to print diagnostics, but
        # on the v6e-4 TPU runtime this sequence has been observed to hang
        # silently right after "Connected to Ray cluster". The probe isn't
        # load-bearing — vLLM does its own ray.init internally.
        env_vars.setdefault("VLLM_SKIP_RAY_PROBE", "1")

        # Forward sandbox-backend / external-API credentials from the
        # launcher's shell env into the iris worker. Iris auto-injects
        # HF_TOKEN, WANDB_API_KEY, HF_DATASETS_TRUST_REMOTE_CODE, and
        # TOKENIZERS_PARALLELISM but nothing else, so harbor's Daytona
        # client and other API-key-driven integrations need explicit
        # passthrough. The user typically loads these via
        # `source ~/Documents/secrets.env` before invoking the launcher.
        # Missing-from-env entries are skipped silently — harbor will
        # surface its own "DAYTONA_API_KEY not set" error if it actually
        # needs one. setdefault keeps any explicit -e overrides above.
        _LAUNCHER_ENV_PASSTHROUGH = (
            "DAYTONA_API_KEY",
            "DAYTONA_JWT_TOKEN",
            "DAYTONA_ORGANIZATION_ID",
            "DAYTONA_API_URL",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "GOOGLE_API_KEY",
            "GEMINI_API_KEY",
            "TOGETHER_API_KEY",
            "FIREWORKS_API_KEY",
            "SUPABASE_URL",
            "SUPABASE_KEY",
            "SUPABASE_SERVICE_ROLE_KEY",
        )
        for _k in _LAUNCHER_ENV_PASSTHROUGH:
            _v = os.environ.get(_k)
            if _v:
                env_vars.setdefault(_k, _v)

        # --secrets-env loader. SkyPilot mounted this file into the container
        # and sourced it remotely; iris has no file_mounts so we parse it
        # client-side and copy KEY=VALUE pairs into env_vars. File entries
        # override the os.environ passthrough above (an explicit file is more
        # intentional than an inherited shell env).
        if getattr(args, "secrets_env", None):
            secrets_path = Path(args.secrets_env).expanduser().resolve()
            if not secrets_path.exists():
                raise FileNotFoundError(f"--secrets-env file not found: {secrets_path}")
            loaded: list[str] = []
            for line_no, raw_line in enumerate(secrets_path.read_text().splitlines(), 1):
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[len("export "):].lstrip()
                if "=" not in line:
                    continue  # malformed; skip
                k, _, v = line.partition("=")
                k = k.strip()
                v = v.strip()
                # Strip matching surrounding quotes if present.
                if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
                    v = v[1:-1]
                if not k:
                    continue
                env_vars[k] = v  # file values override passthrough
                loaded.append(k)
            print(
                f"[iris] Secrets:    loaded {len(loaded)} entries from "
                f"{secrets_path}: {', '.join(sorted(loaded))}",
                flush=True,
            )

        vm_count = parse_tpu_vm_count(args.tpu)

        local_dest = LOCAL_PATHS.runs / job_name

        print(f"[iris] Job:        /{user}/{job_name}", flush=True)
        print(f"[iris] Cluster:    {args.cluster_config}", flush=True)
        print(f"[iris] Image:      {args.task_image}", flush=True)
        print(f"[iris] TPU:        {args.tpu}  (vm_count={vm_count})", flush=True)
        print(f"[iris] Priority:   {args.priority}", flush=True)
        print(f"[iris] Extras:     {extras or '(none)'}", flush=True)
        print(f"[iris] Output:     {remote_output_dir}", flush=True)
        print(f"[iris] Fetch dest: {local_dest}/  (via hpc.iris_fetch_daemon)", flush=True)
        print(f"[iris] Command:    {shlex.join(command)}", flush=True)

        if args.dry_run:
            print("[iris] --dry-run: not submitting", flush=True)
            return 0

        if vm_count > 1:
            print(
                "[iris] NOTE: multi-host TPU slice (vm_count > 1). Validated on v6e-8 "
                "(2026-05-22 smoke #10); larger slices need their own validation pass.",
                file=sys.stderr, flush=True,
            )

        # Defer the heavy iris imports so --dry-run / --help stay snappy.
        from iris.client import IrisClient
        from iris.cluster.config import IrisConfig
        from iris.cluster.types import EnvironmentSpec, Entrypoint
        from iris.cli.job import build_resources, build_job_constraints, resolve_multinode_defaults, build_tpu_alternatives
        from iris.rpc import job_pb2

        # Tunnel to the controller via the documented IrisConfig pattern
        # (see lib/iris/.../cluster/config.py:IrisConfig docstring).
        iris_config = IrisConfig.load(args.cluster_config)
        bundle = iris_config.provider_bundle()
        controller_proto = iris_config.proto.controller
        if controller_proto.WhichOneof("controller") == "local":
            from iris.cluster.providers.local.cluster import LocalCluster
            local_cluster = LocalCluster(iris_config.proto)
            controller_address = local_cluster.start()
        else:
            controller_address = (
                iris_config.controller_address()
                or bundle.controller.discover_controller(controller_proto)
            )

        with bundle.controller.tunnel(controller_address) as controller_url:
            resources = build_resources(args.tpu, None, cpu=args.cpu, memory=args.memory, disk=args.disk)
            tpu_variants = build_tpu_alternatives(args.tpu)
            primary_tpu = tpu_variants[0] if tpu_variants else None
            replicas, coscheduling = resolve_multinode_defaults(primary_tpu, None, None)
            resources_proto = resources.to_proto()
            constraints = build_job_constraints(
                resources_proto=resources_proto,
                tpu_variants=tpu_variants,
                replicas=replicas,
                regions=None, zone=None,
                preemptible=args.preemptible,
            )

            priority_band = job_pb2.PRIORITY_BAND_UNSPECIFIED
            if args.priority:
                # Map name → enum the same way iris/cli/job.py does.
                _PRIO = {
                    "production": job_pb2.PRIORITY_BAND_PRODUCTION,
                    "interactive": job_pb2.PRIORITY_BAND_INTERACTIVE,
                    "batch": job_pb2.PRIORITY_BAND_BATCH,
                }
                priority_band = _PRIO.get(args.priority, priority_band)

            client = IrisClient.remote(controller_url, workspace=self.repo_root)

            # Wrap the user command in a bash bootstrap that:
            #   (1) re-syncs deps with --link-mode=copy to materialize wheel
            #       contents into /app/.venv, replacing the broken symlinks
            #       iris's build phase left behind (iris hardcodes
            #       --link-mode symlink at lib/iris/.../runtime/entrypoint.py
            #       and its DockerRuntime runs setup in a build container, so
            #       the symlinked /root/.cache/uv/archive-v0/... targets do
            #       not exist in the run container — every `import pydantic`
            #       resolves to a namespace package and `from pydantic import
            #       BaseModel` raises "unknown location"). Confirmed via
            #       eval/local/_iris_diag.py.
            #   (2) runs the original user command via a python -c
            #       bootstrap that appends /app to sys.path. See block
            #       above for why we can't just set PYTHONPATH=/app.
            # The first command arg is the entrypoint script (e.g.
            # eval/local/run_eval.py); the rest are passed through as
            # argv[1:]. `python -c '<bootstrap>' <script> args...` makes
            # sys.argv = ['-c', <script>, *args], so the bootstrap rewrites
            # sys.argv to drop the '-c' and run the script via
            # runpy.run_path with __name__ == '__main__'.
            if command and command[0] == "python" and len(command) >= 2:
                script_path = command[1]
                script_argv = command[2:]
                py_bootstrap = (
                    "import sys; "
                    "sys.path.append('/app'); "
                    "sys.argv = sys.argv[1:]; "
                    "import runpy; "
                    "runpy.run_path(sys.argv[0], run_name='__main__')"
                )
                # Build the uv sync flags to mirror what iris runs, but with
                # --link-mode=copy. --all-packages + --extra entries are the
                # only project-shape flags that matter here; everything else
                # (python version, frozen) iris's build already validated.
                extras_flags = " ".join(
                    f"--extra {shlex.quote(e.split(':', 1)[-1])}" for e in extras
                )
                # Use --reinstall to force uv to rewrite every package into
                # the venv as copies, replacing the broken symlinks iris's
                # build phase produced. Without --reinstall uv sees the
                # existing .dist-info entries, declares "already installed",
                # and skips — the broken symlinks stay broken.
                # IRIS_DEBUG_UV_SYNC=1 turns this on; defaults to quiet so
                # the run-phase resync logs don't drown the user output.
                quiet = "" if os.environ.get("IRIS_DEBUG_UV_RESYNC") else "--quiet"
                resync_cmd = (
                    "cd /app && "
                    f"uv sync {quiet} --frozen --reinstall --link-mode=copy "
                    f"--all-packages --no-group dev {extras_flags}".rstrip()
                )
                # Quote the python -c body and script argv for the bash -c
                # invocation. We use a single shlex.join for the python
                # invocation so spaces/quotes in argv survive.
                py_invoke = shlex.join(
                    ["python", "-c", py_bootstrap, script_path, *script_argv]
                )
                bash_cmd = f"set -e; {resync_cmd}; exec {py_invoke}"
                wrapped = ["bash", "-c", bash_cmd]
                entrypoint = Entrypoint.from_command(*wrapped)
            else:
                entrypoint = Entrypoint.from_command(*command)

            job = client.submit(
                entrypoint=entrypoint,
                name=job_name,
                resources=resources,
                environment=EnvironmentSpec(env_vars=env_vars, extras=extras),
                constraints=constraints,
                coscheduling=coscheduling,
                replicas=replicas,
                max_retries_failure=args.max_retries,
                # Iris auto-retries on preemption; leave at default (1000).
                task_image=args.task_image,
                priority_band=priority_band,
                timeout=None if args.timeout == 0 else _seconds_to_duration(args.timeout),
            )
            full_job_id = str(job.job_id)
            print(f"[iris] Submitted: {full_job_id}", flush=True)

            # Record the job in the local registry so the fetch daemon
            # knows where to pull outputs from on completion. Failures
            # here are non-fatal — the job is already submitted, and the
            # user can re-register later via `python -m hpc.iris_fetch_daemon
            # fetch <job-id>` once that module lands.
            try:
                register_submission(
                    job_id=full_job_id,
                    job_name=job_name,
                    submitted_at_iso=datetime.now(timezone.utc).isoformat(),
                    gcs_output_dir=remote_output_dir,
                    local_dest=local_dest,
                    cluster_config=str(args.cluster_config),
                )
            except Exception as e:
                print(f"[iris] WARN: could not register job locally: {e}", file=sys.stderr, flush=True)

            if args.no_wait:
                return 0

            try:
                status = job.wait(stream_logs=True, timeout=float("inf"))
                exit_code = 0 if status.state == job_pb2.JOB_STATE_SUCCEEDED else 1
            except KeyboardInterrupt:
                print(f"[iris] Terminating job {full_job_id}...", file=sys.stderr, flush=True)
                client.terminate_job(job.job_id)
                exit_code = 130

            print(f"[iris] Job exit: {exit_code}", flush=True)
            return exit_code

# Imported lazily inside .run() to keep CLI startup fast, but tiny enough
# to define here.
def _seconds_to_duration(secs: int):
    from iris.cluster.types import Duration
    return Duration.from_seconds(secs)
