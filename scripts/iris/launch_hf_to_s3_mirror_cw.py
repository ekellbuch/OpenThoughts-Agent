#!/usr/bin/env python3
"""Submit ``mirror_hf_to_s3.py`` as a ONE-OFF job on the CoreWeave cw-us-east-02a cluster.

This seeds the CW object store (``s3://marin-us-east-02a/models/<org>--<name>/``) with a
model's weights so the RL controller's warm path can pull them IN-REGION on every node
instead of cold-pulling from HF Hub. It MUST run on cw-us-east-02a — that is where the
object store lives and where the pod gets BOTH HF egress AND the CW-S3 creds +
``AWS_ENDPOINT_URL`` (the iris-task-env Secret); the GCP TPU pools (mirror_hf_to_gcs's
target) cannot reach marin-us-east-02a.

Reuses the exact iris-client submission machinery from ``rl.cloud.launch_rl_iris`` (same
cluster config, gpu-rl image with huggingface_hub + boto3 baked, /app workspace sync so
this script is present in the pod, secrets passthrough). It requests ONE H100 (the CW
cluster is GPU-typed; the mirror uses only CPU + network, but a single-GPU non-exclusive
request schedules quickly) and streams the mirror.

Usage::

    export DC_AGENT_SECRET_ENV=~/Documents/secrets.env
    set -a; source "$DC_AGENT_SECRET_ENV"; set +a
    export KUBECONFIG=~/.kube/coreweave-iris-gpu
    python scripts/iris/launch_hf_to_s3_mirror_cw.py \\
        --repo Qwen/Qwen3-Next-80B-A3B-Thinking
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.append(str(_repo_root))

from hpc.iris_launch_utils import IrisLauncher
from hpc.launch_utils import PROJECT_ROOT
from rl.cloud.launch_rl_iris import (  # reuse the proven CW submission constants/helpers
    DEFAULT_CLUSTER,
    DEFAULT_RL_DOCKER_IMAGE,
    _default_secrets_env,
    _resolve_cluster_config_default,
    _seconds_to_duration,
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", action="append", required=True,
                   help="HF model repo id (org/name); repeatable.")
    p.add_argument("--s3-prefix", "--s3_prefix", dest="s3_prefix",
                   default="s3://marin-us-east-02a/models",
                   help="Destination s3://bucket/prefix (default s3://marin-us-east-02a/models). "
                        "Each repo lands under <prefix>/<org>--<name>/.")
    p.add_argument("--job-name", "--job_name", dest="job_name", default=None,
                   help="Iris job name (auto-derived if unset).")
    p.add_argument("--cluster", default=DEFAULT_CLUSTER)
    p.add_argument("--cluster-config", "--cluster_config", dest="cluster_config",
                   default=_resolve_cluster_config_default())
    p.add_argument("--task-image", "--task_image", dest="task_image",
                   default=DEFAULT_RL_DOCKER_IMAGE,
                   help=f"Container image (default {DEFAULT_RL_DOCKER_IMAGE}).")
    p.add_argument("--gpu-variant", dest="gpu_variant", default="H100")
    p.add_argument("--gpus", type=int, default=1,
                   help="GPUs to request (default 1; the mirror uses only CPU+network, "
                        "but the CW cluster is GPU-typed).")
    p.add_argument("--cpu", type=float, default=16)
    p.add_argument("--memory", default="128GB")
    p.add_argument("--disk", default="500GB",
                   help="Ephemeral disk (holds one shard at a time; 500GB is ample).")
    p.add_argument("--priority", default="batch",
                   choices=["production", "interactive", "batch"])
    p.add_argument("--timeout", type=int, default=0, help="Job timeout in seconds (0=none).")
    p.add_argument("--no-wait", dest="no_wait", action="store_true", default=False,
                   help="Submit and detach instead of streaming logs.")
    p.add_argument("--secrets-env", "--secrets_env", dest="secrets_env",
                   default=_default_secrets_env())
    p.add_argument("--dry-run", dest="dry_run", action="store_true", default=False)
    return p


def _build_command(args: argparse.Namespace) -> list[str]:
    # Run via the RL venv python (absolute) inside the synced /app workspace, so the
    # freshly-synced scripts/iris/mirror_hf_to_s3.py is the one that runs.
    cmd = [
        "bash", "-c",
        "set -e; cd /app; "
        "export PYTHONPATH=/app:${PYTHONPATH:-}; "
        f"exec /opt/openthoughts/envs/rl/bin/python scripts/iris/mirror_hf_to_s3.py "
        + " ".join(
            [f"--repo {r}" for r in args.repo]
            + [f"--s3-prefix {args.s3_prefix}"]
        ),
    ]
    return cmd


def main() -> int:
    args = _build_parser().parse_args()
    if not args.job_name:
        args.job_name = f"hf2s3-{time.strftime('%Y%m%d-%H%M%S')}"

    IrisLauncher.load_secrets_env_into_os_environ(args.secrets_env)
    command = _build_command(args)

    gpu_spec = f"{args.gpu_variant}x{args.gpus}"
    print(f"[hf2s3-cw] Job:      /{os.environ.get('USER', 'user')}/{args.job_name}", flush=True)
    print(f"[hf2s3-cw] Cluster:  {args.cluster}  ({args.cluster_config})", flush=True)
    print(f"[hf2s3-cw] Image:    {args.task_image}", flush=True)
    print(f"[hf2s3-cw] Repos:    {args.repo} -> {args.s3_prefix}", flush=True)
    print(f"[hf2s3-cw] Resources: {gpu_spec} cpu={args.cpu} mem={args.memory} disk={args.disk}", flush=True)
    print(f"[hf2s3-cw] Command:  {command}", flush=True)

    if args.dry_run:
        print("[hf2s3-cw] --dry-run: not submitting", flush=True)
        return 0

    from iris.client import IrisClient
    from iris.cluster.config import load_config
    from iris.cluster.composer import provider_bundle
    from iris.cluster.local_cluster import LocalCluster
    from iris.cluster.types import EnvironmentSpec, Entrypoint
    from iris.cli.job import build_resources, build_job_constraints, resolve_multinode_defaults
    from iris.rpc import job_pb2

    resources = build_resources(None, gpu_spec, cpu=args.cpu, memory=args.memory, disk=args.disk)
    replicas, coscheduling = resolve_multinode_defaults(None, args.gpu_variant, 1)
    resources_proto = resources.to_proto()
    constraints = build_job_constraints(
        resources_proto=resources_proto, tpu_variants=[], replicas=replicas,
        regions=None, zone=None, preemptible=None,
    )
    priority_band = {
        "production": job_pb2.PRIORITY_BAND_PRODUCTION,
        "interactive": job_pb2.PRIORITY_BAND_INTERACTIVE,
        "batch": job_pb2.PRIORITY_BAND_BATCH,
    }.get(args.priority, job_pb2.PRIORITY_BAND_UNSPECIFIED)

    # Forward HF_TOKEN so the HF download authenticates. Do NOT forward AWS_*/R2_* — the
    # cluster injects the correct in-cluster CW creds + AWS_ENDPOINT_URL via iris-task-env
    # (forwarding launch-host AWS_* would clobber them; see launch_rl_iris.py note).
    env_vars: dict[str, str] = {}
    for k in ("HF_TOKEN", "HF_HUB_ENABLE_HF_TRANSFER"):
        v = os.environ.get(k)
        if v:
            env_vars[k] = v

    iris_config = load_config(args.cluster_config)
    bundle = provider_bundle(iris_config)
    if iris_config.controller.controller_kind() == "local":
        controller_address = LocalCluster(iris_config).start()
    else:
        controller_address = (
            iris_config.controller_address()
            or bundle.controller.discover_controller(iris_config.controller)
        )

    with bundle.controller.tunnel(address=controller_address) as controller_url:
        client = IrisClient.remote(controller_url, workspace=PROJECT_ROOT)
        entrypoint = Entrypoint.from_command(*command)
        job = client.submit(
            entrypoint=entrypoint,
            name=args.job_name,
            resources=resources,
            environment=EnvironmentSpec(env_vars=env_vars, extras=[]),
            constraints=constraints or None,
            coscheduling=coscheduling,
            replicas=replicas,
            max_retries_failure=0,
            task_image=args.task_image,
            priority_band=priority_band,
            timeout=None if args.timeout == 0 else _seconds_to_duration(args.timeout),
        )
        full_job_id = str(job.job_id)
        print(f"[hf2s3-cw] Submitted: {full_job_id}", flush=True)
        if args.no_wait:
            return 0
        try:
            status = job.wait(stream_logs=True, timeout=float("inf"))
            exit_code = 0 if status.state == job_pb2.JOB_STATE_SUCCEEDED else 1
        except KeyboardInterrupt:
            print(f"[hf2s3-cw] Terminating {full_job_id}...", file=sys.stderr, flush=True)
            client.terminate_job(job.job_id)
            exit_code = 130
        print(f"[hf2s3-cw] Job exit: {exit_code}", flush=True)
        return exit_code


if __name__ == "__main__":
    sys.exit(main())
