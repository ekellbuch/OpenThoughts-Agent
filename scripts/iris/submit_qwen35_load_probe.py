#!/usr/bin/env python
"""Submit the Rung-0a Qwen3.6-35B load probe as a 1-node gpu-rl iris job.

Reuses `rl.cloud.launch_rl_iris`'s pinned image digest + resource/constraint
builders + `IrisClient.submit` path so the probe runs in an environment IDENTICAL
to the real RL FSDP worker (same gpu-rl image, same `/opt/skyrl` checkout at
`--skyrl-ref`, same `SKYRL_QWEN3_5_VLM_UNWRAP`/`SKYRL_GDN_MASK_FLA` env). 1 node /
8x H100 requested (the probe only uses 1 GPU, but iris nodes are whole-node
exclusive); no rendezvous needed (single node).

The in-container command does, under `set -e`:
  git -C /opt/skyrl checkout <skyrl-ref> + purge baked .pyc (mirrors the launcher)
  -> /opt/openthoughts/envs/rl/bin/python /app/scripts/iris/probe_qwen35_text_load.py

Run from the otagent env on the Mac with KUBECONFIG=~/.kube/coreweave-iris-gpu.
"""

from __future__ import annotations

import argparse
import os
import shlex
import sys

# Reuse the launcher's constants + helpers (image digest, paths, env forwarding).
from rl.cloud.launch_rl_iris import (
    DEFAULT_RL_DOCKER_IMAGE,
    DEFAULT_GPU_VARIANT,
    RL_PYTHON,
    SKYRL_HOME,
    APP_DIR,
    load_config_extra_env,
)
from hpc.launch_utils import PROJECT_ROOT
from hpc.iris_launch_utils import IrisLauncher


PROBE_REL = "scripts/iris/probe_qwen35_text_load.py"


def build_command(skyrl_ref: str, model_id: str) -> list[str]:
    refresh = ""
    if skyrl_ref:
        ref = shlex.quote(skyrl_ref)
        refresh = (
            f"git -C {shlex.quote(SKYRL_HOME)} fetch --quiet --all || true; "
            f"git -C {shlex.quote(SKYRL_HOME)} checkout {ref}; "
            f"find {shlex.quote(SKYRL_HOME)}/skyrl-train -name '*.pyc' -delete 2>/dev/null || true; "
            f"find {shlex.quote(SKYRL_HOME)}/skyrl-train -name __pycache__ -type d -prune "
            f"-exec rm -rf {{}} + 2>/dev/null || true; "
            f"echo \"[probe] MarinSkyRL now at $(git -C {shlex.quote(SKYRL_HOME)} rev-parse HEAD)\"; "
        )
    pythonpath = f"{APP_DIR}:{SKYRL_HOME}/skyrl-train"
    bash = (
        f"set -e; cd {APP_DIR}; "
        f"{refresh}"
        f"export SKYRL_HOME={shlex.quote(SKYRL_HOME)}; "
        f"export PYTHONPATH={shlex.quote(pythonpath)}:${{PYTHONPATH:-}}; "
        f"export PROBE_MODEL_ID={shlex.quote(model_id)}; "
        f"exec {RL_PYTHON} {APP_DIR}/{PROBE_REL}"
    )
    return ["bash", "-c", bash]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--rl_config", default="hpc/skyrl_yaml/iris/8node_qwen3_6_35b_a3b_baseline_seqnorm_tis.yaml")
    p.add_argument("--model_path", default="Qwen/Qwen3.6-35B-A3B")
    p.add_argument("--skyrl-ref", dest="skyrl_ref", default="df8b661")
    p.add_argument("--job-name", dest="job_name", default="rl-35b-a3b-131k-rung0a-probe")
    p.add_argument("--cluster", default="cw-us-east-02a")
    p.add_argument("--cluster-config", dest="cluster_config", default=None)
    p.add_argument("--gpus_per_node", type=int, default=8)
    # A 1-GPU CPU load+remap probe needs ~80GB (34GB bf16 model + transient
    # grouped-MoE duplication). The RL-worker default 1800GB makes the pod
    # UNSCHEDULABLE under cluster contention (every node rejects on `Insufficient
    # memory` when the big RL gangs occupy the low-overhead nodes). Request a
    # modest, always-schedulable footprint; the node is still whole-node-exclusive
    # (8 GPU) but we reserve only what the probe uses so it fits alongside other
    # gangs.
    p.add_argument("--cpu", type=float, default=8.0)
    p.add_argument("--memory", default="256GB")
    p.add_argument("--disk", default="256GB")
    p.add_argument("--priority", default="interactive")
    p.add_argument("--max-retries", dest="max_retries", type=int, default=1)
    p.add_argument("--secrets-env", dest="secrets_env", default=os.environ.get("DC_AGENT_SECRET_ENV"))
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if args.cluster_config is None:
        from rl.cloud.launch_rl_iris import _resolve_cluster_config_default
        args.cluster_config = _resolve_cluster_config_default()

    IrisLauncher.load_secrets_env_into_os_environ(args.secrets_env)

    command = build_command(args.skyrl_ref, args.model_path)
    gpu_variant = DEFAULT_GPU_VARIANT
    gpu_spec = f"{gpu_variant}x{args.gpus_per_node}"

    user = os.environ.get("USER") or "user"
    print(f"[probe-submit] Job:      /{user}/{args.job_name}", flush=True)
    print(f"[probe-submit] Cluster:  {args.cluster}  ({args.cluster_config})", flush=True)
    print(f"[probe-submit] Image:    {DEFAULT_RL_DOCKER_IMAGE}", flush=True)
    print(f"[probe-submit] Topology: 1 node x {gpu_spec}", flush=True)
    print(f"[probe-submit] skyrl-ref:{args.skyrl_ref}", flush=True)
    print(f"[probe-submit] Command:  {shlex.join(command)}", flush=True)

    if args.dry_run:
        print("[probe-submit] --dry-run: not submitting", flush=True)
        return 0

    from iris.client import IrisClient
    from iris.cluster.config import IrisConfig
    from iris.cluster.types import EnvironmentSpec, Entrypoint
    from iris.cli.job import build_resources, build_job_constraints, resolve_multinode_defaults
    from iris.rpc import job_pb2

    resources = build_resources(None, gpu_spec, cpu=args.cpu, memory=args.memory, disk=args.disk)
    replicas, coscheduling = resolve_multinode_defaults(None, gpu_variant, 1)
    resources_proto = resources.to_proto()
    constraints = build_job_constraints(
        resources_proto=resources_proto, tpu_variants=[], replicas=replicas,
        regions=None, zone=None, preemptible=False,
    )
    priority_band = {
        "production": job_pb2.PRIORITY_BAND_PRODUCTION,
        "interactive": job_pb2.PRIORITY_BAND_INTERACTIVE,
        "batch": job_pb2.PRIORITY_BAND_BATCH,
    }.get(args.priority, job_pb2.PRIORITY_BAND_UNSPECIFIED)

    env_vars: dict[str, str] = {}
    config_extra_env = load_config_extra_env(args.rl_config)
    if config_extra_env:
        env_vars.update(config_extra_env)
        print(f"[probe-submit] Config extra_env: {', '.join(sorted(config_extra_env))}", flush=True)
    for k in ("HF_TOKEN", "WANDB_API_KEY", "WANDB_ENTITY", "WANDB_PROJECT"):
        v = os.environ.get(k)
        if v:
            env_vars[k] = v

    iris_config = IrisConfig.load(args.cluster_config)
    bundle = iris_config.provider_bundle()
    controller_proto = iris_config.proto.controller
    if controller_proto.WhichOneof("controller") == "local":
        from iris.cluster.providers.local.cluster import LocalCluster
        controller_address = LocalCluster(iris_config.proto).start()
    else:
        controller_address = (
            iris_config.controller_address()
            or bundle.controller.discover_controller(controller_proto)
        )

    with bundle.controller.tunnel(controller_address) as controller_url:
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
            max_retries_failure=args.max_retries,
            task_image=DEFAULT_RL_DOCKER_IMAGE,
            priority_band=priority_band,
            timeout=None,
        )
        print(f"[probe-submit] Submitted: {job.job_id}  (replicas={replicas})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
