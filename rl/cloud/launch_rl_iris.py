#!/usr/bin/env python3
"""Launch a MarinSkyRL RL training job on Marin's Iris GPU cluster (CoreWeave).

This is the GPU/Iris analog of ``rl/cloud/launch_rl_cloud.py`` (the SkyPilot RL
launcher). It combines:
  - the RL-job structure from ``launch_rl_cloud.py`` (gpu-rl venv, run_rl.py
    entrypoint, rl_config / model_path / train_data / overrides), and
  - the Iris SDK submission mechanics from ``eval/cloud/launch_eval_iris.py``
    (controller tunnel, IrisClient.submit, --secrets-env injection, --no-wait,
    job-name, max-retries, workspace source-sync to /app).

It does NOT subclass ``hpc.iris_launch_utils.IrisLauncher``: that base is
TPU-shaped (build_resources(tpu=...), build_tpu_alternatives, the uv-project
``/app/.venv`` bootstrap). The gpu-rl image is a conda-venv image
(/opt/openthoughts/envs/rl), and the target is GPU, so we drive the iris SDK's
GPU helpers (build_resources(gpu=...), gpu_device, the leafgroup-coscheduling
``resolve_multinode_defaults``) directly. Where it overlaps, the flag names and
secrets handling mirror the two templates exactly.

Multi-node / gang scheduling
----------------------------
Iris HAS a native gang mechanism for GPUs (verified via `iris job run --help`
and lib/iris/src/iris/cli/job.py):
  - ``--gpu H100x8`` requests a whole CoreWeave node (8 H100 + IB) per task.
  - ``--replicas N`` (the `--help` text: "Number of tasks for gang scheduling")
    requests N such tasks.
  - For GPUs with replicas>1, ``resolve_multinode_defaults`` returns
    ``CoschedulingConfig(group_by="leafgroup")`` — the H100/InfiniBand
    colocation level — so all N replicas are co-scheduled together on the same
    IB leaf fabric, all-or-nothing.
  - The cw-us-east-02a cluster config enables **Kueue gang admission**
    (``kueue.cluster_queue: iris-cq``, ``host_network: true`` for NCCL/IB), so
    the N-task gang is admitted atomically: either all N whole nodes are
    granted or the job queues — true exclusive, co-scheduled multi-node.

So this launcher requests ``--num-nodes N`` whole H100x8 nodes EXCLUSIVELY: one
iris task per node (``replicas=N``), each holding all 8 GPUs of its node (no
co-tenants), coscheduled by leafgroup. The RL topology (one cross-node Ray
cluster, NCCL over IB) is wired by an in-container controller
(``scripts/iris/start_rl_iris_controller.py``): rank 0 starts the Ray head and
publishes its IP to a shared rendezvous; ranks 1..N-1 join; then rank 0 runs the
SkyRL/MarinSkyRL driver (``run_rl.py --num_nodes N``) attached to that cluster.

Usage
-----
    set -a; source "${DC_AGENT_SECRET_ENV:?see .claude/secret.md}"; set +a

    python -m rl.cloud.launch_rl_iris \
        --rl_config hpc/skyrl_yaml/iris/<config>.yaml \
        --model_path Qwen/Qwen3-8B \
        --train_data '["mlfoundations-dev/dataset"]' \
        --num-nodes 4 \
        --job-name my-rl-iris-run \
        --no-wait
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shlex
import sys
import time
from pathlib import Path
from typing import List, Optional

# Add repo root to sys.path for imports
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.append(str(_repo_root))

from hpc.launch_utils import PROJECT_ROOT
from hpc.iris_launch_utils import IrisLauncher  # reused only for the secrets-env loader

# Defaults for the CoreWeave H100 GPU cluster.
DEFAULT_CLUSTER = "cw-us-east-02a"
# Pin the RL image by IMMUTABLE DIGEST, not the floating ``:gpu-rl`` tag.
#
# WHY (the floating-tag stale-cache trap): the iris k8s backend always stamps
# the task pod with ``imagePullPolicy: IfNotPresent``
# (marin lib/iris .../backends/k8s/tasks.py) and we cannot override it from here.
# With a FLOATING tag, IfNotPresent means a node that already has *some* image
# under that tag name will NOT re-pull when the tag is later retagged to new
# bytes — so a node that cached an OLD ``:gpu-rl`` keeps running stale code
# (observed: a launcher run executed MarinSkyRL 4c668f4 with NO flash_attn_2_cuda
# while the freshly-retagged ``:gpu-rl`` pointed at the good build).
#
# A content-addressed ``@sha256:`` reference is self-verifying: IfNotPresent only
# treats the cache as a hit when the cached bytes hash to exactly this digest, so
# it always runs the intended image regardless of node cache state — sidestepping
# the stale-tag problem entirely without needing imagePullPolicy: Always.
#
# This digest == the immutable gitsha tag ``:gpu-rl-44c06ea8`` (OT-Agent commit
# 44c06ea8, "bump gpu-rl SKYRL_COMMIT 2d9feef -> 78d83a5"): flash_attn 2.8.3 +
# flash_attn_2_cuda present, /opt/skyrl baked at MarinSkyRL 78d83a5 — which ADDS
# the two fixes that deterministically crashed CoreWeave RL at build_models:
# 518179d (default norm_topk_prob=True for Qwen3.5/3.6 MoE) + 0b2b05b (retry around
# rank-0 HF weight-index resolution). Also still includes 2d9feef's trials_dir
# raw-str fix; harbor BAKED at 342729d5 (reward-zeroing trial.paths.trial_dir fix).
# When the gpu-rl image is rebuilt, bump this digest (use the immutable
# ``:gpu-rl-<gitsha>`` tag's digest, never the floating ``:gpu-rl``).
#
# This digest BAKES torchtitan a1fdd7e (+ tyro): the `ExpertParallel` import-assert
# (step-4a of Dockerfile.gpu-rl) PASSED in-build → the EP>1 MoE unblock is proven.
# The CoreWeave EP=8 RL jobs (30B-A3B 131k, 35B) no longer hit
# `ModuleNotFoundError: torchtitan`. Also baked: vLLM-fork 76259c63 + flash-attn
# 2.8.3 (flash_attn_2_cuda present) + MarinSkyRL 39faff7d + harbor 342729d5.
# MarinSkyRL 39faff7d (bumped from 78d83a5) carries the VALIDATED MoE forward-spill
# fix + deterministic-dtype hardening. In-build asserts ran green: flash_attn_2_cuda
# OK (from cached wheel), torch 2.11.0+cu128 / vllm 0.1.dev16611+g76259c63a /
# skyrl_train import OK, torchtitan ExpertParallel import OK, baked MarinSkyRL HEAD
# == 39faff7d.
#
# BUILT IN-CLUSTER ON COREWEAVE (not the arm64 Mac): the image is amd64 + a
# from-source x86 CUDA build QEMU/Docker-Desktop can't do locally, and iris has NO
# in-cluster build primitive (`iris build` = LOCAL buildx). The build ran as an iris
# job with KANIKO (BuildKit needs CAP_SYS_ADMIN/bind-mounts the cluster denies —
# privileged is silently downgraded; nodes run gVisor). Context = the iris-synced
# /app bundle (cpu48/mem512GB/disk400GB). FAST no-nvcc PREBUILT-WHEELHOUSE path: the
# kaniko script (docker/build_gpu_rl_kaniko.sh) fetched the prebuilt vLLM-fork +
# flash-attn wheels (from laion/gpu-rl-build-wheels) into the context and ran with
# WHEEL_SOURCE=prebuilt-wheelhouse + --skip-unused-stages → ZERO nvcc (~minutes, not
# ~3h); the SKYRL_COMMIT-only bump did not change the wheel cache-key, so the wheels
# stayed ABI-correct. ghcr push via the GitHub PAT (`gh auth token`, write:packages),
# NOT the Docker-Hub DOCKER_TOKEN in secrets.env.
#
# Single-platform linux/amd64 manifest, 13 layers ~21.5 GB. The floating :gpu-rl
# tag resolves to the same digest. When the image is rebuilt, bump this digest
# (use the immutable :gpu-rl-<gitsha> tag's digest, never the floating :gpu-rl).
DEFAULT_RL_DOCKER_IMAGE = (
    "ghcr.io/open-thoughts/openthoughts-agent"
    # gpu-rl-efd77b98 (built 2026-07-03, kaniko job gpurl-kaniko-efd77b98): the PULLABLE re-layering of
    # gpu-rl-69634c0b (@sha256:d9c7e604…, harbor 0729a3e9 = poll fix + tmux-bake). Same baked contents
    # (harbor 0729a3e9, MarinSkyRL 39faff7d, vLLM-fork 76259c63, flash-attn 2.8.3, torch 2.11.0+cu128,
    # rl env pinned via rl_env_constraints.txt) but built with SINGLE_SNAPSHOT=0 (per-instruction layers)
    # + torch's nvidia-CUDA deps split into 3 pre-install RUNs, so the MAX layer is 3.46 GB (was one
    # 16.6 GB --single-snapshot layer). WHY: the 16.6 GB single layer CANNOT be pulled over the
    # CoreWeave→ghcr egress — containerd restarts the single-blob GET from 0 and it dies at 8-11 GB every
    # attempt (diagnosed restart-from-0 across all 8 r4 pods → ImagePullBackOff; the incremental-base
    # rebuild ALSO failed because the build pod had to pull the same 16.6 GB base). 48 small layers each
    # pull+retry independently. Build asserts green (baked harbor 0.8.0 @ 0729a3e9). Digest below.
    # gpu-rl-a003838c (built 2026-07-03, kaniko job gpurl-kaniko-a003838c): HARBOR_COMMIT-only bump →
    # harbor 9416d5f3 "default-OFF episode logging" (descends from a4957ef1, so keeps 1ms poll + tmux-bake
    # + persistent exec session, AND gates the 3 big synchronous per-turn S3 writes — debug.json + prompt
    # + response — behind default-OFF enable_episode_logging, the real throughput lever py-spy found: a
    # sync S3 write per LLM call blocking the shared asyncio loop). Same PULLABLE recipe (SINGLE_SNAPSHOT=0
    # + torch nvidia-CUDA split, max layer 3.46 GB, 48 layers). Everything else unchanged (MarinSkyRL
    # 39faff7d baked, vLLM-fork 76259c63, flash-attn 2.8.3, torch 2.11.0+cu128).
    # gpu-rl-722fec34 (built 2026-07-04, kaniko gpurl-kaniko-722fec34): harbor 2e42d312 (cheap reaper
    # DEFAULT-ON — fixes the n=384 O(N) coordinator-reaper bottleneck py-spy found; opt out via
    # HARBOR_CHEAP_REAPER=0) + skyrl 3caeb79f (TIS served-id splice, now baked-default — no --skyrl-ref
    # needed for 35B). Same PULLABLE recipe; vLLM-fork 76259c63, flash-attn 2.8.3, torch 2.11.0+cu128 unchanged.
    # gpu-rl-dc56d265 (built 2026-07-05, kaniko gpurl-kaniko-dc56d265): harbor 2e42d312 (unchanged, cheap
    # reaper DEFAULT-ON) + skyrl 7b7d627b (load-aware power-of-two-choices inference-engine routing — fixes
    # the sticky hash-at-birth session routing that pinned agentic-RL rollout load onto one vLLM engine
    # while siblings idled at n=384; preserves per-session stickiness for prefix-cache reuse). Same PULLABLE
    # recipe (SINGLE_SNAPSHOT=0, max layer <8 GB; pull-verified 4m0s, 22.5 GB, skyrl HEAD=7b7d627b in-pod).
    # vLLM-fork 76259c63, flash-attn 2.8.3, torch 2.11.0+cu128 unchanged.
    # gpu-rl-bd888d27 (built 2026-07-05, kaniko gpurl-kaniko-bd888d27): HARBOR_COMMIT-only bump → harbor
    # d58043c3, TWO Daytona fixes: (1) connection-pool cap 250→2048 (fleet knob
    # HARBOR_DAYTONA_CONNECTION_POOL_MAXSIZE) — the SDK's 250-connection aiohttp pool starved the verifier's
    # upload/exec/download round-trip for a socket at n_concurrent>>250 → 100% VerifierTimeoutError on the slow
    # 35B (verifier isn't slow; its HTTP calls can't get a connection). 2048 lets the grid run clean at n=768.
    # (2) auto_stop_interval_mins 0→5 — killed-job orphaned sandboxes idle-stop then auto-delete (self-clean)
    # instead of leaking forever. Same PULLABLE recipe; skyrl 7b7d627b, vLLM-fork 76259c63, flash-attn 2.8.3,
    # torch 2.11.0+cu128 unchanged (harbor-only bump — wheels + rl_env_constraints untouched).
    # gpu-rl-2712998d (built 2026-07-05, kaniko gpurl-kaniko-2712998d): PULLABLE re-layer of bd888d27 —
    # bd888d27 (@sha256:a8f76d48…) baked identical contents but with --single-snapshot → one 16.6 GB layer that
    # EOFs on the CoreWeave→ghcr pull (ImagePullBackOff + whiteout conflict). This is SINGLE_SNAPSHOT=0
    # (48 layers, max 3.5 GB), same baked harbor d58043c3 + wheels. Verified pullable; pods reached Running.
    # gpu-rl-4e505a4e (built 2026-07-06, kaniko gpurl-kaniko-4e505a4e, SINGLE_SNAPSHOT=0 pullable): SKYRL_COMMIT
    # bump 7b7d627b→cdca0b3a = EPDIAG per-phase fwd/modelfwd instrumentation (LOGGING-ONLY, EPDIAG-gated, no
    # routing/correctness change) for the EP16-vs-EP8 fwd-op diagnostic (FINDING #2). Strict superset of 861656ba
    # (instrumentation is a no-op when EPDIAG unset). wheels + harbor d58043c3 + rl_env_constraints unchanged.
    # gpu-rl-f9806065 (built 2026-07-06, kaniko gpurl-kaniko-f9806065, SINGLE_SNAPSHOT=0 pullable): SKYRL_COMMIT
    # bump cdca0b3a→b2ff8bf2 = abort_generation DRAIN fix — drains the vLLM engine (poll has_unfinished_requests
    # until idle, bounded 60s fail-loud) before the caller meta-izes params in the layerwise weight-sync reload.
    # Fixes the _C::rms_norm meta-tensor crash on eager decode (grid-30b-c) + the masked stale-weight read under
    # cudagraph replay (BOTH 35B rungs died at gs1's weight sync on 84ffafac). wheels + harbor d58043c3 +
    # rl_env_constraints unchanged (skyrl-only, prebuilt-wheelhouse).
    # NOTE: gpu-rl-f9806065 @sha256:37cdc3e6 was UN-PULLABLE (built --single-snapshot by DEFAULT = one 16 GB
    # layer -> ImagePullBackOff on CoreWeave). gpu-rl-addb348e below is the SINGLE_SNAPSHOT=0 re-layer (48
    # layers, max 3.5 GB, pull-verified) with IDENTICAL contents (SKYRL b2ff8bf2 drain fix).
    # gpu-rl-cf1ecea6 (built 2026-07-07, kaniko gpurl-kaniko-cf1ecea6, SINGLE_SNAPSHOT=0 pullable):
    # SKYRL_COMMIT bump 613e225d->822221a0 = engine-readiness gate in ray_wrapped_inference_engine.py
    # (already-validated; previously runtime-only via --skyrl-ref, now baked-default). Parent is exactly
    # 613e225d, so this ADDS one commit and preserves everything baked in gpu-rl-1a32669c. wheels + harbor
    # + rl_env_constraints UNCHANGED (skyrl-only, prebuilt-wheelhouse). Pull-verified: 48 layers, max 3.46
    # GB, 22.6 GB total. Build asserts green (skyrl_train/vllm/flash_attn/torchtitan.ExpertParallel import).
    # gpu-rl-7d15b25a (built 2026-07-08, kaniko gpurl-kaniko-7d15b25a, SINGLE_SNAPSHOT=0 pullable): the
    # InfiniBand ENABLE image. Adds `rdma-core ibverbs-providers libibverbs1 librdmacm1 ibverbs-utils` to the
    # rl-stage apt-get so NCCL's built-in IB transport can DLOPEN libibverbs.so.1 + the libmlx5 provider at
    # runtime — WITHOUT it (all prior images) NCCL silently disabled IB and fell back to NET/Socket (TCP over
    # enp157s0np0), the cross-node throughput bottleneck. Diagnosed in a live grid-30b-c-cp4-timing-v5 pod:
    # CoreWeave exposes the RDMA devices (/dev/infiniband/{uverbs0..8,rdma_cm}, 9x mlx5 ports ACTIVE @ 100Gb/s
    # EDR) but the image shipped NO verbs userspace (`find / -name libibverbs*` empty, `ibv_devices` not found).
    # NO external libnccl-net.so/OFI plugin is needed on Mellanox IB. Also SKYRL_COMMIT 822221a0->272bf011
    # (penfever/working HEAD, direct child: CP>1 _C::rms_norm Meta-kernel fix) so --skyrl-ref 272bf011 is a
    # no-op safety belt. wheels + harbor d58043c3 + rl_env_constraints UNCHANGED (fast prebuilt-wheelhouse,
    # NO nvcc). Pull-verified: 48 layers, max 3.46 GB, 22.66 GB total. Build asserts green (flash_attn_2_cuda /
    # skyrl_train / vllm / torchtitan.ExpertParallel; baked MarinSkyRL HEAD == 272bf011; harbor 0.8.0). Expected
    # next-launch signal (NCCL_DEBUG=INFO): `NET/IB : Using [0]mlx5_0:1/IB` + `GPU Direct RDMA Enabled`.
    # gpu-rl-19bd8c5e (built 2026-07-13, kaniko gpurl-kaniko-19bd8c5e, SINGLE_SNAPSHOT=0 pullable): SKYRL_COMMIT
    # bump 272bf011->de40d31c (penfever/working HEAD, linear descendant — safe superset). Substantive change: a
    # LOG-CAPTURE-SAFE tqdm fallback (skyrl_train/utils/progress.py). On a non-TTY stderr (CoreWeave/Iris captured
    # container logs, SLURM) every SkyRL progress bar (Generation Buffer / Training Step / Generating Trajectories /
    # Evaluation) now emits THROTTLED newline-terminated loguru lines instead of invisible \r-in-place frames, so
    # progress finally shows up in the captured job logs; delegates to real tqdm on a TTY (auto-gated by
    # sys.stderr.isatty(), no launcher env wiring needed). wheels + harbor + rl_env_constraints UNCHANGED (fast
    # prebuilt-wheelhouse, NO nvcc). Pull-verified: 48 layers, max 3.46 GB. Build asserts green (flash_attn_2_cuda /
    # skyrl_train / vllm / torchtitan.ExpertParallel). NOTE: floating :gpu-rl tag was deliberately NOT moved
    # (PUSH_FLOATING=0) — promote it after a live smoke: `crane tag ...@sha256:98adaa38... gpu-rl`.
    "@sha256:98adaa38c6e8f57e52430d3b63829d918a7cfa7db26cf5382becb8ba022ddf19"  # noqa: E501  (gpu-rl-19bd8c5e, PULLABLE; log-capture-safe tqdm + SKYRL de40d31c)
    # (prev: gpu-rl-7d15b25a @sha256:17a46200 IB userspace + SKYRL 272bf011; gpu-rl-cf1ecea6 @sha256:74d6d3e2 SKYRL 822221a0 engine-readiness gate; gpu-rl-1a32669c @sha256:9a96ad1f SKYRL 613e225d)
)
_SUPERSEDED_RL_IMAGES = (
    # gpu-rl-69634c0b (built 2026-07-02, kaniko job gpurl-kaniko-69634c0b): a HARBOR_COMMIT-ONLY bump
    # of the prior gpu-rl-1af0ae2d (@sha256:d77b34dd…) — baked harbor f7f51f13 → 0729a3e9, which sits
    # on top of ef42e75e and carries BOTH Daytona throughput fixes: (1) ef42e75e "replace 1s exec poll
    # with 1ms+jitter" (per-exec runtime+RTT vs ceil(runtime)+1s → feeds the RL engines faster), and
    # (2) 0729a3e9 "bake tmux+asciinema into every snapshot at BUILD time" (terminus-2 per-episode
    # tmux install short-circuits → kills the ~401s agent_setup / 63% AgentSetupTimeout; takes effect
    # only on a FRESH snapshot mint). Everything else is unchanged: MarinSkyRL 39faff7d (baked),
    # vLLM-fork 76259c63 (compiled) + flash-attn 2.8.3 + torch 2.11.0+cu128, and the rl env stays
    # PINNED to the gpu-rl-81045a29 known-good freeze via docker/rl_env_constraints.txt (UV_CONSTRAINT)
    # — so the NCCL regression of the deleted gpu-rl-00220aac cannot recur. Harbor is a --no-deps
    # source-only swap, so the wheel cache-key is untouched → the prebuilt vLLM-fork + flash-attn
    # wheels (laion/gpu-rl-build-wheels) stayed ABI-correct → FAST no-nvcc prebuilt-wheelhouse path.
    # Build asserts ran green: flash_attn_2_cuda OK, torch 2.11.0+cu128 / vllm 0.1.dev16611+g76259c63a
    # / skyrl_train import OK, torchtitan ExpertParallel import OK (EP>1 MoE unblock),
    # baked harbor 0.8.0 @ commit 0729a3e9, baked MarinSkyRL HEAD == 39faff7d.
    "@sha256:d9c7e6046e8392f3bb50567fa46e8ef3d39e49bd7fdc34409bf40f380a8596a2"
)
DEFAULT_GPU_VARIANT = "H100"
DEFAULT_GPUS_PER_NODE = 8           # gd-8xh100ib-i128 = 8x H100-80GB + IB
# These H100 nodes are requested WHOLE-NODE-EXCLUSIVE (no co-tenants) — so request ALL the
# node's allocatable resources; don't under-request (wasted capacity + a too-low --memory
# caused a container-cgroup OOM at FSDP weight-load on the 30B run). Node allocatable ≈ 128
# CPU / ~2014 GiB mem / 8 GPU.
#   - CPU 48 (NOT 64): ~64-68 of the 128 cores are persistent daemonset reservation, so a
#     request >~60 FAILS the single-IB-leaf gang admission (observed: 64 unplaceable, 48 admits).
#   - MEMORY 1700GB (≈1583 GiB cgroup) — RAISED from 1400GB on 2026-07-11 because 1400 was
#     UNDER-allocating. It must clear TWO opposing footguns:
#     (a) too LOW → container-cgroup OOM at the training forward. The old 512GB OOM'd at FSDP
#         weight-load; and 1400GB (≈1303 GiB cgroup) sits RIGHT AT the ~1303 GiB forward peak
#         at 8 ranks/node (the 80B cp1 128-GPU EP8×FSDP8 geometry) — the r2 rankspread run
#         measured a 1028 GiB forward peak at only 4 ranks/node (2026-07-11 diagnosis), so at
#         8 ranks/node the peak clears 1400's cgroup → no headroom = OOM risk; and
#     (b) too HIGH (1800GB ≈ 1676 GiB) → sits so close to node-allocatable (~2014 GiB) that
#         after daemonset/persistent-reservation overhead a leafgroup gang (all-or-nothing,
#         one IB leaf) can't fit all pods → Kueue SchedulingGated stall (cost multiple
#         60-120min stalls overnight 2026-06-26, on a 1-GPU probe AND 8-node gangs).
#     1700GB fits with headroom (≈1583 GiB < 2014 GiB allocatable) AND clears the forward
#     peak. Lower toward the real need on an admission stall; do NOT raise toward 1800.
#     (1000-1200GB suffices for 2-node smokes.) See .claude/ops/iris/coreweave_gpu_ops.md.
#   - DISK defaults to "auto" = 80% of the node's live allocatable ephemeral-storage (~27.2 TiB
#     → ~21 TiB). WHY NOT the old 512GB: the long MoE training step's Ray object store spills to
#     /tmp (a metered emptyDir that counts against the --disk ephemeral-storage limit), growing
#     to >1.6 TB and EVICTING the pod (2026-06-28). Whole-node-exclusive gangs have NO co-tenants,
#     so reserving disk is pure waste — claim ~80%. (R2 object-spilling, the durable fix, is also
#     on; this headroom is belt-and-suspenders.) Pass --disk explicitly to override.
DEFAULT_CPU_PER_NODE = 48.0
DEFAULT_MEMORY_PER_NODE = "1700GB"
# --disk "auto" → DISK_FRACTION of the GPU node's live allocatable ephemeral-storage at launch
# (FALLBACK_DISK_GIB iff the node query fails). See _resolve_default_disk().
DEFAULT_DISK_PER_NODE = "auto"
DISK_FRACTION = 0.80
FALLBACK_DISK_GIB = 21800  # ~80% of the h100-8x ~27.2 TiB allocatable, used only if kubectl is unavailable
DEFAULT_PRIORITY = "interactive"


def _parse_quantity_to_gib(q: str) -> float:
    """Parse a k8s resource quantity (plain bytes, or Ki/Mi/Gi/Ti binary / k/M/G/T decimal suffix) to GiB."""
    q = q.strip()
    for suf, mult in (("Ki", 2**10), ("Mi", 2**20), ("Gi", 2**30), ("Ti", 2**40), ("Pi", 2**50)):
        if q.endswith(suf):
            return float(q[: -len(suf)]) * mult / 2**30
    for suf, mult in (("k", 1e3), ("M", 1e6), ("G", 1e9), ("T", 1e12), ("P", 1e15)):
        if q.endswith(suf):
            return float(q[: -len(suf)]) * mult / 2**30
    return float(q) / 2**30  # plain bytes


def _resolve_default_disk(fraction: float = DISK_FRACTION) -> str:
    """``fraction`` of the GPU node's LIVE allocatable ephemeral-storage, as a ``"<N>Gi"`` string.

    Whole-node-exclusive gangs have no co-tenants, so claim most of the node NVMe (the old fixed
    512GB default evicted long MoE steps once Ray's object store spilled to the metered /tmp).
    Queries kubectl for the MIN allocatable across 8-GPU nodes (never over-request a smaller node);
    falls back to FALLBACK_DISK_GIB if kubectl is unavailable (requires KUBECONFIG)."""
    import subprocess

    try:
        out = subprocess.run(
            ["kubectl", "get", "nodes", "-o",
             r'jsonpath={range .items[*]}{.status.capacity.nvidia\.com/gpu}{" "}'
             r'{.status.allocatable.ephemeral-storage}{"\n"}{end}'],
            capture_output=True, text=True, timeout=20, check=True,
        ).stdout
        allocs = [
            _parse_quantity_to_gib(p[1])
            for p in (line.split() for line in out.splitlines())
            if len(p) == 2 and p[0] == "8"
        ]
        if allocs:
            gib = int(min(allocs) * fraction)
            print(f"[rl-iris] --disk auto: {fraction:.0%} of node allocatable "
                  f"(min {min(allocs):.0f}GiB across {len(allocs)} GPU nodes) = {gib}Gi", flush=True)
            return f"{gib}Gi"
        print("[rl-iris] --disk auto: no 8-GPU nodes returned by kubectl; "
              f"using fallback {FALLBACK_DISK_GIB}Gi", flush=True)
    except Exception as exc:  # noqa: BLE001 - best-effort; fall back rather than block a launch
        print(f"[rl-iris] --disk auto: kubectl node query failed ({type(exc).__name__}: {exc}); "
              f"using fallback {FALLBACK_DISK_GIB}Gi", flush=True)
    return f"{FALLBACK_DISK_GIB}Gi"
# The gpu-rl image's RL venv (deps-only: torch 2.11 + vLLM fork + skyrl editable).
RL_PYTHON = "/opt/openthoughts/envs/rl/bin/python"
SKYRL_HOME = "/opt/skyrl"
# In-container source sync target. iris syncs the launcher's `workspace`
# (the OT-Agent repo) to /app and sets IRIS_WORKDIR=/app; putting /app first on
# PYTHONPATH makes the live synced OT-Agent code win over the image's baked
# /opt/openthoughts copy.
APP_DIR = "/app"


def _resolve_cluster_config_default() -> str:
    """Find the marin repo's cw-us-east-02a cluster YAML."""
    rel = f"lib/iris/config/{DEFAULT_CLUSTER}.yaml"
    candidates = [
        Path.home() / "Documents/marin" / rel,
        Path("/Users/benjaminfeuer/Documents/marin") / rel,
        Path(os.environ.get("MARIN_ROOT", "")) / rel,
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return rel


def _default_secrets_env() -> Optional[str]:
    cand = os.environ.get("OT_AGENT_SECRETS_ENV") or os.path.expanduser("~/Documents/secrets.env")
    return cand if os.path.isfile(cand) else None


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch a MarinSkyRL RL training job on the Iris CoreWeave H100 cluster.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- RL job args (mirror launch_rl_cloud.py) ---
    parser.add_argument(
        "--rl_config", required=True,
        help="Path to SkyRL/MarinSkyRL config YAML (repo-relative or absolute).",
    )
    parser.add_argument("--rl-config", dest="rl_config", help=argparse.SUPPRESS)

    parser.add_argument(
        "--model_path", required=True,
        help="Model path or HuggingFace ID (e.g., Qwen/Qwen3-8B).",
    )
    parser.add_argument("--model-path", dest="model_path", help=argparse.SUPPRESS)

    parser.add_argument(
        "--train_data", default="[]",
        help="Training data paths as a JSON list (e.g., '[\"org/dataset\"]').",
    )
    parser.add_argument("--train-data", dest="train_data", help=argparse.SUPPRESS)

    parser.add_argument(
        "--val_data", default="[]",
        help="Validation data paths as a JSON list.",
    )
    parser.add_argument("--val-data", dest="val_data", help=argparse.SUPPRESS)

    parser.add_argument(
        "--skyrl_override", action="append", default=[],
        help="SkyRL Hydra override (repeatable).",
    )
    parser.add_argument("--skyrl-override", dest="skyrl_override", action="append", help=argparse.SUPPRESS)

    parser.add_argument(
        "--experiments_dir", default="/app/experiments",
        help="In-container experiments output dir (on the synced /app workspace).",
    )
    parser.add_argument("--experiments-dir", dest="experiments_dir", help=argparse.SUPPRESS)

    # --- Resource / topology args (GPU multi-node) ---
    parser.add_argument(
        "--num-nodes", "--num_nodes", dest="num_nodes", type=int, default=1,
        help="Number of WHOLE H100 nodes to request EXCLUSIVELY, gang/co-scheduled "
             "(one iris task per node, all 8 GPUs each, coscheduled by leafgroup/IB).",
    )
    parser.add_argument(
        "--gpus-per-node", "--gpus_per_node", dest="gpus_per_node", type=int,
        default=DEFAULT_GPUS_PER_NODE,
        help="GPUs per node (CoreWeave nodes are 8x H100).",
    )
    parser.add_argument(
        "--gpu-variant", "--gpu_variant", dest="gpu_variant", default=DEFAULT_GPU_VARIANT,
        help="GPU variant (default H100).",
    )
    parser.add_argument(
        "--cpu", type=float, default=DEFAULT_CPU_PER_NODE,
        help="CPU cores per node.",
    )
    parser.add_argument(
        "--memory", default=DEFAULT_MEMORY_PER_NODE,
        help="Memory per node.",
    )
    parser.add_argument(
        "--disk", default=DEFAULT_DISK_PER_NODE,
        help=f"Ephemeral disk per node. Default 'auto' = {int(DISK_FRACTION * 100)}%% of the GPU "
             "node's live allocatable ephemeral-storage (whole-node-exclusive gangs have no "
             "co-tenants, so claim most of the node NVMe — keeps Ray object-spill / checkpoints "
             "clear of the ephemeral-storage eviction). Pass an explicit value (e.g. 4000GB) to override.",
    )
    parser.add_argument(
        "--ray-port", "--ray_port", dest="ray_port", type=int, default=6379,
        help="Port the cross-node Ray head binds.",
    )
    parser.add_argument(
        "--rendezvous-dir", "--rendezvous_dir", dest="rendezvous_dir", default=None,
        help="Shared object-store/path (gs://, s3://, or shared dir) for the multi-node "
             "Ray head/worker rendezvous. Required for --num-nodes>1. On cw-us-east-02a "
             "use an s3:// URI under the cluster's default bucket, e.g. "
             "s3://marin-us-east-02a/iris/rl-rdv/<job>; the cluster injects working creds "
             "+ AWS_ENDPOINT_URL into every task pod (iris-task-env Secret), so no external "
             "creds are needed. NOTE: the default object store moved R2 (s3://marin-na) -> "
             "CW (s3://marin-us-east-02a) on 2026-07-05 (marin c7caecc95a); pods now inject "
             "CW creds+endpoint and can NO LONGER reach s3://marin-na (R2).",
    )
    parser.add_argument(
        "--rendezvous-timeout", "--rendezvous_timeout", dest="rendezvous_timeout",
        type=int, default=None,
        help="Seconds the worker ranks poll for rank-0's Ray-head rendezvous file "
             "(forwarded to start_rl_iris_controller.py --rendezvous-timeout). Unset = the "
             "controller default (1800s). RAISE it (e.g. 3600) for a big model whose rank-0 "
             "pre-stage/snapshot_download can legitimately take >30 min, so a SLOW-but-not-hung "
             "head prestage completes inside the window instead of the workers timing out and "
             "killing the gang (the 80B rank-spread bring-up flake, 2026-07-11).",
    )
    parser.add_argument(
        "--trials-dir", "--trials_dir", dest="trials_dir", default="auto",
        help="Where Harbor writes per-trial agentic-RL rollout artifacts "
             "(terminal_bench_config.trials_dir). 'auto' (default) = "
             "s3://marin-us-east-02a/iris/<job_name>/trace_jobs — a DURABLE path the cw-us-east-02a "
             "pods reach via auto-injected creds, so rollouts survive pod GC and are inspectable "
             "post-hoc. 'local'/'off' = keep the config default (node-local "
             "/app/experiments/<run>/trace_jobs, EPHEMERAL — lost on GC, no shared FS/PVC). Or pass "
             "an explicit s3://, gs://, or path URI. NOTE: cw uses s3://marin-us-east-02a (CW "
             "object store; R2 s3://marin-na is no longer reachable — store moved 2026-07-05, "
             "marin c7caecc95a), NOT gs://; "
             "ignored if you already set terminal_bench_config.trials_dir via --skyrl_override.",
    )

    # --- Iris submission args (mirror launch_eval_iris.py / IrisLauncher) ---
    parser.add_argument(
        "--cluster", default=DEFAULT_CLUSTER,
        help="Iris cluster name (default cw-us-east-02a).",
    )
    parser.add_argument(
        "--cluster-config", "--cluster_config", dest="cluster_config",
        default=_resolve_cluster_config_default(),
        help="Path to the iris cluster YAML (default: cw-us-east-02a in the marin repo).",
    )
    parser.add_argument(
        "--task-image", "--task_image", "--docker_image", "--docker-image",
        dest="task_image", default=DEFAULT_RL_DOCKER_IMAGE,
        help=f"Container image (default {DEFAULT_RL_DOCKER_IMAGE}).",
    )
    parser.add_argument(
        "--job-name", "--job_name", dest="job_name", default=None,
        help="Job name (auto-derived if not set).",
    )
    parser.add_argument(
        "--priority", default=DEFAULT_PRIORITY,
        choices=["production", "interactive", "batch"],
        help="Iris priority band.",
    )
    parser.add_argument(
        "--max-retries", "--max_retries", dest="max_retries", type=int, default=0,
        help="Max retries on failure (iris auto-retries preemptions separately).",
    )
    parser.add_argument(
        "--timeout", type=int, default=0,
        help="Job timeout in seconds (0 = no timeout).",
    )
    parser.add_argument(
        "--no-wait", dest="no_wait", action="store_true", default=False,
        help="Submit and detach instead of streaming logs.",
    )
    parser.add_argument(
        "--preemptible", dest="preemptible", action="store_true", default=None,
        help="Force scheduling on preemptible workers.",
    )
    parser.add_argument(
        "--no-preemptible", dest="preemptible", action="store_false",
        help="Force scheduling on non-preemptible workers.",
    )
    parser.add_argument(
        "--secrets-env", "--secrets_env", dest="secrets_env", default=_default_secrets_env(),
        help="KEY=VALUE env file injected into the task (HF_TOKEN, WANDB_API_KEY, etc.). "
             "Defaults to $OT_AGENT_SECRETS_ENV, else ~/Documents/secrets.env.",
    )
    parser.add_argument(
        "--daytona-api-key-env", "--daytona_api_key_env", dest="daytona_api_key_env",
        default=os.environ.get("DAYTONA_KEY_OVERRIDE"),
        help="Name of an env var whose VALUE is forwarded to the pod as DAYTONA_API_KEY "
             "(routes agentic RL onto a dedicated Daytona org, e.g. "
             "--daytona-api-key-env DAYTONA_RL_API_KEY). Applied AFTER --secrets-env is "
             "re-sourced (which does 'file overrides shell'), so the override actually STICKS "
             "where a plain shell `export DAYTONA_API_KEY=...` is silently clobbered. "
             "Referenced by NAME only; no key value on the command line. "
             "Defaults to $DAYTONA_KEY_OVERRIDE.",
    )
    parser.add_argument(
        "--skyrl-ref", "--skyrl_ref", dest="skyrl_ref", default=None,
        help="If set, `git fetch && git checkout <ref>` the baked MarinSkyRL clone at "
             "/opt/skyrl BEFORE running, so the live editable install picks up a newer "
             "(or pinned) commit than the one baked into the image. Use to apply a "
             "MarinSkyRL fix that landed AFTER the image was built without waiting for an "
             "image rebuild (deps are baked, but skyrl-train is an editable git clone, so "
             "a checkout is live). Default: unset = use whatever commit the image baked.",
    )
    # ----------------------------------------------------------------------- #
    # MarinSkyRL runtime-knob flags (deslop stage 3). Each promotes a live      #
    # SKYRL_* runtime env var to a first-class CLI flag. ALL default to None    #
    # ("unspecified") so an all-defaults launch injects NOTHING and the pod env #
    # is byte-identical to today (the SkyRL code's own default applies). A       #
    # config's `extra_env:` block is overlaid on TOP of these, so extra_env      #
    # still wins: precedence is  env/extra_env > flag > code-default.            #
    # ----------------------------------------------------------------------- #
    g = parser.add_argument_group("MarinSkyRL runtime knobs (SKYRL_* -> flags)")
    g.add_argument(
        "--r3-transport", "--r3_transport", dest="r3_transport",
        choices=["by_value", "resident", "decentral"], default=None,
        help="R3 (rollout routed-experts) transport for MoE async RL. 'decentral' "
             "(code default) routes the captured routed-experts generation-worker -> "
             "node-resident consumer (head holds ~0 R3); 'resident' de-dups to 1 "
             "copy/dp-group on the driver head plasma; 'by_value' is the old per-actor "
             "by-value dispatch. Folds SKYRL_R3_RESIDENT + SKYRL_R3_DECENTRAL. "
             "Default: unset = code default (decentral).",
    )
    g.add_argument(
        "--r3-put-timeout-s", "--r3_put_timeout_s", dest="r3_put_timeout_s", type=int, default=None,
        help="Bounded ray.put() timeout (s) for an R3 dp-chunk dispatch "
             "(SKYRL_DISPATCH_PUT_TIMEOUT_S). Default: unset = 600.",
    )
    g.add_argument(
        "--nccl-timeout-s", "--nccl_timeout_s", dest="nccl_timeout_s", type=int, default=None,
        help="Worker NCCL-collective timeout in seconds (SKYRL_WORKER_NCCL_TIMEOUT_IN_S). "
             "Default: unset = 1800.",
    )
    g.add_argument(
        "--host-ram-monitor", dest="host_ram_monitor", choices=["on", "off"], default=None,
        help="Policy-worker host-RAM/cgroup-mem monitor thread "
             "(SKYRL_POLICY_HOST_RAM_MONITOR). Default: unset = on.",
    )
    g.add_argument(
        "--host-ram-monitor-interval-s", dest="host_ram_monitor_interval_s", type=int, default=None,
        help="Host-RAM monitor sample interval, s (SKYRL_POLICY_HOST_RAM_MONITOR_INTERVAL). "
             "Default: unset = 60.",
    )
    g.add_argument(
        "--tis-splice", dest="tis_splice", choices=["on", "off"], default=None,
        help="TIS served-id splice policy (SKYRL_TIS_SPLICE) — use vLLM's raw served "
             "token ids as the generated region for exact-by-id TIS alignment. "
             "Default: unset = on (no-op on non-thinking turns).",
    )
    g.add_argument(
        "--gdn-mask-fla", dest="gdn_mask_fla", choices=["auto", "on", "off"], default=None,
        help="Force the pure-torch GatedDeltaNet path / mask the broken fla wheel "
             "(SKYRL_GDN_MASK_FLA). 'auto' (and unset) derive it from the model arch "
             "(on for Qwen3-Next/GDN, off for dense). Default: unset = auto.",
    )
    g.add_argument(
        "--gdn-flashqla", dest="gdn_flashqla", choices=["on", "off"], default=None,
        help="Opt-in FlashQLA fused GDN tilelang kernel (SKYRL_GDN_FLASHQLA); needs the "
             "fla_tilelang overlay. Default: unset = off.",
    )
    g.add_argument(
        "--forward-dispatch-fix", dest="forward_dispatch_fix", choices=["on", "off"], default=None,
        help="MoE async-dispatch forward fix (SKYRL_FORWARD_DISPATCH_FIX), a correctness "
             "knob. Default: unset = on. Pass off only for an A/B.",
    )
    g.add_argument(
        "--weightsync-drain-barrier", dest="weightsync_drain_barrier", choices=["on", "off"], default=None,
        help="Post-weight-sync async drain barrier (SKYRL_WEIGHTSYNC_DRAIN_BARRIER), a "
             "correctness knob. Default: unset = on.",
    )
    g.add_argument(
        "--cp-require-right-align", dest="cp_require_right_align", choices=["on", "off"], default=None,
        help="Require right-aligned attention mask under context-parallel "
             "(SKYRL_CP_REQUIRE_RIGHT_ALIGN), a correctness knob. Default: unset = on.",
    )
    g.add_argument(
        "--w13-reload-bracket", dest="w13_reload_bracket", choices=["on", "off"], default=None,
        help="Bracket the MoE weight-sync with layerwise-reload init/finalize so FusedMoE "
             "w13 is re-swapped exactly once (SKYRL_W13_RELOAD_BRACKET), a correctness "
             "knob. Default: unset = on.",
    )
    g.add_argument(
        "--ep-loader-chunk-rows", dest="ep_loader_chunk_rows", type=int, default=None,
        help="Per-broadcast dim-0 row budget for the streamed EP full-state-dict loader "
             "(SKYRL_EP_LOADER_CHUNK_ROWS). Default: unset = 8.",
    )
    g.add_argument(
        "--collective-count-diag", dest="collective_count_diag", choices=["on", "off"], default=None,
        help="GC-proof per-rank default-PG collective-count instrumentation "
             "(SKYRL_COLLECTIVE_COUNT_DIAG), a DIAGNOSTIC knob for the 80B gs1 NCCL "
             "desync. Logs each policy rank's default-PG collective count at forward "
             "phase boundaries (forward/_forward_impl enter+exit + the first MoE-EP "
             "all-to-all per forward) to the finelog, which survives pod GC — diffing "
             "the counts across ranks at the wedge localizes the divergent EP group. "
             "O(phases), reads torch's own PG seq counter (no perturbation). "
             "Default: unset = off.",
    )

    parser.add_argument(
        "--dry-run", "--dry_run", dest="dry_run", action="store_true", default=False,
        help="Print the resolved config + in-container command without submitting.",
    )

    return parser


def build_skyrl_flag_env(args: argparse.Namespace) -> dict[str, str]:
    """Translate the MarinSkyRL runtime-knob CLI flags into SKYRL_* env vars for the
    pod. Only flags that were explicitly set (non-None) emit an entry, so an
    all-defaults invocation returns {} and the pod env stays byte-identical to today.
    The caller overlays the config's ``extra_env:`` on top of this, so a config's
    explicit value still wins (precedence: env/extra_env > flag > code default)."""
    env: dict[str, str] = {}

    def _onoff(name: str, value) -> None:
        if value is not None:
            env[name] = "1" if value == "on" else "0"

    # R3 transport: fold the nested resident && decentral gating into one choice.
    if args.r3_transport == "by_value":
        env["SKYRL_R3_RESIDENT"] = "0"
    elif args.r3_transport == "resident":
        env["SKYRL_R3_RESIDENT"] = "1"
        env["SKYRL_R3_DECENTRAL"] = "0"
    elif args.r3_transport == "decentral":
        env["SKYRL_R3_RESIDENT"] = "1"
        env["SKYRL_R3_DECENTRAL"] = "1"
    if args.r3_put_timeout_s is not None:
        env["SKYRL_DISPATCH_PUT_TIMEOUT_S"] = str(args.r3_put_timeout_s)
    if args.nccl_timeout_s is not None:
        env["SKYRL_WORKER_NCCL_TIMEOUT_IN_S"] = str(args.nccl_timeout_s)
    _onoff("SKYRL_POLICY_HOST_RAM_MONITOR", args.host_ram_monitor)
    if args.host_ram_monitor_interval_s is not None:
        env["SKYRL_POLICY_HOST_RAM_MONITOR_INTERVAL"] = str(args.host_ram_monitor_interval_s)
    _onoff("SKYRL_TIS_SPLICE", args.tis_splice)
    # GDN mask: 'auto' (like unset) leaves the env unset so the code auto-derives.
    if args.gdn_mask_fla in ("on", "off"):
        env["SKYRL_GDN_MASK_FLA"] = "1" if args.gdn_mask_fla == "on" else "0"
    _onoff("SKYRL_GDN_FLASHQLA", args.gdn_flashqla)
    _onoff("SKYRL_FORWARD_DISPATCH_FIX", args.forward_dispatch_fix)
    _onoff("SKYRL_WEIGHTSYNC_DRAIN_BARRIER", args.weightsync_drain_barrier)
    _onoff("SKYRL_CP_REQUIRE_RIGHT_ALIGN", args.cp_require_right_align)
    _onoff("SKYRL_W13_RELOAD_BRACKET", args.w13_reload_bracket)
    if args.ep_loader_chunk_rows is not None:
        env["SKYRL_EP_LOADER_CHUNK_ROWS"] = str(args.ep_loader_chunk_rows)
    _onoff("SKYRL_COLLECTIVE_COUNT_DIAG", args.collective_count_diag)
    return env


def load_config_extra_env(rl_config_path: str) -> dict[str, str]:
    """Read a top-level ``extra_env:`` mapping from the RL config YAML.

    On the SLURM/Apptainer path the runtime env lives under ``container.extra_env``
    and is emitted as shell ``export`` lines (hpc/rl_launch_utils.py). The Iris path
    has NO ``container:`` block (the gpu-rl Docker image is the runtime), so that
    plumbing never runs — without this, env declared in the YAML is silently
    dropped and only the launcher's hardcoded passthrough (HF/WANDB/DAYTONA) reaches
    the pod. This forwards a top-level ``extra_env:`` block (and, defensively,
    ``container.extra_env`` if a ported config still carries one) into the iris
    EnvironmentSpec so e.g. EPDIAG probe arms + R3/DCP guard env take effect.

    Values are coerced to str (YAML may parse "1"/true as int/bool). Returns {} if
    the file is unreadable or declares no extra_env (byte-identical behavior for the
    existing extra_env-less iris configs).
    """
    try:
        full = PROJECT_ROOT / rl_config_path
        path = full if full.exists() else Path(rl_config_path)
        import yaml
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
    except Exception as exc:  # noqa: BLE001
        print(f"[rl-iris] WARNING: could not read extra_env from {rl_config_path}: {exc}",
              file=sys.stderr)
        return {}
    extra = dict(raw.get("extra_env") or {})
    container_env = (raw.get("container") or {}).get("extra_env") or {}
    for k, v in container_env.items():
        extra.setdefault(k, v)
    out: dict[str, str] = {}
    for k, v in extra.items():
        if v is None:
            continue
        if isinstance(v, bool):
            v = int(v)
        out[str(k)] = str(v)
    return out


def _job_scope_fr_dump_path(prefix: str, job_name: str) -> str:
    """Rewrite a JOB-SCOPED NCCL flight-recorder dump path so its slug segment is the
    ACTUAL job name, e.g. ``/tmp/fr_dumps/<slug>/nccl_fr_rank`` -> ``/tmp/fr_dumps/
    <job_name>/nccl_fr_rank``.

    WHY (2026-07-11 FR-slug bug): the 80B configs hardcode
    ``TORCH_NCCL_DEBUG_INFO_TEMP_FILE: /tmp/fr_dumps/80b-next-cp1/nccl_fr_rank`` in
    their ``extra_env:``, so a run launched under a DIFFERENT ``--job-name`` (e.g.
    ``80b-next-cp1-r3d2``) still wrote its FR dumps under the stale ``80b-next-cp1``
    slug (harmless there, but wrong — a future FR dump would land under the wrong
    slug). Deriving the slug from the live job name keeps the dump under the right
    per-job dir. The controller's ``ensure_fr_dump_dir`` mkdir -p's whatever dirname
    the cvar carries, so overriding the cvar here is sufficient.

    ONLY rewrites the job-scoped ``.../fr_dumps/<slug>/<file>`` pattern; a bare
    generic path (``/tmp/nccl_fr_rank``, which every non-80B iris config uses) has no
    slug segment and is returned UNCHANGED (byte-identical for those configs)."""
    parent = os.path.dirname(prefix)          # e.g. /tmp/fr_dumps/<slug>
    grandparent = os.path.dirname(parent)     # e.g. /tmp/fr_dumps
    if os.path.basename(grandparent) != "fr_dumps":
        return prefix                         # not a job-scoped fr_dumps path; leave it
    return os.path.join(grandparent, job_name, os.path.basename(prefix))


def normalize(args: argparse.Namespace) -> None:
    """Validate + normalize. Keep rl_config repo-relative so it resolves on /app."""
    # Resolve rl_config to a repo-relative path (it must exist on the synced
    # /app workspace, NOT be an absolute host path).
    rl_cfg = Path(args.rl_config)
    if rl_cfg.is_absolute():
        try:
            args.rl_config = str(rl_cfg.resolve().relative_to(PROJECT_ROOT))
        except ValueError:
            raise SystemExit(
                f"--rl_config {args.rl_config!r} is absolute and not under the repo "
                f"({PROJECT_ROOT}); pass a repo-relative path so it resolves on /app."
            )
    # Verify it exists locally (so we fail fast before submitting).
    if not (PROJECT_ROOT / args.rl_config).exists():
        # Fall back to hpc/skyrl_yaml/<name>[.yaml] like launch_rl_cloud.py.
        yaml_dir = Path("hpc/skyrl_yaml")
        for cand in (yaml_dir / args.rl_config, yaml_dir / f"{args.rl_config}.yaml"):
            if (PROJECT_ROOT / cand).exists():
                args.rl_config = str(cand)
                break
        else:
            print(f"[rl-iris] WARNING: --rl_config {args.rl_config!r} not found under "
                  f"{PROJECT_ROOT}; the worker will error if it isn't on /app.",
                  file=sys.stderr)

    if args.num_nodes < 1:
        raise SystemExit("--num-nodes must be >= 1.")
    if args.num_nodes > 1 and not args.rendezvous_dir:
        raise SystemExit(
            "--num-nodes>1 requires --rendezvous-dir (a shared gs://, s3://, or path URI "
            "both head and worker nodes can reach) for the multi-node Ray rendezvous."
        )


def build_task_command(args: argparse.Namespace) -> List[str]:
    """Build the in-container command, multi-node-aware.

    The full pipeline that runs inside each task container:
      cd /app
      && export SKYRL_HOME + PYTHONPATH (live /app + skyrl-train win)
      && <RL_PYTHON> scripts/iris/start_rl_iris_controller.py
            --ray-port ... --rendezvous-dir ...
            -- <RL_PYTHON> -m rl.local.run_rl --rl_config ... --num_nodes N ...

    Rank 0 (IRIS_TASK_ID==0) starts the Ray head and runs run_rl.py (which, with
    RAY_ADDRESS set + --num_nodes>1, attaches to the cluster instead of starting a
    local one). Workers join Ray and park. We invoke the gpu-rl venv python by
    absolute path so it is used regardless of whichever venv iris's setup phase
    activates.
    """
    total_gpus = args.num_nodes * args.gpus_per_node

    # The MarinSkyRL training command rank 0 runs (run_rl.py owns config parse,
    # hydra-arg build, HF data resolution, and the SkyRL entrypoint launch).
    train_cmd: List[str] = [
        RL_PYTHON, "-m", "rl.local.run_rl",
        "--rl_config", args.rl_config,
        "--model_path", args.model_path,
        "--job_name", args.job_name,
        "--gpus", str(total_gpus),
        "--num_nodes", str(args.num_nodes),
        "--gpus_per_node", str(args.gpus_per_node),
        "--experiments_dir", args.experiments_dir,
        "--ray_port", str(args.ray_port),
    ]
    if args.train_data and args.train_data != "[]":
        train_cmd.extend(["--train_data", args.train_data])
    if args.val_data and args.val_data != "[]":
        train_cmd.extend(["--val_data", args.val_data])
    for override in (args.skyrl_override or []):
        train_cmd.extend(["--skyrl_override", override])

    # Durable Harbor rollout artifacts. The default (config trials_dir: null) resolves to a
    # node-local path on the rank-0 pod (/app/experiments/<run>/trace_jobs); cw-us-east-02a has
    # no shared FS/PVC and GCs pods on terminal, so those per-trial rollouts are lost when the
    # job ends. Point terminal_bench_config.trials_dir at a remote R2 URI (the cluster injects
    # working R2 creds, same store as the rendezvous) so rollouts persist + are inspectable
    # post-hoc. Skip if the user opted out (--trials-dir local) or already set it explicitly.
    trials_dir = (args.trials_dir or "auto").strip()
    user_set_trials = any("terminal_bench_config.trials_dir=" in o for o in (args.skyrl_override or []))
    if trials_dir.lower() not in ("local", "off", "none", "") and not user_set_trials:
        if trials_dir.lower() == "auto":
            trials_dir = f"s3://marin-us-east-02a/iris/{args.job_name}/trace_jobs"
        train_cmd.extend(["--skyrl_override", f"++terminal_bench_config.trials_dir={trials_dir}"])

    # The controller wraps the training command for the multi-node Ray bootstrap.
    controller_cmd: List[str] = [
        RL_PYTHON, "scripts/iris/start_rl_iris_controller.py",
        "--ray-port", str(args.ray_port),
    ]
    if args.rendezvous_dir:
        controller_cmd.extend(["--rendezvous-dir", args.rendezvous_dir])
    # Worker rendezvous poll deadline. Unset = controller default (1800s). Raise it when
    # rank-0's per-node pre-stage of a large model can legitimately exceed 30 min, so a
    # slow-but-not-hung head prestage completes before the workers give up + kill the gang.
    if args.rendezvous_timeout is not None:
        controller_cmd.extend(["--rendezvous-timeout", str(args.rendezvous_timeout)])
    # Per-NODE task-dataset staging. On a multi-node iris/CoreWeave slice each task
    # pod has its OWN node-local filesystem ($DCFT=/opt/openthoughts in the gpu-rl
    # image) — there is NO shared scratch like SLURM's GPFS. run_rl.py's
    # resolve_rl_train_data() extracts the HF task dataset to /opt/openthoughts/tasks/
    # but it runs ONLY on rank 0 (the head), so the Ray-scheduled rollout workers on
    # ranks 1..N-1 find an empty tasks dir and every rollout dies with
    # FileNotFoundError: .../task.toml -> reward always 0 (data-starved, doomed run).
    # Fix: forward --train-data to the controller so it can run the SAME extraction
    # on EVERY node before Ray starts, populating the identical node-local path on
    # all pods. Idempotent (on_exist=skip) — rank-0's later run_rl re-resolve is a
    # cheap no-op.
    if args.train_data and args.train_data != "[]":
        controller_cmd.extend(["--train-data", args.train_data])
    # Per-NODE model pre-staging, coupled to HF_HUB_OFFLINE. A config that runs the
    # FSDP ranks offline (extra_env HF_HUB_OFFLINE=1) has NO warm cache unless the
    # weights are pulled first; without pre-staging each of the N*8 ranks would race
    # HF Hub online inside init_model and a slow straggler blows the 20-min c10d store
    # barrier (the 80B init-straggle kill, 2026-07-10). When the config is offline,
    # forward the model repo-id so the controller pre-downloads it ONCE PER NODE into
    # the node-local HF cache before Ray — off the collective critical path. Online
    # configs are byte-identical (no flag forwarded).
    _cfg_env = load_config_extra_env(args.rl_config)
    if str(_cfg_env.get("HF_HUB_OFFLINE", "")).strip().lower() in ("1", "true", "yes", "on"):
        if args.model_path and not args.model_path.startswith(("s3://", "gs://", "gcs://")):
            controller_cmd.extend(["--prestage-model", args.model_path])
    controller_cmd.append("--")
    controller_cmd.extend(train_cmd)

    # Wrap in a bash bootstrap: cd to the synced workspace and set PYTHONPATH so
    # live /app + skyrl-train win over the image's baked copies. Use the absolute
    # RL venv python (set above) — independent of iris's activated venv.
    pythonpath = f"{APP_DIR}:{SKYRL_HOME}/skyrl-train"
    # Optional: refresh the baked MarinSkyRL editable clone to a newer/pinned commit
    # before running (deps are baked, but skyrl-train is `pip install -e` over a git
    # clone, so a checkout is live without reinstall). Fetch is best-effort but the
    # checkout MUST succeed (the ref is the whole point), so it's under `set -e`.
    skyrl_refresh = ""
    if args.skyrl_ref:
        ref = shlex.quote(args.skyrl_ref)
        skyrl_refresh = (
            f"git -C {shlex.quote(SKYRL_HOME)} fetch --quiet --all || true; "
            f"git -C {shlex.quote(SKYRL_HOME)} checkout {ref}; "
            # Purge baked bytecode after the checkout. The gpu-rl image bakes
            # `.pyc` for the editable skyrl-train at its build-time commit; if those
            # were compiled with hash-based (UNCHECKED_HASH) invalidation, Python
            # does NOT recompile when `git checkout` swaps the `.py` underneath, so
            # a `--skyrl-ref` checkout SILENTLY runs the stale baked bytecode (proven
            # 2026-06-25: the norm_topk_prob fix at 518179d checked out, but the pod
            # raised at the pre-fix line numbers). Delete the cache so the live `.py`
            # is recompiled. Best-effort (|| true) — must not block on a read-only fs.
            f"find {shlex.quote(SKYRL_HOME)}/skyrl-train -name '*.pyc' -delete 2>/dev/null || true; "
            f"find {shlex.quote(SKYRL_HOME)}/skyrl-train -name __pycache__ -type d -prune -exec rm -rf {{}} + 2>/dev/null || true; "
            f"echo \"[rl-iris] MarinSkyRL now at $(git -C {shlex.quote(SKYRL_HOME)} rev-parse HEAD)\"; "
        )
    bash = (
        f"set -e; cd {APP_DIR}; "
        f"{skyrl_refresh}"
        f"export SKYRL_HOME={shlex.quote(SKYRL_HOME)}; "
        f"export PYTHONPATH={shlex.quote(pythonpath)}:${{PYTHONPATH:-}}; "
        f"export VLLM_USE_V1=1; "
        f"exec {shlex.join(controller_cmd)}"
    )
    return ["bash", "-c", bash]


def main() -> int:
    parser = create_parser()
    args = parser.parse_args()
    normalize(args)

    if not args.job_name:
        args.job_name = f"rl-iris-{time.strftime('%Y%m%d-%H%M%S')}"

    # Load --secrets-env into os.environ on the launch host (so launch-host
    # hooks see it) AND collect them for injection into the task. Reuse the
    # IrisLauncher static helper (same semantics as launch_eval_iris.py).
    IrisLauncher.load_secrets_env_into_os_environ(args.secrets_env)

    # Daytona org re-route (robust). load_secrets_env_into_os_environ() above does
    # "file overrides shell" (hpc/iris/env.py) — so a pre-launch `export
    # DAYTONA_API_KEY="$DAYTONA_RL_API_KEY"` is CLOBBERED by secrets.env's main-org
    # value, which is then what the passthrough (below) forwards to the pod. To route
    # onto the dedicated RL Daytona org we must remap DAYTONA_API_KEY *after* the
    # re-source, referencing the source var by NAME only (no key value in code/CLI).
    override_src = getattr(args, "daytona_api_key_env", None)
    if override_src:
        override_val = os.environ.get(override_src)
        if not override_val:
            raise SystemExit(
                f"[rl-iris] --daytona-api-key-env={override_src} but that env var is "
                f"empty/unset. `source {args.secrets_env}` first; it must define {override_src}."
            )
        os.environ["DAYTONA_API_KEY"] = override_val
        _fp = hashlib.sha1(override_val.encode()).hexdigest()[:12]
        print(
            f"[rl-iris] Daytona re-route: DAYTONA_API_KEY <- ${override_src} (sha1={_fp})",
            flush=True,
        )

    command = build_task_command(args)

    # Per-task resources: a WHOLE node (8 H100 + IB), one task per node.
    gpu_spec = f"{args.gpu_variant}x{args.gpus_per_node}"

    # Resolve the "auto" disk default to ~80% of the node's live allocatable ephemeral-storage.
    # (Whole-node-exclusive gangs have no co-tenants → reserving disk is wasted; a too-low fixed
    # default evicted long MoE steps once Ray spilled to the metered /tmp. See _resolve_default_disk.)
    if str(args.disk).strip().lower() == "auto":
        args.disk = _resolve_default_disk()

    user = os.environ.get("USER") or os.environ.get("USERNAME") or "user"
    print(f"[rl-iris] Job:        /{user}/{args.job_name}", flush=True)
    print(f"[rl-iris] Cluster:    {args.cluster}  ({args.cluster_config})", flush=True)
    print(f"[rl-iris] Image:      {args.task_image}", flush=True)
    print(f"[rl-iris] Topology:   {args.num_nodes} node(s) x {gpu_spec}  "
          f"(= {args.num_nodes * args.gpus_per_node} GPUs, exclusive, gang/leafgroup)", flush=True)
    print(f"[rl-iris] Per node:   cpu={args.cpu} memory={args.memory} disk={args.disk}", flush=True)
    print(f"[rl-iris] Priority:   {args.priority}", flush=True)
    print(f"[rl-iris] RL config:  {args.rl_config}  model={args.model_path}", flush=True)
    # Surface the resolved SKYRL_* runtime-knob flag env here (before the --dry-run
    # return) so a dry-run confirms e.g. --collective-count-diag actually resolves.
    # This is display-only; main() re-derives it (idempotent, pure fn of args) below.
    _flag_env_preview = build_skyrl_flag_env(args)
    if _flag_env_preview:
        print("[rl-iris] SKYRL flag env: "
              f"{', '.join(f'{k}={v}' for k, v in sorted(_flag_env_preview.items()))}",
              flush=True)
    if args.num_nodes > 1:
        print(f"[rl-iris] Rendezvous: {args.rendezvous_dir}", flush=True)
    print(f"[rl-iris] Command:    {shlex.join(command)}", flush=True)

    if args.dry_run:
        print("[rl-iris] --dry-run: not submitting", flush=True)
        return 0

    # Defer heavy iris imports so --dry-run / --help stay snappy.
    #
    # NOTE: post iris PR #6652 (pydantic config parsing) + #6730 (multi-backend
    # controller) the old submit API moved. The config is now a pydantic
    # ``IrisClusterConfig`` loaded via the MODULE-LEVEL ``load_config(path)``
    # (the ``IrisConfig`` class + its ``.load()`` / ``.provider_bundle()`` /
    # ``.proto`` are gone). The provider bundle is now built by the module-level
    # ``iris.cluster.composer.provider_bundle(config)``, and ``LocalCluster``
    # moved to ``iris.cluster.local_cluster``. The job-build helpers
    # (build_resources / build_job_constraints / resolve_multinode_defaults /
    # EnvironmentSpec / Entrypoint / job_pb2) and the ``IrisClient.remote(...)`` /
    # ``client.submit(...)`` surface are UNCHANGED — see how the marin CLI itself
    # now submits in iris/cli/job.py + iris/cli/connect.py, which this mirrors.
    from iris.client import IrisClient
    from iris.cluster.config import load_config
    from iris.cluster.composer import provider_bundle
    from iris.cluster.local_cluster import LocalCluster
    from iris.cluster.types import EnvironmentSpec, Entrypoint
    from iris.cli.job import build_resources, build_job_constraints, resolve_multinode_defaults
    from iris.rpc import job_pb2

    # Per-task resources: whole node, all GPUs (no co-tenant → exclusive).
    resources = build_resources(
        None, gpu_spec, cpu=args.cpu, memory=args.memory, disk=args.disk
    )

    # Multi-node gang: replicas=num_nodes; for GPUs with replicas>1 this returns
    # CoschedulingConfig(group_by="leafgroup") — co-schedule all nodes on one IB
    # leaf fabric, atomically (Kueue gang admission on cw-us-east-02a).
    replicas, coscheduling = resolve_multinode_defaults(None, args.gpu_variant, args.num_nodes)

    resources_proto = resources.to_proto()
    constraints = build_job_constraints(
        resources_proto=resources_proto,
        tpu_variants=[],
        replicas=replicas,
        regions=None,
        zone=None,
        preemptible=args.preemptible,
    )

    priority_band = {
        "production": job_pb2.PRIORITY_BAND_PRODUCTION,
        "interactive": job_pb2.PRIORITY_BAND_INTERACTIVE,
        "batch": job_pb2.PRIORITY_BAND_BATCH,
    }.get(args.priority, job_pb2.PRIORITY_BAND_UNSPECIFIED)

    # Env: secrets file values + the standard RL/iris-serve signals. iris injects
    # IRIS_TASK_ID / IRIS_NUM_TASKS / IRIS_ADVERTISE_HOST per task automatically.
    env_vars: dict[str, str] = {}
    # MarinSkyRL runtime-knob flags (deslop stage 3) -> SKYRL_* env vars. Seeded
    # FIRST (below the config extra_env) so a config's explicit extra_env value still
    # OVERRIDES a flag; an all-defaults launch contributes {} (byte-identical).
    flag_env = build_skyrl_flag_env(args)
    if flag_env:
        env_vars.update(flag_env)
        print(f"[rl-iris] SKYRL flag env: "
              f"{', '.join(f'{k}={v}' for k, v in sorted(flag_env.items()))}", flush=True)
    # Forward the RL config YAML's top-level `extra_env:` block (the Iris analog of
    # the SLURM container.extra_env exports — see load_config_extra_env). Overlaid
    # ON TOP of the flag env so an explicit config value wins; the launcher's own
    # signals (rendezvous/secrets, below) then win over both on any collision.
    config_extra_env = load_config_extra_env(args.rl_config)
    if config_extra_env:
        env_vars.update(config_extra_env)
        print(f"[rl-iris] Config extra_env: {', '.join(sorted(config_extra_env))}", flush=True)
    # FR-slug fix: a config may hardcode a JOB-SCOPED NCCL flight-recorder dump path
    # (/tmp/fr_dumps/<slug>/nccl_fr_rank) with a STALE slug from the config it was
    # copied from. Re-scope the slug to the live --job-name so a future FR dump lands
    # under the right per-job dir (the controller mkdir -p's the cvar's dirname). No-op
    # for the bare generic /tmp/nccl_fr_rank path every non-80B config uses.
    for _fr_cvar in ("TORCH_NCCL_DEBUG_INFO_TEMP_FILE", "TORCH_FR_DUMP_TEMP_FILE"):
        _old = env_vars.get(_fr_cvar)
        if _old:
            _new = _job_scope_fr_dump_path(_old, args.job_name)
            if _new != _old:
                env_vars[_fr_cvar] = _new
                print(f"[rl-iris] FR-slug re-scope: {_fr_cvar} {_old} -> {_new}", flush=True)
    if args.rendezvous_dir:
        env_vars["OT_AGENT_IRIS_RENDEZVOUS_DIR"] = args.rendezvous_dir
    env_vars["OT_AGENT_IRIS_RAY_PORT"] = str(args.ray_port)
    # Forward the launch host's secrets (mirrors launch_eval_iris.py passthrough).
    #
    # IMPORTANT — do NOT forward AWS_*/R2_* here. The cw-us-east-02a cluster
    # projects an `iris-task-env` k8s Secret into EVERY task pod via `envFrom`
    # (because storage.remote_state_dir is an s3:// URI), and that secret already
    # carries the correct in-cluster R2 credentials + endpoint
    # (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_ENDPOINT_URL / AWS_REGION /
    # FSSPEC_S3). In K8s, explicit container `env` entries take precedence over
    # `envFrom`, so forwarding the launch host's AWS_* (which point at a
    # DIFFERENT account and lack AWS_ENDPOINT_URL) would CLOBBER the pod's
    # injected creds and make the s3://marin-us-east-02a rendezvous (multi-node)
    # silently target real AWS S3 instead of the cluster store. NOTE: the default
    # object store moved R2 (s3://marin-na) -> CW (s3://marin-us-east-02a) on
    # 2026-07-05 (marin c7caecc95a) — pods now inject CW creds+AWS_ENDPOINT_URL and
    # can no longer reach R2. Let the cluster-injected creds win; the
    # fsspec rendezvous in start_rl_iris_controller.py uses default credential
    # discovery and picks them up.
    #
    # Daytona credentials MUST be forwarded: agentic RL (terminal_bench / Harbor)
    # builds a Daytona sandbox per trial, and iris injects only HF/WANDB into the
    # task pod — nothing else. Without DAYTONA_API_KEY the worker's harbor client
    # raises DaytonaAuthenticationError on every env build, so no sandbox comes
    # up, the verifier never runs, and EVERY trajectory finalizes as
    # VerificationNotCompletedError with reward 0 (observed zeroing an entire
    # reverify rollout). Mirror the base IrisLauncher passthrough set
    # (hpc/iris_launch_utils.py) so the same creds reach the RL worker.
    #
    # WANDB routing default: the iris RL configs log to wandb (trainer.logger: wandb;
    # CoreWeave has egress). SkyRL's wandb.init passes project= but NOT entity=
    # (MarinSkyRL tracking.py), so without WANDB_ENTITY the run silently lands in the
    # API key's DEFAULT entity (e.g. nyu-dice-lab), not the team org. Default both to
    # the OT-Agent team here (matches hpc/dotenv/perlmutter.env) so every run lands in
    # dogml/OpenThoughts-Agent; an explicitly-set launch-host WANDB_ENTITY/PROJECT wins.
    os.environ.setdefault("WANDB_ENTITY", "dogml")
    os.environ.setdefault("WANDB_PROJECT", "OpenThoughts-Agent")
    for k in (
        "HF_TOKEN", "WANDB_API_KEY", "WANDB_ENTITY", "WANDB_PROJECT",
        "DAYTONA_API_KEY", "DAYTONA_JWT_TOKEN", "DAYTONA_ORGANIZATION_ID",
        "DAYTONA_API_URL",
        "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY", "GEMINI_API_KEY", "TOGETHER_API_KEY",
    ):
        v = os.environ.get(k)
        if v:
            env_vars[k] = v

    # Load the cluster config (pydantic IrisClusterConfig) and build the provider
    # bundle, then discover + tunnel to the controller. This mirrors the marin
    # CLI's own path (iris/cli/connect.py::require_controller_url): for a local
    # controller start an in-process LocalCluster; otherwise use the config's
    # controller_address() (defaults.worker.controller_address) if set, else fall
    # back to the backend's discover_controller(). cw-us-east-02a's controller
    # kind is "coreweave" (non-local, no IAP auth) → the discover path.
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
            max_retries_failure=args.max_retries,
            task_image=args.task_image,
            priority_band=priority_band,
            timeout=None if args.timeout == 0 else _seconds_to_duration(args.timeout),
        )
        full_job_id = str(job.job_id)
        print(f"[rl-iris] Submitted: {full_job_id}  (replicas={replicas}, "
              f"coscheduling={getattr(coscheduling, 'group_by', None)})", flush=True)

        if args.no_wait:
            return 0
        try:
            status = job.wait(stream_logs=True, timeout=float("inf"))
            exit_code = 0 if status.state == job_pb2.JOB_STATE_SUCCEEDED else 1
        except KeyboardInterrupt:
            print(f"[rl-iris] Terminating job {full_job_id}...", file=sys.stderr, flush=True)
            client.terminate_job(job.job_id)
            exit_code = 130
        print(f"[rl-iris] Job exit: {exit_code}", flush=True)
        return exit_code


def _seconds_to_duration(secs: int):
    from rigging.timing import Duration
    return Duration.from_seconds(secs)


if __name__ == "__main__":
    sys.exit(main())
