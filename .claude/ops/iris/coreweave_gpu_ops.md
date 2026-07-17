# Iris CoreWeave GPU cluster (`cw-us-east-02a`) — access & particulars

Cluster particulars for the **GPU RL path** on Marin's Iris: the CoreWeave
`cw-us-east-02a` H100 cluster (8× H100-80GB + InfiniBand per node), driven via
`python -m cloud.iris.launch_rl_iris` (run **from the MarinSkyRL repo root**) + the
`gpu-rl` Docker image. This is the ACCESS/HARDWARE/SCHEDULING reference; the launch
HOW-TO (flag set, config-authoring rules, bring-up checklist) lives in the
**`rl-agentic-launch-iris`** skill — ops = particulars, skill = procedure.

> **⚠ Launcher moved (cutover 2026-07-16).** The canonical live entry point is now the
> **self-contained MarinSkyRL launcher** `cloud/iris/launch_rl_iris.py` (repo
> `~/Documents/MarinSkyRL`, on `main` / any branch containing `cloud/iris/`), invoked as
> **`python -m cloud.iris.launch_rl_iris`** with `--rl_config cloud/iris/configs/<cfg>.yaml`.
> The old OT-Agent copy `python -m rl.cloud.launch_rl_iris` has been **REMOVED** — launch only
> from the MarinSkyRL module above. The env, kubeconfig,
> secrets, priority bands, node cap, rendezvous, and Daytona particulars below are unchanged;
> only the repo + module path + config location change. Validated end-to-end 2026-07-16
> (1-node Qwen3-8B FSDP2 smoke reached training via the MarinSkyRL launcher). The controller
> + helper scripts (`start_rl_iris_controller.py`, `tilelang_cache_sync.py`, `run_rl.py`,
> config translation) are now `cloud/iris/*` in MarinSkyRL, synced to `/app`;
> `PYTHONPATH=/app:/opt/skyrl/skyrl-train` (skyrl-train stays baked in the image at `/opt/skyrl`).

> **Scope — this is the GPU cloud, not the TPU cloud.** The rest of `.claude/ops/iris/`
> (`iris_job_lifecycle.md`, `iris_google_tpu_cloud_hardware.md`,
> `iris_eval_fixed_snapshot_template_scoping.md`) is the Iris **Google TPU** cloud
> (the `marin` cluster: datagen/eval via `data/cloud/launch_tracegen_iris.py` /
> `eval/cloud/launch_eval_iris.py`, regional `gs://` buckets, preemptible v5p/v6e
> slices). THIS doc is a DIFFERENT physical cluster — CoreWeave H100 GPUs, the
> `cw-us-east-02a` cluster, CW-object-store/`s3://` rendezvous (store moved R2→CW
> 2026-07-05), gang/Kueue admission — reached through the same `iris` SDK but with none
> of the TPU regional-egress / XLA-cache / 100 GB-node-disk mechanics. Don't cross-apply
> the TPU doc's region/disk/preemption rules here.

---

## Access

Launch from the **local Mac**, **otagent py3.12 conda env**
(`/Users/benjaminfeuer/miniconda3/envs/otagent/bin/python` — symlinks fail in the
sandbox; use the full interpreter path), **`cd ~/Documents/MarinSkyRL`** first (the
`-m cloud.iris.launch_rl_iris` module resolves from the MarinSkyRL repo root; the
`iris` SDK itself is installed in the otagent env). The local MarinSkyRL checkout must
be on **`main`** (or a branch that contains `cloud/iris/`) — the current working branch
`feuer/megatron-backend-transformers5` does NOT have the port. There is **no cluster
login / no SSH** — you talk to the cluster through the `iris` SDK over a controller
tunnel, and the launcher uploads your **local MarinSkyRL** workspace to `/app` (a local
commit takes effect on the next launch immediately — there is no Iris clone to pull).
The runtime is self-contained: `/app` provides `cloud.iris.*` + configs; `skyrl_train`
imports from the baked `/opt/skyrl` (swap with `--skyrl-ref`). No OpenThoughts-Agent
workspace is uploaded any more.

**Pre-launch preamble:**
```bash
source "$DC_AGENT_SECRET_ENV"   # HF_TOKEN, WANDB_*, DAYTONA_* — forwarded into the pod
export KUBECONFIG=~/.kube/coreweave-iris-gpu         # REQUIRED — the CoreWeave GPU cluster kubeconfig
```
- **`KUBECONFIG=~/.kube/coreweave-iris-gpu` is a HARD PREREQUISITE for every CoreWeave
  job/query.** This Mac's default `~/.kube/config` points at a different context
  (TPU/`marin`/other); without the export, `kubectl` inspects the wrong cluster and
  `iris` cw commands open the tunnel against the wrong backend → misleading "0 pods /
  not found" / auth errors that look like a dead job but are really the wrong
  kubeconfig. Re-export in any fresh shell or background call.
- **`source "$DC_AGENT_SECRET_ENV"` is load-bearing:** the launcher forwards
  `HF_TOKEN` / `WANDB_*` / `DAYTONA_*` from the launch host's env into the task pod.
  Without `DAYTONA_*` every agentic trajectory finalizes
  `VerificationNotCompletedError` reward 0 (no sandbox comes up); without `HF_TOKEN`
  weight/data resolution fails.
- **`iris` binary** = `/Users/benjaminfeuer/miniconda3/envs/otagent/bin/iris`. Use the
  otagent-env binary, NOT `marin/.venv/bin/iris`: cw is a k8s controller backend
  (reaching it instantiates `CloudK8sService`, which imports `kubernetes`), and the
  marin `.venv` ships a broken `kubernetes` (dist-info present, module not importable)
  → every cw `iris` command (`job summary`, `query`, `job list`, …) dies with
  `ImportError: Install iris[controller]`. The otagant env has a working `kubernetes`
  35.0.0 + the editable iris package. (`conda activate marin && uv run iris` also works
  IF that venv has the controller extra; the otagant binary is the reliable default.)
  All `iris`/`kubectl` calls must be **SYNCHRONOUS** — never background them.
- **Cluster config** auto-resolves to `~/Documents/marin/lib/iris/config/cw-us-east-02a.yaml`
  (`launch_rl_iris.py:_resolve_cluster_config_default`); override with `--cluster-config`
  only if it moved.

---

## Sibling CoreWeave GPU cluster: `cw-rno2a` (RNO2A / Reno) — 512 H100, added 2026-07 (marin PR #6909)

A **second, larger** CoreWeave GPU cluster is now live alongside `cw-us-east-02a`. Same
KubernetesProvider / Kueue-gang model; different region, kubeconfig, and node shape.

- **Fleet:** **64× `gd-8xh100ib-i128` (8× H100-80GB + IB each) = 512 H100** + 1 `turin-gp-l`
  CPU controller node (65 nodes total). The H100 pool is **pinned fully warm**
  (`buffer_slices: 64` = the whole reservation stays provisioned even when idle) — so a
  gang admits without a cold node-provision wait. **4× the East cluster (which is ~36 nodes / ~256 GPU).**
- **Access — DIFFERENT kubeconfig + context from East (don't reuse East's):**
  - `KUBECONFIG=~/.kube/coreweave-iris` (NOTE: **no `-gpu` suffix** — East is `~/.kube/coreweave-iris-gpu`).
    This is a 5-context file (a fresh `…-july-token` covering East/West/rno2a/oa-*); perms 600.
  - `kube_context: marin-rn02a_RNO2A` → endpoint `https://208261-6670debc.k8s.rno2a.coreweave.com`.
  - Query nodes/pods exactly like East: `KUBECONFIG=~/.kube/coreweave-iris kubectl --context marin-rn02a_RNO2A get nodes`.
  - Namespace `iris` (controller `iris-controller-*` + `finelog-cw-rno2a-*` both run there).
  - **⚠ SECRET-BEARING kubectl (all iris clusters):** `kubectl describe pod <task-pod>` and `kubectl get pod <task-pod> -o yaml` dump `IRIS_JOB_ENV` — i.e. the pod's **live API keys** (`DAYTONA_*`, `HF_TOKEN`, `WANDB_API_KEY`, `OPENAI_API_KEY`, …) — into their output. Do NOT run `describe`/`-o yaml` on a task pod when you only need state; use `kubectl … get pods` (state) or read ONE var by name via `kubectl -n iris exec <pod> -- printenv <VAR>` (never echo the value). Bake this bound into subagent prompts — a leaked `describe` puts secrets in a transcript.
- **Cluster config** = `lib/iris/config/cw-rno2a.yaml` (on marin **origin/main**). ⚠ To drive it
  via the `iris` SDK the **local marin clone must be on `main`** (or a branch carrying the new
  `IrisClusterConfig` schema) — an older clone's pydantic **rejects** the new fields
  (`kube_context`, `external_object_storage_endpoint`, `checkpoint_interval_seconds`,
  `signing_key`, `trusted_cidrs`) → `extra_forbidden` at config-load. Direct `kubectl` (above)
  works regardless. Object store: `object_storage_endpoint: http://cwlota.com`,
  `external_object_storage_endpoint: https://cwobject.com`. Finelog server: `finelog deploy up cw-rno2a`.
- **⚠ NCCL bring-up gotcha (do NOT carry East's value over):** the RNO2A bring-up bug (marin
  **#6940**, fixed 2026-07-04) was a wrong `NCCL_SOCKET_IFNAME` (`=enp90s0np0`) that broke
  multi-node NCCL bootstrap (`Bootstrap: no socket interface found`). The **correct socket
  interface is cluster-specific** — a port of our RL configs (which set `NCCL_SOCKET_IFNAME`)
  must use the rno2a-correct PF, not `cw-us-east-02a`'s. Also fixed: **#6950** (128k grug-MoE
  gangs hung on first-step NCCL collective — bad IB leaf-group 71). Both closed; but **large-gang
  MoE stability is still being shaken out** by the marin team (several `jaxpp-rno2a-*` W&B runs
  crashed ~2026-07-10) — treat a first big multi-node run here as UNTESTED (deep-dive it).
- **Ownership:** this is a **marin-team** reservation (rjpower/dlwh/Rafal Wojdyla run grug/jaxpp
  MoE experiments on it). Not our dedicated cluster — coordinate before parking a large long run.
- **Status snapshot (2026-07-13):** 512 H100 / 64 nodes **entirely idle** (0 GPUs in use; only
  controller + finelog pods in `iris`). Verified via the `kubectl … get pods -A` GPU-request tally.

---

## Hardware (node shape)

CoreWeave `cw-us-east-02a` node = **8× H100-80GB + InfiniBand**, requested **whole-node
exclusive** (`H100x8`, one iris task per node, no co-tenants). ~**36 H100 nodes** total.

- **~128 CPU cores per node, BUT ~64–68 cores are persistent system/daemonset overhead**
  → only ~48–60 cores are actually free per node. This is why `--cpu 48` admits a
  multi-node gang and `--cpu 64` does not (see Scheduling).
- **Whole-node-exclusive ⇒ REQUEST ALL the node's allocatable resources** (no
  co-tenants, so under-requesting is wasted capacity AND a footgun). Node allocatable ≈
  **128 CPU / ~2014 GiB mem / 8 GPU**. Launcher defaults (`launch_rl_iris.py`):
  **`--cpu 48`** (max-admittable — >~60 fails the IB gang), **`--memory 1400GB`** (≈
  full ~2 TB leaving daemonset headroom — see Binding gotchas for the validated middle),
  **`--gpus_per_node 8`**, `--disk 512GB` (rendezvous/ckpts go to the CW object store
  `s3://marin-us-east-02a`, not node-local).
- **NVLink intra-node + InfiniBand inter-node.** This is the headline difference from
  Jupiter's GH200 4-GPU nodes: a TP=8 vLLM engine places **intra-node on ONE 8-GPU
  node** over NVLink (decode), no cross-node TP — exactly the placement Jupiter's 4-GPU
  nodes could never satisfy for the MoE DCP=2 arm.
- **⚡ IB (NET/IB GPUDirect RDMA) — the userspace is now BAKED (gpu-rl-7d15b25a, 2026-07-08).**
  CoreWeave auto-exposes the RDMA devices into every pod (`/dev/infiniband/{uverbs0..8,rdma_cm}`,
  `/sys/class/infiniband/mlx5_0..8`, all 9 ports `state=ACTIVE` @ `100 Gb/sec 4X EDR`) — the
  pod/config side needs NO change (no `rdma/*` resource request, no hostNetwork edit). But EVERY
  gpu-rl image BEFORE gpu-rl-7d15b25a shipped **no IB verbs userspace** (`ldconfig` had no
  libibverbs/librdmacm/libmlx5; `ibv_devices` not found), so NCCL's built-in IB transport — which
  DLOPENS `libibverbs.so.1` + the `libmlx5` provider at runtime — silently fell back to
  **`NET/Socket`** (TCP over `enp157s0np0`). Fix = the rl-stage apt-get now installs
  `rdma-core ibverbs-providers libibverbs1 librdmacm1 ibverbs-utils` (Dockerfile.gpu-rl). **No
  external `libnccl-net.so`/OFI plugin is needed on Mellanox IB — the "Could not find
  libnccl-net.so" health-probe line is BENIGN.** Expected on-launch signal with `NCCL_DEBUG=INFO`:
  `NET/IB : Using [0]mlx5_0:1/IB` + `GPU Direct RDMA Enabled` (was
  `NET/Socket : Using [0]enp157s0np0:…`). Ref: `/Users/benjaminfeuer/Documents/agent_logs/2026-07-08_gpu-rl-ib-enable.md`.
- **NCCL DEFAULTS — use them (MoE-salad doubt FALSIFIED 2026-06-27).** On H100+IB, do
  NOT set the GH200/SIF disables (`NCCL_P2P_DISABLE` / `NCCL_NVLS_ENABLE=0` /
  `NCCL_COLLNET_ENABLE=0`): they cripple the intra-node NVLink all-reduce a TP=8 (DCP)
  engine depends on. NCCL defaults give NVLink intra-node + IB inter-node. Keep the
  observability/raised-timeout env (`NCCL_DEBUG=INFO`, `SKYRL_WORKER_NCCL_TIMEOUT_IN_S`,
  `TORCH_NCCL_*`). *(The CJK token-salad observed on `rl-131k-cpdcp2r3-think2507-r9`
  was NOT an NCCL issue — it was the FusedMoE `w13` gate/up swap not re-applied on the
  disaggregated RL weight update, fixed by `SKYRL_W13_RELOAD_BRACKET` [MarinSkyRL
  `2bb70a88`; default on]. Record: `/Users/benjaminfeuer/Documents/agent_logs/2026-06-27_coreweave_nccl_defaults_doubt.md`.)*
- **Egress: CoreWeave nodes have internet.** Models/data are pulled from HF **online** —
  do NOT set `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE` (contrast Leonardo/Jupiter compute
  nodes, which have none). The cost is the transient HF-weight-resolution flake below.
- **Storage/scratch:** ephemeral per-node disk via the `--disk` request (default
  `512GB`); multi-node Ray rendezvous + banked traces go through the CW `s3://` object
  store (`s3://marin-us-east-02a`; store moved R2→CW 2026-07-05, see Scheduling), not
  node-local disk. No shared persistent POSIX scratch like Leonardo's `$WORK` —
  checkpoints/exports go to HF / the object store.

---

## Scheduling & multi-node particulars

- **Gang scheduling.** `--num-nodes N` → `replicas=N` whole `H100x8` tasks. For GPUs
  with replicas>1, `resolve_multinode_defaults` returns
  **`CoschedulingConfig(group_by="leafgroup")`** — all N nodes co-scheduled on **one
  InfiniBand leaf fabric**, all-or-nothing. `cw-us-east-02a` enables **Kueue gang
  admission** (`kueue.cluster_queue: iris-cq`, `host_network: true` for NCCL/IB), so the
  N-task gang admits **atomically** (all N whole nodes granted, or it queues). At submit
  you see `replicas=N, coscheduling=leafgroup`; pods then sit **SchedulingGated**
  (normal Kueue gang pre-admit) until admitted.
- **⚠ JUST SUBMIT — do NOT pre-check free-node count and withhold the submit (operator 2026-07-10).**
  A gang sitting `SchedulingGated`/pending because a transient `cw-hpc-verification`/`nhc-*` health-check
  sweep or another tenant's job holds nodes is NORMAL — **Kueue admits it atomically the moment N whole
  nodes free, with zero babysitting.** Polling the free-node count and refusing to submit until ≥N are free
  is the WRONG pattern (it was mine, corrected): it keeps the job OUT of the queue entirely, so it never
  gets its turn, and every transient sweep looks like a permanent block. **Submit, let Iris/Kueue schedule,
  report the id + state (running or SchedulingGated — both fine), move on.** The health checks are NOT
  blocking — the scheduler handles them. The ONLY real "doomed gang" is a CONFIG mis-size that can't fit one
  IB leaf (e.g. `--cpu 64` → the `topology 'infiniband' allows to fit only 2 out of N` message below);
  `--cpu 48` avoids it. Transient occupancy is not that — never gate a submit on it.
- **The single-IB-leaf gang constraint is what `--cpu 48` is about.** The gang must fit
  on ONE IB leaf; with `--cpu 64` only ~2/32 nodes have ≥64 free cores (the daemonset
  overhead above), so an N-node single-leaf gang sits SchedulingGated forever with a
  Kueue `topology 'infiniband' allows to fit only 2 out of N pod(s)` message. `--cpu 48`
  fits all nodes → admits immediately (QuotaReserved=True).
- **Multi-node Ray rendezvous via an `s3://` object-store bucket.** `--num-nodes>1`
  REQUIRES `--rendezvous-dir` (the launcher hard-errors otherwise). Use an `s3://` URI
  under the cluster's default bucket (`marin-us-east-02a`), e.g.
  `s3://marin-us-east-02a/iris/rl-<slug>/<run>`.
  **✅ DURABLE PATTERN (prefer this) — DON'T hardcode the region bucket; derive the storage root
  from `marin_prefix()` (`rigging.filesystem`, returns `data_config().resolved_root()`), which
  AUTO-RESOLVES to the active cluster's correct bucket for your job.** Build `--rendezvous-dir` /
  `--s3-output-dir` / `--gcs-output-dir` (and read paths) off `marin_prefix()` and a launch follows
  a store migration automatically — future-proofing against exactly the R2→CW break below. Region
  helpers: `marin_prefix_for_region(region)` (`marin.rl.placement`), `marin_region()`. Hardcode the
  `s3://marin-us-east-02a` / `gs://marin-models-us` literal only as a fallback when you can't call it.
  **⚠ Store moved R2 (`s3://marin-na`) →
  CW (`s3://marin-us-east-02a`) on 2026-07-05 (marin `c7caecc95a`):** pods now inject CW
  creds + `AWS_ENDPOINT_URL=cwlota.com` and can NO LONGER reach `s3://marin-na` (R2) —
  a `marin-na` rendezvous PUT resolves to the nonexistent `marin-na.cwlota.com` and
  STALLS (this killed CP4 v1/v2/v3). The cluster injects working creds into every task
  pod via the **`iris-task-env` k8s Secret** (`envFrom`, because
  `storage.remote_state_dir` is an `s3://` URI), so **no external creds are needed** —
  and you must **NOT forward `AWS_*`/`R2_*`**: explicit container `env` overrides
  `envFrom`, so forwarding the launch host's `AWS_*` (different account, no
  `AWS_ENDPOINT_URL`) clobbers the pod's injected creds and silently targets real AWS
  S3. Use a **fresh sub-path per run** so a stale head file from a prior attempt isn't
  picked up. Mechanism: one `start_rl_iris_controller.py` per node; rank 0 writes
  `ray_head.json` to the rendezvous, workers poll for it and join; rank 0 publishes
  `ray_head.done` on completion.
  **⚠ Inspecting a `marin-us-east-02a` object from the Mac (2026-07-13): you CAN'T — it's
  in-cluster-only.** The store's endpoint `AWS_ENDPOINT_URL=cwlota.com` is CoreWeave's
  **in-cluster LOTA** address; from the laptop it just **connect-times-out** (`http://cwlota.com/...`
  is unroutable). Sourcing the `iris-task-env` / `finelog-cw-use02a-env` secret Mac-side does NOT
  help: the finelog secret's public CloudFlare-R2 endpoint (`…r2.cloudflarestorage.com`) is for
  `s3://marin-na` only; `marin-us-east-02a` lives on LOTA. And a plain `aws s3 ls s3://marin-us-east-02a/…`
  with the laptop's default creds hits **real AWS S3** → `AccessDenied` (the bucket isn't there).
  **To list/read a `marin-us-east-02a` object, `kubectl -n iris exec` into a Running TASK pod**
  (any of your own `iris-<user>-…` workers — it already has `AWS_*` + `cwlota.com` reach via
  `iris-task-env`; do NOT exec into someone else's active training pod, e.g. dlwh's `…grug-train-cw…`)
  and use **boto3 with `Config(s3={"addressing_style":"virtual"})`** — LOTA **rejects path-style**
  (`PathStyleRequestNotAllowed`), which is boto3's default, so an unconfigured client fails even
  in-cluster. One-liner: `kubectl -n iris exec <pod> -- python -c 'import boto3,os;from botocore.config import Config;s3=boto3.client("s3",endpoint_url=os.environ["AWS_ENDPOINT_URL"],config=Config(s3={"addressing_style":"virtual"}));print([o["Prefix"] for o in s3.list_objects_v2(Bucket="marin-us-east-02a",Prefix="<prefix>/",Delimiter="/").get("CommonPrefixes",[])])'` (subdirs are `CommonPrefixes`→`["Prefix"]`; top-level files are `Contents`→`["Key"]`).
  (Aside — the on-disk shape of a marin/Levanter *training* checkpoint there: `<run>-<hash>/.executor_info`
  + `checkpoints/step-N/{metadata.json, manifest.ocdbt, d/<content-addressed blobs>}` = **Orbax OCDBT /
  TensorStore JAX** format, NOT HF transformers — no `config.json`/`*.safetensors`/`tokenizer.json`;
  needs an explicit Levanter→HF export to become loadable by `transformers`.)
- **The `gpu-rl` image is deps-only; source is synced at runtime.** The image
  (`ghcr.io/open-thoughts/openthoughts-agent`, pinned by **immutable `@sha256:` digest**
  in `launch_rl_iris.py:DEFAULT_RL_DOCKER_IMAGE` — NOT the floating `:gpu-rl` tag, which
  stale-caches under `imagePullPolicy: IfNotPresent`) bakes the RL conda venv
  (`/opt/openthoughts/envs/rl`: torch 2.11 + the **vLLM fork built from source** +
  flash-attn 2.8.3), **MarinSkyRL editable** at `/opt/skyrl`, and harbor. The launcher
  syncs the **local OT-Agent workspace to `/app`** (first on PYTHONPATH) → first-party
  edits live on the next launch **without an image rebuild**. A MarinSkyRL fix that
  landed after the image build can be picked up live via `--skyrl-ref <commit>` (editable
  checkout); only the compiled vLLM fork requires an image rebuild (then **bump the
  digest**, using the immutable `:gpu-rl-<gitsha>` tag's digest).
- **⚠ BUILD THE IMAGE MULTI-LAYER (`SINGLE_SNAPSHOT=0`) — a single >8 GB layer is
  UN-PULLABLE cold over the CoreWeave→ghcr egress.** A kaniko `--single-snapshot` build
  collapses everything kaniko adds into ONE ~16.6 GB layer; the first **fresh** pull of
  that single-stream layer never completes — containerd restarts each attempt from byte
  0 (`short read: expected … got <N>`) and dies at 8–11 GB every time, so all pods sit
  `ImagePullBackOff` indefinitely (NOT transient — don't wait it out; a relaunch
  re-pulls the same blob). The incremental-`FROM`-base trick does NOT help (same 16.6 GB
  base pull). **Fix = re-layer:** build `SINGLE_SNAPSHOT=0` (per-instruction layers) and
  split the big torch/nvidia-CUDA installs into a few pinned pre-install RUNs so **no
  single layer exceeds ~8 GB** (validated r5 image `gpu-rl-efd77b98` = 48 layers, max
  3.46 GB → pulled clean, Running in ~5 min). Quality-gate any new image with a
  throwaway 1-pod pull test BEFORE swapping a live job onto it.

---

## Observability (verify, monitor, fetch logs)

**Verify access** (cheap, before submitting):
```bash
# iris-side: my live jobs (JobState: 0=UNSPECIFIED 1=PENDING 2=BUILDING 3=RUNNING 4=SUCCEEDED
#                          5=FAILED 6=KILLED 7=WORKER_FAILED 8=UNSCHEDULABLE)
/Users/benjaminfeuer/miniconda3/envs/otagent/bin/iris --cluster=cw-us-east-02a query \
  "SELECT job_id,state FROM jobs WHERE state IN (1,2,3) AND job_id LIKE '/benjaminfeuer/%'" -f csv

# k8s-side: H100 node headroom (an N-node gang needs N WHOLE free 8-GPU nodes)
kubectl get nodes        # with KUBECONFIG=~/.kube/coreweave-iris-gpu  (Ready count ONLY — see trap below)
```

**Are nodes actually free? Use Kueue + a per-node free-GPU count — NOT `kubectl get nodes` or a pod request-sum.**
`kubectl get nodes` shows *Ready*, not *free*; a naive "sum `requests.nvidia.com/gpu`
over running pods vs 36×8" is wrong twice over (verified 2026-06-26): (a) allocatable
GPUs is **~256, not 288** — only **~32 of the ~36 Ready nodes carry 8 GPUs** (the rest
are util/control nodes with 0 GPU), and (b) the request-sum **undercounts** because some
pods declare GPUs via `limits`, not `requests`. Both errors make a busy cluster look
free → you relaunch into contention and get **preempted by higher-priority `/power`
jobs** (interactive < production). Use the two authoritative signals instead:

```bash
# (1) Kueue ClusterQueue = the SCHEDULER'S OWN accounting — literally what decides gang admission.
kubectl get clusterqueue                 # PENDING WORKLOADS column: 0 = no admission backlog (good sign)
kubectl get clusterqueue iris-cq -o json | python3 -c 'import json,sys; d=json.load(sys.stdin)["status"]; \
print("admitted:",d.get("admittedWorkloads"),"pending:",d.get("pendingWorkloads")); \
print([{r["name"]:r.get("total") for r in f["resources"]} for f in d.get("flavorsUsage",[])])'
#   (nominalQuota lives under .spec.resourceGroups[].flavors[].resources[].nominalQuota; quotas mix units
#    like "1G" for memory — parse GPU separately, don't int() the whole map.)

# (2) CORRECT free whole-nodes = per-node (allocatable_gpu - sum of bound-pod GPU req/limit), count nodes with >=8 free.
kubectl get nodes -o json | python3 -c '
import json,sys,subprocess
nodes=json.load(sys.stdin)["items"]
alloc={n["metadata"]["name"]:int(n["status"]["allocatable"].get("nvidia.com/gpu",0)) for n in nodes}
pods=json.loads(subprocess.run(["kubectl","get","pods","-A","-o","json"],capture_output=True,text=True).stdout)["items"]
used={}
for p in pods:
    if p.get("status",{}).get("phase") in ("Succeeded","Failed"): continue
    nn=p.get("spec",{}).get("nodeName")
    if not nn: continue
    g=0
    for c in p["spec"].get("containers",[]):
        r=c.get("resources",{}); req=r.get("requests",{}) or {}; lim=r.get("limits",{}) or {}
        g+=int(req.get("nvidia.com/gpu", lim.get("nvidia.com/gpu",0)))
    used[nn]=used.get(nn,0)+g
gpu_nodes=sum(1 for a in alloc.values() if a>0)
free=sum(1 for n,a in alloc.items() if a>0 and a-used.get(n,0)>=8)
print(f"GPU nodes:{gpu_nodes}  fully-free 8-GPU nodes:{free}  total free GPUs:{sum(max(0,a-used.get(n,0)) for n,a in alloc.items())}")'
```
Decision rule for relaunching an idle gang: only submit when **`pendingWorkloads == 0`**
AND **fully-free 8-GPU nodes ≥ N** (the gang size; e.g. ≥16 for a 30B+35B pair). If
`/power` is bursting (free nodes oscillating), either wait for it to drain or escalate
to `--priority production` — do NOT churn-relaunch at interactive into contention. The
in-container invocation the launcher ultimately drives is
`uv run iris --cluster=cw-us-east-02a job run …` (the SDK `IrisClient.submit` path);
you do not type that by hand — `python -m cloud.iris.launch_rl_iris` builds it.

**⚑ Priority bands + node cap (operator 2026-07-16).** `--priority {production, interactive, batch}`
(`cloud/iris/launch_rl_iris.py` → `PRIORITY_BAND_*`; default `interactive`). Bands are ordered
production > interactive > batch; a higher band preempts a lower one.
- **`--priority batch` → NO node cap. Surge freely.** Batch is the lowest band, fully preemptible — it
  yields to every higher-priority job (ours or a teammate's), so it can never block anyone. Submit as many
  batch nodes as the clusters will hold across BOTH CW clusters.
- **Non-batch (`interactive`/`production`) → the ~24-node soft cap applies** across ALL my non-batch CW
  jobs combined (these contend with / preempt other tenants, so keep the footprint bounded; 24 is the
  current operator-set value — treat the latest authorization as binding).
- **Accounting:** sum only **non-batch** node usage against the ~24 cap; batch jobs don't count.
- **Cross-cluster moves are autonomous.** Relocate a job between `cw-us-east-02a` and `cw-rno2a` freely when one
  cluster is packed and the other has room (East fills with larger teammate runs while rno2a's 65 nodes sit
  near-empty; a job gated `SchedulingGated` for hours on one admits in seconds on the other). Placement is
  launch-flag-level (`--cluster` / `--cluster-config` / `--rendezvous-dir`); the rl_config is mostly
  cluster-agnostic. **Before any move, grep the config for hardcoded interface/PF/host/region values in
  `extra_env`.** The trap is **`NCCL_SOCKET_IFNAME`**: East's Ethernet PF `enp157s0np0` does not exist on rno2a,
  so a hardcoded PF dies upstream of everything with NCCL `Bootstrap: no socket interface found`. Use the
  cluster-PORTABLE exclusion list `"^ibs,ibp,lo,docker,veth,cilium,lxc"` (matches cw-rno2a.yaml's default;
  auto-picks Ethernet on either cluster) instead of a hardcoded PF. Kubeconfigs: East
  `~/.kube/coreweave-iris-gpu` (`--cluster=cw-us-east-02a`), rno2a `~/.kube/coreweave-iris` (`--cluster=cw-rno2a`).
- **⚑ Priority preemption now actually WORKS (marin #7207 + #7206, merged 2026-07-16) — this is what makes
  "batch yields to everyone" true.** BEFORE #7207 it did NOT: multi-host gangs are admitted through Kueue (pods
  held `SchedulingGated` until the whole Workload admits), so the native kube-scheduler PriorityClass preemption
  never saw them, and the ClusterQueue carried no preemption policy → a higher-priority gang could NOT evict
  running lower-priority `batch` gangs, and a full cluster of batch would **starve** it indefinitely (Kueue's
  default `BestEffortFIFO` even backfilled freed nodes with the next small batch gang). #7207 makes **Kueue
  MANDATORY on the k8s backend** (composer / LocalQueue reconcile / pod builder all fail fast if
  `kubernetes_provider.kueue.cluster_queue` is unset — for us it's `iris-cq`; no non-Kueue path remains, which
  is what makes preemption sound — a single-pod GPU job that bypassed Kueue used to silently defeat preemption
  of the gangs beside it) and adds `preemption.withinClusterQueue: LowerPriority` so higher bands evict lower.
  #7206 makes preempt-and-place **atomic** (the preemptor binds to the worker its victim frees — no gap where
  the freed worker is stranded, stolen by a solo first-fit task, or over-preempts fresh victims tick after
  tick; gangs re-form on the freed slice). **Implications for us:** (a) the "surge freely on batch, it never
  blocks anyone" model is now genuinely sound — batch WILL be evicted the moment a higher band needs the nodes;
  (b) precisely because batch now gets preempted for real, the ckpt→s3 durable-resume fix below is load-bearing
  for any long batch run.
- **⚠ Preemption trade-off — checkpoint-resume is NOT yet plumbed for CoreWeave preemption (2026-07-16).** A
  batch job WILL get preempted when a higher band needs the nodes. But our RL runs write the resumable
  checkpoint to **ephemeral pod-local disk**, not s3: `ckpt_path` is null in `hpc/skyrl_yaml/iris/*.yaml` →
  auto-derives to `{experiments_dir}/{job}/checkpoints` and `experiments_dir` defaults to the in-container
  `/app/experiments` (`cloud/iris/launch_rl_iris.py`), which a preempt/re-bring-up wipes. The trainer's
  save+resume is correct and s3-capable (`resume_mode: latest` is set; fsspec supports s3://), so the ONE fix
  is to point `trainer.ckpt_path` at a **stable per-job** `s3://marin-us-east-02a/iris/<job_name>/checkpoints`
  (auto-derive in the launcher, mirroring the `trials_dir` pattern at `launch_rl_iris.py:879`, OR set it in the
  YAMLs). Until that lands, a preempted batch job **restarts from step 0** — so batch is safe only for
  short/smoke runs; do NOT park a long run on batch expecting resume. (SLURM resumes fine because
  `experiments_dir` there is durable `$WORK`.) Verify the s3-ckpt round-trip at 80B FSDP scale at a smoke
  before trusting it in production.

**Monitor liveness — state-poll, NOT a log-string watch.** Poll the authoritative iris
job lifecycle state, never grep rank-0 logs for a content string. A clean kill /
eviction / preemption / early crash often emits **no** terminal log line, and the pods
are reaped, so a content-watch sits idle while the job is gone (this is how the
`rl-131k-cpdcp2r3` watch missed the run ending `killed`/"Terminated by user" with 0
pods). The watch primitive:
```bash
PY=/Users/benjaminfeuer/miniconda3/envs/otagent/bin/python
$PY scripts/iris/watch_job_state.py /benjaminfeuer/<job> --once --json    # authoritative state now
$PY scripts/iris/watch_job_state.py /benjaminfeuer/<job> --interval 60     # watch until terminal
#   wraps `iris job summary --json` (auth) + SQL `query` fallback + kubectl pod cross-check;
#   "no record AND 0 pods" => terminal `absent`. Importable: get_job_state() / watch().
```
Log-content greps (`scripts/iris/analyze_job_history.py`) are for sel_rows / EPDIag /
throughput **science only** — never for liveness/terminal detection. (The launch
HOW-TO's §8 carries the full monitoring rule.)

**⚠ Empty job-state ≠ still-running — the `jobs` table is PRUNED, not a durable
ledger. Absence-after-existence is TERMINAL; never hand-roll a `state IN (4,5,6)`
watcher.** A completion watcher that polls `iris query "SELECT state FROM jobs WHERE
job_id='...'"` and only fires on a terminal `state` (4 SUCCEEDED / 5 FAILED / 6 KILLED)
will **poll forever past a real completion**: the controller's background pruner DELETES
a terminal job's row, so once pruned the query returns an **EMPTY result** (no `state`
row) and the terminal case never matches. Empty ≠ "still running" — it means "no record",
which for a job you have already seen means **"finished + aged out."** (`job summary`
reads the same DB, so it too returns "no record" once the row is pruned — this is
consistent, not a summary-vs-query discrepancy.)
- **Pruning mechanics (marin `lib/iris/src/iris/cluster/controller/`):** the pruner
  (`pruner.py:_prune_terminal_jobs`) runs every `prune_interval` (**1 h**,
  `controller.py:236`) and deletes terminal jobs older than `job_retention` (**7 days**
  default, `controller.py:239`; **no override in `lib/iris/config/cw-us-east-02a.yaml`**).
  **BUT a FEDERATED job ages out sooner + less predictably:** `find_prunable_job`
  (`reads.py:488-505`) excludes non-local rows (`jobs.cluster == 'local'`,
  `types.py:107`) from the 7-day time-prune — a federated job's parent-side **mirror row
  is instead deleted the moment the PEER issues a tombstone** (`federation_changelog.tombstone=1`,
  `schema.py:641`) that federation-sync mirrors. A job submitted via the `iris.oa.dev`
  meta-scheduler (parent-minting → routed to a peer CW cluster) is exactly this case: its
  row can vanish right after the peer prunes it, well inside a 7-day window. (This is how a
  ~120 s SQL-state watcher silently missed a Grug HF-export completion — the job finished,
  published, aged out, and the watcher polled an empty row forever.)
- **ROBUST recipe:**
  1. **Prefer `watch_job_state.py` — it already handles absence-as-terminal.** Its order
     is `job summary` → `query` fallback → if BOTH have no record **AND** `kubectl` reports
     **0 pods**, it returns `state="absent"` (`is_terminal=True`, exit **2**). It never reads
     an empty row as "running." **REQUIRES a correct `KUBECONFIG`** (`~/.kube/coreweave-iris-gpu`
     for cw-us-east-02a; the **peer's** kubeconfig for a federated job whose pods live on the
     peer) — if kubectl can't run, `count_live_pods` returns `None`, `get_job_state` RAISES a
     *transient* error, and the watch loop **keeps waiting** (absence can't be confirmed). Export
     the right kubeconfig, and for `--once`/`watch` do NOT pass `--no-pods` when you need the
     absent verdict.
  2. **If you must hand-roll:** treat the row present with a terminal `state` **OR the row
     ABSENT (empty result) with 0 live pods** as terminal→inspect. Never treat empty as "keep
     waiting."
- **⚠ `absent` proves the job ENDED, not that it SUCCEEDED** — the terminal state (4/5/6) was
  pruned along with the row, so a watcher that raced the prune learns only "gone." For a
  pass/fail verdict either read the state **inside** the retention window, or use the artifact
  signal below.
- **✅ Artifact-producing jobs (HF export, s3/gcs write) — watch the ARTIFACT directly; it is
  the ground-truth completion signal AND it is DURABLE (the job row is not).** For an HF
  export, poll the repo/revision existence rather than the ephemeral job-state:
  ```bash
  PY=/Users/benjaminfeuer/miniconda3/envs/otagent/bin/python
  $PY -c 'from huggingface_hub import HfApi; import sys; print(HfApi().repo_exists(sys.argv[1]))' <repo_id>
  # stricter — require the expected files landed (not just an empty repo shell):
  $PY -c 'from huggingface_hub import HfApi; import sys; print(HfApi().list_repo_files(sys.argv[1], revision="main"))' <repo_id>
  ```
  For an s3/gcs object write, poll the object path exists (for `marin-us-east-02a`, in-pod
  boto3 with the LOTA `virtual` addressing style — see §Scheduling). Prefer the artifact watch
  for these jobs: it fires on the thing you actually want (artifact published), it is immune to
  job-row aging, and it works even after the job-state has been pruned to `absent`.

**Finelog retains the FULL job log** — retrievable by **time-window pagination** with
the cw-capable iris binary (`/Users/benjaminfeuer/miniconda3/envs/otagent/bin/iris`).
The only real truncation is `--tail`'s line cap; `--since-ms <submitted_at_ms> --no-tail`
returns everything:
```bash
IRIS=/Users/benjaminfeuer/miniconda3/envs/otagent/bin/iris   # KUBECONFIG=~/.kube/coreweave-iris-gpu
$IRIS --cluster cw-us-east-02a query \
  "SELECT job_id,submitted_at_ms,started_at_ms,finished_at_ms,error,exit_code FROM jobs WHERE job_id LIKE '%<job>%'"
$IRIS --cluster cw-us-east-02a job logs /benjaminfeuer/<job> --since-ms <submitted_at_ms> --max-lines 500000 --no-tail
```
(e.g. on dead `rl-131k-cpdcp2r3-v2` one `--no-tail --since-ms <submit>` recovered all
2275 lines spanning the full 10:09:08 → 10:16:21 lifetime, revealing the rank-0 fatal
`ModuleNotFoundError: No module named 'torchtitan'` in `fsdp_utils.py:667 apply_ep` —
the MoE EP path needs torchtitan, not in the gpu-rl image.)

**⚠ Poll / tooling pitfalls (esp. cw-rno2a + pure-RL jobs) — learned 2026-07-14:**
- **`job summary` is FLAKY on cw-rno2a** (intermittent `execute_unary` controller blips). Drive rno2a liveness through `iris query` / `watch_job_state.py --once` instead; retry-or-fall-back rather than trusting one `job summary` failure.
- **jobs-table columns are the `*_ms` names** (`submitted_at_ms`, `started_at_ms`, `finished_at_ms`, `error`, `exit_code`) — a bare `submitted_at`/`failure_count` errors `no such column`. If a `query` column errors, drop it and re-select `name, state` (don't abandon the query).
- **`analyze_job_history.py` does NOT "just work" for cw-rno2a or pure-RL jobs** (supersedes the older "works out of the box" note):
  - Region resolver passes `--config <cluster-name>` to iris, which wants a PATH → it errors `Path 'cw-rno2a' does not exist`. Workaround: pass the config YAML **path** as `--cluster` (e.g. `--cluster /Users/benjaminfeuer/Documents/marin/lib/iris/config/cw-rno2a.yaml`) AND put the cw iris on PATH (`export PATH=/Users/benjaminfeuer/miniconda3/envs/otagent/bin:$PATH`).
  - Even then it **cannot resolve a pure-RL job's output dir** (no `--jobs-dir`/`--gcs-output-dir` in the baked command) → `LookupError`. It is built for jobs with a GCS trace/output dir. **For RL-job metrics, fetch the finelog directly** via `iris job logs --since-ms <submitted_at_ms> --no-tail` (bounded) instead.
- **`iris job logs` has NO server-side grep** (only `--since-ms`/`--since-seconds`/`--max-lines`/`--tail`). Never `--max-lines >~200 | grep` a long job into the Mac (OOM-crash risk — see §Monitoring & debugging practices / the log-resource-discipline memory). Bound every fetch with a tight `--since-ms` window + `--max-lines`.

**Per-trial `TimingInfo` duty-cycle read — CoreWeave in-pod access (recipe lives in the harbor project doc):**
The reusable recipe — the `result.json` `TimingInfo` field→phase map, the duty-cycle fraction math, the
lease-race / burst≠churn checks, and the aggregates-only discipline — is a **harbor trial artifact**, so it lives in
**`.claude/projects/harbor/harbor.md` §"Per-trial `TimingInfo` duty-cycle recipe"**. Only the cluster-specific
*access* to the trials bucket is here:
- **Trials bucket is in-cluster-only from the Mac** (`marin-us-east-02a`, LOTA — see §Scheduling), so **aggregate
  IN-POD and transfer aggregates only** (never sync raw trials/logs to the Mac).
- **Trials path:** `s3://marin-us-east-02a/iris/<run-name>/trace_jobs/<trial>/{config.json,result.json,agent/,artifacts/,verifier/}`. The `<run-name>` dir is set from `trainer.run_name` (e.g. `rno2a-30b-coder-v0l`) — NOT the config's literal `terminal_bench.trials_dir` (the coordinator rewrites it). Discover it by listing `iris/` in the bucket and matching the job name.
- **In-pod S3 access:** `kubectl --context marin-rn02a_RNO2A -n iris exec <task-pod> -c task -- python -c '...'` using boto3 with `endpoint_url="http://cwlota.com"` + `Config(s3={"addressing_style":"virtual"})` (LOTA rejects path-style, same as §Scheduling). List `result.json` keys with one paginated `list_objects_v2` over `.../trace_jobs/`, sort by `LastModified` desc, read the newest ~200 (steady-state duty cycle) / ~500 (error + re-provision tails); print only computed medians/percentiles.

Then apply the harbor.md recipe to the fetched `result.json` fields.

---

## Binding gotchas

> **⚠ `--max-retries ≥1` for the transient HF weight-resolution flake.** At scale (e.g.
> 32 FSDP ranks each resolving sharded safetensors online) one rank can hit a transient
> HF Hub HTTP/EOF failure → transformers reports the generic `… does not appear to have
> a file named model.safetensors`; with `max_retries=0` that one rank SIGKILLs the whole
> gang. `--max-retries 1` re-brings-up the gang on that failure (time-only cost). A
> weight-resolution retry wrapper landed in MarinSkyRL (commit `0b2b05b`); keep
> `--max-retries ≥1` as belt-and-suspenders. (Durable alternative: pre-stage the model
> into the image's HF cache / a shared snapshot before the FSDP workers start, or raise
> `HF_HUB_DOWNLOAD_TIMEOUT`.)

> **⚠ `--memory` default is `1400GB`** (`launch_rl_iris.py:DEFAULT_MEMORY_PER_NODE`).
> `1800GB` (≈1676 GiB) sits so close to node-allocatable (~2014 GiB) that after the
> daemonset + persistent-reservation overhead a leafgroup (all-or-nothing, one IB leaf)
> gang can't fit all its pods → Kueue `topology 'infiniband' allows to fit only K of N
> … excluded: resource "memory"` → **SchedulingGated stall**. **`1400GB` is validated**
> for the 8-node 131k EP8 run (admits cleanly + does the full weight-load with no
> cgroup-OOM); drop to `1000–1200GB` for 2-node smokes. On an admission stall, LOWER
> `--memory` toward the real need — **never raise a cap**. (The old `512GB` was the
> opposite footgun — a weight-load cgroup-OOM; `1400GB` is the validated middle, and
> now the default so no flag is needed.)

- **Ray agent ports collide with `worker_ports` nondeterministically — pin them all.**
  `ray start` (head AND worker) lets Ray RANDOMIZE several system ports
  (`metrics_export`, `runtime_env_agent`, `dashboard_agent_grpc`, …) from the ephemeral
  zone that overlaps the default `worker_ports` range **10002–19999**. A random landing
  inside it aborts the node (`ValueError: Ray component worker_ports is trying to use a
  port number <N> that is used by other components`) — **nondeterministic** (passes or
  fails run-to-run; a likely cause of intermittent "long build then die" CoreWeave
  deaths). A head-only or single-port pin is INSUFFICIENT (the randomized port just
  moves to another agent). **Fix (committed `beda7a7f`,
  `cloud/iris/start_rl_iris_controller.py:_ray_port_flags`):** pin ALL of them outside
  the range on head+worker — `metrics_export=8090, runtime_env_agent=8092,
  dashboard_agent_grpc=8093, dashboard_agent_listen=8094, node_manager=8076,
  object_manager=8077`. Rides the `/app` upload (no rebuild).
- **Nodes have NODE-LOCAL storage (no shared GPFS) → stage the agentic task dataset on
  EVERY node.** Unlike the SLURM clusters (shared GPFS — one rank extracts, all nodes
  see it), CoreWeave's `/opt/openthoughts/tasks` is node-local, so a rank-0-only
  `parquet→tasks` extraction leaves the 7 workers with EMPTY task dirs → every rollout
  throws `FileNotFoundError: /opt/openthoughts/tasks/<dataset>/<instance>/task.toml` →
  reward 0. This is a SILENT data-starvation: the compute path looks green
  (grouped-mm/R3 fine, no crash) but `avg_num_tokens≈1.0` and all rewards are 0. **Fix
  (committed `7c135780`):** the launcher forwards `--train-data` to the controller,
  which stages on every node before Ray via `resolve_rl_train_data`. Verify in bring-up:
  each rank logs `Staging train_data on this node (rank N/8)` → `[extract] … Done`
  before rollouts.
- **Transient self-healing on bring-up is NORMAL, not a fault to salvage:** a `ghcr.io`
  blob EOF → `ImagePullBackOff` self-heals (kubelet retries); `shm_broadcast: No
  available shared memory broadcast block found in 60s` is **benign** (engines idle-wait
  while the policy mesh loads weights).
- **k8s does NOT shell-expand `$VAR` in injected env.** Hardcode literal paths in a
  config's `extra_env:` (e.g. `LD_LIBRARY_PATH: /opt/openthoughts/envs/rl/lib`, **not**
  `$CONDA_PREFIX/lib`) — the launcher injects env as literal k8s values. (Config-authoring
  detail; full rules in the launch skill §4.)

---

## Daytona (orgs + sandbox lifecycle)

**Org routing — RL ALWAYS on the RL org.** Agentic RL (Daytona backend) runs on the
dedicated **Daytona RL org** (`DAYTONA_RL_API_KEY`, sha1 `f8f0296c1680`) — ~6000 fast
sandboxes, so many concurrent gangs fit; do NOT spread RL across other orgs to "balance
load." Route it via the committed **`--daytona-api-key-env DAYTONA_RL_API_KEY`** flag
(`cloud/iris/launch_rl_iris.py`), NOT a pre-launch `export DAYTONA_API_KEY=…` — the
launcher re-sources `secrets.env` after any shell export (`hpc/iris/env.py` "file
overrides shell") and CLOBBERS it. VERIFY in-pod:
`kubectl exec … printenv DAYTONA_API_KEY | sha1sum` == the RL key hash.
- **DATA (`DAYTONA_DATA_API_KEY`) and main (`DAYTONA_API_KEY`) are interchangeable**
  general-purpose pools (evals use main, datagen uses DATA; either takes the other's
  overflow).
- **B org (`DAYTONA_B_KEY`) is USER-ONLY — NO agents / no automated jobs there, ever**
  (~250-sandbox cap; do not confuse it with the RL org). Never pass `DAYTONA_B_KEY` in
  an agent launch.

**Killed jobs ORPHAN their in-flight sandboxes → reap after every kill.** Harbor
auto-destroys a trial's sandbox only when the trial COMPLETES normally (the live
RolloutCoordinator tears it down). An `iris --cluster coreweave job stop` HARD-kills →
the coordinator dies **before** destroying its in-flight sandboxes → ~250–384 sandboxes
ORPHAN and linger. (Verified 2026-07-04: killing 3 probes in a day left ~579 stale
>120min-idle sandboxes on the RL org, while a concurrent live job cycled cleanly —
teardown WORKS for completed trials; the pile was 100% kill-orphans.) Root cause:
`DaytonaEnvironment` set `auto_stop_interval_mins=0` (auto-stop OFF) +
`auto_delete_interval_mins=0` (delete-immediately-on-stop armed but never fires —
`0` = immediate, `-1` = never; auto_stop=0 defeated delete-on-stop). **Fix — harbor
`1143aba8`** (marin-community/harbor `penfever/working`): `auto_stop_interval_mins`
default 0→5, so an idle orphan stops after 5 min → auto_delete removes it (the idle
timer resets on every sandbox exec, so active trials never trip it). **Takes effect on
the NEXT image rebake** (harbor is baked into the gpu-rl RL + eval images). **Until
rebaked, manually reap after every kill:**
`python scripts/daytona/cleanup_stale_sandboxes.py --api-key-env DAYTONA_RL_API_KEY --threshold 120 --delete`
(`--threshold ≥120` so you never reap an active trial while OTHER jobs run — active
agentic trials idle 15–60 min, never >2h). Orphans take ~1h to cross the idle threshold,
so a kill + immediate reap MISSES them — reap ~1–2h later (or let the
monitor/harvest cron catch them at its >1200-count trigger).

## Monitoring & debugging practices

- **A post-bring-up TRAINER wedge stays `state 3 RUNNING` and emits NO terminal log — RUNNING ≠ stepping.**
  A job can complete vLLM engine bring-up cleanly (weights loaded, KV cache sized, CUDA graphs captured) and then
  the trainer side (megatron/FSDP2 re-sync, first ppo_train step, mesh init) HANGS — the last real line is a
  benign heartbeat (e.g. an `fd-monitor` from `skyrl_entrypoint`), then silence for the whole wall-clock until the
  job goes terminal. Do NOT read "vLLM up + state 3" as healthy. Liveness for a trainer = **forward advancement**
  (a fresh training step / rising banked-gs / the run's `finished_at` horizon moving), never engine-bring-up
  completion. (2026-07-17: `megatron-parity-v0m-mcore-east16/east17` both looked "up" then wedged post-bring-up
  for ~68 min → terminal `FAILED(5)`; the log-string watcher never fired.)
- **While ANY debug thread is in flight, a single 30-minute cron re-verifies EVERY active debug job's authoritative
  state** (jobs-table `state` / `watch_job_state.py --once`), independent of each job's own watcher — the cron
  backstops watchers that go silent on a clean kill/eviction/post-bring-up wedge. Per-job watchers are still armed,
  but the cron is the catch-all. Retire the cron when the debug roster drains. (Behavioral rule:
  memory `watcher-and-debug-monitoring`.)
- **Throughput / max-concurrency probes: fixed 60/120-min check-ins + DIRECT log reads
  — NEVER watcher-park a subagent.** Parked "bring-up/generation watcher" subagents
  re-invoke unreliably and once stalled a 35B max-conc probe **~8h** with no number. The
  throughput signal (vLLM scheduler `Running:/Waiting:/GPU KV cache usage:` lines +
  in-flight Daytona sandbox count) is readable DIRECTLY in one command
  (`iris job logs --since-ms … | grep 'loggers.py.*Running:'` + a Daytona `list()`
  count). Take the measurement at ~60 min (bring-up ~10–15 + sandbox ramp ~15–30 +
  steady-state), confirm at ~120 min, then move on; drive it with scheduled ticks, NOT
  an indefinite park.
- **Debug on the REAL failing config with the fix as the SOLE variable.** A/B:
  identical config, two runs differing only in the fix (env flag or `--skyrl-ref`) —
  fix-OFF must FAIL and fix-ON must PASS; one arm alone proves nothing. Do NOT build
  reduced/faster "canary" configs (smaller ctx/batch/steps) to debug a bug you have NOT
  proven they reproduce — a green result with no failing control is INCONCLUSIVE
  (`canary_moe_dispatchfix_8k` [config retired 2026-07-08] "validated" a MoE-wedge fix by completing 2 steps but was
  never run WITHOUT the fix → never proved the 8k config reproduces the 131k wedge). A
  slower GUARANTEED repro beats a fast UNCERTAIN one; reduced configs are for SPEED of
  a *proven* repro only.
- **py-spy forensic on a wedged/hung job — capture BEFORE the kill.** A kill destroys the only live evidence of a
  hang (NCCL flight-recorder dumps are lost to pod GC), so on a suspected deadlock / collective-desync, py-spy the
  stuck ranks first, then recommend the kill with the stacks as evidence. VERIFIED working on CoreWeave despite
  `ptrace_scope=1` (the `task` container carries `CAP_SYS_PTRACE`):
  `kubectl exec -n iris <pod> -c task -- /opt/openthoughts/.venv/bin/py-spy dump --pid <ray::skyrl_entrypoint PID>`.
  Dump several ranks to `agent_logs/`, and compare a LEADING vs LAGGING rank to pin which rank is stuck in which
  collective (the desync source). **A py-spy snapshot is NOT a wedge verdict:** ranks at `dist.barrier()` while
  others are mid-`forward`, plus a single NCCL `Watchdog caught collective … ran for N ms` LOG LINE, does not prove
  a terminal deadlock — a real tripped watchdog ABORTS the process (pod crash/restart). Require pod-restarts==0 +
  an actual abort/terminal state + stalled FRESH logs (all nodes) before calling wedge, and reconcile any cited
  timeout against the run's own timeline (a 3600 s collective can't predate the phase it is in). On Leonardo py-spy
  is BLOCKED (`ptrace_scope=2`) — use another forensic.

## Cross-reference

- **Launch procedure** (the flag set, config map + node-count derivation,
  config-authoring rules for `hpc/skyrl_yaml/iris/`, gang/rendezvous walkthrough,
  bring-up checklist, monitoring + completion) → the **`rl-agentic-launch-iris`** skill.
- **TPU (datagen/eval) Iris** → `iris_job_lifecycle.md` + `iris_google_tpu_cloud_hardware.md`
  (a DIFFERENT cluster; see the scope banner above).
- **Code (canonical, MarinSkyRL `cloud/iris/`):** `cloud/iris/launch_rl_iris.py` (launcher,
  digest pin, `extra_env` forwarding, AWS_* warning), `cloud/iris/start_rl_iris_controller.py`
  (the per-node Ray rendezvous controller), `cloud/iris/run_rl.py` + `cloud/iris/rl_config_translation.py`
  (in-pod config parse → Hydra args), `cloud/iris/configs/*.yaml` (recipes). The former OT-Agent copies
  (`rl/cloud/launch_rl_iris.py`, `scripts/iris/start_rl_iris_controller.py`, `scripts/iris/tilelang_cache_sync.py`)
  have been removed — MarinSkyRL `cloud/iris/` is the sole home.
- **Standing constraints** (≤6 RUNNING RL/cluster, `enable_db_registration: false`, a3
  CONCLUDED, Daytona snapshot caps HARD, never kill a RUNNING job / `iris cluster
  restart` without permission) — see `CLAUDE.md §Always` + the launch skill §7.

## Marin GitHub-issue monitoring + research (external skills)

CoreWeave/Iris is **Marin's** cluster — when monitoring/triaging/updating the Marin
GitHub issues this work touches (e.g. the RL-launch-on-Iris workflow issue,
cluster/quota threads, upstream fixes we depend on), use the **mumwelt skills** over
the local offline Marin mirror. Which skill + how to invoke = **`.claude/projects/mum/mum.md`**
(prefer `marin-research`; `marin-context` is the faster/less-accurate lookup).

These give the **context** to monitor + decide; the actual issue/PR **update** still
goes through the `gh` CLI.

## iris.oa.dev federated GPU submission (NEW 2026-07-09, operator — IN FLUX, bugs expected)
- **New path:** submit to `iris.oa.dev` requesting GPUs; an **H100 request auto-routes to a CW cluster** via a **meta-scheduler** (a simple scheduler over the per-cluster schedulers that only decides "which cluster can I go to"). `--target-cluster <name>` pins a specific CW cluster. **Only OpenAthena accounts are authorized for CW** (we are `ben.feuer@openathena.ai` → authorized).
- **Auth:** the new path needs `iris login` with the openathena.ai gmail (kludgy OAuth). **Iris REJECTS jobs submitted via the OLD-STYLE SSH TUNNEL** (to keep legacy non-OA users off CW).
- **Our EXISTING paths still work (operator + validated 2026-07-09):** the current CW submission (MarinSkyRL `cloud/iris/launch_rl_iris.py` → `bundle.controller.tunnel()` + `KUBECONFIG=~/.kube/coreweave-iris-gpu`) and the marin-TPU eval submission (`launch_eval_iris` `--cluster=marin`) — 80B **v5** launched + a **TPU eval refill (r438)** both succeeded via the existing path this session, so we are NOT being rejected. `iris.oa.dev` is an EASIER path that becomes the default as bugs are fixed; NOT mandatory yet.
- ⚠ **Known-rough (operator, 2026-07-09):** (1) NO queuing at the main server yet — an H100 request WITHOUT `--target-cluster` is RANDOMLY dispatched to a CW cluster (fix pending: hold at main until a cluster reports enough free GPUs) → on the new path ALWAYS pass an explicit cluster; (2) log-forwarding slow (finelog push pending, ~that night); (3) a long tail of rollout bugs.
- **Our exposure:** we always pass an explicit `--cluster=cw-us-east-02a` (RL) / `--cluster=marin` (TPU-eval), so the random-routing does NOT hit us; SLURM clusters (Leonardo/TACC) don't touch iris. **⚠ OPEN:** confirm whether our `bundle.controller.tunnel()` submission IS the rejected "old-style SSH tunnel" — recent submissions succeeded, so evidently a DIFFERENT (still-valid) mechanism; watch the next submission for an auth/tunnel rejection → if so, migrate the launchers to `iris login` + `iris.oa.dev`. (Serving ingress already uses `iris.oa.dev/proxy/t/*` — see `iris_ingress.md`.)
