# Iris CoreWeave GPU cluster (`cw-us-east-02a`) — access & particulars

Cluster particulars for the **GPU RL path** on Marin's Iris: the CoreWeave
`cw-us-east-02a` H100 cluster (8× H100-80GB + InfiniBand per node), driven via
`python -m rl.cloud.launch_rl_iris` + the `gpu-rl` Docker image. This is the
ACCESS/HARDWARE/SCHEDULING reference; the launch HOW-TO (flag set, config-authoring
rules, bring-up checklist) lives in the **`rl-agentic-launch-iris`** skill — ops =
particulars, skill = procedure.

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
sandbox; use the full interpreter path). There is **no cluster login / no SSH** — you
talk to the cluster through the `iris` SDK over a controller tunnel, and the launcher
uploads your **local** workspace to `/app` (a local commit takes effect on the next
launch immediately — there is no Iris clone to pull).

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
  `NET/Socket : Using [0]enp157s0np0:…`). Ref: `agent_logs/2026-07-08_gpu-rl-ib-enable.md`.
- **NCCL DEFAULTS — use them (MoE-salad doubt FALSIFIED 2026-06-27).** On H100+IB, do
  NOT set the GH200/SIF disables (`NCCL_P2P_DISABLE` / `NCCL_NVLS_ENABLE=0` /
  `NCCL_COLLNET_ENABLE=0`): they cripple the intra-node NVLink all-reduce a TP=8 (DCP)
  engine depends on. NCCL defaults give NVLink intra-node + IB inter-node. Keep the
  observability/raised-timeout env (`NCCL_DEBUG=INFO`, `SKYRL_WORKER_NCCL_TIMEOUT_IN_S`,
  `TORCH_NCCL_*`). *(The CJK token-salad observed on `rl-131k-cpdcp2r3-think2507-r9`
  was NOT an NCCL issue — it was the FusedMoE `w13` gate/up swap not re-applied on the
  disaggregated RL weight update, fixed by `SKYRL_W13_RELOAD_BRACKET` [MarinSkyRL
  `2bb70a88`; default on]. Record: `agent_logs/2026-06-27_coreweave_nccl_defaults_doubt.md`.)*
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
- **The single-IB-leaf gang constraint is what `--cpu 48` is about.** The gang must fit
  on ONE IB leaf; with `--cpu 64` only ~2/32 nodes have ≥64 free cores (the daemonset
  overhead above), so an N-node single-leaf gang sits SchedulingGated forever with a
  Kueue `topology 'infiniband' allows to fit only 2 out of N pod(s)` message. `--cpu 48`
  fits all nodes → admits immediately (QuotaReserved=True).
- **Multi-node Ray rendezvous via an `s3://` object-store bucket.** `--num-nodes>1`
  REQUIRES `--rendezvous-dir` (the launcher hard-errors otherwise). Use an `s3://` URI
  under the cluster's default bucket (`marin-us-east-02a`), e.g.
  `s3://marin-us-east-02a/iris/rl-<slug>/<run>`. **⚠ Store moved R2 (`s3://marin-na`) →
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
you do not type that by hand — `python -m rl.cloud.launch_rl_iris` builds it.

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
the MoE EP path needs torchtitan, not in the gpu-rl image.) The analyzer takes
`--cluster` + `--iris-bin` (auto-resolves the cw-capable iris) so it works for cw out of
the box.

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
  `scripts/iris/start_rl_iris_controller.py:_ray_port_flags`):** pin ALL of them outside
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
(`rl/cloud/launch_rl_iris.py`), NOT a pre-launch `export DAYTONA_API_KEY=…` — the
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

## Cross-reference

- **Launch procedure** (the flag set, config map + node-count derivation,
  config-authoring rules for `hpc/skyrl_yaml/iris/`, gang/rendezvous walkthrough,
  bring-up checklist, monitoring + completion) → the **`rl-agentic-launch-iris`** skill.
- **TPU (datagen/eval) Iris** → `iris_job_lifecycle.md` + `iris_google_tpu_cloud_hardware.md`
  (a DIFFERENT cluster; see the scope banner above).
- **Code:** `rl/cloud/launch_rl_iris.py` (launcher, digest pin, `extra_env` forwarding,
  AWS_* warning), `scripts/iris/start_rl_iris_controller.py` (the per-node Ray
  rendezvous controller).
- **Standing constraints** (≤6 RUNNING RL/cluster, `enable_db_registration: false`, a3
  CONCLUDED, Daytona snapshot caps HARD, never kill a RUNNING job / `iris cluster
  restart` without permission) — see `CLAUDE.md §Always` + the launch skill §7.

## Marin GitHub-issue monitoring + research (external skills)

CoreWeave/Iris is **Marin's** cluster — when monitoring/triaging/updating the Marin
GitHub issues this work touches (e.g. the RL-launch-on-Iris workflow issue,
cluster/quota threads, upstream fixes we depend on), use the **mumwelt skills at
`/Users/benjaminfeuer/Documents/mumwelt/mumwelt/skills/`** (invocable by name; they
shell out to the `mum` CLI over a local offline mirror of all Marin activity — GitHub
issues/PRs/comments, Discord, W&B, weekly summaries):
- **`marin-context`** — search + cite Marin activity (a specific issue/PR/run, "what
  was decided + why", who did what). The default for monitoring a Marin issue/PR's
  state + history before commenting on or updating it.
- **`marin-research`** — multi-subagent deep dive for broad/ambiguous "full picture of
  X" / retro questions one query won't cover.
- **`marin-publish`** — render a finished writeup to a linkable gist (htmlpreview) to
  share.

These give the **context** to monitor + decide; the actual issue/PR **update** still
goes through the `gh` CLI. Check the mirror's freshness first (the skill prompts if
stale). They live OUTSIDE this repo (the `mumwelt` checkout), so they're a referenced
tool, not a committed part of ot-agent.
