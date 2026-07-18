---
name: rl-agentic-launch-iris
description: >-
  Launch / relaunch agentic MarinSkyRL (SkyRL GRPO) RL on Marin's Iris / CoreWeave GPU cluster
  (cw-us-east-02a, 8x H100-80GB + InfiniBand per node) via `python -m cloud.iris.launch_rl_iris`
  (run from the MarinSkyRL repo root) + the gpu-rl Docker image (NO Apptainer SIF). Covers the dense 8B
  FSDP2 arms (seqnorm + TIS) and the MoE 30B-A3B arms (CP + DCP=2 + R3 @ 131k) — the exact launcher flag set
  (`--rl_config`, `--model_path`, `--train_data`, `--num-nodes`, `--rendezvous-dir`, `--job-name`, `--priority`,
  `--cpu`, `--max-retries`), the gang/leafgroup/Kueue multi-node Ray rendezvous, the iris config-authoring
  rules (NO container block, load-bearing top-level `extra_env:` forwarding, disaggregated placement +
  explicit `num_inference_engines`, SIF→Docker env translation), and the bring-up gotchas (`--cpu 48`,
  `--max-retries ≥1`). Use when asked to launch / relaunch an agentic SkyRL RL run on Iris / CoreWeave.
  Cluster access/hardware particulars live in .claude/ops/iris/ops.md (this skill defers to
  it). Reference: cloud/iris/launch_rl_iris.py, cloud/iris/start_rl_iris_controller.py,
  cloud/iris/configs/, .claude/ops/iris/ops.md.
---

> ⚠ **Do not add comments to YAMLs. Report recommendations directly to the supervisor.**

# rl-agentic-launch-iris

> **📍 Iris orientation — read first.** Read the Iris **tools catalog** (`.claude/ops/iris/ops.md`)
> and the **ops directory** (`.claude/ops/iris/` — CoreWeave GPU particulars in `ops.md`, TPU
> lifecycle in `ops.md`) before acting. They carry the binding access/preamble/gotchas +
> helper-script inventory the steps below rely on.

> **⚠ Local clone = ground truth (CLAUDE.md §Always).** ALL code/config edits go in the local Mac checkouts →
> commit. The Iris launcher **uploads the LOCAL workspace to `/app`** (PYTHONPATH puts `/app` +
> `/opt/skyrl/skyrl-train` first), so a local commit takes effect on the next launch immediately — no
> push-then-pull-on-cluster (there is no Iris clone to pull). Still never hand-edit on a remote, leave
> divergent state, or patch-by-rsync. The vLLM fork (compiled) is fixed only by an image rebuild + digest bump.

Agentic RL on Iris runs through **`python -m cloud.iris.launch_rl_iris`** — the self-contained MarinSkyRL
launcher (`cloud/iris/`), run **from the MarinSkyRL repo root** (`~/Documents/MarinSkyRL`, on `main` / any
branch containing `cloud/iris/`). (SkyRL, GRPO, FSDP2 or MoE-EP.) Each rollout is a real **Harbor** agent
episode against a **Daytona** sandbox (`terminal_bench` generator). Target: Marin's **CoreWeave
`cw-us-east-02a`** cluster — **8x H100-80GB + InfiniBand per node**, whole-node exclusive,
gang/leafgroup-coscheduled (NOT SLURM, NOT TPU). The **gpu-rl Docker image IS the runtime** — no Apptainer
SIF, no `hpc.launch`. Cluster-access/hardware/scheduling particulars →
`.claude/ops/iris/ops.md` (kubeconfig, H100 node shape + NCCL rationale, gang/Kueue/rendezvous
mechanics, the `--cpu 48` / `--max-retries` gotchas). The TPU-centric Iris lifecycle (datagen/eval) is a
DIFFERENT cluster — `.claude/ops/iris/ops.md`.

## 1. Prereqs (the pre-launch preamble)

Launch from the local Mac, **otagent py3.12 conda env**, **from the MarinSkyRL repo root**:
```bash
cd ~/Documents/MarinSkyRL                              # `-m cloud.iris.launch_rl_iris` resolves from here; on main (or a cloud/iris branch)
source "${DC_AGENT_SECRET_ENV:?set DC_AGENT_SECRET_ENV to the secrets file first}"   # HF_TOKEN, WANDB_*, DAYTONA_* (forwarded into the pod)
export KUBECONFIG=~/.kube/coreweave-iris-gpu           # CoreWeave GPU cluster kubeconfig (see ops doc)
# otagent python = /Users/benjaminfeuer/miniconda3/envs/otagent/bin/python (symlinks fail in the sandbox); the `iris` SDK is installed in this env
```
- **Confirm access** before submitting (synchronous `iris`/`kubectl` only — never background them; the
  ~36-node H100 headroom check is in the ops doc's "Verify access").
- **Cluster config** auto-resolves to `~/Documents/marin/lib/iris/config/cw-us-east-02a.yaml`; override with
  `--cluster-config` only if it moved.
- **gpu-rl image:** deps-only (RL venv `/opt/openthoughts/envs/rl` + vLLM fork + MarinSkyRL editable
  `/opt/skyrl` + harbor), **pinned by immutable `@sha256:` digest** in
  `cloud/iris/launch_rl_iris.py:DEFAULT_RL_DOCKER_IMAGE` (NOT the floating `:gpu-rl` tag — it stale-caches).
  Source syncs at runtime so first-party edits live without a rebuild; **bump the digest on an image rebuild**.

## 2. The canonical launch

Lifted from each iris config's header (carries its own ready-to-run command):
```bash
cd ~/Documents/MarinSkyRL   # canonical launcher repo (on main / a cloud/iris branch)
source "${DC_AGENT_SECRET_ENV:?set DC_AGENT_SECRET_ENV to the secrets file first}"
python -m cloud.iris.launch_rl_iris \
  --rl_config cloud/iris/configs/<cfg>.yaml \
  --model_path <hf-id> \
  --train_data '["<HF-repo-or-harbor-task-set>"]' \
  --num-nodes N \
  --gpus_per_node 8 \
  --cpu 48 \
  --max-retries 1 \
  --rendezvous-dir s3://marin-us-east-02a/iris/rl-<slug>/<run> \
  --job-name <name> \
  --priority interactive \
  --no-wait
```
**VARIES per arm:** `--rl_config` (the recipe), `--model_path`, `--train_data`, `--num-nodes` (must match the
config's GPU budget — §3). **Fixed on Iris:** `--gpus_per_node 8` (8× H100 nodes; also FORCES policy/ref
gpus-per-node, §4), `--cpu 48` (NOT the 64 default — §6), `--max-retries 1` (transient HF weight-flake — §6),
`--priority interactive` (band; `production`/`batch` exist), `--no-wait` (submit + detach; without it the
launcher streams logs and a `KeyboardInterrupt` terminates the job). Flag glossary:

- **`--rl_config`** — repo-relative path under `cloud/iris/configs/`. Must exist on the synced `/app`
  workspace; the launcher resolves an absolute path back to repo-relative and fails fast if outside the repo.
  (`--rl-config` hyphenated alias also accepted; same for `--model-path`/`--train-data` etc.)
- **`--model_path`** — HF id (e.g. `Qwen/Qwen3-8B`, `Qwen/Qwen3-Coder-30B-A3B-Instruct`). CoreWeave nodes
  have egress → model pulled from HF **online** (do NOT set `HF_HUB_OFFLINE`).
- **`--train_data`** — JSON-list string `'["..."]'` (HF repo `DCAgent/…` / `laion/…`, or a harbor task set).
- **`--num-nodes N`** — number of **WHOLE H100 nodes** requested EXCLUSIVELY, gang/leafgroup-coscheduled (one
  iris task per node, all 8 GPUs each). (See §3 + §5.)
- **`--rendezvous-dir`** — **REQUIRED for `--num-nodes>1`** (launcher hard-errors otherwise). Shared store the
  multi-node Ray head/workers rendezvous through. **Build this off `marin_prefix()`** (`rigging.filesystem` —
  auto-resolves the active cluster's storage root, so a store migration is followed automatically); the literal
  below is an illustrative fallback. On cw-us-east-02a use an **`s3://` URI under the `marin-us-east-02a`
  bucket** (e.g. `s3://marin-us-east-02a/iris/rl-<slug>/<run>`); the cluster injects working creds into every
  task pod (the `iris-task-env` Secret) so **no external creds** are needed and you must **NOT forward
  `AWS_*`/`R2_*`** (clobbers the pod's injected creds and silently targets real AWS S3). Use a fresh sub-path
  per run so a stale head file from a prior attempt isn't picked up. **⚠ Store is `s3://marin-us-east-02a`
  (CW), NOT the old `s3://marin-na` (R2) — a `marin-na` rendezvous PUT resolves to the nonexistent
  `marin-na.cwlota.com` and STALLS.**
- **`--job-name`** — controls the iris job id `/benjaminfeuer/<name>`; set it explicitly so monitoring +
  teardown land on a predictable name. (Auto-derived `rl-iris-<ts>` if unset.)
- **`--priority`** — `production` / `interactive` / `batch` band.
- **`--cpu` / `--memory` / `--disk`** — per-node resources (defaults 64 / 512GB / 512GB; **set `--cpu 48`**).
- **`--max-retries K`** — iris re-brings-up the gang on a FAILURE up to K times (preemptions retry
  separately). **Use ≥1** (§6).
- **`--skyrl-ref <git-ref>`** — `git fetch && checkout` the baked `/opt/skyrl` MarinSkyRL clone to a
  newer/pinned commit BEFORE running (deps are baked, but skyrl-train is editable → the checkout is live; the
  launcher also purges stale `.pyc` so the checkout isn't shadowed by baked bytecode). Use to pick up a
  MarinSkyRL fix that landed AFTER the image build without rebuilding.
  > **⚠ FOOTGUN — omitting `--skyrl-ref` silently runs STALE baked MarinSkyRL** (currently **`2d9feef`**). If a
  > needed fix landed after the image build, the gang runs OLD code and can **DETERMINISTICALLY crash at
  > `build_models`** ~70s in (`exit_code 0`, no exception surfaced — the `RayTaskError` traceback truncates at
  > `ray.get(refs)`; pull the FULL finelog with `--no-tail` to see the real cause). **RULE:** before any
  > launch, diff local MarinSkyRL HEAD (`git -C ~/Documents/MarinSkyRL log --oneline`) against the image's
  > baked commit; if HEAD is ahead on a code path you exercise (MoE swap, weight load, CP/EP), pass
  > `--skyrl-ref <local-HEAD>` (or rebuild + bump the digest). R3/MoE/Qwen3.5 arms are the usual suspects.
- **`--skyrl_override '++a.b.c=val'`** — repeatable Hydra override (last-wins over the yaml).
- **`--dry-run`** — print the resolved config + in-container command without submitting (always dry-run a new
  config first: confirm the hydra args show the placement / `num_inference_engines` / TP / extra_env you
  intend — see the resume log's VALIDATED block for the pattern).

## 3. Config map + node count (`--num-nodes` MUST match the config)

`--num-nodes = total_GPUs_in_config / 8`. Derive the GPU budget from the yaml (`policy_num_nodes`,
`ref_num_nodes`, `num_inference_engines × inference_engine_tensor_parallel_size`):

| Config (`cloud/iris/configs/…`, MarinSkyRL) | Model | Layout | GPUs → `--num-nodes` |
|---|---|---|---|
| `56GPU_qwen3_8b.yaml` | dense 8B (seqnorm + TIS) | disaggregated: 1 node policy + 1 node ref + 48×TP1 engines | 56 → **7** |
| `32GPU_qwen3_coder_30b_a3b_ep4.yaml` | **Qwen3-Coder-30B-A3B (MoE)** — CANONICAL 30B (EP4) | disaggregated (EP4) | 32 → **4** |

- Only these two configs were carried into MarinSkyRL `cloud/iris/configs/` (32GPU EP4 is the canonical 30B).
  The OT-Agent `hpc/skyrl_yaml/iris/` still holds `64GPU_qwen3_6_35b_a3b.yaml` +
  `128GPU_80B_A3B_next_cp{1,4}.yaml`; if you need one, **port it into `cloud/iris/configs/` first** (copy +
  confirm any referenced asset — e.g. a chat-template jinja — is present in the MarinSkyRL tree; §4) rather
  than reaching back into the frozen OT-Agent copy.
- **Smoke first.** No dedicated toy smoke config ships. Validate via `--dry-run` + a short real arm before a
  full run. Author a throwaway `cloud/iris/configs/*.yaml` (colocate_all, small max_steps,
  `trace_upload.enabled: false`) if you need a live bring-up check.
- Dense-8B model is typically `Qwen/Qwen3-8B` (or `laion/…` 8B); MoE arm is
  `Qwen/Qwen3-Coder-30B-A3B-Instruct`. Common train sets: `DCAgent/exp_rpt_pymethods2test-large`, etc.
- These iris configs are **ports of the Jupiter prod configs** (same experiment) — each header documents what
  was carried verbatim vs. changed. Read it.

## 4. Config-authoring rules for `cloud/iris/configs/` (MarinSkyRL)

Porting a Jupiter (Apptainer SIF) config to Iris (Docker) — the load-bearing rules:

- **Every referenced FILE ASSET must exist in the MarinSkyRL tree (self-containment).** In-pod, `run_rl.py`
  parses the config from `/app` (the synced MarinSkyRL workspace, `PROJECT_ROOT`), and the trainer opens
  asset paths relative to `/app`. A `generator.engine_init_kwargs.custom_chat_template_chat_completion_path`
  (e.g. `chat_templates/qwen3_thinking_acc.jinja2`) is `open()`ed directly — a MISSING file is a hard
  `FileNotFoundError` at vLLM engine init. `chat_templates/qwen3_thinking_acc.jinja2` (thinking-accumulation
  template with `reasoning_content` splicing) lives at the **MarinSkyRL repo root** `chat_templates/` — **NOT**
  the simpler `skyrl-train/examples/terminal_bench/qwen3_thinking_acc.jinja2` (they differ; do not
  substitute). When porting any OT-Agent config, copy its referenced assets into the MarinSkyRL tree at the
  config-relative path.
- **NO `container:` / SIF / apptainer / conda / binds / pydeps block.** The gpu-rl image IS the container (RL
  venv `/opt/openthoughts/envs/rl`, MarinSkyRL `/opt/skyrl`, workspace synced to `/app`).
- **Top-level `extra_env:` is FORWARDED and LOAD-BEARING.** The SLURM path emits runtime env as shell
  `export`s under `container.extra_env`; the Iris path has no `container:` block, so that plumbing never
  runs. **`launch_rl_iris.py:load_config_extra_env()`** reads a **top-level `extra_env:`** mapping (and,
  defensively, a leftover `container.extra_env`) and merges it into the iris `EnvironmentSpec`. **Without it
  the YAML's env is SILENTLY DROPPED** — only the launcher's hardcoded HF/WANDB/DAYTONA passthrough reaches
  the pod. The launcher seeds `extra_env` FIRST so its own signals (rendezvous/secrets) win on a collision.
  Values coerced to str (k8s env are strings).
- **SIF→Docker env translation (what to DROP / KEEP / hardcode in `extra_env:`):**
  - **DROP** all `APPTAINERENV_*` duplicates, `TRITON_LIBCUDA_PATH`, `LIBRARY_PATH=/.singularity-d/libs`
    (SIF-only), and `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE` (CoreWeave has egress).
  - **DROP** the GH200/SIF NCCL disables **`NCCL_P2P_DISABLE` / `NCCL_NVLS_ENABLE` / `NCCL_COLLNET_ENABLE`** —
    on H100+IB they CRIPPLE the intra-node NVLink all-reduce a TP=8 engine (DCP) depends on. Use **NCCL
    defaults** (NVLink intra-node + IB inter-node); keep `NCCL_DEBUG=INFO` + the observability/raised
    timeouts (`SKYRL_WORKER_NCCL_TIMEOUT_IN_S`, `TORCH_NCCL_*`). `disable_custom_all_reduce: true` STAYS. Do
    NOT add these for MoE (falsified A/B — the salad was the w13 gate/up swap below).
  - **KEEP `SKYRL_W13_RELOAD_BRACKET` ON (default `1`) for MoE.** Re-applies the FusedMoE `w13` gate/up kernel
    swap (`process_weights_after_loading`) on the disaggregated RL weight update. WITHOUT it, the served MoE
    policy emits CJK token-salad (100% reward-0) on H100/FlashInfer-CUTLASS (fixed MarinSkyRL `2bb70a88`).
    Swap-inert on triton/dense → byte-identical there, so just leave it on; do NOT set `0`. Bring-up check:
    engine log shows `initialize_layerwise_reload` / `finish_weight_reload`.
  - **HARDCODE** `LD_LIBRARY_PATH: /opt/openthoughts/envs/rl/lib` (the RL conda prefix) — **NOT
    `$CONDA_PREFIX/lib`**: k8s does NOT shell-expand `$VAR`, so a literal `$CONDA_PREFIX/lib` is a broken path.
  - **KEEP** the vLLM serve flags (`VLLM_USE_FLASHINFER_SAMPLER=0`, `VLLM_ATTENTION_BACKEND=FLASH_ATTN`),
    `PYTORCH_CUDA_ALLOC_CONF`, and (for the MoE R3/DCP arm) the guard env `VLLM_ALLOW_ROUTED_EXPERTS_DCP=1`
    (+ `VLLM_MQ_MAX_CHUNK_BYTES_MB`, `VLLM_ROUTED_EXPERTS_SIDE_TIMEOUT_SECONDS`,
    `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS`) and the EPDIAG probe arm — all as **plain top-level `extra_env:`**.
- **Disaggregated (`colocate_all: false`) vs colocated (`true`):** the smoke is colocated (policy/ref/engines
  share GPUs via vLLM sleep-mode, all placement counts null → derived). The real arms are **disaggregated**
  (policy/ref FSDP on a disjoint set of nodes from the vLLM generation engines).
- **Set `num_inference_engines` EXPLICITLY when disaggregated.** `build_skyrl_hydra_args` only DERIVES it
  (`= total_gpus / TP`) when the YAML leaves it `null`. On a disaggregated split that derivation is WRONG — it
  counts the policy-only nodes too. E.g. the 30B config: `(8×8)//8 = 8` would be derived, but only **4**
  engines exist (4 nodes are policy-only). **4 is authoritative; pin it.** (The colocated smoke can leave it null.)
- **`--gpus_per_node 8` FORCES `policy_num_gpus_per_node` / `ref_num_gpus_per_node` to 8** (`build_skyrl_hydra_args`
  overrides the YAML values when the flag is set). `policy_num_nodes` / `ref_num_nodes` ARE honored from the
  YAML. So in a port: divide the Jupiter node counts by 2 (GH200 4-GPU → H100 8-GPU nodes) for
  `*_num_nodes`; list `*_num_gpus_per_node: 8` for documentation only (the flag overrides it).
- **MoE / DCP placement constraints** (hard — don't hand-edit around them): a **TP=8 engine must place
  intra-node on ONE 8-GPU node** (NVLink decode, no cross-node TP — what CoreWeave's 8-GPU nodes unblock vs
  Jupiter's 4-GPU). `inference_engine_mp_backend: false` (RAY executor, R3+DCP mandatory) +
  `async_scheduling: false` (R3 guard). DCP ceiling `dcp ≤ tp/num_kv_heads`; MoE dim-0 guard
  `(num_experts // ep) % fsdp == 0`; `policy_strict_spread_pg: true` reserves policy nodes up front so engines
  land on the disjoint gen nodes.

## 5. Gang scheduling + multi-node Ray rendezvous

- **`--num-nodes N` → `replicas=N`** whole H100x8 tasks. For GPUs with replicas>1,
  `resolve_multinode_defaults` returns **`CoschedulingConfig(group_by="leafgroup")`** — all N nodes
  co-scheduled on one InfiniBand leaf fabric, all-or-nothing. cw-us-east-02a enables **Kueue gang admission**
  (`host_network: true` for NCCL/IB) → the N-task gang is admitted **atomically** (all N whole nodes granted,
  or the job queues). At submit: `replicas=N, coscheduling=leafgroup`, then pods sit **SchedulingGated**
  (normal Kueue gang pre-admit) until admitted.
- **One controller runs on every node** (`cloud/iris/start_rl_iris_controller.py`); iris injects
  `IRIS_TASK_ID` / `IRIS_NUM_TASKS` / `IRIS_ADVERTISE_HOST` per task. **Rank 0** starts the Ray head, writes
  the head IP to the rendezvous file (`ray_head.json` under `--rendezvous-dir`), waits for all N nodes to
  join, then runs the MarinSkyRL driver (`run_rl.py`) with `RAY_ADDRESS` set so SkyRL's bare `ray.init()`
  **attaches** to the cluster. **Ranks 1..N-1** poll the rendezvous for the head IP, `ray start
  --address=…`, then park (contributing their 8 GPUs) until rank 0 publishes the `ray_head.done` marker. On a
  retry, rank 0 purges the stale rendezvous and rewrites (workers ignore stale files via `written_at`).

## 6. Bring-up gotchas (the pre-flight checklist)

- **`--cpu 48`, NOT 64.** CoreWeave nodes are ~128 cores but carry ~64–68 cores of persistent
  system/daemonset overhead, so at the **64 default only ~2/32 nodes have ≥64 free cores** → an N-node
  leafgroup gang can't be satisfied and sits **SchedulingGated** forever (`topology 'infiniband' allows to
  fit only 2 out of N pod(s)`). **48 cores fits all nodes** and admits immediately. Memory 512GB is fine.
- **`--max-retries ≥1`** for the transient HF weight-resolution flake. At scale (e.g. 32 FSDP ranks each
  resolving sharded safetensors online) a single rank can hit a transient HF Hub HTTP/EOF failure →
  transformers reports `… does not appear to have a file named model.safetensors`; with `max_retries=0` that
  one rank kills the gang. `--max-retries 1` re-brings-up the gang (time-only cost). A first-party
  retry-wrapper landed in MarinSkyRL (`0b2b05b`); keep `--max-retries ≥1` as belt-and-suspenders. *(DURABLE
  alternative: pre-stage the model into the image's HF cache / a shared snapshot before the FSDP workers
  start, or raise `HF_HUB_DOWNLOAD_TIMEOUT`.)*
- **TP=8 must place intra-node on an 8-GPU node** (NVLink decode) — guaranteed by `--gpus_per_node 8` + the
  per-engine STRICT_PACK PG; do not split a TP=8 engine across nodes.
- **DCP / CP / R3 / EPDIAG env must reach the pod** — that's the `extra_env:` forwarding (§4). After a launch,
  confirm from the rank-0 log that the guard ENGAGED (not rejected): SkyRL `_validate_dcp_cfg
  VLLM_ALLOW_ROUTED_EXPERTS_DCP=1: allowing …` AND vLLM-fork `vllm.py … allowing --enable-return-routed-experts
  with decode_context_parallel_size=2`. The `envs.py "Unknown vLLM environment variable detected:
  VLLM_ALLOW_ROUTED_EXPERTS_DCP"` line is ONLY the env-registry whitelist not listing the fork-added var — it
  is **NOT** a no-op when the fork's own code reads the var (it does).
- **Transient self-healing on bring-up is normal:** a `ghcr.io` blob EOF → `ImagePullBackOff` self-heals
  (kubelet retries); `shm_broadcast: No available shared memory broadcast block found in 60s` is **benign**
  (engines idle-wait while the policy mesh loads weights). Don't salvage these.

## 7. Standing constraints (do NOT violate)

- **Daytona RL concurrency ≤ 6 RUNNING per cluster.** PENDING/SchedulingGated gangs that haven't admitted
  don't count, but a RUNNING gang does — don't launch a 7th concurrent RL job.
- **`enable_db_registration: false`** in every iris RL yaml (it is). DB registration is a **manual cleanup
  step**, never a launch flag.
- **The a3 series is CONCLUDED** — do NOT launch / refill / auto-advance a3 rows. Active iris arms: the
  seqnorm/TIS dense-8B port + the CP+DCP2+R3 MoE port.
- **Never alter config/hparams mid-series** — propose a separate experiment.
- **Daytona snapshot caps are HARD** — clean STALE snapshots, never raise the cap
  (`.claude/projects/daytona/daytona.md`).
- **Never kill a RUNNING job without explicit permission**; **never** `iris cluster restart`/stop/bounce
  (kills every running job). `iris job kill /benjaminfeuer/<job>` is job-scoped (with permission).

## 8. Monitoring + completion

> **⚠ LIVENESS / TERMINAL DETECTION = AUTHORITATIVE STATE-POLL, NEVER A LOG-STRING WATCH.** Poll the
> authoritative iris job lifecycle state (`iris job summary --json` → `state`), NOT grep rank-0 logs for a
> content string (`EPDIAG_FWD1` / `DEADLOCK` / `TERMINAL`). A clean kill/eviction/preemption or an early
> crash that never prints one of those strings leaves the watcher believing "still running" while the job +
> pods have already left the cluster. **Log-content greps are ONLY for sel_rows / EPDIAG / throughput science
> (via `analyze_job_history.py`) — never for liveness or terminal detection.**

- **Watch a run (the primitive — use this, do not grep logs for liveness):**
  ```bash
  PY=/Users/benjaminfeuer/miniconda3/envs/otagent/bin/python   # otagent env: ships iris + a WORKING kubernetes
  # one-shot authoritative state (state + error + per-task + pod cross-check):
  $PY scripts/iris/watch_job_state.py /benjaminfeuer/<job> --once --json
  # watch until the run leaves RUNNING, emit a line on every transition, exit terminal:
  $PY scripts/iris/watch_job_state.py /benjaminfeuer/<job> --interval 60
  #   exit 0 succeeded · 1 failed/killed/worker_failed/unschedulable · 2 absent (disappeared/0-pods) · 3 error
  ```
  It polls `iris job summary --json` (falls back to the SQL `query`) and treats **"no controller record AND 0
  pods" as a TERMINAL `absent` verdict** — the case the old content-watch missed. Importable:
  `from scripts.iris.watch_job_state import get_job_state, watch`.

> **⚠ FRESH-LAUNCH 15-MIN + 30-MIN CHECK-INS MUST PARSE THE ROLLOUT LOGS — lifecycle state is necessary but
> NOT sufficient.** `watch_job_state` reports `running, pods=8, failure_count=0` for a job that is **silently
> dead**: a *throughput-starvation wedge* (engines decode but an oversubscribed queue never drains → the
> step-0 batch never assembles), *node-local data starvation* (a rank-0-only task-dataset stage → 7 ranks see
> empty task dirs → every rollout `reward 0`), or vLLM simply never serving. State-polling alone reports all
> of these as "healthy." For ANY fresh Iris RL launch, the 15-min AND 30-min check-ins capture + parse the
> logs. Dispatch a subagent armed with **`rl-job-health-deep-dive`** (operationalizes the ladder below into a
> KILL/NO-KILL: syncs trace_jobs + logs via `peek … pull`, live-polls the GPUs against the serving-throughput
> LUT, reads the literal rollouts). Procedure:
> ```bash
> # capture finelog + per-rank pod logs (our RL jobs use a REMOTE s3 trials_dir, so peek's ls/cat/grep bail —
> # `pull` still grabs the logs, where the live bring-up signal is). Override IRIS_BIN: the script defaults to
> # the marin .venv iris, which CANNOT drive cw (broken kubernetes import).
> IRIS_BIN=/Users/benjaminfeuer/miniconda3/envs/otagent/bin/iris \
>   bash scripts/iris/peek_rl_rollouts.sh <pod-name-substring> pull     # → ~/Documents/experiments/traces/<job>_<stamp>/logs/iris_finelog.log
> ```
> Then grep the finelog for the milestone ladder — each rung **rules out a specific silent-death mode**:
> 1. **`[rl-iris] MarinSkyRL now at <sha>`** (+ `HEAD is now at <sha>`) — confirms `--skyrl-ref` took (the fix is live). Absent ⇒ forgot the flag / it failed → stale baked code.
> 2. **`Staging train_data on this node (rank N/8)` for ALL N ranks** — else node-local data starvation → silent `reward 0`. Must see every rank.
> 3. **`Ray nodes alive: N/N`** — rendezvous complete.
> 4. **HF weight load:** an `OSError … does not appear to have a file named model.safetensors` FOLLOWED by `load_pretrained_with_retry … retrying` → the `0b2b05b` retry **catching** it (BENIGN). The SAME error with **no** retry wrapper + a task failure ~70s in = the stale-image `build_models` crash. (MoE arm: `AttributeError: … 'norm_topk_prob'` is the *other* stale-image crash — must be ABSENT.)
> 5. **vLLM ACTUALLY GENERATING** = `loggers.py … Avg generation throughput: >0 tokens/s, Running: R reqs, Waiting: W` recurring. `Waiting` persistently ≫ `Running` with flat throughput = the throughput-starvation WEDGE (de-oversubscribe: lower `n_concurrent_trials` / raise `max_num_seqs`). `Waiting ≈ 0` = healthy.
> 6. **MoE DCP arm only:** `_validate_dcp_cfg VLLM_ALLOW_ROUTED_EXPERTS_DCP=1: allowing …` + `decode_context_parallel_size=2` — the R3+DCP guard engaged (the `Unknown vLLM environment variable` line is the benign whitelist note, NOT a no-op).
> 7. **Train driver:** `Resumed training from global_step 0` + `TerminalBenchGenerator initialized … Concurrent trials: K` (Harbor RolloutCoordinator up; K×(#engines) = your `n_concurrent_trials`).
> 8. **Trials completing:** first `result.json` / reward written + **`global_step` 0→1**. At 15/30 min a 131k arm usually has **ZERO** completed (episodes are long) — report as "rollouts executing, 0 trials completed yet," NEVER "healthy/done." Completed-trial artifacts land in the **remote** `s3://marin-us-east-02a/iris/<job>/trace_jobs` (read via `aws s3 ls --endpoint-url <R2>`), not the pod.
>
> **Verdict rule:** rungs 1–7 green + generation throughput >0 + `Waiting≈0` ⇒ genuinely progressing (even with 0 completed trials). Generation throughput **0** after engines are up, or `Waiting≫Running`, or no RolloutCoordinator, or any rank missing its data-stage line ⇒ **escalate now** (wedge / starvation), do not wait for the next sweep.

- **Manual state / logs** (synchronous calls only — no background `iris`/`kubectl`; use the **otagent** iris
  binary — the bare marin `.venv/bin/iris` lacks a working `kubernetes` for the cw k8s controller backend):
  ```bash
  # iris-side state (0=UNSPECIFIED 1=PENDING 2=BUILDING 3=RUNNING 4=SUCCEEDED 5=FAILED 6=KILLED
  #                  7=WORKER_FAILED 8=UNSCHEDULABLE) — the watcher already wraps this:
  /Users/benjaminfeuer/miniconda3/envs/otagent/bin/iris --cluster=cw-us-east-02a query \
    "SELECT job_id,state FROM jobs WHERE job_id LIKE '/benjaminfeuer/<job>%'" -f csv
  # richest single-job authoritative call (state + error + exit + per-task states + finished_at):
  /Users/benjaminfeuer/miniconda3/envs/otagent/bin/iris --cluster=cw-us-east-02a job summary /benjaminfeuer/<job> --json
  # via the python SDK (matches the launcher's client):
  #   IrisClient.status(JobName.from_wire("/benjaminfeuer/<job>"))
  #   IrisClient.fetch_task_logs(JobName.from_wire("/benjaminfeuer/<job>/0"))   # rank 0 = the driver
  ```
  Rank-0 (`/…/<job>/0`) is the training driver; ranks 1..N-1 are Ray workers that clean-exit on the head's
  done-marker (a healthy run's head failing shows `1 task failed, N-1 succeeded`).
- **What healthy bring-up looks like** (from the rank-0 log, in order): gang **admitted**
  (QuotaReserved/Admitted=True, pods leave SchedulingGated) → all N pods **Running** → Ray **rendezvous +
  all N nodes joined** (`Ray nodes alive: N/N`) → SkyRL driver up (`total training steps …`) → **engines up**
  (`InferenceEngineClient initialized with K engines`, each loading its shard 100%; for DCP the worker name
  is `Worker_TP0_DCP0`) → **policy mesh loaded** (FSDP ranks load weights 5→100%) → first training step. A
  30B/131k arm can take ~1h to the first rollout and ~2.5–3h to the gs1 forward.
- **Mandatory progress columns** (per `monitor-job-tables`): entropy / log_ratio / grad_norm + reward — not
  just step. For full-history **science** (sel_rows / EPDIAG / throughput stats) use the windowed pagination
  (`scripts/iris/analyze_job_history.py`), **NOT** `iris job logs --tail` (under-samples 10–100×).
- **On completion → `rl-agentic-job-cleanup`** (best-ckpt selection, HF upload, the **manual** Supabase DB
  registration, trace export). `enable_db_registration` stays false at launch.
- **Teardown:** `iris job kill /benjaminfeuer/<job>` (with permission). Rescue banked traces from the s3://
  jobs/rendezvous path if the trace upload didn't auto-run (`ops.md` §4).

---

## Operating notes
- **Always `--dry-run` a new/edited config** and eyeball the resolved hydra args: `colocate_all`,
  `policy_num_nodes`, the EXPLICIT `num_inference_engines`, TP/DCP/CP/EP/FSDP, and that the intended
  `extra_env:` keys forward with **no `NCCL_P2P_DISABLE` leak**. (See the resume log's VALIDATED block.)
- **This is the agentic terminal_bench/Harbor/Daytona path.** `DAYTONA_*` MUST be forwarded (the launcher
  passes it) — without it every trajectory finalizes `VerificationNotCompletedError` reward 0. Do NOT forward
  `AWS_*`/`R2_*` (clobbers the pod's injected R2 creds → the s3:// rendezvous silently hits real AWS).
