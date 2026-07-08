---
name: build-tpu-image-iris
description: >-
  Build + push the datagen/eval TPU image (`ghcr.io/open-thoughts/openthoughts-agent:tpu`) — the Iris TPU
  runtime (vLLM-TPU 0.20.0 PyTorch/XLA+JAX + Harbor + the Daytona sandbox backend) — AS AN IRIS KANIKO JOB on
  CoreWeave `cw-us-east-02a` (a CPU-only amd64 build; NO CUDA/nvcc, so it's minutes not hours, and the Mac
  CANNOT build it — arm64 + linux/amd64-only). Covers WHY kaniko not buildkit (shared with the gpu-rl skill),
  the crane-export-over-ubuntu recipe, and — the load-bearing lesson this skill exists for — how to make a
  canonical fast-moving dep (Harbor, our `marin-community/harbor` fork on `penfever/working`) ACTUALLY reach
  the worker instead of a STALE CACHED WHEEL: prefer a fresh-from-GitHub editable/source install, bust the
  kaniko layer cache (bump `HARBOR_COMMIT` + `--force-reinstall`, or a version bump so `uv` re-resolves), and
  VERIFY the fix by grepping the INSTALLED file inside the built image (never trust a build-log line). Also:
  PINNED-first push (`:tpu-<gitsha>` immutable, promote floating `:tpu` via `crane tag` only after a live
  smoke), capturing the digest, monitoring, and WHEN a rebuild is required vs a runtime `/app` bundle sync.
  Use when asked to build / rebuild / push the `:tpu` image, or after bumping harbor / the [datagen-tpu]
  resolve / vLLM-TPU. Reference: docker/Dockerfile.tpu, docker/build_tpu_kaniko.sh, the sibling
  build-gpu-rl-image-iris skill, .claude/ops/iris/iris_job_lifecycle.md.
---

# build-tpu-image-iris

> **📍 Iris orientation — read first.** Before acting on anything in this skill, read the Iris **tools
> catalog** (`.claude/ops/iris/iris_tools.md`) and the Iris **ops directory** (`.claude/ops/iris/` — the
> CoreWeave GPU particulars in `coreweave_gpu_ops.md`, the TPU `marin` particulars in `iris_job_lifecycle.md`).
> They carry the binding access/preamble/gotchas and the helper-script inventory the steps below rely on.

> **⚠ Local clone = ground truth.** ALL `Dockerfile.tpu` / `build_tpu_kaniko.sh` edits go in the local Mac
> checkout on `penfever/working` → commit → push. **The iris job bundles the LOCAL workspace to `/app`** via
> `git ls-files --cached --others --exclude-standard` (respects `.gitignore`; reads WORKING-TREE content, so
> uncommitted *tracked* edits ARE included — you do NOT have to commit/push a Dockerfile edit before a build).
> Never hand-edit on a remote / patch-by-rsync.

The **`:tpu` image** is the Iris **TPU datagen + eval runtime** — `linux/amd64` (TPU host VMs are x86_64),
single-stage `python:3.12-slim` + `uv pip install -e ".[datagen-tpu,cloud]"` (vLLM-TPU 0.20.0 PyTorch/XLA+JAX,
tpu-inference 0.20.0, transformers 5.5.3 for Qwen3.5 hybrid GDN) + Harbor with the Daytona sandbox backend. It
is consumed by every datagen/eval launch (`data/cloud/launch_tracegen_iris.py`, `eval/cloud/launch_eval_iris.py`
— see the `datagen-launch-iris` skill). This skill is the **how-to to BUILD + PUSH it**.

> **This image is MUCH simpler than gpu-rl.** No CUDA / nvcc / vLLM-fork wheels / `WHEEL_SOURCE` / wheelhouse
> — `Dockerfile.tpu` is single-stage and TPU slice size is a *submit-time* iris arg (`--tpu v5p-8`), not a
> build choice. So this skill omits all the gpu-rl wheel-cache machinery. For the shared kaniko mechanics
> (WHY kaniko not buildkit, the crane-export trick, ghcr creds, `SINGLE_SNAPSHOT=0` layering), the sibling
> **`build-gpu-rl-image-iris`** skill is the deep reference — read its §1/§2 once; they apply verbatim here.

---

## 0. Where the build runs, and why it can't run elsewhere

- **The Mac CANNOT build it.** It's arm64; the image is `linux/amd64`-only. Build in-cluster.
- **There is NO in-cluster image-build primitive in iris** (`iris build` = LOCAL `docker buildx`). So the build
  runs **AS AN IRIS JOB** — a CPU-only task that does the Docker build itself, via **kaniko** (the cluster
  denies `CAP_SYS_ADMIN`/bind-mounts + runs gVisor, so buildkit fails; kaniko snapshots its own rootfs). Same
  decisive reasoning as the gpu-rl skill §1.
- **CPU-only, minutes not hours.** Unlike gpu-rl (~3 h of nvcc), the `:tpu` build is just `apt` + a `uv`
  resolve/install — dominated by the dependency download/install (~15–30 min). No GPU, no big-memory
  wheel-packaging step.
- **Cluster:** run on **`cw-us-east-02a`** (proven kaniko + ghcr-egress + kubeconfig path; the TPU image is a
  plain amd64 CPU build so it does not need the marin TPU cluster). `KUBECONFIG=~/.kube/coreweave-iris-gpu`
  and the **otagent-env iris binary** are hard prereqs (see `.claude/ops/iris/coreweave_gpu_ops.md`).

## 1. Mechanism (crane-export kaniko) — what `build_tpu_kaniko.sh` does

The task image is `docker.io/library/ubuntu:22.04` (kaniko's executor is distroless / no bash, so it can't be
the task image directly). `docker/build_tpu_kaniko.sh` runs, in order:

1. `apt-get install` crane → `crane export gcr.io/kaniko-project/executor:latest` over `/`.
2. Write `$DOCKER_CONFIG/config.json` **AFTER the overlay** (kaniko clobbers `/kaniko` otherwise), with the
   ghcr auth `{"auths":{"ghcr.io":{"auth":"<base64 penfever:$DOCKER_TOKEN>"}}}` and
   `export DOCKER_CONFIG=/kaniko/.docker`.
3. `exec /kaniko/executor --context dir:///app --dockerfile docker/Dockerfile.tpu --skip-unused-stages
   --compressed-caching=false --cache=true --cache-repo=…/cache-tpu --destination …:tpu-<GITSHA>`.

Load-bearing flags (already in the script — recognize them):
- **`SINGLE_SNAPSHOT=0` (default) = per-instruction layers.** Dodges the un-pullable ~16 GB single-blob layer
  (containerd EOFs the single-blob GET over ghcr egress → `ImagePullBackOff`). Keep the default; do NOT set
  `SINGLE_SNAPSHOT=1` unless you accept that pull risk.
- **`--compressed-caching=false`** — writes layers to disk so multi-layer snapshotting doesn't blow memory.
- **`--skip-unused-stages`** — harmless here (single-stage) but kept for parity.

### ghcr push creds (the load-bearing cred gotcha — shared with gpu-rl)
- **`secrets.env DOCKER_TOKEN` is a Docker Hub PAT (`dckr_pat_…`) — WRONG for ghcr.io.** Pass the **GitHub
  PAT** as `-e DOCKER_TOKEN "$(gh auth token)"` (user `penfever`, scope `write:packages`). The script's
  `DOCKER_TOKEN` env is what it base64s into the ghcr auth — so feed it the GH token, not the Docker Hub one.

## 2. ⚑ THE CORE LESSON — a canonical dep must reach the worker FRESH, never as a stale cached wheel

This skill exists because a harbor fix (`c49064a8`, the `_maybe_init_existing_job` resume/export-push
job-config-drift fix at `job.py:263`) **built green but never actually shipped** — the deployed image still
ran the OLD raising code, and every ≥1-preempt datagen job kept dying at resume. Root cause: `c49064a8` did
NOT bump harbor's version (still `0.8.0`), so kaniko's cached `uv pip install harbor` layer (keyed on the
unchanged COPYed `pyproject.toml` + RUN string) was **reused**, baking a STALE pre-fix `0.8.0` wheel. The
"Built harbor @ c49064a8" build-log line was misleading.

**Principle (user directive, 2026-07-08): for our frequently-updated canonical repos, prefer installing FRESH
FROM GITHUB (editable / from-source), not a cached wheel.** Harbor is exactly such a repo — our
`marin-community/harbor` fork, canonical branch `penfever/working`, changed almost every campaign. It must be
baked from source at the intended commit, with the cache guaranteed busted, every build.

**Two mechanisms that guarantee freshness (the Dockerfile now does BOTH for harbor):**
1. **Pin `ENV HARBOR_COMMIT=<sha>` and bump it every harbor change.** Changing the env line changes THIS
   RUN-layer's cache key → kaniko is forced to rebuild the layer (this is the cache-bust).
2. **`uv pip install --no-deps --force-reinstall "harbor[daytona] @ git+https://github.com/marin-community/harbor.git@${HARBOR_COMMIT}"`.**
   `--force-reinstall` makes `uv` rebuild harbor from the exact pinned commit regardless of any cached wheel;
   `--no-deps` keeps it a surgical swap that won't refloat the pinned resolve.
   Then a **build-time assert** prints the baked version + commit.
   ```dockerfile
   ENV HARBOR_COMMIT=2dde0bbf11d48a4e44df6ad0a56cee31e7ab609e   # 0.8.1, carries c49064a8 (resume/export-push fix)
   RUN . /opt/openthoughts/.venv/bin/activate && \
       uv pip install --no-deps --force-reinstall \
           "harbor[daytona] @ git+https://github.com/marin-community/harbor.git@${HARBOR_COMMIT}" && \
       python -c "import harbor, importlib.metadata as m; print('baked harbor', m.version('harbor'))"
   ```
   **Belt-and-suspenders that ALSO helped:** bump harbor's package version on `penfever/working`
   (`0.8.0 → 0.8.1`, commit `2dde0bbf`) so `uv`'s resolver treats it as a new package and won't reuse a
   same-version cached wheel even on the base `[datagen-tpu]` resolve layer.

> **Do NOT rely on a same-commit version-only bump alone, and do NOT rely on `--no-cache` alone either.** The
> durable, reproducible fix is the `HARBOR_COMMIT` env-line bump (cache-bust) **+** `--force-reinstall`
> (source rebuild) **+** a version bump (resolver-level freshness). All three are cheap; use them together.

> **Applying the principle beyond harbor.** For any other canonical fast-moving repo baked into this image,
> mirror this shape: a pinned `ENV <REPO>_COMMIT` on the canonical branch + `--force-reinstall` from
> `git+https://…@${COMMIT}`, or an editable clone the runtime resyncs. Never let such a dep ride a cached
> wheel keyed on an unchanged `pyproject.toml`. (First-party OT-Agent code is already editable `-e` + synced
> to `/app` at launch — it is never stale; this lesson is specifically for *third-party-installed* canonical
> deps like harbor.)

## 3. The launch command (verbatim shape)

```bash
source "${DC_AGENT_SECRET_ENV:?set DC_AGENT_SECRET_ENV to the secrets file first}"
export KUBECONFIG=~/.kube/coreweave-iris-gpu                      # HARD prereq (see ops doc)
IRIS=/Users/benjaminfeuer/miniconda3/envs/otagent/bin/iris        # the cw-capable (otagent-env) iris binary
GHCR_TOKEN=$(gh auth token)                                       # GitHub PAT, NOT the Docker Hub DOCKER_TOKEN
GITSHA=$(git rev-parse --short HEAD)                              # REQUIRED — build_tpu_kaniko.sh has `: "${GITSHA:?}"` (the immutable :tpu-<gitsha> tag)
B64=$(base64 -i docker/build_tpu_kaniko.sh | tr -d '\n')          # the kaniko build script, base64'd in

$IRIS --cluster=cw-us-east-02a job run \
  --task-image docker.io/library/ubuntu:22.04 --no-sync --enable-extra-resources \
  --cpu 32 --memory 256GB --disk 300GB \
  --job-name tpu-kaniko-$GITSHA \
  --max-retries 0 --timeout 10800 \
  -e DOCKER_USER_ID penfever -e DOCKER_TOKEN "$GHCR_TOKEN" -e BUILD_B64 "$B64" \
  -e GITSHA "$GITSHA" --no-wait \
  -- bash -lc 'echo "$BUILD_B64" | base64 -d > /tmp/build.sh && exec bash /tmp/build.sh'
```
> - **`-e GITSHA` is MANDATORY** (`: "${GITSHA:?}"` — the immutable `:tpu-<gitsha>` tag; the job dies without it).
> - **These are the verbatim flags that built `:tpu-abd6dc86` (2026-07-08, ~8.5 min, 9.8 MB bundle).** They're
>   **over-provisioned** — a CPU-only `uv` resolve + layered snapshot needs nowhere near 256 GB / 300 GB
>   (gpu-rl's sizing was inherited). Trimming toward ~16 CPU / 64 GB / 120 GB is safe if you want faster
>   scheduling; bump back only if the snapshot step reports pressure.
> - **`SINGLE_SNAPSHOT` and `DOCKERFILE` were NOT set** → they take `build_tpu_kaniko.sh` defaults
>   (`SINGLE_SNAPSHOT=0` = per-instruction pullable layers, `DOCKERFILE=docker/Dockerfile.tpu`). Leave them.
> - **`--timeout 10800` (3 h)** is generous headroom for a ~8–30 min build.
> - The build **pushes ONLY the immutable `:tpu-<GITSHA>` tag** (PINNED-first). The floating `:tpu` is promoted
>   separately in §5, so a botched build cannot break other users' `:tpu` jobs.

## 4. ⚑ VERIFY THE BUILT IMAGE — grep the INSTALLED file, never trust a build-log line

This is the whole point of the skill — the last build was declared good on a build-log line while the deployed
file was stale. After the build SUCCEEDS + pushes, **pull the image and grep the ACTUAL installed source**:

**Use the venv's python explicitly** — a bare `bash -lc python …` resolves a non-venv interpreter and
`import harbor` fails. Either call `/opt/openthoughts/.venv/bin/python` directly or
`. /opt/openthoughts/.venv/bin/activate` first.

```bash
NEW=ghcr.io/open-thoughts/openthoughts-agent:tpu-<gitsha>
# 1. installed harbor version is the bumped one:
docker run --rm "$NEW" /opt/openthoughts/.venv/bin/python -c \
  "import importlib.metadata as m; print('harbor', m.version('harbor'))"          # expect 0.8.1 (not 0.8.0)
# 2. the FIX is in the installed job.py (logger.warning, NOT a raise) at the guard:
docker run --rm "$NEW" sh -lc \
  "grep -n -A4 -B2 'existing_config != self.config' /opt/openthoughts/.venv/lib/python3.12/site-packages/harbor/job.py"
#    → the `if existing_config != self.config:` guard must be followed by logger.warning(...), NOT
#      raise FileExistsError(...). (On :tpu-abd6dc86 this is job.py:275-276.)
```
> **The file has TWO guards — check the RIGHT one.** The job-config-drift guard (the resume killer) is the
> `existing_config != self.config` one, now `logger.warning`. A `raise FileExistsError` still legitimately
> remains FAR AWAY (job.py:841 on abd6dc86) in the per-trial reconciler the fix intentionally defers to — so a
> plain `grep raise FileExistsError` will show a hit; that's expected, NOT a regression. Grep the specific
> guard line as above.
> Confirm the site-packages path with
> `docker run --rm "$NEW" /opt/openthoughts/.venv/bin/python -c "import harbor,os;print(os.path.dirname(harbor.__file__))"`
> if the venv layout has moved. **Do not declare success on the build log alone** — grep the file. That a
> pull-and-run verify job even executes ALSO proves the image is pullable (`SINGLE_SNAPSHOT=0`); optionally
> confirm layering with `docker buildx imagetools inspect "$NEW" --raw` → many layers, max well under 8 GB.

## 5. Capture the digest + PROMOTE the floating `:tpu` (PINNED-first)

The build pushed only `:tpu-<gitsha>`. Promote the floating `:tpu` to it **only AFTER a live smoke** (a real
datagen job on `:tpu-<gitsha>` serves and — for a resume/export-push fix — SURVIVES a preempt-resume and
populates literals end-to-end). Record the pre-promote digest first so rollback is a one-liner.

```bash
# resolve the immutable digest of the pinned tag:
crane digest ghcr.io/open-thoughts/openthoughts-agent:tpu-<gitsha>            # sha256:… (record it)
# record the CURRENT floating :tpu digest BEFORE promoting (rollback target):
crane digest ghcr.io/open-thoughts/openthoughts-agent:tpu
# PROMOTE (re-point floating :tpu → the validated pinned digest):
crane tag ghcr.io/open-thoughts/openthoughts-agent@sha256:<new-digest> tpu
# ROLLBACK (if the smoke regresses): re-point :tpu back to the recorded pre-promote digest:
crane tag ghcr.io/open-thoughts/openthoughts-agent@sha256:<old-digest> tpu
```
Then update the campaign tracker's **Runtime** block with the new `:tpu-<gitsha>` + digest + what it bakes
(harbor commit, OT-Agent bundle commit) and the recorded rollback digest.

> **Rollback to a digest that PREDATES the fix is a no-op** — it dies identically. When rolling back, roll back
> to the last digest that actually WORKED for your purpose, not just "the previous tag." (2026-07-08: the
> tempting `879ebaba` rollback was a strict no-op because it, too, predated `c49064a8`.)

## 6. Monitoring the build job

> **⚠ State-poll, NEVER a log-string watch.** A clean kill / OOM-reap emits no terminal log line — poll the
> authoritative iris lifecycle state.

```bash
PY=/Users/benjaminfeuer/miniconda3/envs/otagent/bin/python   # otagent env: iris + a WORKING kubernetes client
export KUBECONFIG=~/.kube/coreweave-iris-gpu                 # HARD prereq for every cw call
$PY scripts/iris/watch_job_state.py /benjaminfeuer/tpu-kaniko-<gitsha> --once --json    # authoritative state now
$PY scripts/iris/watch_job_state.py /benjaminfeuer/tpu-kaniko-<gitsha> --interval 60     # watch until terminal
$IRIS --cluster=cw-us-east-02a job summary /benjaminfeuer/tpu-kaniko-<gitsha> --json     # state+error+exit+finished_at
# full log by time-window (NOT --tail, which under-samples): --since-ms <submit_ms> --no-tail
```
- Use the **otagent iris binary** for cw (the bare `marin/.venv/bin/iris` has a broken `kubernetes` import and
  cannot drive cw). All `iris`/`kubectl` calls **synchronous** — never background them.
- Kaniko's `--destination` push is the TERMINAL step → SUCCEEDED = pushed. Still run §4 verification.

## 7. WHEN a rebuild is actually required (vs a runtime `/app` bundle sync)

Rebuild the `:tpu` image ONLY for a change to something the image **BAKES into the venv**:
- **harbor** (the `HARBOR_COMMIT` pin — third-party, installed into the venv, NOT synced from `/app`). ⇒ the
  common case; follow §2's freshness recipe.
- **the `[datagen-tpu,cloud]` resolve** (vLLM-TPU / tpu-inference / jax/libtpu / transformers pin in
  `pyproject.toml`).
- **base-layer changes** (apt packages, the python base, `uv` venv layout).

Do **NOT** rebuild for a **first-party OT-Agent source change** (`hpc/`, `data/`, `scripts/`, `eval/`, the
launchers, the RecordProxy/correlator). Those are editable `-e` and the iris worker **syncs the local
working-tree to `/app` at launch**, overriding the image-baked copy — so a code edit ships on the NEXT launch
with no rebuild. (This is why datagen fixes usually don't need an image build; harbor is the exception because
it's a third-party install, not first-party `/app` code.)

## 8. Standing constraints

- **`penfever/working` is canonical** for both OT-Agent and (harbor) `marin-community/harbor` — `git fetch` +
  FF-check before any push; leave the editable harbor checkout at `/Users/benjaminfeuer/Documents/harbor`
  clean on `penfever/working` (datagen rescues depend on it).
- **PINNED-first, promote-after-smoke.** Never push straight to floating `:tpu`; other users' live datagen/eval
  jobs pull it.
- **`source "$DC_AGENT_SECRET_ENV"`**; NEVER echo secret
  values. ghcr = `gh auth token`, not Docker Hub `DOCKER_TOKEN`.
- **Never `iris cluster restart`/stop/bounce** (kills every running job). The build job is its own iris job;
  `iris job kill /benjaminfeuer/tpu-kaniko-<gitsha>` is job-scoped (with permission).
- **`--cache-repo=…/cache-tpu` is shared** — don't delete it.

---

## Cross-reference
- **Consumes this image:** `datagen-launch-iris` + `eval-agentic-launch-iris` (the `:tpu` runtime). Update the
  campaign tracker Runtime block + rollback digest after a promote.
- **Shared kaniko mechanics (deep reference):** `build-gpu-rl-image-iris` SKILL §1 (kaniko-not-buildkit),
  §2 (crane-export + ghcr creds + `SINGLE_SNAPSHOT=0`), §6 (build-asserts-as-validation).
- **Cluster access / kubeconfig / iris-binary:** `.claude/ops/iris/coreweave_gpu_ops.md`.
- **TPU datagen lifecycle / preemption / resume:** `.claude/ops/iris/iris_job_lifecycle.md`.
- **Code:** `docker/Dockerfile.tpu`, `docker/build_tpu_kaniko.sh`, `scripts/iris/watch_job_state.py`.
