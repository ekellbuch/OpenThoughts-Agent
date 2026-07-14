#!/usr/bin/env bash
# build_gpu_rl_kaniko.sh — in-cluster kaniko build of the gpu-rl image.
#
# Runs INSIDE an iris job whose task-image is docker.io/library/ubuntu:22.04
# (kaniko's executor image is distroless / has no bash, so it cannot be the task
# image directly). We crane-export the kaniko executor rootfs over / and run
# /kaniko/executor. See .claude/skills/build-gpu-rl-image-iris/SKILL.md.
#
# Required env (passed by the iris launch as -e):
#   DOCKER_USER_ID  ghcr user (penfever)
#   DOCKER_TOKEN    a GitHub PAT with write:packages (NOT the Docker Hub dckr_pat_)
#   GITSHA          OT-Agent commit sha for the immutable :gpu-rl-<gitsha> tag
# SECURITY: NO `set -x` here — command-tracing would echo DOCKER_TOKEN (the ghcr PAT)
# and the base64 `AUTH` into the R2-persisted finelog. Tracing is re-enabled AFTER the
# token is consumed (the config.json write below), so the build steps are traced but the
# secret never is.
set -euo pipefail

: "${DOCKER_USER_ID:?}"
: "${DOCKER_TOKEN:?}"
: "${GITSHA:?}"

# WHEEL_SOURCE selects how the vLLM-fork + flash-attn wheels enter the rl stage:
#   prebuilt-wheelhouse (FAST): COPY docker/wheelhouse/*.whl + uv pip install — NO
#     nvcc. Requires the wheels to be present in the /app bundle; since
#     docker/wheelhouse/* is gitignored, force-stage them first (git add -f) so the
#     iris bundle (git ls-files --cached) includes them.
#   wheel-builder (SLOW, default): compile both wheels inline with nvcc (~3 h). Use
#     only when no prebuilt wheels can be force-staged.
WHEEL_SOURCE="${WHEEL_SOURCE:-wheel-builder}"

# SINGLE_SNAPSHOT=1 (default) => kaniko collapses the whole build into ONE final
# layer (--single-snapshot; smallest image, but that ONE layer can be ~16 GB and
# CANNOT be pulled over the CoreWeave->ghcr egress — containerd restarts the
# single-blob GET from 0 and it dies at 8-11 GB every time). Set SINGLE_SNAPSHOT=0
# to produce PER-INSTRUCTION layers (each small enough to pull+retry independently)
# — the fix for the un-pullable 16 GB layer. --compressed-caching=false (below)
# writes layers to disk so multi-layer snapshotting does NOT blow up memory.
SINGLE_SNAPSHOT="${SINGLE_SNAPSHOT:-1}"
if [ "$SINGLE_SNAPSHOT" = "1" ]; then SNAPSHOT_FLAG="--single-snapshot"; else SNAPSHOT_FLAG=""; fi

CACHE_REPO=ghcr.io/open-thoughts/openthoughts-agent/cache
# KANIKO_CACHE=1 (default) reuses cached RUN layers from CACHE_REPO (the ~3 h nvcc
# reuse on the full-build path). Set KANIKO_CACHE=0 to FORCE every RUN to execute
# fresh (no cache read/write) — use when you need the in-build asserts to actually
# run + print (e.g. verifying an incremental FlashQLA layer's tilelang pin), rather
# than kaniko silently extracting a previously-cached layer.
if [ "${KANIKO_CACHE:-1}" = "0" ]; then CACHE_FLAGS="--cache=false"; else CACHE_FLAGS="--cache=true --cache-repo=${CACHE_REPO}"; fi
DEST_FLOATING=ghcr.io/open-thoughts/openthoughts-agent:gpu-rl
DEST_PINNED=ghcr.io/open-thoughts/openthoughts-agent:gpu-rl-${GITSHA}

# PUSH_FLOATING=1 (default) also moves the floating :gpu-rl tag to this build.
# Set PUSH_FLOATING=0 to push ONLY the immutable :gpu-rl-<gitsha> tag and LEAVE
# :gpu-rl pointing where it was — use this for a build that must be smoke-tested
# BEFORE it becomes the floating default (consumers pin the digest, not :gpu-rl).
FLOATING_DEST_FLAG="--destination ${DEST_FLOATING}"
if [ "${PUSH_FLOATING:-1}" != "1" ]; then FLOATING_DEST_FLAG=""; fi

# --- 1. fetch crane (static binary) ---
apt-get update -y && apt-get install -y --no-install-recommends ca-certificates curl tar
cd /tmp
CRANE_VER=v0.20.2
curl -fsSL "https://github.com/google/go-containerregistry/releases/download/${CRANE_VER}/go-containerregistry_Linux_x86_64.tar.gz" -o crane.tgz
tar -xzf crane.tgz crane
install -m 0755 crane /usr/local/bin/crane

# --- 2. crane-export the kaniko executor rootfs over / ---
# Overlays kaniko's /kaniko/... onto the ubuntu rootfs.
crane export gcr.io/kaniko-project/executor:latest - | tar -xf - -C / || true

# --- 3. write the ghcr auth config AFTER the overlay (kaniko clobbers /kaniko otherwise) ---
export DOCKER_CONFIG=/kaniko/.docker
mkdir -p "$DOCKER_CONFIG"
AUTH=$(printf '%s:%s' "$DOCKER_USER_ID" "$DOCKER_TOKEN" | base64 | tr -d '\n')
cat > "$DOCKER_CONFIG/config.json" <<EOF
{"auths":{"ghcr.io":{"auth":"${AUTH}"}}}
EOF
unset AUTH
set -x  # ghcr PAT is now consumed — safe to trace the build steps below (token never traced)

# --- 3.5. populate docker/wheelhouse/ for the prebuilt-wheelhouse (no-nvcc) path ---
# The iris /app bundle has a 25 MB cap, so the ~900 MB prebuilt wheels CANNOT ride
# it. Instead fetch them from the public HF dataset mirror into the build context
# (/app/docker/wheelhouse/) so kaniko's `COPY docker/wheelhouse/` (rl stage) finds
# them => ZERO nvcc. Skip if the wheels are already present (e.g. a real x86 host
# that ran build_wheels.sh). Mirror: laion/gpu-rl-build-wheels (vLLM-fork +
# flash-attn wheels are open-source; public, no token needed for download).
if [ "$WHEEL_SOURCE" = "prebuilt-wheelhouse" ]; then
  WH=/app/docker/wheelhouse
  mkdir -p "$WH"
  HF_BASE="https://huggingface.co/datasets/laion/gpu-rl-build-wheels/resolve/main"
  FLASH_WHL=flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl
  VLLM_WHL=vllm-0.1.dev16611+g76259c63a.d20260625.cu128-cp312-cp312-linux_x86_64.whl
  for f in "$FLASH_WHL" "$VLLM_WHL" MANIFEST; do
    if [ ! -s "$WH/$f" ]; then
      echo "fetching wheelhouse artifact: $f"
      curl -fSL --retry 5 --retry-delay 5 "$HF_BASE/$(printf '%s' "$f" | sed 's/+/%2B/g')" -o "$WH/$f"
    fi
  done
  echo "=== wheelhouse contents ==="; ls -la "$WH"
  # fail fast if a wheel is missing/empty -> do NOT silently fall through to a compile
  test -s "$WH/$FLASH_WHL" && test -s "$WH/$VLLM_WHL" || { echo "FATAL: wheelhouse not populated"; exit 1; }
fi

# --- 4. run kaniko ---
# --skip-unused-stages is LOAD-BEARING for the prebuilt-wheelhouse path: kaniko
# builds EVERY Dockerfile stage by default (unlike BuildKit), so without it the
# `wheel-builder` (nvcc) stage compiles even when the rl stage takes its wheels
# from prebuilt-wheelhouse. With it, the unreferenced wheel-builder stage is
# pruned => ZERO nvcc on the prebuilt-wheelhouse path.
exec /kaniko/executor \
  --context dir:///app \
  --dockerfile "${DOCKERFILE:-docker/Dockerfile.gpu-rl}" \
  --build-arg WHEEL_SOURCE="$WHEEL_SOURCE" \
  --skip-unused-stages \
  $SNAPSHOT_FLAG \
  --compressed-caching=false \
  $CACHE_FLAGS \
  $FLOATING_DEST_FLAG \
  --destination "${DEST_PINNED}"
