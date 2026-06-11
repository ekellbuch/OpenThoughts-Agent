#!/bin/bash
# =============================================================================
# build_vllm0202rc0_r3_sif.sbatch
#
# Reproduces the (purged) #208/#211 "shelf asset": a SkyRL+Megatron+TE SIF whose
# vLLM is the penfever fork at v0.20.2rc0 + native R3 routed-experts capture, with
# Gemma4 / Qwen3-MoE / Qwen3-Next model support, on Jupiter GH200 (aarch64, cu13).
#
# STRATEGY
#   base SIF  = skyrl_megatron.sif      (NGC 25.09: torch 2.9.0a0+nv25.09 cu130,
#                                        triton 3.4.0, Megatron-core 0.14.0, TE 2.7,
#                                        apex, flash_attn 2.7.4, transformers 5.10.1
#                                        [gemma4/qwen3_next/qwen3_moe], SkyRL editable
#                                        at /opt/SkyRL, nvcc CUDA 13.0). NO vLLM.
#   + build the vLLM 0.20.2rc0 fork FROM SOURCE against the in-SIF torch 2.9
#     (use_existing_torch.py strips the fork's torch==2.11.0 pin so we keep the
#     NGC torch the whole Megatron/TE/apex/flash_attn/flashinfer stack is built on).
#     The fork's HEAD (v0.20.2rc0-305-g1948bebd1) already carries the R3 capture
#     rail (routed_experts_capturer.py + routed_experts emission in the four
#     chat/completion serving+protocol files) and the gemma4 models — so NO extra
#     R3 patch is applied; R3 is native.
#   + merge the FlashQLA / tilelang / tvm_ffi / fla GDN overlay (fla_tilelang_overlay.img)
#     via debugfs rdump (no fuse, no root) — same as bake_r3_sif.sbatch.
#   then apptainer build the final SIF.
#
# OUTPUT: $CONTAINERS/skyrl_megatron_vllm0202rc0_r3.sif
#
# The fork SOURCE is rsync'd to the cluster separately (per CLAUDE.md, vLLM syncs
# via rsync, not git-to-cluster). This sbatch expects it at $VLLM_SRC below.
# =============================================================================
#SBATCH --job-name=build_vllm0202rc0_r3
#SBATCH --account=reformo
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=05:00:00
#SBATCH --output=/e/scratch/jureap59/feuer1/sif_build/logs/build0202_%j.out
#SBATCH --error=/e/scratch/jureap59/feuer1/sif_build/logs/build0202_%j.err

set -euo pipefail

CONTAINERS=/e/scratch/jureap59/feuer1/containers
BASE_SIF=$CONTAINERS/skyrl_megatron.sif                 # NGC base, NO vLLM
OVL_FLA=$CONTAINERS/fla_tilelang_overlay.img            # FlashQLA/tilelang/tvm_ffi/fla GDN
OUT_SIF=$CONTAINERS/skyrl_megatron_vllm0202rc0_r3.sif
VLLM_SRC=/e/scratch/jureap59/feuer1/sif_build/vllm_src  # rsync'd fork @ v0.20.2rc0-305-g1948bebd1

# Build scratch on GPFS (node-local /tmp too small for sandbox+extract+SIF).
BUILD=/e/scratch/jureap59/feuer1/sif_build/job_$SLURM_JOB_ID
SANDBOX=$BUILD/sandbox
mkdir -p "$BUILD"
export APPTAINER_TMPDIR=$BUILD/aptmp
export APPTAINER_CACHEDIR=$BUILD/apcache
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"
LOCAL=/tmp/sif_build_$SLURM_JOB_ID
mkdir -p "$LOCAL"

cleanup() { rm -rf "$LOCAL" "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR" 2>/dev/null || true; }
trap cleanup EXIT

echo "=== $(date) host=$(hostname) job=$SLURM_JOB_ID ==="
echo "BASE_SIF=$BASE_SIF ($(stat -c%s $BASE_SIF) bytes)"
echo "OVL_FLA=$OVL_FLA ($(stat -c%s $OVL_FLA) bytes)"
echo "VLLM_SRC=$VLLM_SRC"
[ -d "$VLLM_SRC/vllm" ] || { echo "FATAL: vLLM fork source not found at $VLLM_SRC (rsync it first)"; exit 2; }
cat "$VLLM_SRC/.vllm_commit" 2>/dev/null || true
df -h /tmp | tail -1; df -h /e/scratch | tail -1

# ---------------------------------------------------------------------------
# 1. Writable sandbox from base SIF
# ---------------------------------------------------------------------------
echo "=== [1] building sandbox from base SIF ==="
rm -rf "$SANDBOX"
apptainer build --sandbox "$SANDBOX" "$BASE_SIF"
echo "sandbox built."

# Stage the fork source INTO the sandbox (writable; pip build writes egg-info etc.)
echo "=== [1b] staging vLLM fork source into sandbox ==="
mkdir -p "$SANDBOX/opt/vllm_build"
cp -a "$VLLM_SRC"/. "$SANDBOX/opt/vllm_build/"

# Stage the PEP517 build-backend wheels (setuptools_scm + deps) INTO the sandbox.
# The booster compute node has NO internet, so pip cannot fetch these at build
# time. They are pre-downloaded on the (internet-enabled) login node into
# $BUILD_DEPS_WHEELS and installed offline (--no-index) before the editable
# vLLM build, which runs setup.py in-env under --no-build-isolation and needs
# setuptools_scm present. (Root cause of the job 848577 failure.)
BUILD_DEPS_WHEELS=/e/scratch/jureap59/feuer1/sif_build/build_deps_wheels
echo "=== [1c] staging build-backend wheels into sandbox ==="
[ -d "$BUILD_DEPS_WHEELS" ] || { echo "FATAL: build-deps wheelhouse not found at $BUILD_DEPS_WHEELS (pre-download on login node: apptainer exec $BASE_SIF python -m pip download --dest $BUILD_DEPS_WHEELS setuptools_scm setuptools wheel packaging)"; exit 4; }
ls "$BUILD_DEPS_WHEELS"/*.whl >/dev/null 2>&1 || { echo "FATAL: no .whl in $BUILD_DEPS_WHEELS"; exit 4; }
mkdir -p "$SANDBOX/opt/build_deps_wheels"
cp -a "$BUILD_DEPS_WHEELS"/. "$SANDBOX/opt/build_deps_wheels/"
echo "staged $(ls "$SANDBOX/opt/build_deps_wheels"/*.whl | wc -l) build-dep wheels."

# ---------------------------------------------------------------------------
# 2. Build + install vLLM 0.20.2rc0 fork FROM SOURCE against in-SIF torch 2.9
# ---------------------------------------------------------------------------
echo "=== [2] building vLLM from source inside sandbox (against existing torch) ==="
# --writable-tmpfs is not enough (pip needs to write into /usr/local/lib); use
# --writable on the sandbox. Bind nothing external; compile fully offline.
# MAX_JOBS bounded to keep peak RSS under the node; nvcc + ninja from the SIF.
# NOTE: the rsync'd source has NO .git, so setuptools_scm cannot derive the
# version. Pin it explicitly so the wheel reports 0.20.2rc0 (matches the fork's
# `git describe` tag v0.20.2rc0). Without this, the build errors or reports 0.0.0.
apptainer exec --writable --no-home \
  --env MAX_JOBS=48 \
  --env CMAKE_BUILD_TYPE=Release \
  --env VLLM_TARGET_DEVICE=cuda \
  --env TORCH_CUDA_ARCH_LIST="9.0+PTX" \
  --env CCACHE_DISABLE=1 \
  --env PIP_NO_BUILD_ISOLATION=1 \
  --env SETUPTOOLS_SCM_PRETEND_VERSION=0.20.2rc0 \
  --env SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM=0.20.2rc0 \
  "$SANDBOX" bash -lc '
    set -euo pipefail
    cd /opt/vllm_build
    echo "--- pre-build torch / nvcc ---"
    python -c "import torch;print(\"torch\",torch.__version__,\"cuda\",torch.version.cuda)"
    nvcc --version | tail -2
    echo "--- strip torch pins so the build uses the in-SIF NGC torch 2.9 ---"
    python use_existing_torch.py
    echo "--- install PEP517 build-backend deps OFFLINE (setuptools_scm + deps) ---"
    # With --no-build-isolation the editable build runs setup.py in THIS env, so
    # setuptools_scm (+ setuptools/wheel/packaging) must be present here explicitly.
    # The base SIF lacks setuptools_scm -> job 848577 failed at "Preparing editable
    # metadata (pyproject.toml)" with ModuleNotFoundError: No module named
    # setuptools_scm. The compute node has no internet; install from the staged
    # wheelhouse. NOT guarded by `|| true` -- this MUST succeed or the build is
    # doomed at metadata generation anyway.
    python -m pip install --no-index --find-links /opt/build_deps_wheels \
      setuptools_scm setuptools wheel packaging
    # VERIFY: just confirm the import succeeds + report the version via the
    # packaging-metadata API. setuptools_scm v8 REMOVED the module-level
    # `__version__` attribute, so `setuptools_scm.__version__` AttributeErrors
    # under set -e (job 849267). Use importlib.metadata, and do NOT gate the
    # build on the version string -- a successful import is the only thing the
    # editable build actually needs.
    # NB: the whole in-sandbox block is wrapped in bash -lc + a single-quoted string,
    # so these python -c snippets must use ESCAPED DOUBLE QUOTES (\") ONLY. Any bare
    # single-quote char anywhere in this block (even in a comment) closes the outer
    # quote and the shell runs import as a command (job 849776: unexpected EOF).
    python -c "import setuptools_scm; print(\"setuptools_scm import OK\")"  # the REAL gate (import must work)
    python -c "import importlib.metadata as md; print(\"setuptools_scm\", md.version(\"setuptools-scm\"))" || true  # version report is non-fatal
    echo "--- install python build deps (no torch) ---"
    python -m pip install --no-build-isolation -r requirements/build.txt || true
    echo "--- build + install vLLM (editable, no-build-isolation, existing torch) ---"
    # precompiled kernels are unavailable for this torch; compile from source.
    python -m pip install --no-build-isolation -v -e . 2>&1 | tail -120
    echo "--- post-install version probe ---"
    python -c "import vllm; print(\"vllm\", vllm.__version__)"
  '
echo "vLLM build/install complete."

# ---------------------------------------------------------------------------
# 3. Merge FlashQLA / tilelang / tvm_ffi / fla GDN overlay via debugfs rdump
# ---------------------------------------------------------------------------
merge_overlay() {
  local IMG=$1 NAME=$2
  echo "=== [3.$NAME] extracting $IMG via debugfs ==="
  local D=$LOCAL/x_$NAME
  rm -rf "$D"; mkdir -p "$D"
  debugfs -R "rdump /upper $D" "$IMG" 2>&1 | grep -v "^debugfs 1\." || true
  local UPPER="$D/upper"
  local n=$(find "$UPPER" -type f 2>/dev/null | wc -l)
  echo "$NAME extracted upper file count: $n"
  [ "$n" -gt 0 ] || { echo "FATAL: 0 files extracted for $NAME"; exit 3; }
  cp -a "$UPPER"/. "$SANDBOX"/
  echo "$NAME merged into sandbox."
  rm -rf "$D"
}
merge_overlay "$OVL_FLA" fla

echo "=== [3b] verifying merged GDN overlay paths ==="
for d in \
  usr/local/lib/python3.12/dist-packages/tilelang \
  usr/local/lib/python3.12/dist-packages/flash_qla \
  usr/local/lib/python3.12/dist-packages/tvm_ffi \
  usr/local/lib/python3.12/dist-packages/tilelang-0.1.8.dist-info \
  usr/local/lib/python3.12/dist-packages/flash_qla-0.1.0+6ef4858.dist-info \
  usr/local/lib/python3.12/dist-packages/apache_tvm_ffi-0.1.9.dist-info ; do
  [ -e "$SANDBOX/$d" ] && echo "OK  $d" || echo "MISSING $d"
done

# Drop the bulky source tree from the final image (keep the editable install
# pointer though — editable -e installs reference /opt/vllm_build/vllm at import
# time, so we KEEP the source but prune build artifacts to shrink the SIF).
echo "=== [3c] pruning vLLM build artifacts (keep source for editable import) ==="
rm -rf "$SANDBOX/opt/vllm_build/build" "$SANDBOX/opt/vllm_build/.deps" \
       "$SANDBOX"/root/.cache/pip "$SANDBOX"/opt/vllm_build/*.egg-info/SOURCES.txt 2>/dev/null || true

# ---------------------------------------------------------------------------
# 4. Rebuild final SIF
# ---------------------------------------------------------------------------
echo "=== [4] building final SIF -> $OUT_SIF ==="
rm -f "$OUT_SIF"
apptainer build "$OUT_SIF" "$SANDBOX"
echo "final SIF built: $(stat -c%s $OUT_SIF) bytes"
ls -la "$OUT_SIF"

# ---------------------------------------------------------------------------
# 5. Validate
# ---------------------------------------------------------------------------
echo "=== [5] validating baked SIF ==="
# Post-build validation is informational only -- the SIF is already written at
# this point, so a probe hiccup (e.g. no GPU visible on the build node for
# torch.cuda.is_available, or an unexpected import surprise) must NOT flip the
# whole job to FAILED and mask a successful build. Guarded with `|| true`.
apptainer exec --nv "$OUT_SIF" python - <<'PYEOF' || echo "WARN: validation block exited non-zero (SIF already built; inspect output above)"
import importlib
def chk(n):
    try:
        m=importlib.import_module(n); print(f"IMPORT OK  {n}  v={getattr(m,'__version__','n/a')}"); return m
    except Exception as e:
        print(f"IMPORT FAIL {n}: {type(e).__name__}: {e}"); return None
print("--- base stack ---")
import torch; print("torch", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available())
v=chk("vllm")
import vllm; print("VLLM_VERSION:", vllm.__version__)
chk("transformers")
import transformers
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES as C
print("transformers gemma4:", "gemma4" in C, "gemma4_text:", "gemma4_text" in C,
      "qwen3_next:", "qwen3_next" in C, "qwen3_moe:", "qwen3_moe" in C)
chk("skyrl_train"); chk("skyrl_gym")
print("--- GDN overlay ---")
chk("tilelang"); chk("tvm_ffi")
try:
    import flash_qla; print("IMPORT OK flash_qla", getattr(flash_qla,'__version__','n/a'))
except Exception as e: print("flash_qla:", type(e).__name__, e)
print("--- vLLM model registry archs ---")
try:
    from vllm import ModelRegistry
    archs=set(ModelRegistry.get_supported_archs())
    for a in ("Gemma4ForConditionalGeneration","Gemma4ForCausalLM","Qwen3MoeForCausalLM","Qwen3NextForCausalLM"):
        print(f"  REGISTRY {a}: {a in archs}")
except Exception as e: print("registry probe FAIL:", type(e).__name__, e)
print("--- R3 routed_experts native capture (serving + protocol) ---")
import os
sp="/opt/vllm_build"  # editable source root
for rel in ("vllm/entrypoints/openai/chat_completion/serving.py",
            "vllm/entrypoints/openai/chat_completion/protocol.py",
            "vllm/entrypoints/openai/completion/serving.py",
            "vllm/entrypoints/openai/completion/protocol.py",
            "vllm/model_executor/layers/fused_moe/routed_experts_capturer.py"):
    p=os.path.join(sp,rel)
    if os.path.exists(p):
        print(f"  {rel}: routed_experts_count={open(p).read().count('routed_experts')}")
    else:
        # fall back to installed location
        import vllm as _v, pathlib
        ip=os.path.join(os.path.dirname(_v.__file__), rel.split('vllm/',1)[1])
        print(f"  {rel}: src-missing installed_exists={os.path.exists(ip)} "
              f"count={open(ip).read().count('routed_experts') if os.path.exists(ip) else 'n/a'}")
print("--- GDN overlay version parity ---")
import importlib.metadata as md
def ver(p):
    try: return md.version(p)
    except Exception as e: return f"ERR:{e}"
for pkg,exp in [("tilelang","0.1.8"),("apache-tvm-ffi","0.1.9"),("flash_qla","0.1.0+6ef4858")]:
    print(f"  {pkg} = {ver(pkg)} (expect {exp})")
PYEOF

echo "=== $(date) BUILD COMPLETE ==="
echo "OUT_SIF=$OUT_SIF"
echo "=== removing sandbox ==="
rm -rf "$SANDBOX" "$BUILD/aptmp" "$BUILD/apcache" 2>/dev/null || true
echo "=== done ==="
