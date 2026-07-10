#!/usr/bin/env bash
# peek_rl_rollouts.sh — inspect (or fully capture) the Harbor rollout artifacts (trace_jobs) of a
# running MarinSkyRL agentic-RL job on cw-us-east-02a, by reaching its rank-0 pod.
#
# WHY: agentic RL (terminal_bench / Harbor) writes per-trial rollout artifacts (the literal agent
# trajectory + prompts/responses + verifier_output + result.json reward) to
# terminal_bench_config.trials_dir. Our jobs launch with a REMOTE object-store trials_dir
# (s3://marin-us-east-02a/iris/<job>/trace_jobs via launch_rl_iris.py --trials-dir auto) — DURABLE (survives
# pod GC), unlike the old node-local ephemeral path (trials_dir: null). NOTE: the default store moved
# R2 (s3://marin-na) -> CW (s3://marin-us-east-02a) on 2026-07-05 (marin c7caecc95a). The rank-0 pod carries the
# cluster-injected creds + AWS_ENDPOINT_URL (iris-task-env Secret), but the LAUNCH HOST (Mac) does
# NOT have working cluster (CW) object-store creds. So this script does all object-store ops INSIDE the pod via boto3 (the
# proven path). Legacy jobs that wrote to a node-local trials_dir are still handled via the pod's
# local path. `result.json` is the COMPLETED-trial marker (it carries the reward) — its count is the
# real "how many trials finished" answer (a started trial has config/prompt/debug but no result.json).
#
# USAGE:
#   peek_rl_rollouts.sh <pod-name-substring>                  # SUMMARY: trial dirs started + COMPLETED (result.json) + breakdown
#   peek_rl_rollouts.sh <substr> ls   [glob]                  # list trial dirs (+ started/completed counts)
#   peek_rl_rollouts.sh <substr> cat  <trial-dir>             # dump a trial's json artifacts (the literal rollout)
#   peek_rl_rollouts.sh <substr> grep <pattern>               # list trial json files whose body matches a regex
#   peek_rl_rollouts.sh <substr> cp   <trial-dir> [dest]      # pull a single trial dir to the launch host
#   peek_rl_rollouts.sh <substr> pull [out-base-dir]          # FULL CAPTURE -> date-stamped subdir:
#                                                             #   complete iris finelog + per-rank pod logs
#                                                             #   + ALL trace_jobs (synced from R2) + MANIFEST.md
#                                                             #   + any torch NCCL flight-recorder dumps (see frdump)
#   peek_rl_rollouts.sh <substr> frdump [dest] [fr-base]      # torch NCCL flight-recorder dump ONLY (fast, no
#                                                             #   S3/trace_jobs/finelog): kubectl-cp's
#                                                             #   TORCH_NCCL_DEBUG_INFO_TEMP_FILE-prefixed files off
#                                                             #   EVERY pod of this job (not just rank-0) into
#                                                             #   dest/fr_dumps/<pod>/. Run this THE MOMENT a
#                                                             #   "ProcessGroupNCCL's watchdog got stuck" /
#                                                             #   "Fatal Python error: Aborted" signature appears in
#                                                             #   the finelog -- BEFORE killing the job -- the FR
#                                                             #   dump is node-local (CoreWeave has no shared POSIX
#                                                             #   mount) and is lost once the pod is reaped.
#                                                             #   fr-base defaults to /tmp/fr_dumps/<jobname>/nccl_fr_rank
#                                                             #   (the 2026-07-10 job-scoped FR path convention --
#                                                             #   override if a config still uses the old bare
#                                                             #   /tmp/nccl_fr_rank).
#
# NOTE: <substr> matches the POD name (iris-benjaminfeuer-<name>-<rank>-<hash>-0), which can differ
# from the iris job_id display name. With no match the script lists candidate rl pods.
#
# ENV: PEEK_KUBECONFIG (default ~/.kube/coreweave-iris-gpu), NS (default iris), CONTAINER (default task),
#      PEEK_CLUSTER (default cw-us-east-02a), IRIS_BIN (default the otagent cw-capable iris),
#      PEEK_OUT (default ~/Documents/experiments/traces),
#      PEEK_TRIALS_S3 (override the remote trials_dir; default s3://marin-us-east-02a/iris/<jobname>/trace_jobs),
#      PEEK_MAX_OBJECT_BYTES (pull: skip any single object larger than this; default 20MB=20971520,
#                             set 0 to fetch everything incl. the 100s-of-MB result.json blobs)
#      PEEK_FR_BASE (frdump/pull: override the in-pod FR dump path prefix; default
#                    /tmp/fr_dumps/<jobname>/nccl_fr_rank)
set -euo pipefail

JOB="${1:-}"
ACTION="${2:-ls}"
# Force the CoreWeave kubeconfig. Do NOT honor an inherited $KUBECONFIG — the login shell's default
# points at a different cluster (→ 'no pods'); override only via PEEK_KUBECONFIG.
export KUBECONFIG="${PEEK_KUBECONFIG:-$HOME/.kube/coreweave-iris-gpu}"
NS="${NS:-iris}"
CONTAINER="${CONTAINER:-task}"
CLUSTER="${PEEK_CLUSTER:-cw-us-east-02a}"
# Default to the OTAGENT iris (the marin .venv iris has a broken `kubernetes` import → cannot drive cw).
IRIS_BIN="${IRIS_BIN:-/Users/benjaminfeuer/miniconda3/envs/otagent/bin/iris}"
PEEK_OUT="${PEEK_OUT:-/Users/benjaminfeuer/Documents/experiments/traces}"

if [ -z "$JOB" ]; then
  echo "usage: peek_rl_rollouts.sh <pod-name-substring> [ls|cat|grep|cp|pull] [args]" >&2
  echo "running rl pods in ns/$NS:" >&2
  kubectl get pods -n "$NS" -o name 2>/dev/null | grep -iE "rl-|cpdcp|resmoke|a3b" | sed 's#^pod/#  #' >&2 || true
  exit 64
fi

# rank-0 pod = the rank that owns the Harbor coordinator / trials_dir writes.
# Match rank-0 of the LATEST generation: the pod-name suffix is the iris retry generation
# (`-0` first attempt, `-1` after a --max-retries re-bring-up, …), so a hardcoded `-0$` misses
# a retried job's live pod. Take the highest generation.
POD=$(kubectl get pods -n "$NS" -o name 2>/dev/null | grep -E "iris-.*${JOB}.*-0-[0-9a-f]+-[0-9]+$" | sort | tail -1 || true)
if [ -z "$POD" ]; then
  echo "[peek] no running rank-0 pod matching '*${JOB}*-0-*' in ns/$NS." >&2
  echo "[peek] (job terminal? then a node-local trials_dir is GC'd — only a REMOTE R2 trials_dir survives,"
  echo "[peek]  inspect it with: PEEK_TRIALS_S3=s3://… and a still-running pod, or aws/boto3 against R2.) Candidate rl pods:" >&2
  kubectl get pods -n "$NS" -o name 2>/dev/null | grep -iE "rl-|cpdcp|resmoke|a3b" | sed 's#^pod/#  #' >&2 || true
  exit 1
fi
POD="${POD#pod/}"
echo "[peek] pod=$POD  ns=$NS  container=$CONTAINER"

# Derive the iris job_id (/<user>/<jobname>) from the pod name for finelog + dest naming.
USER_FROM_POD=$(printf '%s' "$POD" | sed -E 's/^iris-([a-z0-9]+)-.*/\1/')
JOBNAME=$(printf '%s' "$POD" | sed -E 's/^iris-[a-z0-9]+-(.+)-[0-9]+-[0-9a-f]+-[0-9]+$/\1/')
JOBID="/${USER_FROM_POD}/${JOBNAME}"

kexec() { kubectl exec -n "$NS" "$POD" -c "$CONTAINER" -- bash -lc "$1"; }

# In-pod torch NCCL flight-recorder dump path prefix. Default matches the 2026-07-10
# job-scoped convention (TORCH_NCCL_DEBUG_INFO_TEMP_FILE: /tmp/fr_dumps/<jobname>/nccl_fr_rank
# in e.g. hpc/skyrl_yaml/iris/128GPU_80B_A3B_next_cp1.yaml); override via PEEK_FR_BASE for
# configs still on the old bare /tmp/nccl_fr_rank. torch appends the GLOBAL RANK to this
# prefix per-rank (no separator), so we glob "${FR_BASE}*".
FR_BASE="${PEEK_FR_BASE:-/tmp/fr_dumps/${JOBNAME}/nccl_fr_rank}"

# pull_fr_dumps <dest-dir> — kubectl-cp any files matching $FR_BASE* off EVERY pod of this
# job (each pod = one node = up to 8 local ranks) into <dest-dir>/fr_dumps/<pod>/. No-op
# (0 files) on a healthy run that never hit TORCH_NCCL_DUMP_ON_TIMEOUT. Safe to call anytime
# the job's pods are still up -- this is the retrieval step that was MISSING for the
# 2026-07-09 80b-next-cp1 wedge (dump path was node-local /tmp and nobody pulled it before
# the pod was reaped).
pull_fr_dumps() {
  local dest="$1"
  local nfound=0
  echo "[frdump] base=${FR_BASE}*  (across all pods of job matching '*${JOB}*')"
  for p in $(kubectl get pods -n "$NS" -o name 2>/dev/null | grep -E "iris-.*${JOB}.*-[0-9]+-[0-9a-f]+-[0-9]+$" | sed 's#pod/##' | sort); do
    local files
    files=$(kubectl exec -n "$NS" "$p" -c "$CONTAINER" -- sh -c "ls ${FR_BASE}* 2>/dev/null" 2>/dev/null | tr -d '\r' || true)
    if [ -z "$files" ]; then
      continue
    fi
    local poddir="${dest}/fr_dumps/${p}"
    mkdir -p "$poddir"
    while IFS= read -r f; do
      [ -z "$f" ] && continue
      if kubectl cp -c "$CONTAINER" "$NS/$p:$f" "$poddir/$(basename "$f")" >/dev/null 2>&1; then
        nfound=$((nfound + 1))
        echo "[frdump]   ${p}:${f} -> ${poddir}/$(basename "$f")"
      else
        echo "[frdump]   WARN: kubectl cp failed for ${p}:${f}" >&2
      fi
    done <<< "$files"
  done
  if [ "$nfound" -eq 0 ]; then
    echo "[frdump] no FR dump files found (either a healthy run, or the pods are already gone)."
  else
    echo "[frdump] pulled ${nfound} FR dump file(s) -> ${dest}/fr_dumps/"
  fi
  return 0
}

# --- trials_dir discovery: prefer a node-local path (legacy trials_dir: null); else REMOTE R2. ---
TJ_LOCAL=$(kexec 'ls -d /app/experiments/*/trace_jobs 2>/dev/null | head -1' 2>/dev/null | tr -d '\r' || true)
S3_TJ="${PEEK_TRIALS_S3:-s3://marin-us-east-02a/iris/${JOBNAME}/trace_jobs}"
if [ -n "$TJ_LOCAL" ]; then
  MODE_LOCAL=1
  echo "[peek] LOCAL trials_dir=$TJ_LOCAL"
else
  MODE_LOCAL=0
  echo "[peek] REMOTE trials_dir=$S3_TJ  (R2 via rank-0 pod boto3; Mac lacks cluster (CW) object-store creds)"
fi

# Run an R2 op INSIDE the rank-0 pod (it has AWS_ENDPOINT_URL + injected R2 creds + boto3).
#   r2_op count              -> trial-dir + COMPLETED (result.json) counts + artifact breakdown + episode range
#   r2_op listdirs           -> one trial-dir name per line
#   r2_op download <pod-dir> -> download every object under the trials_dir prefix into <pod-dir>; echoes the object count
#   r2_op catdir <trial>     -> print every *.json under that trial (key header + body)
#   r2_op grep <regex>       -> print trial-relative keys of *.json objects whose body matches <regex>
r2_op() {
  kubectl exec -i -n "$NS" "$POD" -c "$CONTAINER" -- python - "$S3_TJ" "$@" <<'PYEOF'
import sys, os, re, collections, boto3
from botocore.config import Config
s3url = sys.argv[1]
mode  = sys.argv[2] if len(sys.argv) > 2 else "count"
arg   = sys.argv[3] if len(sys.argv) > 3 else ""
assert s3url.startswith("s3://"), s3url
BUCKET, _, PREFIX = s3url[5:].partition("/")
PREFIX = PREFIX.rstrip("/") + "/"
# CW object store requires VIRTUAL-hosted addressing (path-style -> PathStyleRequestNotAllowed);
# same fix as 398a0481/a9f4c8c5/2e194e31 for the training S3 clients.
c = boto3.client("s3", endpoint_url=os.environ["AWS_ENDPOINT_URL"],
                 config=Config(s3={"addressing_style": "virtual"}))
keys = []
for page in c.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=PREFIX):
    keys += [o["Key"] for o in page.get("Contents", [])]
rel  = [k[len(PREFIX):] for k in keys if k[len(PREFIX):]]
dirs = sorted(set(r.split("/")[0] for r in rel))
done = [k for k in keys if k.endswith("result.json")]
if mode == "count":
    print(f"trials_dir          : {s3url}")
    print(f"trial dirs started  : {len(dirs)}")
    print(f"COMPLETED (result.json w/ reward) : {len(done)}")
    print("artifact breakdown  :", dict(collections.Counter(r.rsplit('/', 1)[-1] for r in rel).most_common(10)))
    eps = [int(m.group(1)) for r in rel for m in [re.search(r'episode-(\d+)', r)] if m]
    if eps:
        print(f"episode range       : {min(eps)}..{max(eps)}")
elif mode == "listdirs":
    for d in dirs:
        print(d)
elif mode == "listkeys":
    # "<size> <trial-relative-key>" per object (size first; keys have no spaces, so
    # the Mac splits on the first space). Used by `pull` to fetch + size-verify each object.
    for page in c.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=PREFIX):
        for o in page.get("Contents", []):
            r = o["Key"][len(PREFIX):]
            if r:
                print(f"{o['Size']} {r}")
elif mode == "download":
    dest = arg or "/tmp/peek_tj"
    n = 0
    for k in keys:
        r = k[len(PREFIX):]
        if not r:
            continue
        p = os.path.join(dest, r)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        c.download_file(BUCKET, k, p)
        n += 1
    print(n)  # object count -> stdout (last line)
elif mode == "catdir":
    for k in keys:
        r = k[len(PREFIX):]
        if r.split("/")[0] == arg and k.endswith(".json"):
            print(f"\n# {r}")
            try:
                print(c.get_object(Bucket=BUCKET, Key=k)["Body"].read().decode("utf-8", "replace"))
            except Exception as e:
                print(f"<read error: {e}>")
elif mode == "grep":
    pat = re.compile(arg)
    for k in keys:
        if not k.endswith(".json"):
            continue
        try:
            body = c.get_object(Bucket=BUCKET, Key=k)["Body"].read().decode("utf-8", "replace")
        except Exception:
            continue
        if pat.search(body):
            print(k[len(PREFIX):])
PYEOF
}

case "$ACTION" in
  ls)
    if [ "$MODE_LOCAL" = 1 ]; then
      GLOB="${3:-*}"
      kexec "ls -d $TJ_LOCAL/$GLOB/ 2>/dev/null | sed 's#$TJ_LOCAL/##'" || true
      echo "[peek] total trial dirs: $(kexec "ls -d $TJ_LOCAL/*/ 2>/dev/null | wc -l" | tr -d ' ')"
    else
      r2_op count
    fi
    ;;
  cat)
    TR="${3:?cat needs <trial-dir>}"
    if [ "$MODE_LOCAL" = 1 ]; then
      kexec "find '$TJ_LOCAL/$TR' -maxdepth 2 -name '*.json' -print -exec sh -c 'echo; cat \"\$1\"; echo' _ {} \; 2>/dev/null"
    else
      r2_op catdir "$TR"
    fi
    ;;
  grep)
    PAT="${3:?grep needs <pattern>}"
    if [ "$MODE_LOCAL" = 1 ]; then
      kexec "grep -rls --include='*.json' -e '$PAT' '$TJ_LOCAL' 2>/dev/null | sed 's#$TJ_LOCAL/##' | head -40" || true
    else
      r2_op grep "$PAT" | head -40
    fi
    ;;
  cp)
    TR="${3:?cp needs <trial-dir>}"; DEST="${4:-./$TR}"
    if [ "$MODE_LOCAL" = 1 ]; then
      kubectl cp -c "$CONTAINER" "$NS/$POD:$TJ_LOCAL/$TR" "$DEST"
    else
      mkdir -p "$DEST"
      POD_TMP="/tmp/peek_cp_${TR//\//_}"
      kubectl exec -i -n "$NS" "$POD" -c "$CONTAINER" -- python - "$S3_TJ/$TR" /dev/null download "$POD_TMP" <<'PYEOF' >/dev/null
import sys, os, boto3
from botocore.config import Config
s3url = sys.argv[1]; dest = sys.argv[3]
BUCKET, _, PREFIX = s3url[5:].partition("/"); PREFIX = PREFIX.rstrip("/") + "/"
c = boto3.client("s3", endpoint_url=os.environ["AWS_ENDPOINT_URL"],
                 config=Config(s3={"addressing_style": "virtual"}))
for page in c.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=PREFIX):
    for o in page.get("Contents", []):
        r = o["Key"][len(PREFIX):]
        if not r: continue
        p = os.path.join(dest, r); os.makedirs(os.path.dirname(p), exist_ok=True)
        c.download_file(BUCKET, o["Key"], p)
PYEOF
      kubectl exec -n "$NS" "$POD" -c "$CONTAINER" -- tar cf - -C "$POD_TMP" . 2>/dev/null | tar xf - -C "$DEST/" || true
      kubectl exec -n "$NS" "$POD" -c "$CONTAINER" -- rm -rf "$POD_TMP" 2>/dev/null || true
    fi
    echo "[peek] copied -> $DEST"
    ;;
  pull)
    # FULL CAPTURE into a fresh date-stamped subdir: complete iris finelog + per-rank pod logs + ALL
    # trace_jobs (synced from R2, or tar'd from a legacy node-local path) + a provenance MANIFEST.
    OUTBASE="${3:-$PEEK_OUT}"
    STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
    DEST="${OUTBASE}/${JOBNAME}_${STAMP}"
    mkdir -p "$DEST/logs" "$DEST/trace_jobs"
    echo "[pull] dest=$DEST  jobid=$JOBID  cluster=$CLUSTER"

    # 1) Complete iris/finelog job log (full history, no tail).
    echo "[pull] capturing iris finelog ..."
    "$IRIS_BIN" --cluster="$CLUSTER" job logs "$JOBID" --max-lines 10000000 --no-tail \
      > "$DEST/logs/iris_finelog.log" 2> "$DEST/logs/iris_finelog.stderr" \
      || echo "[pull] WARN: iris finelog returned nonzero (see logs/iris_finelog.stderr)" >&2
    echo "[pull]   finelog: $(wc -l < "$DEST/logs/iris_finelog.log" | tr -d ' ') lines"

    # 2) Per-pod container stdout for every rank of this job (rank-0 = harbor coordinator).
    echo "[pull] capturing per-rank pod logs ..."
    for p in $(kubectl get pods -n "$NS" -o name 2>/dev/null | grep -E "iris-.*${JOB}.*-[0-9]+-[0-9a-f]+-[0-9]+$" | sed 's#pod/##' | sort); do
      rank=$(printf '%s' "$p" | sed -E 's/.*-([0-9]+)-[0-9a-f]+-[0-9]+$/\1/')
      kubectl logs -n "$NS" "$p" -c "$CONTAINER" --tail=-1 > "$DEST/logs/pod_rank${rank}.log" 2>/dev/null &
    done
    wait

    # 3) Capture ALL trace_jobs. REMOTE (R2): download via the rank-0 pod's boto3 into a pod tmp dir,
    #    then tar-stream it to the Mac. LOCAL (legacy): tar-stream the pod's node-local path directly.
    N_TRIALS=0; N_DONE=0
    if [ "$MODE_LOCAL" = 1 ]; then
      echo "[pull] tar-streaming node-local trace_jobs ($TJ_LOCAL) ..."
      PARENT="$(dirname "$TJ_LOCAL")"; BASE="$(basename "$TJ_LOCAL")"
      { kubectl exec -n "$NS" "$POD" -c "$CONTAINER" -- tar cf - -C "$PARENT" "$BASE" 2>/dev/null | tar xf - -C "$DEST/"; } \
        || echo "[pull] WARN: trace_jobs tar returned nonzero (capture may be partial)" >&2
    else
      echo "[pull] downloading trace_jobs DIRECT from R2 ($S3_TJ) — Mac<-R2 via boto3 ..."
      # Bulk artifacts download DIRECTLY from R2 to the Mac (boto3 download_file = native
      # multipart + retries), NOT through `kubectl exec`: result.json can be 100s of MB and
      # truncates over the exec/SPDY stream. The Mac has no cluster object-store creds of its own, so we
      # lift the rank-0 pod's injected R2 creds (endpoint+key+secret) into THIS process's env
      # only (never printed). R2 (Cloudflare) is internet-reachable and egress-free; force
      # region=auto (the Mac's default AWS_REGION e.g. us-east-2 is rejected by R2).
      PEEK_PY="${PEEK_PY:-$(dirname "$IRIS_BIN")/python}"
      creds=$(kubectl exec -n "$NS" "$POD" -c "$CONTAINER" -- sh -c \
        'printf "%s\n%s\n%s\n" "$AWS_ENDPOINT_URL" "$AWS_ACCESS_KEY_ID" "$AWS_SECRET_ACCESS_KEY"' 2>/dev/null | tr -d '\r')
      R2_ENDPOINT=$(printf '%s\n' "$creds" | sed -n 1p)
      R2_KEY=$(printf '%s\n' "$creds" | sed -n 2p)
      R2_SECRET=$(printf '%s\n' "$creds" | sed -n 3p)
      if [ -z "$R2_ENDPOINT" ] || [ -z "$R2_KEY" ] || [ -z "$R2_SECRET" ]; then
        echo "[pull] WARN: could not lift R2 creds from pod; skipping trace_jobs download." >&2
      else
        AWS_ENDPOINT_URL="$R2_ENDPOINT" AWS_ACCESS_KEY_ID="$R2_KEY" AWS_SECRET_ACCESS_KEY="$R2_SECRET" \
          AWS_REGION=auto AWS_DEFAULT_REGION=auto \
          "$PEEK_PY" - "$S3_TJ" "$DEST/trace_jobs" "$DEST/.r2_failed.tsv" "$DEST/.r2_skipped.tsv" <<'PYEOF'
import os, sys, boto3, botocore, concurrent.futures as cf
from boto3.s3.transfer import TransferConfig
s3url, dest, faillog, skiplog = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
bucket, _, prefix = s3url[5:].partition("/"); prefix = prefix.rstrip("/") + "/"
# Skip any SINGLE object larger than this (default 20 MB). The huge result.json blobs
# (full rollout_details/logprobs, 100s of MB) dominate size but are rarely needed for
# analysis; the trajectory/reward/exception artifacts are all small. Override via
# PEEK_MAX_OBJECT_BYTES (set 0 to disable skipping and fetch everything).
MAXB = int(os.environ.get("PEEK_MAX_OBJECT_BYTES", str(20 * 1024**2)))
cfg = botocore.config.Config(region_name="auto", connect_timeout=15, read_timeout=120,
                             retries={"max_attempts": 5, "mode": "standard"}, max_pool_connections=64,
                             s3={"addressing_style": "virtual"})  # CW store: virtual-hosted, not path-style
c = boto3.client("s3", endpoint_url=os.environ["AWS_ENDPOINT_URL"], config=cfg)
tcfg = TransferConfig(multipart_threshold=16 * 1024**2, multipart_chunksize=16 * 1024**2,
                      max_concurrency=4, use_threads=True)
objs, skipped, total, skipbytes = [], [], 0, 0
for page in c.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
    for o in page.get("Contents", []):
        rel = o["Key"][len(prefix):]
        if not rel:
            continue
        if MAXB and o["Size"] > MAXB:
            skipped.append((rel, o["Size"])); skipbytes += o["Size"]; continue
        objs.append((o["Key"], rel, o["Size"])); total += o["Size"]
with open(skiplog, "w") as f:
    for rel, sz in skipped:
        f.write(f"{rel}\t{sz}\n")
msg = f"[pull]   fetching {len(objs)} objects ({total/1e9:.2f} GB)"
if MAXB:
    msg += f"; SKIPPING {len(skipped)} files >{MAXB//1024**2}MB ({skipbytes/1e9:.1f} GB, see .r2_skipped.tsv)"
print(msg, flush=True)
fails = []
def fetch(item):
    key, rel, size = item
    out = os.path.join(dest, rel); os.makedirs(os.path.dirname(out), exist_ok=True)
    for _ in range(3):
        try:
            c.download_file(bucket, key, out, Config=tcfg)
            if os.path.getsize(out) == size:
                return True
        except Exception:
            pass
    # A size mismatch is expected for objects a LIVE job is still writing — log, keep going.
    fails.append((rel, size, os.path.getsize(out) if os.path.exists(out) else 0)); return False
ok = 0
with cf.ThreadPoolExecutor(max_workers=12) as ex:
    for i, r in enumerate(ex.map(fetch, objs), 1):
        ok += 1 if r else 0
        if i % 1000 == 0:
            print(f"[pull]   ... {i}/{len(objs)} done ({ok} verified)", flush=True)
with open(faillog, "w") as f:
    for rel, want, got in fails:
        f.write(f"{rel}\twant={want}\tgot={got}\n")
print(f"[pull]   R2 objects verified: {ok}/{len(objs)}  (size-mismatch/failed: {len(fails)})", flush=True)
PYEOF
        NFAIL=$([ -f "$DEST/.r2_failed.tsv" ] && wc -l < "$DEST/.r2_failed.tsv" | tr -d ' ' || echo 0)
        [ "${NFAIL:-0}" -gt 0 ] && echo "[pull]   note: $NFAIL objects failed size-verify (usually live-job churn); see .r2_failed.tsv" >&2
      fi
    fi
    N_TRIALS=$(ls -d "$DEST"/trace_jobs/*/ 2>/dev/null | wc -l | tr -d ' ')
    # COMPLETED = trials with a result.json, counting BOTH downloaded ones and any
    # skipped (large) ones recorded in .r2_skipped.tsv (result.json is usually >20MB → skipped).
    N_DONE_DISK=$(find "$DEST/trace_jobs" -name result.json 2>/dev/null | wc -l | tr -d ' ')
    N_DONE_SKIP=$(grep -c 'result\.json	' "$DEST/.r2_skipped.tsv" 2>/dev/null || echo 0)
    N_DONE=$(( N_DONE_DISK + N_DONE_SKIP ))
    N_SKIP=$([ -f "$DEST/.r2_skipped.tsv" ] && wc -l < "$DEST/.r2_skipped.tsv" | tr -d ' ' || echo 0)
    echo "[pull]   trial dirs=$N_TRIALS  COMPLETED(result.json)=$N_DONE  (large files skipped: $N_SKIP)"

    # 4) torch NCCL flight-recorder dumps (if any), off EVERY pod, not just rank-0. No-op on
    #    a healthy run. See pull_fr_dumps() above / the `frdump` action for a fast, S3-free
    #    variant to run immediately on a suspected hang, before killing the job.
    pull_fr_dumps "$DEST"

    # 5) Provenance manifest.
    cat > "$DEST/MANIFEST.md" <<EOF
# Capture: ${JOBNAME} (${CLUSTER})

- Captured (UTC): $(date -u +%Y-%m-%dT%H:%M:%SZ)
- Job: ${JOBID}
- Rank-0 pod: ${POD}
- trials_dir: ${TJ_LOCAL:-$S3_TJ}  ($([ "$MODE_LOCAL" = 1 ] && echo "node-local (ephemeral)" || echo "REMOTE R2 (durable)"))

## Contents
- trace_jobs/  : ${N_TRIALS} Harbor trial dirs, ${N_DONE} COMPLETED (have result.json + reward).
                 REMOTE jobs: downloaded DIRECT from R2 (${S3_TJ}) via boto3. LOCAL jobs: tar-streamed
                 from the rank-0 pod's ephemeral path — that copy is the only durable one.
- ${N_SKIP} large files (>$(( ${PEEK_MAX_OBJECT_BYTES:-20971520} / 1048576 ))MB) were SKIPPED — listed in .r2_skipped.tsv (path + size).
                 These are mostly the giant result.json (full rollout_details). Re-fetch all with
                 PEEK_MAX_OBJECT_BYTES=0, or a single one with boto3 against its key.
- logs/iris_finelog.log   : complete iris/finelog job log (--no-tail)
- logs/pod_rank*.log      : per-pod container stdout at capture time (rank-0 = harbor coordinator)
- fr_dumps/<pod>/         : torch NCCL flight-recorder dumps (base ${FR_BASE}*), if any were
                 present on that pod. Empty/absent = healthy run (no TORCH_NCCL_DUMP_ON_TIMEOUT
                 fired) or the FR path predates the 2026-07-10 job-scoped convention (see
                 PEEK_FR_BASE). Inspect with torch.distributed.debug.parse_fr_trace_from_file /
                 the FR pickle-to-JSON tooling (torch>=2.6).

## Reproduce
$(basename "$0") ${JOB} pull ${OUTBASE}
EOF

    echo "[pull] DONE — $DEST"
    echo "[pull]   trials: ${N_TRIALS} started / ${N_DONE} completed   total size: $(du -sh "$DEST" 2>/dev/null | cut -f1)"
    ;;
  frdump)
    # Fast, S3-free capture of JUST the torch NCCL flight-recorder dumps off every pod of
    # this job. Intended to be run THE MOMENT a hang is suspected (watchdog-stuck / Aborted
    # in the finelog), before any kill — see the usage header.
    OUTBASE="${3:-$PEEK_OUT}"
    if [ -n "${4:-}" ]; then
      FR_BASE="$4"
    fi
    STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
    DEST="${OUTBASE}/${JOBNAME}_${STAMP}_frdump"
    mkdir -p "$DEST"
    echo "[frdump] dest=$DEST  jobid=$JOBID  cluster=$CLUSTER"
    pull_fr_dumps "$DEST"
    ;;
  *)
    echo "[peek] unknown action '$ACTION' (ls|cat|grep|cp|pull|frdump)" >&2; exit 2;;
esac
