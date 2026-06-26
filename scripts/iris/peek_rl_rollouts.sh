#!/usr/bin/env bash
# peek_rl_rollouts.sh — inspect (or fully capture) the Harbor rollout artifacts (trace_jobs) of a
# running MarinSkyRL agentic-RL job on cw-us-east-02a, by reaching its rank-0 pod.
#
# WHY: agentic RL (terminal_bench / Harbor) writes per-trial rollout artifacts (the literal agent
# trajectory + prompts/responses + verifier_output + result.json reward) to
# terminal_bench_config.trials_dir. Our jobs launch with a REMOTE R2 trials_dir
# (s3://marin-na/iris/<job>/trace_jobs via launch_rl_iris.py --trials-dir auto) — DURABLE (survives
# pod GC), unlike the old node-local ephemeral path (trials_dir: null). The rank-0 pod carries the
# cluster-injected R2 creds + AWS_ENDPOINT_URL (iris-task-env Secret), but the LAUNCH HOST (Mac) does
# NOT have working marin-na R2 creds. So this script does all R2 ops INSIDE the pod via boto3 (the
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
#
# NOTE: <substr> matches the POD name (iris-benjaminfeuer-<name>-<rank>-<hash>-0), which can differ
# from the iris job_id display name. With no match the script lists candidate rl pods.
#
# ENV: PEEK_KUBECONFIG (default ~/.kube/coreweave-iris-gpu), NS (default iris), CONTAINER (default task),
#      PEEK_CLUSTER (default cw-us-east-02a), IRIS_BIN (default the otagent cw-capable iris),
#      PEEK_OUT (default ~/Documents/experiments/traces),
#      PEEK_TRIALS_S3 (override the remote trials_dir; default s3://marin-na/iris/<jobname>/trace_jobs)
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
POD=$(kubectl get pods -n "$NS" -o name 2>/dev/null | grep -E "iris-.*${JOB}.*-0-[0-9a-f]+-0$" | head -1 || true)
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
JOBNAME=$(printf '%s' "$POD" | sed -E 's/^iris-[a-z0-9]+-(.+)-[0-9]+-[0-9a-f]+-0$/\1/')
JOBID="/${USER_FROM_POD}/${JOBNAME}"

kexec() { kubectl exec -n "$NS" "$POD" -c "$CONTAINER" -- bash -lc "$1"; }

# --- trials_dir discovery: prefer a node-local path (legacy trials_dir: null); else REMOTE R2. ---
TJ_LOCAL=$(kexec 'ls -d /app/experiments/*/trace_jobs 2>/dev/null | head -1' 2>/dev/null | tr -d '\r' || true)
S3_TJ="${PEEK_TRIALS_S3:-s3://marin-na/iris/${JOBNAME}/trace_jobs}"
if [ -n "$TJ_LOCAL" ]; then
  MODE_LOCAL=1
  echo "[peek] LOCAL trials_dir=$TJ_LOCAL"
else
  MODE_LOCAL=0
  echo "[peek] REMOTE trials_dir=$S3_TJ  (R2 via rank-0 pod boto3; Mac lacks marin-na R2 creds)"
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
s3url = sys.argv[1]
mode  = sys.argv[2] if len(sys.argv) > 2 else "count"
arg   = sys.argv[3] if len(sys.argv) > 3 else ""
assert s3url.startswith("s3://"), s3url
BUCKET, _, PREFIX = s3url[5:].partition("/")
PREFIX = PREFIX.rstrip("/") + "/"
c = boto3.client("s3", endpoint_url=os.environ["AWS_ENDPOINT_URL"])
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
s3url = sys.argv[1]; dest = sys.argv[3]
BUCKET, _, PREFIX = s3url[5:].partition("/"); PREFIX = PREFIX.rstrip("/") + "/"
c = boto3.client("s3", endpoint_url=os.environ["AWS_ENDPOINT_URL"])
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
    for p in $(kubectl get pods -n "$NS" -o name 2>/dev/null | grep -E "iris-.*${JOB}.*-[0-9]+-[0-9a-f]+-0$" | sed 's#pod/##'); do
      rank=$(printf '%s' "$p" | sed -E 's/.*-([0-9]+)-[0-9a-f]+-0$/\1/')
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
      echo "[pull] syncing trace_jobs from R2 ($S3_TJ) via rank-0 pod ..."
      POD_TMP="/tmp/peek_tj_capture_$$"
      kubectl exec -n "$NS" "$POD" -c "$CONTAINER" -- rm -rf "$POD_TMP" 2>/dev/null || true
      NOBJ=$(r2_op download "$POD_TMP" 2>/dev/null | tail -1 | tr -dc '0-9' || true)
      { kubectl exec -n "$NS" "$POD" -c "$CONTAINER" -- tar cf - -C "$POD_TMP" . 2>/dev/null | tar xf - -C "$DEST/trace_jobs/"; } \
        || echo "[pull] WARN: trace_jobs tar-stream returned nonzero (capture may be partial)" >&2
      kubectl exec -n "$NS" "$POD" -c "$CONTAINER" -- rm -rf "$POD_TMP" 2>/dev/null || true
      echo "[pull]   R2 objects downloaded: ${NOBJ:-0}"
    fi
    N_TRIALS=$(ls -d "$DEST"/trace_jobs/*/ 2>/dev/null | wc -l | tr -d ' ')
    N_DONE=$(find "$DEST/trace_jobs" -name result.json 2>/dev/null | wc -l | tr -d ' ')
    echo "[pull]   trial dirs=$N_TRIALS  COMPLETED(result.json)=$N_DONE"

    # 4) Provenance manifest.
    cat > "$DEST/MANIFEST.md" <<EOF
# Capture: ${JOBNAME} (${CLUSTER})

- Captured (UTC): $(date -u +%Y-%m-%dT%H:%M:%SZ)
- Job: ${JOBID}
- Rank-0 pod: ${POD}
- trials_dir: ${TJ_LOCAL:-$S3_TJ}  ($([ "$MODE_LOCAL" = 1 ] && echo "node-local (ephemeral)" || echo "REMOTE R2 (durable)"))

## Contents
- trace_jobs/  : ${N_TRIALS} Harbor trial dirs, ${N_DONE} COMPLETED (have result.json + reward).
                 REMOTE jobs: synced from R2 (${S3_TJ}). LOCAL jobs: tar-streamed from the rank-0 pod's
                 ephemeral path — that copy is the only durable one.
- logs/iris_finelog.log   : complete iris/finelog job log (--no-tail)
- logs/pod_rank*.log      : per-pod container stdout at capture time (rank-0 = harbor coordinator)

## Reproduce
$(basename "$0") ${JOB} pull ${OUTBASE}
EOF

    echo "[pull] DONE — $DEST"
    echo "[pull]   trials: ${N_TRIALS} started / ${N_DONE} completed   total size: $(du -sh "$DEST" 2>/dev/null | cut -f1)"
    ;;
  *)
    echo "[peek] unknown action '$ACTION' (ls|cat|grep|cp|pull)" >&2; exit 2;;
esac
