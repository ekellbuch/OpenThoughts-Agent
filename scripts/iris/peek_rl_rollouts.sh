#!/usr/bin/env bash
# peek_rl_rollouts.sh — inspect (or fully capture) the LIVE Harbor rollout artifacts
# (trace_jobs) of a running MarinSkyRL agentic-RL job on cw-us-east-02a, by exec'ing
# into its rank-0 pod.
#
# WHY: agentic RL (terminal_bench / Harbor) writes per-trial rollout artifacts (the literal
# agent trajectory + observations + verifier_output + result.json reward — same layout as
# datagen trials) to terminal_bench_config.trials_dir. With the default (trials_dir: null),
# that resolves to a NODE-LOCAL path on the rank-0 pod (/app/experiments/<run>/trace_jobs),
# which is EPHEMERAL — there is no shared FS/PVC on cw-us-east-02a and pods GC on terminal,
# so the rollouts vanish when the job ends, and iris has no exec/cp. This script is the
# stopgap for jobs ALREADY running with a local trials_dir.
#
# DURABLE FIX (preferred): launch with a remote trials_dir so rollouts persist + are
# inspectable post-hoc — `launch_rl_iris.py --trials-dir auto` (default) writes to
# s3://marin-na/iris/<job>/trace_jobs (R2; the cw pods carry auto-injected R2 creds). Then
# read with `aws s3 ls --endpoint-url <R2>` / fsspec + the harbor trace tooling, no pod exec.
#
# USAGE:
#   peek_rl_rollouts.sh <pod-name-substring>                  # list trial dirs + count (default)
#   peek_rl_rollouts.sh <substr> ls   [glob]                  # list (optional glob, e.g. 'pymethods*')
#   peek_rl_rollouts.sh <substr> cat  <trial-dir>             # dump a trial's json artifacts (the literal rollout)
#   peek_rl_rollouts.sh <substr> grep <pattern>               # list trial json files matching a pattern
#   peek_rl_rollouts.sh <substr> cp   <trial-dir> [dest]      # pull a single trial dir to the launch host
#   peek_rl_rollouts.sh <substr> pull [out-base-dir]          # FULL CAPTURE -> date-stamped subdir:
#                                                             #   complete iris finelog + per-rank pod logs
#                                                             #   + tar of ALL trace_jobs + MANIFEST.md
#
# NOTE: <substr> matches the POD name (iris-benjaminfeuer-<name>-<rank>-<hash>-0), which can
# differ from the iris job_id display name. With no match the script lists candidate rl pods.
#
# ENV: PEEK_KUBECONFIG (default ~/.kube/coreweave-iris-gpu), NS (default iris),
#      CONTAINER (default task), PEEK_CLUSTER (default cw-us-east-02a),
#      IRIS_BIN (default the marin venv iris), PEEK_OUT (default ~/Documents/experiments/traces)
set -euo pipefail

JOB="${1:-}"
ACTION="${2:-ls}"
# Force the CoreWeave kubeconfig. Do NOT honor an inherited $KUBECONFIG — the login shell's
# default is ~/.kube/lambdaconfig (wrong cluster → 'no pods'); override only via PEEK_KUBECONFIG.
export KUBECONFIG="${PEEK_KUBECONFIG:-$HOME/.kube/coreweave-iris-gpu}"
NS="${NS:-iris}"
CONTAINER="${CONTAINER:-task}"
CLUSTER="${PEEK_CLUSTER:-cw-us-east-02a}"
IRIS_BIN="${IRIS_BIN:-/Users/benjaminfeuer/Documents/marin/.venv/bin/iris}"
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
  echo "[peek] (job terminal? then rollouts are GC'd — only a remote trials_dir survives.) Candidate rl pods:" >&2
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

# Discover the (local) trials_dir. Empty ⇒ the run likely uses a REMOTE trials_dir.
# Fatal for the read actions; for `pull` we still capture logs and just skip the local tar.
TJ=$(kexec 'ls -d /app/experiments/*/trace_jobs 2>/dev/null | head -1' || true)
TJ="$(printf '%s' "$TJ" | tr -d '\r')"
if [ -z "$TJ" ]; then
  echo "[peek] no /app/experiments/*/trace_jobs in pod — this run probably writes to a REMOTE trials_dir." >&2
  if [ "$ACTION" != "pull" ]; then
    echo "[peek] check: aws s3 ls --endpoint-url <R2> s3://marin-na/iris/<job>/trace_jobs/  (or the configured trials_dir)" >&2
    exit 3
  fi
  echo "[peek] pull: will still capture logs; skipping local trace_jobs tar." >&2
else
  echo "[peek] trials_dir=$TJ"
fi

case "$ACTION" in
  ls)
    GLOB="${3:-*}"
    kexec "ls -d $TJ/$GLOB/ 2>/dev/null | sed 's#$TJ/##'" || true
    echo "[peek] total trial dirs: $(kexec "ls -d $TJ/*/ 2>/dev/null | wc -l" | tr -d ' ')"
    ;;
  cat)
    TR="${3:?cat needs <trial-dir>}"
    kexec "find '$TJ/$TR' -maxdepth 2 -name '*.json' -print -exec sh -c 'echo; cat \"\$1\"; echo' _ {} \; 2>/dev/null"
    ;;
  grep)
    PAT="${3:?grep needs <pattern>}"
    kexec "grep -rls --include='*.json' -e '$PAT' '$TJ' 2>/dev/null | sed 's#$TJ/##' | head -40" || true
    ;;
  cp)
    TR="${3:?cp needs <trial-dir>}"; DEST="${4:-./$TR}"
    kubectl cp -c "$CONTAINER" "$NS/$POD:$TJ/$TR" "$DEST"
    echo "[peek] copied -> $DEST"
    ;;
  pull)
    # FULL CAPTURE into a fresh date-stamped subdir: complete iris finelog + per-rank pod
    # logs + tar of ALL trace_jobs + a provenance MANIFEST. Network ops are wrapped so a
    # single failure still leaves a usable (partial) capture + manifest.
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

    # 3) Tar-stream ALL trace_jobs off the rank-0 pod (only if a local trials_dir exists).
    N_TRIALS=0
    if [ -n "$TJ" ]; then
      echo "[pull] tar-streaming trace_jobs ..."
      PARENT="$(dirname "$TJ")"; BASE="$(basename "$TJ")"
      { kubectl exec -n "$NS" "$POD" -c "$CONTAINER" -- tar cf - -C "$PARENT" "$BASE" 2>/dev/null | tar xf - -C "$DEST/"; } \
        || echo "[pull] WARN: trace_jobs tar returned nonzero (capture may be partial)" >&2
      N_TRIALS=$(ls -d "$DEST"/trace_jobs/*/ 2>/dev/null | wc -l | tr -d ' ')
    else
      rmdir "$DEST/trace_jobs" 2>/dev/null || true
      echo "[pull] (no local trace_jobs — remote trials_dir; logs captured only)"
    fi

    # 4) Provenance manifest.
    cat > "$DEST/MANIFEST.md" <<EOF
# Capture: ${JOBNAME} (${CLUSTER})

- Captured (UTC): $(date -u +%Y-%m-%dT%H:%M:%SZ)
- Job: ${JOBID}
- Rank-0 pod: ${POD}
- Local trials_dir on pod: ${TJ:-<none / remote trials_dir>}

## Contents
- trace_jobs/  : all ${N_TRIALS} Harbor trial dirs, tar-streamed live from the rank-0 pod's
                 EPHEMERAL local path (${TJ:-n/a}). If this job ran with trials_dir: null,
                 this capture is the only durable copy.
- logs/iris_finelog.log   : complete iris/finelog job log (--no-tail)
- logs/pod_rank*.log      : per-pod container stdout at capture time (rank-0 = harbor coordinator)

## Reproduce
$(basename "$0") ${JOB} pull ${OUTBASE}
EOF

    echo "[pull] DONE — $DEST"
    echo "[pull]   trials: ${N_TRIALS}   total size: $(du -sh "$DEST" 2>/dev/null | cut -f1)"
    ;;
  *)
    echo "[peek] unknown action '$ACTION' (ls|cat|grep|cp|pull)" >&2; exit 2;;
esac
