#!/usr/bin/env bash
# peek_rl_rollouts.sh — inspect the LIVE Harbor rollout artifacts (trace_jobs) of a
# running MarinSkyRL agentic-RL job on cw-us-east-02a, by exec'ing into its rank-0 pod.
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
#   peek_rl_rollouts.sh <substr> cp   <trial-dir> [dest]      # pull a trial dir to the launch host
#
# NOTE: <substr> matches the POD name (iris-benjaminfeuer-<name>-<rank>-<hash>-0), which can
# differ from the iris job_id display name. With no match the script lists candidate rl pods.
#
# ENV: PEEK_KUBECONFIG (default ~/.kube/coreweave-iris-gpu), NS (default iris), CONTAINER (default task)
set -euo pipefail

JOB="${1:-}"
ACTION="${2:-ls}"
# Force the CoreWeave kubeconfig. Do NOT honor an inherited $KUBECONFIG — the login shell's
# default is ~/.kube/lambdaconfig (wrong cluster → 'no pods'); override only via PEEK_KUBECONFIG.
export KUBECONFIG="${PEEK_KUBECONFIG:-$HOME/.kube/coreweave-iris-gpu}"
NS="${NS:-iris}"
CONTAINER="${CONTAINER:-task}"

if [ -z "$JOB" ]; then
  echo "usage: peek_rl_rollouts.sh <pod-name-substring> [ls|cat|grep|cp] [args]" >&2
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

kexec() { kubectl exec -n "$NS" "$POD" -c "$CONTAINER" -- bash -lc "$1"; }

# Discover the (local) trials_dir. If empty, the run likely uses a REMOTE trials_dir — say so.
TJ=$(kexec 'ls -d /app/experiments/*/trace_jobs 2>/dev/null | head -1' || true)
TJ="$(printf '%s' "$TJ" | tr -d '\r')"
if [ -z "$TJ" ]; then
  echo "[peek] no /app/experiments/*/trace_jobs in pod — this run probably writes to a REMOTE trials_dir." >&2
  echo "[peek] check: aws s3 ls --endpoint-url <R2> s3://marin-na/iris/<job>/trace_jobs/  (or the configured trials_dir)" >&2
  exit 3
fi
echo "[peek] trials_dir=$TJ"

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
  *)
    echo "[peek] unknown action '$ACTION' (ls|cat|grep|cp)" >&2; exit 2;;
esac
