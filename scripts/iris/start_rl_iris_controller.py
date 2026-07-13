#!/usr/bin/env python3
"""Bootstrap a multi-node MarinSkyRL RL job on an iris GPU slice.

This is the RL analog of ``scripts/vllm/start_vllm_iris_controller.py`` (which
serves vLLM across a multi-host iris TPU slice). The crucial shared fact: iris
gang-schedules a multi-node job as N coscheduled tasks — one task per whole
node — and runs THIS SAME entrypoint on every node, injecting ``IRIS_TASK_ID``
/ ``IRIS_NUM_TASKS`` per task. SkyRL/MarinSkyRL (skyrl-train) is Ray-native: it
wants ONE cross-node Ray cluster and a single training driver that fans
policy/ref/inference actors across the Ray nodes.

So this script bootstraps that one cluster, then runs the driver on rank 0:

- **rank 0 (head):** ``ray start --head``; publish the head IP to a shared
  rendezvous file; wait until ``ray.nodes()`` shows all ``IRIS_NUM_TASKS``
  nodes joined; then ``exec`` the MarinSkyRL training command (``python -m
  <entrypoint> <hydra args>``) with ``RAY_ADDRESS`` pointing at the head, so
  skyrl-train's ``initialize_ray`` (which calls bare ``ray.init()``) attaches
  to the existing multi-node cluster instead of starting a fresh local one.
- **ranks 1..N-1 (workers):** read the head IP from the rendezvous, run
  ``ray start --address=<head_ip>:<port>``, verify they joined, then BLOCK
  until the head finishes (signalled via the rendezvous ``done`` marker) or
  SIGTERM. They contribute their 8 H100s to the Ray cluster; the driver on
  rank 0 schedules engine/policy workers onto them. They do NOT run the
  training driver.

Head-IP discovery
-----------------
iris injects ``IRIS_ADVERTISE_HOST`` (the task's routable IP under
``host_network: true`` — required for NCCL/IB on the CoreWeave slice) into
every task, so rank 0 uses it directly as the Ray head IP. iris does NOT inject
a peer/host list into the task env, so rank 0 publishes the head IP to a small
rendezvous file on a shared object store the launcher passes in
(``--rendezvous-dir`` / ``OT_AGENT_IRIS_RENDEZVOUS_DIR``). Ranks 1..N poll for
it. The rendezvous URI may be ``gs://``, ``s3://`` (CoreWeave R2), or a shared
local/NFS path — it is opened via ``fsspec`` so the storage backend is whatever
the URI scheme resolves to (CoreWeave uses R2/S3, not GCS).
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time

RENDEZVOUS_FILENAME = "ray_head.json"
DONE_FILENAME = "ray_head.done"


def _ray_bin() -> str:
    """Resolve the ``ray`` CLI from the SAME venv as the running interpreter.

    The launcher runs this controller via the RL venv python by absolute path
    (e.g. /opt/openthoughts/envs/rl/bin/python), but iris's uv-sync setup
    activates a DIFFERENT venv (/app/.venv) and leaves only that on $PATH — which
    has no `ray`. So a bare `ray` command resolves to nothing (FileNotFoundError).
    Use the `ray` binary that sits next to this interpreter; fall back to PATH.
    """
    import shutil

    candidate = os.path.join(os.path.dirname(sys.executable), "ray")
    if os.path.exists(candidate):
        return candidate
    found = shutil.which("ray")
    return found or "ray"

# Cold GPU nodes (image pull + setup) can take several minutes to reach the
# rendezvous, so these are generous.
DEFAULT_RENDEZVOUS_TIMEOUT = 1800
DEFAULT_CLUSTER_JOIN_TIMEOUT = 1800
POLL_INTERVAL = 5
# Tolerates clock skew between nodes and the time rank-0 needs to start Ray.
RENDEZVOUS_FRESHNESS_SLACK = 60
# Bound the rank-0 rendezvous PutObject: an unbounded fsspec/s3fs put can wedge the
# head forever (no connect/read timeout), invisibly, until every worker times out at
# the 1800s rendezvous deadline. Each attempt runs under a hard futures timeout with
# bounded retries + backoff; final failure raises LOUD so the gang fails fast.
RENDEZVOUS_WRITE_ATTEMPTS = 5
RENDEZVOUS_WRITE_TIMEOUT = 30  # seconds per attempt
# Bound `ray start --head` so a hung Ray CLI fails loud (TimeoutExpired) instead of
# silently stalling bring-up until the worker rendezvous deadline.
RAY_START_HEAD_TIMEOUT = 300  # seconds


def _log(msg: str) -> None:
    print(f"[start_rl_iris_controller] {msg}", flush=True)


def stage_train_data(train_data_json: str) -> None:
    """Extract the HF task dataset(s) to this NODE's local task dir on EVERY node.

    WHY THIS EXISTS (the data-starvation bug, 2026-06-26): the agentic
    terminal_bench task dataset (e.g. ``DCAgent/exp_rpt_pymethods2test-large``) is
    an HF *parquet* repo that must be extracted into the on-disk
    ``$DCFT/tasks/<repo>/<instance>/{instruction.md,task.toml,...}`` layout the
    Harbor rollout reads. ``hpc.rl_launch_utils.resolve_rl_train_data`` does that
    extraction, but on the iris/CoreWeave path it runs only inside rank-0's
    ``run_rl.py`` driver, writing to ``$DCFT=/opt/openthoughts/tasks`` on the HEAD
    pod's NODE-LOCAL filesystem. CoreWeave task pods do NOT share a filesystem (no
    SLURM-style GPFS ``$SCRATCH``), so the Ray-scheduled rollout workers on ranks
    1..N-1 saw an empty tasks dir and every rollout died with
    ``FileNotFoundError: .../task.toml`` -> reward always 0 (doomed, data-starved).

    Fix: the controller runs on EVERY node, so stage here (before Ray bootstrap)
    using the SAME extraction routine. CoreWeave nodes have egress, so each pod
    fetches+extracts to the identical node-local path; the path strings rank-0's
    dataset object ships to the workers then resolve on every pod. Idempotent
    (``on_exist=skip`` + the stat short-circuit in ``_fix_task_permissions``), so a
    re-run / rank-0's later run_rl re-resolve is a cheap no-op.
    """
    import json as _json

    try:
        train_data = _json.loads(train_data_json)
    except (ValueError, TypeError):
        train_data = [train_data_json] if train_data_json else []
    if not train_data:
        return

    # Reuse the exact, SLURM-proven staging logic. PYTHONPATH already includes
    # /app (set by the launcher bootstrap), so hpc is importable.
    from hpc.rl_launch_utils import resolve_rl_train_data

    _log(f"Staging train_data on this node (rank {_rank()}/{_num_tasks()}): {train_data}")
    # The pod env carries HF_HUB_OFFLINE=1 / TRANSFORMERS_OFFLINE=1 (config extra_env,
    # for the training ranks so they load the pre-staged model from the warm node-local
    # cache). But this HF *parquet* dataset is NOT pre-cached — it must be fetched from
    # the Hub here, so force-clear both for the extraction only (symmetric to
    # stage_model; the ranks' env is untouched). Without this the snapshot_download
    # inside resolve_rl_train_data dies OfflineModeIsEnabled and kills the whole gang
    # in the controller pre-Ray phase.
    saved = {k: os.environ.get(k) for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")}
    for k in saved:
        os.environ.pop(k, None)
    try:
        resolved = resolve_rl_train_data(train_data, on_exist="skip", verbose=True)
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v
    _log(f"train_data staged to node-local paths: {resolved}")


def stage_model(model_path: str) -> None:
    """Pre-download the policy model into this NODE's local HF cache on EVERY node.

    WHY THIS EXISTS (the init-straggle bug, 2026-07-10): with the model resolved
    ONLINE by HF repo-id (``trainer.policy.model.path=<org/name>``, no
    HF_HUB_OFFLINE), each of the N*8 FSDP ranks independently pulls the ~160 GB
    of weights from HF Hub inside ``FSDPPolicyWorkerBase.init_model``. On a
    network-saturated cluster the per-rank variance blew out — slow ranks were
    still in ``from_pretrained`` at 68 min while the lazy policy-PG NCCL comm-init
    (the first fsdp2_load_full_state_dict broadcast) fired a c10d
    ``store->get('0')`` on the fast ranks that timed out at the 20-min store
    barrier, SIGKILLing the whole gang before global_step 1.

    Fix (symmetric to ``stage_train_data``): the controller runs on EVERY node
    before Ray bootstrap, so pre-download the weights ONCE PER NODE here (16 pulls,
    not 128) into the node-local HF cache. The download runs in the controller's
    pre-Ray phase where no collective barrier is active, so a slow pull can no
    longer race the init barrier. With ``HF_HUB_OFFLINE=1`` set for the training
    ranks (config extra_env), all 8 ranks on a node then read the pre-populated
    node-local cache — uniform + fast, zero HF contention at init. Idempotent:
    ``snapshot_download`` skips already-complete cached files, so a re-brought-up
    gang (``--max-retries``) re-runs it as a cheap cache-validate.

    The pod env carries ``HF_HUB_OFFLINE=1`` / ``TRANSFORMERS_OFFLINE=1`` for the
    ranks; this download MUST reach the Hub, so it runs ``snapshot_download`` in a
    SUBPROCESS whose env has both flags stripped (an in-process env-pop is too late —
    huggingface_hub caches HF_HUB_OFFLINE into a module constant at import; see the
    body). The ranks' env is untouched.
    """
    if not model_path or model_path.startswith(("s3://", "gs://", "gcs://")) or os.path.isdir(model_path):
        # Nothing to pre-download: empty, an object-store URI (handled elsewhere),
        # or already a local directory the ranks read directly.
        _log(f"stage_model: skip (model_path={model_path!r} is empty/local/cloud)")
        return

    # Weights + config + tokenizer + any trust_remote_code modeling files. Mirrors
    # mirror_hf_to_gcs.INCLUDE_PATTERNS so from_pretrained resolves fully offline.
    allow_patterns = ["*.safetensors", "*.json", "*.txt", "*.model", "*.py"]

    # Download in a SUBPROCESS with the offline flags stripped from ITS env. An
    # in-process os.environ.pop does NOT work here: huggingface_hub snapshots
    # HF_HUB_OFFLINE into a module CONSTANT at IMPORT time, and the controller has
    # already imported huggingface_hub, so clearing the env var afterward leaves the
    # cached constant True -> snapshot_download still raises OfflineModeIsEnabled.
    # A fresh child process re-reads the (cleaned) env at its own import. (This is
    # exactly why stage_train_data — which shells out to a subprocess — works with the
    # same env-pop, while stage_model, running in-process, did not: the 80b-next-cp1
    # gang died here on 2026-07-10.) The child inherits HF_HOME/HF_HUB_CACHE, so it
    # populates the SAME node-local cache the offline ranks then read.
    child_env = {k: v for k, v in os.environ.items()
                 if k not in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")}
    child_env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"  # keep captured stderr bounded
    code = (
        "import sys\n"
        "from huggingface_hub import snapshot_download\n"
        "p = snapshot_download(sys.argv[1], allow_patterns=sys.argv[2].split(','))\n"
        "print('PRESTAGE_LOCAL_DIR=' + p)\n"
    )
    _log(f"Pre-staging model on this node (rank {_rank()}/{_num_tasks()}): {model_path}")
    last_err = ""
    for attempt in range(1, 7):
        proc = subprocess.run(
            [sys.executable, "-c", code, model_path, ",".join(allow_patterns)],
            env=child_env, capture_output=True, text=True,
        )
        if proc.returncode == 0:
            local_dir = ""
            for line in proc.stdout.splitlines():
                if line.startswith("PRESTAGE_LOCAL_DIR="):
                    local_dir = line.split("=", 1)[1]
            _log(f"model pre-staged to node-local HF cache: {local_dir}")
            return
        last_err = (proc.stderr or proc.stdout or "")[-800:]
        _log(f"model prestage attempt {attempt}/6 failed (rc={proc.returncode}): {last_err}")
        time.sleep(min(30, 2 ** attempt))
    raise RuntimeError(
        f"model prestage failed after 6 attempts for {model_path}: {last_err}"
    )


def _rank() -> int:
    # IRIS_TASK_ID is the full task path (e.g. "/user/job/0"); on retried tasks
    # iris appends a ":N" retry suffix. The rank is the trailing path segment
    # with any retry suffix stripped.
    return int(os.environ.get("IRIS_TASK_ID", "0").rsplit("/", 1)[-1].split(":", 1)[0])


def _num_tasks() -> int:
    return int(os.environ.get("IRIS_NUM_TASKS", "1"))


def _own_ip() -> str:
    """Routable IP of this node.

    Prefers iris's ``IRIS_ADVERTISE_HOST`` (the routable IP iris computed for
    this task under ``host_network: true``); falls back to a UDP-socket probe.
    """
    advertised = os.environ.get("IRIS_ADVERTISE_HOST")
    if advertised and advertised != "127.0.0.1":
        return advertised
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    finally:
        s.close()


# ---------------------------------------------------------------------------
# Rendezvous — head publishes its IP, workers poll for it. Backend-agnostic via
# fsspec so the URI scheme (gs://, s3://, file://, plain path) selects storage.
# ---------------------------------------------------------------------------


def _fs_and_path(uri: str):
    """Return (fsspec filesystem, path) for ``uri``. Uses default credential
    discovery for the scheme (workload identity / instance creds / env keys).

    For ``s3://`` the CoreWeave object store (marin-us-east-02a) REQUIRES
    virtual-hosted (hostname-based) addressing — path-style PutObject/
    CreateMultipartUpload get rejected with ``PathStyleRequestNotAllowed``
    (observed on the 80b-next-cp1 rendezvous + trace uploads, 2026-07-09). s3fs
    otherwise defaults to path-style for a custom endpoint_url, so pin the
    botocore ``addressing_style`` explicitly. The ``OT_AGENT_S3_ADDRESSING_STYLE``
    env (default ``virtual``) allows an override for a path-only store (e.g. GCS)."""
    import fsspec

    storage_options = None
    if uri.startswith("s3://") or uri.startswith("s3a://"):
        style = os.environ.get("OT_AGENT_S3_ADDRESSING_STYLE", "virtual")
        storage_options = {"config_kwargs": {"s3": {"addressing_style": style}}}
    fs, _, paths = fsspec.get_fs_token_paths(uri, storage_options=storage_options)
    return fs, paths[0]


def _pin_boto3_s3_addressing_style() -> None:
    """Pin virtual-hosted (hostname-based) S3 addressing for the boto3 code path,
    cluster-wide, via an AWS config file + ``AWS_CONFIG_FILE``.

    WHY (boto3 companion to the fsspec/s3fs ``FSSPEC_S3_CONFIG_KWARGS`` + the
    ``_fs_and_path`` rendezvous pin): the CoreWeave object store
    (marin-us-east-02a / R2) REJECTS path-style S3 requests with
    ``PathStyleRequestNotAllowed`` and requires virtual-hosted addressing.
    Ray's object-spill IO workers (``ray._private.external_storage``
    ``ExternalStorageSmartOpenImpl``) call a bare ``boto3.resource("s3")`` — with
    NO botocore ``Config`` — in child processes WE never construct, so we cannot
    pass ``addressing_style`` in code. botocore also has NO environment variable
    for ``addressing_style`` (it is resolved ONLY from the shared-config file's
    ``s3`` section; see botocore ``configprovider.py``). So we WRITE a minimal AWS
    config file that sets ``s3.addressing_style`` and point ``AWS_CONFIG_FILE`` at
    it. Because this runs before ``ray start`` (head + every worker) and mutates
    ``os.environ``, every Ray subprocess / raylet IO worker / the training driver
    inherits it and its ``boto3.resource("s3")`` picks up virtual addressing.

    R2 credentials/region/endpoint come from env vars (``AWS_*_KEY`` /
    ``AWS_REGION`` / ``AWS_ENDPOINT_URL``) INDEPENDENTLY of the config file, so a
    minimal ``[default]`` profile with only the ``s3`` section does not disturb
    credential discovery. Env-overridable via ``OT_AGENT_S3_ADDRESSING_STYLE``
    (default ``virtual``; set ``path``/``auto`` for a path-style store). If an
    operator already exported ``AWS_CONFIG_FILE`` (a real profile/creds file), we
    RESPECT it and skip rather than clobber."""
    style = os.environ.get("OT_AGENT_S3_ADDRESSING_STYLE", "virtual")
    existing = os.environ.get("AWS_CONFIG_FILE")
    if existing and os.path.exists(existing):
        _log(
            f"AWS_CONFIG_FILE already set ({existing}); NOT overriding for S3 "
            f"addressing_style. If Ray spill hits PathStyleRequestNotAllowed, add "
            f"'s3 =\\n    addressing_style = {style}' to that file's [default] profile."
        )
        return
    cfg_path = os.path.join(tempfile.gettempdir(), "ot_agent_aws_config")
    try:
        with open(cfg_path, "w") as f:
            f.write(f"[default]\ns3 =\n    addressing_style = {style}\n")
        os.environ["AWS_CONFIG_FILE"] = cfg_path
        _log(
            f"S3 addressing_style={style} pinned via AWS_CONFIG_FILE={cfg_path} "
            f"(boto3/Ray-object-spill virtual-hosted addressing for CoreWeave R2)."
        )
    except OSError as exc:  # noqa: BLE001 - best-effort; falls back to boto3 default
        _log(f"WARNING: could not write AWS config for S3 addressing_style ({exc}); "
             f"Ray object-spill may hit PathStyleRequestNotAllowed on CoreWeave R2.")


def ensure_fr_dump_dir() -> None:
    """`mkdir -p` the NCCL flight-recorder dump directory on THIS node before torch init.

    WHY THIS EXISTS (2026-07-11, /benjaminfeuer/80b-next-cp1 gs1-optimizer NCCL hang):
    the 80B config enables the flight recorder (``TORCH_NCCL_DUMP_ON_TIMEOUT=1``) and
    points the per-rank dump path at a JOB-SCOPED subdir via
    ``TORCH_NCCL_DEBUG_INFO_TEMP_FILE`` / ``TORCH_FR_DUMP_TEMP_FILE`` (e.g.
    ``/tmp/fr_dumps/<job>/nccl_fr_rank`` — torch appends the global rank). But NOTHING
    created the parent ``/tmp/fr_dumps/<job>/``: torch's ``DebugInfoWriter`` is a plain
    local ``std::ofstream`` that does NOT ``mkdir -p`` its parent, so on the collective
    timeout every rank's dump SILENTLY failed to write (confirmed: the abort logged
    ``[F711] … after attempting to dump debug info`` yet a filesystem-wide find for
    ``nccl_fr_rank*`` on all 16 pods returned nothing) — we lost the one artifact that
    would have fingerprinted the AWOL rank.

    This controller runs on EVERY node in the pre-Ray phase (before any FSDP/vLLM Ray
    actor — and thus before any torch/NCCL init — on that node), and the FR dump path is
    a NODE-LOCAL ``/tmp`` path shared by every rank/actor in this pod, so creating the
    dir here once per node guarantees it exists before the first torch init. The dir is
    DERIVED from the cvar (``dirname`` of the dump-path prefix), so it is robust to the
    path changing per-job — no job name is hardcoded. Best-effort: a failure here must
    never block bring-up (the FR dump is instrumentation, not correctness)."""
    # TORCH_FR_DUMP_TEMP_FILE is the newer alias torch's fallback chain checks first;
    # TORCH_NCCL_DEBUG_INFO_TEMP_FILE is the canonical name. Honor whichever is set
    # (both to the same value in our configs). The value is a per-rank FILENAME PREFIX
    # (torch appends the global rank), so the dir to create is its dirname.
    dump_prefix = (
        os.environ.get("TORCH_FR_DUMP_TEMP_FILE")
        or os.environ.get("TORCH_NCCL_DEBUG_INFO_TEMP_FILE")
    )
    if not dump_prefix:
        _log("FR dump dir: no TORCH_(FR_DUMP|NCCL_DEBUG_INFO)_TEMP_FILE set; nothing to create.")
        return
    dump_dir = os.path.dirname(dump_prefix)
    if not dump_dir:
        # Bare filename with no directory component (e.g. the generic "/tmp/nccl_fr_rank"
        # has dirname "/tmp", which exists; a relative bare name has no dir to make).
        return
    try:
        os.makedirs(dump_dir, exist_ok=True)
        _log(f"FR dump dir ensured (mkdir -p): {dump_dir} (from cvar prefix {dump_prefix!r})")
    except OSError as exc:  # noqa: BLE001 - best-effort; never block bring-up on instrumentation
        _log(f"WARNING: could not create FR dump dir {dump_dir} ({exc}); "
             f"NCCL flight-recorder dumps may fail to write on a collective timeout.")


def _rendezvous_uri(rendezvous_dir: str) -> str:
    return f"{rendezvous_dir.rstrip('/')}/{RENDEZVOUS_FILENAME}"


def _done_uri(rendezvous_dir: str) -> str:
    return f"{rendezvous_dir.rstrip('/')}/{DONE_FILENAME}"


def _write_rendezvous_once(fs, path: str, payload: dict) -> None:
    """Single blocking PutObject of the rendezvous payload (the caller runs this
    under a bounded futures timeout so a stalled put cannot wedge the head)."""
    with fs.open(path, "w") as f:
        json.dump(payload, f)


def write_rendezvous(rendezvous_dir: str, head_ip: str, ray_port: int) -> None:
    uri = _rendezvous_uri(rendezvous_dir)
    payload = {
        "head_ip": head_ip,
        "port": ray_port,
        "num_tasks": _num_tasks(),
        "written_at": time.time(),
    }
    fs, path = _fs_and_path(uri)
    # Bound the object-store PutObject with a hard per-attempt timeout via a DAEMON
    # thread + join(timeout) + bounded retries/backoff. WHY: an unbounded s3fs/fsspec put
    # here deterministically hangs the head (no connect/read timeout on the underlying
    # client), and because it is invisible, every worker only discovers it at the 1800s
    # rendezvous deadline.
    # WHY a daemon thread and NOT a ThreadPoolExecutor: an executor's worker thread is
    # NON-daemon, so even after we abandon a wedged put, Python's atexit `_python_exit`
    # JOINS that thread forever at interpreter shutdown → the process can never exit →
    # ZOMBIE (this bit CP4 v3). A daemon thread is NOT joined by `_python_exit`, so a
    # wedged write can never block process exit; we simply abandon it after the timeout.
    last_exc: BaseException | None = None
    for attempt in range(1, RENDEZVOUS_WRITE_ATTEMPTS + 1):
        t0 = time.time()
        _log(
            f"Writing rendezvous {uri} (attempt {attempt}/{RENDEZVOUS_WRITE_ATTEMPTS}, "
            f"per-attempt timeout {RENDEZVOUS_WRITE_TIMEOUT}s)..."
        )
        result_box: dict = {}

        def _target() -> None:
            try:
                _write_rendezvous_once(fs, path, payload)
                result_box["ok"] = True
            except BaseException as exc:  # noqa: BLE001 - surface to the joiner
                result_box["exc"] = exc

        writer = threading.Thread(
            target=_target, name=f"rendezvous-write-{attempt}", daemon=True
        )
        writer.start()
        writer.join(timeout=RENDEZVOUS_WRITE_TIMEOUT)
        if writer.is_alive():
            # STALLED: the put is still running past the timeout. Abandon the daemon
            # thread (never joined at exit, so it cannot wedge process teardown) and
            # fall through to retry / fail-fast.
            last_exc = TimeoutError(
                f"rendezvous PutObject did not complete within {RENDEZVOUS_WRITE_TIMEOUT}s"
            )
            _log(
                f"Rendezvous write STALLED — timed out after {time.time() - t0:.1f}s "
                f"(attempt {attempt}/{RENDEZVOUS_WRITE_ATTEMPTS}); object-store PutObject "
                f"to {uri} is not completing."
            )
        elif "exc" in result_box:
            last_exc = result_box["exc"]
            _log(
                f"Rendezvous write FAILED after {time.time() - t0:.1f}s "
                f"(attempt {attempt}/{RENDEZVOUS_WRITE_ATTEMPTS}): {last_exc!r}"
            )
        else:
            _log(
                f"Wrote rendezvous {uri}: head_ip={head_ip} port={ray_port} "
                f"(attempt {attempt}, {time.time() - t0:.1f}s)"
            )
            return
        if attempt < RENDEZVOUS_WRITE_ATTEMPTS:
            backoff = min(2 ** (attempt - 1), 10)
            _log(f"Retrying rendezvous write in {backoff}s...")
            time.sleep(backoff)
    raise RuntimeError(
        f"Rank-0 failed to publish the rendezvous to {uri} after "
        f"{RENDEZVOUS_WRITE_ATTEMPTS} attempts (last error: {last_exc!r}). Failing "
        f"fast so the gang aborts with a clear cause instead of hanging to the worker "
        f"rendezvous deadline."
    ) from last_exc


def poll_rendezvous(rendezvous_dir: str, timeout: int, min_written_at: float | None = None) -> dict:
    """Poll for the head's rendezvous file. Returns its parsed payload.

    Payloads with ``written_at`` older than ``min_written_at`` (minus slack) are
    treated as stale (from a prior iris task attempt) and ignored.
    """
    uri = _rendezvous_uri(rendezvous_dir)
    fs, path = _fs_and_path(uri)
    deadline = time.time() + timeout
    threshold = (min_written_at - RENDEZVOUS_FRESHNESS_SLACK) if min_written_at else None
    _log(f"Polling for rendezvous {uri} (timeout {timeout}s)...")
    while time.time() < deadline:
        try:
            if fs.exists(path):
                with fs.open(path, "r") as f:
                    payload = json.load(f)
                if payload.get("head_ip"):
                    written_at = payload.get("written_at", 0)
                    if threshold is not None and written_at < threshold:
                        _log(
                            f"Ignoring stale rendezvous (written_at={written_at:.0f} "
                            f"< threshold={threshold:.0f}); waiting for rank-0 rewrite."
                        )
                    else:
                        _log(f"Found rendezvous: {payload}")
                        return payload
        except Exception as exc:  # transient object-store hiccup
            _log(f"rendezvous poll error (will retry): {exc}")
        time.sleep(POLL_INTERVAL)
    raise TimeoutError(
        f"Worker rank {_rank()} timed out after {timeout}s waiting for "
        f"rank-0 rendezvous at {uri}. Did the head task fail to start?"
    )


def _set_marker(rendezvous_dir: str, name: str) -> None:
    uri = f"{rendezvous_dir.rstrip('/')}/{name}"
    try:
        fs, path = _fs_and_path(uri)
        with fs.open(path, "w") as f:
            f.write(str(time.time()))
    except Exception as exc:
        _log(f"Warning: could not write marker {uri}: {exc}")


def _marker_exists(rendezvous_dir: str, name: str, min_written_at: float | None = None) -> bool:
    uri = f"{rendezvous_dir.rstrip('/')}/{name}"
    try:
        fs, path = _fs_and_path(uri)
        if not fs.exists(path):
            return False
        if min_written_at is None:
            return True
        with fs.open(path, "r") as f:
            written_at = float(f.read().strip() or 0)
        return written_at >= (min_written_at - RENDEZVOUS_FRESHNESS_SLACK)
    except Exception:
        return False


def clear_rendezvous(rendezvous_dir: str) -> None:
    """Best-effort delete of the rendezvous + done markers (rank 0, on entry/exit)."""
    for name in (RENDEZVOUS_FILENAME, DONE_FILENAME):
        uri = f"{rendezvous_dir.rstrip('/')}/{name}"
        try:
            fs, path = _fs_and_path(uri)
            if fs.exists(path):
                fs.rm(path)
                _log(f"Removed {uri}")
        except Exception as exc:
            _log(f"Warning: could not remove {uri}: {exc}")


# ---------------------------------------------------------------------------
# Ray cluster bootstrap (mirrors start_vllm_iris_controller).
# ---------------------------------------------------------------------------


# Ray port allocation on cw-us-east-02a — PIN every named system port OUTSIDE the
# worker_ports range so Ray's own randomized agent ports can never collide with it.
#
# THE BUG: Ray assigns several system components (metrics_export, runtime_env_agent,
# dashboard_agent_grpc, dashboard_agent_listen, node/object_manager) by picking a
# RANDOM free port. By default it draws them from the SAME ephemeral zone as the
# default worker_ports range (10002–19999), and Ray's own pre-start validation then
# aborts the node when a random agent port lands inside worker_ports:
#   ValueError: Ray component worker_ports is trying to use a port number <N>
#   that is used by other components.
# Observed on THREE different components across launches:
#   - head:   metrics_export=19865        (grpmm-fix)
#   - worker: runtime_env_agent=15731     (grpmm-fix3, head-only metrics pin in place)
#   - worker: dashboard_agent_grpc=24543 + runtime_env_agent=28330  (grpmm-fix4)
# grpmm-fix4 proved that merely SHIFTING worker_ports (to 20000–29999) does NOT help:
# the random agent ports simply followed into the new range. The collision can be ANY
# of Ray's randomized agent ports, on EITHER head or worker.
#
# THE FIX (deterministic, complete): keep worker_ports at Ray's DEFAULT 10002–19999
# and PIN every agent port Ray would otherwise randomize to a fixed value in the low
# 8xxx band — OUTSIDE 10002–19999 and distinct from gcs(6379)/dashboard(8265)/
# client_server(10001). With no random port left to draw from the worker range, the
# validation can never trip. Applied on BOTH head and worker (run_worker is where the
# fix3/fix4 collisions hit). 8090 matches the repo precedent RAY_metrics_export_port=
# 8090 in scripts/torch/kimi-k2-tracegen-run-v2.sh; the rest are adjacent free 8xxx.
RAY_METRICS_EXPORT_PORT = 8090
RAY_RUNTIME_ENV_AGENT_PORT = 8092
RAY_DASHBOARD_AGENT_GRPC_PORT = 8093
RAY_DASHBOARD_AGENT_LISTEN_PORT = 8094
RAY_NODE_MANAGER_PORT = 8076
RAY_OBJECT_MANAGER_PORT = 8077


def _ray_port_flags() -> list[str]:
    """Ray port flags shared by head + worker: pin EVERY named system port that Ray
    would otherwise randomize to a fixed value OUTSIDE the default worker_ports range
    (10002–19999), so no random agent port can ever collide with worker_ports (see the
    collision note above). worker_ports is left at Ray's default."""
    return [
        f"--metrics-export-port={RAY_METRICS_EXPORT_PORT}",
        f"--runtime-env-agent-port={RAY_RUNTIME_ENV_AGENT_PORT}",
        f"--dashboard-agent-grpc-port={RAY_DASHBOARD_AGENT_GRPC_PORT}",
        f"--dashboard-agent-listen-port={RAY_DASHBOARD_AGENT_LISTEN_PORT}",
        f"--node-manager-port={RAY_NODE_MANAGER_PORT}",
        f"--object-manager-port={RAY_OBJECT_MANAGER_PORT}",
    ]


# --- R2 object-store spilling (added 2026-06-28) -----------------------------------
# WHY: Ray spills its object store to /tmp/ray/session*/ray_spilled_objects on LOCAL
# disk when plasma (~95GB) overflows. The fully-async RL generator over-produces
# rollouts during the slow (~55-min) first training step, so the spill grows ~370 G/h
# to >1.6 TB and the kubelet EVICTS rank-0 on its ephemeral-storage limit -> gang
# bounce. CoreWeave/iris task pods have R2 (Cloudflare S3-compatible) creds + endpoint
# in env (AWS_ENDPOINT_URL / AWS_*_KEY / AWS_REGION=auto), and boto3 honors
# AWS_ENDPOINT_URL natively, so we redirect Ray's spill to s3://marin-us-east-02a/... instead of
# local disk. Validated 2026-06-28 in a running w13fix-r3 pod: with this exact config
# Ray spilled 25 objects to R2 and 0 to /tmp (see ray._private.external_storage
# ExternalStorageSmartOpenImpl, which does boto3.resource("s3") -> picks up the R2
# endpoint from env). NOTE: requires boto3 in the rl env (baked into the gpu-rl image).
# Set on the HEAD ONLY. `object_spilling_config` lives in Ray's `_system_config`, which
# Ray REJECTS on a worker `ray start --address=...` ("System config parameters can only be
# set on the head node" -> the worker's ray start exits 1 -> the coscheduled gang dies; this
# bit both MoE relaunches 2026-06-28, the first runs on the boto3 image where the spill URI
# was non-None on workers). The head's config propagates cluster-wide via GCS, and each
# worker raylet still spills its OWN objects per-node — the smart_open backend appends
# "_<node_id>" to the prefix at spill time so nodes never collide. So head-only is BOTH
# required (Ray forbids it on workers) and sufficient (config + per-node suffix propagate).
# Gate: OT_AGENT_RAY_SPILL_TO_R2 (default "1" = on); set to "0" to fall back to local
# /tmp spilling. Spill prefix is derived per-job from --rendezvous-dir so runs and
# task-retries within a run share one prefix without colliding across jobs.
RAY_SPILL_BUFFER_SIZE = 100 * 1024 * 1024  # 100MB multipart buffer (>=1MB recommended for remote)


def _ray_spill_uri(rendezvous_dir: str | None) -> str | None:
    """Per-job R2 spill prefix derived from the rendezvous dir, or None if R2 spilling
    is disabled / no rendezvous dir is available (single-node runs with no s3 dir fall
    back to local /tmp spilling)."""
    if os.environ.get("OT_AGENT_RAY_SPILL_TO_R2", "1") != "1":
        return None
    if not rendezvous_dir or not rendezvous_dir.startswith("s3://"):
        return None
    # SELF-GATE on boto3: Ray's smart_open spill backend imports boto3 directly, and it
    # is NOT in the gpu-rl image's rl env until the Dockerfile.gpu-rl boto3 add is baked.
    # On an image without boto3, return None -> clean fallback to local /tmp spill (no
    # `ray start` crash). R2 spilling AUTO-ACTIVATES once the rebuilt image ships boto3.
    try:
        import boto3  # noqa: F401
    except ImportError:
        _log("WARNING: boto3 missing -> Ray R2 object-spilling DISABLED (local /tmp fallback); "
             "rebuild gpu-rl image with boto3 (Dockerfile.gpu-rl) to enable. This run risks "
             "the ephemeral-storage eviction if its object store spills > the --disk limit.")
        return None
    return f"{rendezvous_dir.rstrip('/')}/ray_spill"


def _ray_spill_flags(spill_uri: str | None) -> list[str]:
    """Build the `--system-config` flag that redirects Ray object spilling to R2.

    The object_spilling_config VALUE is itself a JSON STRING (double-encoded), per Ray's
    system-config schema. min_spilling_size=0 forces every overflow to spill remotely
    rather than buffering small objects locally first."""
    if not spill_uri:
        return []
    spilling_config = json.dumps(
        {
            "type": "smart_open",
            "params": {"uri": spill_uri, "buffer_size": RAY_SPILL_BUFFER_SIZE},
        }
    )
    system_config = json.dumps(
        {"object_spilling_config": spilling_config, "min_spilling_size": 0}
    )
    return [f"--system-config={system_config}"]


# --- Ray cgroup-aware memory (added 2026-06-28) ------------------------------------
# WHY: in a memory-cgroup-limited pod, Ray can read the HOST's physical RAM (~2 TB via
# /proc/meminfo) instead of the --memory cgroup limit, and size its plasma object store
# at the default ~30% of that (~600 GB). On top of FSDP `cpu_offload`'s params+optimizer
# (also host RAM, OUTSIDE Ray's accounting) + the first training step's activations, that
# overran the 1400 GiB container limit -> the OOM killer SIGKILLed an FSDP worker ->
# NCCL-watchdog (1800s) gang death. Proven byte-exact 2026-06-28 on the 30B/35B MoE arms
# (rank-0 cgroup memory.peak == memory.max == 1400 GiB). Fix: read the container's cgroup
# limit and pass it to `ray start` as --memory, and BOUND the plasma store so it can't
# balloon off a misread host figure — leaving the bulk of host RAM for cpu_offload.
RAY_OBJECT_STORE_CAP_GIB = 96  # bounded plasma; default can be ~30% of *detected* RAM (huge if host is misread)
# Per-job override (env) for the plasma cap, in GiB. Default = RAY_OBJECT_STORE_CAP_GIB
# (96) -> UNSET is byte-identical to the prior hard-coded behavior for every existing job.
# WHY THIS EXISTS (2026-07-10, 80B v5 @98k dispatch-put stall,
# agent_logs/2026-07-09_80b_v5_98k_nccl_wedge_kill.md "ROOT CAUSE DEFINITIVELY PINNED"):
# the R3-resident forward dispatch (dispatch.py MeshDispatch.dispatch) ray.put()s one
# ~4.6 GB per-dp-group chunk into the DRIVER's (head node's) local plasma. The 8 dp-group
# chunks total only ~37 GB (fits in 96 GiB with room to spare), but at the FIRST training
# forward the head's plasma is already ~91 GiB full of PRIOR-pipeline live objects, so the
# store has room for only ONE more chunk: dp=0's chunk goes in + its forward pins it for
# ~798 s; dp=1's ray.put() then blocks on eviction/R2-spill (can't evict dp=0's in-use
# chunk) and trips the 600 s bounded-put timeout (d13c3586) -> the run dies at global_step 1.
# Because the 8 chunks (37 GB) are NOT themselves the overflow, waving/bounding the dispatch
# (candidate fixes b/c) does NOT help; only giving the store headroom ABOVE the ~91 GiB
# lingering baseline does. Raising the cap to ~160 GiB clears 91+37 = 128 GiB with margin.
# The min(., cgroup//8) OOM guard below still applies as the HARD ceiling (so a too-large
# value is clamped, never an OOM risk); the H100 head/policy nodes have ~1.3 TB cgroup with
# ~180 GB used by cpu_offload -> ~1 TB host headroom, so a 160 GiB plasma is safe.
_RAY_STORE_CAP_ENV = "OT_AGENT_RAY_OBJECT_STORE_CAP_GIB"


def _cgroup_mem_limit_bytes() -> int | None:
    """The container's memory cgroup limit in bytes (cgroup v2 then v1), or None if
    unreadable / unlimited (so callers fall back to Ray's own detection for --memory)."""
    for path in ("/sys/fs/cgroup/memory.max",                    # cgroup v2
                 "/sys/fs/cgroup/memory/memory.limit_in_bytes"):  # cgroup v1
        try:
            raw = open(path).read().strip()
        except OSError:
            continue
        if raw in ("max", ""):           # v2 unlimited
            return None
        try:
            v = int(raw)
        except ValueError:
            continue
        if v <= 0 or v > (1 << 62):      # v1 "unlimited" sentinel is a near-2^63 value
            return None
        return v
    return None


def _ray_mem_flags() -> list[str]:
    """Make `ray start` cgroup-aware (see the block comment above). Always bounds the
    plasma object store; additionally pins --memory to the cgroup limit when readable."""
    try:
        cap_gib = float(os.environ.get(_RAY_STORE_CAP_ENV, RAY_OBJECT_STORE_CAP_GIB))
    except ValueError:
        _log(f"Ray cgroup-aware: {_RAY_STORE_CAP_ENV}={os.environ.get(_RAY_STORE_CAP_ENV)!r} "
             f"not a number; falling back to default {RAY_OBJECT_STORE_CAP_GIB}GiB")
        cap_gib = float(RAY_OBJECT_STORE_CAP_GIB)
    store_cap = int(cap_gib * (1 << 30))
    if cap_gib != RAY_OBJECT_STORE_CAP_GIB:
        _log(f"Ray cgroup-aware: plasma cap overridden via {_RAY_STORE_CAP_ENV} -> "
             f"~{cap_gib:.0f}GiB (pre-guard)")
    limit = _cgroup_mem_limit_bytes()
    flags: list[str] = []
    if limit:
        store_cap = min(store_cap, limit // 8)  # never let the store exceed ~1/8 of the container
        flags.append(f"--memory={limit}")
        _log(f"Ray cgroup-aware: --memory={limit} (~{limit / (1 << 30):.0f}GiB cgroup limit), "
             f"--object-store-memory={store_cap} (~{store_cap / (1 << 30):.0f}GiB plasma cap)")
    else:
        _log(f"Ray cgroup-aware: no cgroup mem limit readable; bounding "
             f"--object-store-memory={store_cap} (~{store_cap / (1 << 30):.0f}GiB) only")
    flags.append(f"--object-store-memory={store_cap}")
    return flags


def ray_start_head(head_ip: str, ray_port: int, spill_uri: str | None = None) -> None:
    cmd = [
        _ray_bin(), "start", "--head",
        f"--node-ip-address={head_ip}",
        f"--port={ray_port}",
        "--dashboard-host=0.0.0.0",
        *_ray_port_flags(),
        *_ray_mem_flags(),
        *_ray_spill_flags(spill_uri),
    ]
    if spill_uri:
        _log(f"Ray object spilling -> R2 prefix {spill_uri} (no local /tmp spill)")
    _log(f"Starting Ray HEAD: {' '.join(cmd)}")
    t0 = time.time()
    subprocess.run(cmd, check=True, timeout=RAY_START_HEAD_TIMEOUT)
    _log(f"Ray HEAD subprocess returned (exit 0) in {time.time() - t0:.1f}s")


def ray_start_worker(head_ip: str, ray_port: int, node_ip: str, spill_uri: str | None = None) -> None:
    # NOTE: do NOT pass _ray_spill_flags here — `--system-config` is head-only in Ray
    # (see the R2-spill block comment). The head's object_spilling_config propagates to
    # this worker's raylet via GCS; the worker still spills its own objects per-node.
    cmd = [
        _ray_bin(), "start",
        f"--address={head_ip}:{ray_port}",
        f"--node-ip-address={node_ip}",
        *_ray_port_flags(),
        *_ray_mem_flags(),
    ]
    if spill_uri:
        _log(f"Ray object spilling -> R2 prefix {spill_uri} (cluster config from head; "
             f"this worker spills per-node, no local /tmp spill)")
    _log(f"Starting Ray WORKER: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def ray_stop() -> None:
    try:
        subprocess.run([_ray_bin(), "stop", "--force"], check=False, timeout=60)
    except subprocess.TimeoutExpired:
        _log("Warning: 'ray stop' timed out")


def wait_for_nodes(ray_address: str, expected_nodes: int, timeout: int, rewrite_cb=None) -> None:
    """Block until the Ray cluster reports ``expected_nodes`` alive nodes.

    ``rewrite_cb`` (head only): a no-arg callable invoked on every poll to RE-PUBLISH
    the rendezvous so its ``written_at`` stays fresh. WHY: a worker pod on a cold node
    can start >RENDEZVOUS_FRESHNESS_SLACK (60s) after the head wrote the rendezvous;
    its ``poll_rendezvous(min_written_at=worker_start)`` then rejects the head's
    one-shot rendezvous as "stale" and waits forever for a rewrite, while the head
    waits forever for that 4th node — a mutual deadlock (observed: ep8 diag, task 3
    started 3 min late on a cold node). Rewriting each poll keeps the timestamp ahead
    of any late worker's freshness threshold without weakening the prior-ATTEMPT
    protection (a stale file from a dead PRIOR attempt is still never refreshed).
    """
    import ray

    deadline = time.time() + timeout
    _log(f"Waiting for {expected_nodes} Ray node(s) at {ray_address} (timeout {timeout}s)...")
    ray.init(address=ray_address, ignore_reinit_error=True)
    try:
        last_count = -1
        while time.time() < deadline:
            if rewrite_cb is not None:
                try:
                    rewrite_cb()
                except Exception as exc:
                    _log(f"Warning: rendezvous rewrite failed (will retry): {exc}")
            alive = [n for n in ray.nodes() if n.get("Alive")]
            count = len(alive)
            if count != last_count:
                _log(f"Ray nodes alive: {count}/{expected_nodes}")
                last_count = count
            if count >= expected_nodes:
                _log(f"All {expected_nodes} Ray node(s) joined. Resources: {ray.cluster_resources()}")
                return
            time.sleep(POLL_INTERVAL)
        raise TimeoutError(f"Only {last_count}/{expected_nodes} Ray nodes joined within {timeout}s.")
    finally:
        ray.shutdown()


# ---------------------------------------------------------------------------
# Roles.
# ---------------------------------------------------------------------------


def capture_termination_artifacts(rendezvous_dir: str | None, reason: str) -> None:
    """On teardown, snapshot a FAST diagnostic summary to the rendezvous store
    BEFORE the pod is reaped. Best-effort; never raises; bounded to finish inside
    the k8s grace period.

    WHY: an iris/k8s-level termination (ephemeral-storage EVICTION, cgroup OOM,
    VRAM OOM) sends the controller a plain SIGTERM and leaves NOTHING in the iris
    finelog — and the per-node Ray logs are deleted with the pod, so the real
    cause is unrecoverable post-mortem (seen 2026-06-28, rl-q36-35b-w13fix: rank-0
    EVICTED for ephemeral-storage>512Gi mid-training-step; no traceback anywhere,
    k8s event TTL'd in ~1h). This persists disk-hogs + GPU mem + df + dmesg-OOM,
    keyed by task id, to ``<rendezvous_dir>/term_artifacts/`` so the next probe
    reads the true cause (disk vs VRAM-OOM vs RAM-OOM)."""
    if not rendezvous_dir:
        return
    import subprocess as _sp

    task_id = os.environ.get("IRIS_TASK_ID", "unknown").replace("/", "_")
    ts = int(time.time())

    def _run(cmd: str, timeout: int = 7) -> str:
        try:
            return _sp.run(cmd, shell=True, capture_output=True, text=True,
                           timeout=timeout).stdout
        except Exception as exc:  # noqa: BLE001 - best-effort
            return f"<{cmd!r} failed: {exc}>"

    summary = "\n".join([
        f"=== TERMINATION ARTIFACT task={task_id} ts={ts} reason={reason} ===",
        "--- df -h /tmp /dev/shm ---", _run("df -h /tmp /dev/shm 2>&1"),
        "--- top /tmp disk hogs (ephemeral-storage eviction cause) ---",
        _run("du -sh /tmp/* /tmp/ray/session*/logs /tmp/ray/session*/*spill* 2>/dev/null | sort -rh | head -25", 9),
        "--- nvidia-smi (VRAM OOM cause) ---",
        _run("nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv 2>&1"),
        "--- top RSS procs (host-RAM OOM cause) ---",
        _run("ps -eo pid,rss,comm --sort=-rss 2>/dev/null | head -12"),
        "--- dmesg OOM/kill tail ---",
        _run("dmesg 2>/dev/null | grep -iE 'oom|killed process|out of memory|Xid' | tail -10"),
    ])
    try:
        uri = f"{rendezvous_dir.rstrip('/')}/term_artifacts/{task_id}_{ts}.txt"
        fs, path = _fs_and_path(uri)
        with fs.open(path, "w") as f:
            f.write(summary)
        _log(f"[term-capture] wrote termination artifact -> {uri}")
    except Exception as exc:  # noqa: BLE001 - still emit to finelog as fallback
        _log(f"[term-capture] upload FAILED ({exc}); emitting inline:\n{summary}")


# --- Ray session-log -> object-store sync (added 2026-07-13) -----------------------
# WHY: the per-actor Ray WORKER logs (/tmp/ray/session_*/logs/worker-*.{out,err},
# raylet.out, ...) are the only place the FSDP policy / rollout actor stdout+tracebacks
# land — the iris finelog AGGREGATES only what reaches the head, and a pod GC / eviction
# DELETES these node-local logs with the pod, so a post-mortem loses the real per-rank
# cause (this is the 80B gs1 host-RAM-OOM + NCCL-desync debug pain: the wedged rank's
# worker log died with its pod). This periodically (+ on SIGTERM) uploads THIS node's
# session logs to the object store under the job's rendezvous prefix, keyed by node id,
# reusing the SAME fsspec/boto3 + AWS_ENDPOINT_URL creds path the rendezvous / spill /
# term-artifact writers already use (_fs_and_path). Per-node: each pod writes its own
# logs under <rendezvous_dir>/ray_session_logs/<node_id>/. Gate: OT_AGENT_RAY_LOG_SYNC
# (default "1"); interval OT_AGENT_RAY_LOG_SYNC_INTERVAL_S (default 300s).
RAY_LOG_SYNC_MAX_FILE_BYTES = 2 * 1024 * 1024 * 1024  # skip a single >2 GiB log (pathological)


def sync_ray_session_logs(rendezvous_dir: str | None, node_id: str, reason: str) -> None:
    """Upload THIS node's /tmp/ray/session_*/logs/ tree to the object store under
    ``<rendezvous_dir>/ray_session_logs/<node_id>/<session>/``. Best-effort; never
    raises; per-file so one bad file can't abort the rest."""
    if not rendezvous_dir:
        return
    if os.environ.get("OT_AGENT_RAY_LOG_SYNC", "1") != "1":
        return
    import glob

    log_dirs = sorted(glob.glob("/tmp/ray/session_*/logs"))
    if not log_dirs:
        return
    dest_base = f"{rendezvous_dir.rstrip('/')}/ray_session_logs/{node_id}"
    try:
        fs, dest_path = _fs_and_path(dest_base)
    except Exception as exc:  # noqa: BLE001 - best-effort
        _log(f"[ray-log-sync] cannot resolve dest {dest_base} ({exc}) [{reason}]")
        return
    n_files = n_bytes = 0
    for ld in log_dirs:
        session = os.path.basename(os.path.dirname(ld))  # session_YYYY-MM-DD_...
        for root, _dirs, files in os.walk(ld):
            for fn in files:
                lp = os.path.join(root, fn)
                try:
                    sz = os.path.getsize(lp)
                    if sz > RAY_LOG_SYNC_MAX_FILE_BYTES:
                        continue
                    rel = os.path.relpath(lp, ld)
                    fs.put(lp, f"{dest_path}/{session}/{rel}")
                    n_files += 1
                    n_bytes += sz
                except Exception:  # noqa: BLE001 - skip a single unreadable/racing file
                    continue
    _log(
        f"[ray-log-sync] uploaded {n_files} file(s) / {n_bytes / 1073741824.0:.2f} GiB "
        f"-> {dest_base} [{reason}]"
    )


def start_ray_log_sync(rendezvous_dir: str | None, node_id: str) -> threading.Event:
    """Start a daemon thread that periodically uploads this node's Ray session logs.
    Returns the stop Event (set it to stop). No-op (returns a set-able event) when
    disabled / no rendezvous dir. Fires a first upload after a short warmup so the
    sync is confirmable during bring-up, then every OT_AGENT_RAY_LOG_SYNC_INTERVAL_S."""
    stop = threading.Event()
    if not rendezvous_dir or os.environ.get("OT_AGENT_RAY_LOG_SYNC", "1") != "1":
        return stop
    interval = int(os.environ.get("OT_AGENT_RAY_LOG_SYNC_INTERVAL_S", "300"))
    if interval <= 0:
        return stop

    def _loop() -> None:
        # brief warmup so Ray has created the session dir + some logs before the first push
        if stop.wait(min(60, interval)):
            return
        sync_ray_session_logs(rendezvous_dir, node_id, "periodic")
        while not stop.wait(interval):
            sync_ray_session_logs(rendezvous_dir, node_id, "periodic")

    threading.Thread(target=_loop, daemon=True, name="ray-log-sync").start()
    _log(
        f"[ray-log-sync] started (every {interval}s, first ~{min(60, interval)}s) -> "
        f"{rendezvous_dir.rstrip('/')}/ray_session_logs/{node_id}"
    )
    return stop


def run_head(args: argparse.Namespace, train_argv: list[str]) -> int:
    num_tasks = _num_tasks()
    head_ip = _own_ip()
    ray_port = args.ray_port
    ray_address = f"{head_ip}:{ray_port}"
    node_id = f"rank0-{socket.gethostname()}"
    _log(f"ROLE=head rank=0/{num_tasks} head_ip={head_ip} ray_port={ray_port}")
    ray_log_sync_stop: threading.Event | None = None

    # Install the SIGTERM/SIGINT handler + termination-artifact capture at the TOP of
    # bring-up (BEFORE clear_rendezvous / ray_start_head / rendezvous write), so a reap
    # ANYWHERE in bring-up produces the py-spy/stack/term artifact instead of dying
    # silently. `process` is None during bring-up; the handler skips the driver kill
    # until the training driver is launched below (closure reads the current value).
    process = None

    def _shutdown(signum, _frame) -> None:
        _log(f"Received signal {signum}; terminating training driver and stopping Ray...")
        # Capture FIRST (before teardown mutates disk/GPU state) — a SIGTERM here is
        # often a k8s eviction / OOM whose cause survives nowhere else.
        capture_termination_artifacts(args.rendezvous_dir, f"signal {signum} (head rank 0)")
        # Flush this node's Ray session logs (per-actor worker stdout/tracebacks) before
        # the pod is reaped — Ray's node-local logs are deleted with the pod.
        sync_ray_session_logs(args.rendezvous_dir, node_id, f"signal {signum} (head)")
        if process is not None:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                process.wait(timeout=60)
            except (subprocess.TimeoutExpired, ProcessLookupError):
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
        if args.rendezvous_dir and num_tasks > 1:
            _set_marker(args.rendezvous_dir, DONE_FILENAME)
        ray_stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # On iris task retry, a rendezvous file from a previous attempt still points
    # at a now-dead head. Purge before starting the new head.
    if num_tasks > 1 and args.rendezvous_dir:
        clear_rendezvous(args.rendezvous_dir)

    ray_start_head(head_ip, ray_port, spill_uri=_ray_spill_uri(args.rendezvous_dir))
    _log("Ray head bootstrap complete; entering rendezvous / cluster-join phase.")

    # Start the periodic Ray session-log -> object-store sync now that the session dir
    # exists (per-node, keyed by node id, under the job's rendezvous prefix).
    ray_log_sync_stop = start_ray_log_sync(args.rendezvous_dir, node_id)

    if num_tasks > 1:
        if not args.rendezvous_dir:
            raise ValueError(
                "Multi-node iris slice (IRIS_NUM_TASKS>1) requires --rendezvous-dir "
                "(or OT_AGENT_IRIS_RENDEZVOUS_DIR) so worker ranks can find the head IP."
            )
        _log(
            "[start_rl_iris_controller] Ray head subprocess returned; writing rendezvous "
            f"-> {_rendezvous_uri(args.rendezvous_dir)}"
        )
        write_rendezvous(args.rendezvous_dir, head_ip, ray_port)
        # Re-publish the rendezvous each poll so a late cold-node worker never sees it
        # as "stale" (see wait_for_nodes docstring — prevents the freshness deadlock).
        wait_for_nodes(
            ray_address, num_tasks, args.cluster_join_timeout,
            rewrite_cb=lambda: write_rendezvous(args.rendezvous_dir, head_ip, ray_port),
        )
    else:
        _log("Single-node slice: skipping rendezvous and multi-node wait.")

    env = os.environ.copy()
    env["RAY_ADDRESS"] = ray_address  # skyrl-train's bare ray.init() attaches here
    env["PYTHONUNBUFFERED"] = "1"

    _log("Launching MarinSkyRL training driver:")
    _log("  " + " ".join(train_argv))
    sys.stdout.flush()
    sys.stderr.flush()

    # The SIGTERM/SIGINT handler is already installed at the top of run_head; assigning
    # `process` here arms its driver-teardown path (the closure reads this value).
    process = subprocess.Popen(train_argv, env=env, start_new_session=True)

    exit_code = process.wait()
    if exit_code != 0:
        capture_termination_artifacts(args.rendezvous_dir, f"driver exit_code={exit_code} (head rank 0)")
    # Final flush of this node's Ray session logs before teardown reaps them.
    if ray_log_sync_stop is not None:
        ray_log_sync_stop.set()
    sync_ray_session_logs(args.rendezvous_dir, node_id, f"driver exit_code={exit_code} (head)")
    # Signal workers to unpark, then tear down.
    if args.rendezvous_dir and num_tasks > 1:
        _set_marker(args.rendezvous_dir, DONE_FILENAME)
    ray_stop()
    if args.rendezvous_dir and num_tasks > 1:
        clear_rendezvous(args.rendezvous_dir)
    return exit_code


def run_worker(args: argparse.Namespace) -> int:
    worker_start = time.time()
    rank = _rank()
    num_tasks = _num_tasks()
    node_ip = _own_ip()
    node_id = f"rank{rank}-{socket.gethostname()}"
    _log(f"ROLE=worker rank={rank}/{num_tasks} node_ip={node_ip}")

    if not args.rendezvous_dir:
        raise ValueError(
            "Worker rank requires --rendezvous-dir (or OT_AGENT_IRIS_RENDEZVOUS_DIR) "
            "to discover the head IP."
        )

    payload = poll_rendezvous(args.rendezvous_dir, args.rendezvous_timeout, min_written_at=worker_start)
    head_ip = payload["head_ip"]
    ray_port = int(payload.get("port", args.ray_port))
    ray_address = f"{head_ip}:{ray_port}"

    ray_start_worker(head_ip, ray_port, node_ip, spill_uri=_ray_spill_uri(args.rendezvous_dir))
    wait_for_nodes(ray_address, num_tasks, args.cluster_join_timeout)
    _log(f"Worker rank {rank} joined Ray cluster at {ray_address}; parking until the head finishes.")

    # Periodic Ray session-log -> object-store sync for THIS worker node (the FSDP/rollout
    # actors on this node log to its local /tmp/ray session, deleted with the pod on GC).
    ray_log_sync_stop = start_ray_log_sync(args.rendezvous_dir, node_id)

    stop = threading.Event()

    def _shutdown(signum, _frame) -> None:
        _log(f"Worker rank {rank} received signal {signum}; stopping Ray.")
        # A SIGTERM on a worker node is often a k8s eviction/OOM of that node (it
        # hosts the training actors' GPUs); capture its disk/GPU state before reap.
        capture_termination_artifacts(args.rendezvous_dir, f"signal {signum} (worker rank {rank})")
        # Flush this node's per-actor Ray worker logs before the pod is reaped.
        sync_ray_session_logs(args.rendezvous_dir, node_id, f"signal {signum} (worker rank {rank})")
        stop.set()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # Block until the head publishes the done marker (training finished) or we
    # are signalled. The training driver on rank 0 schedules actors onto this
    # node's GPUs; this process just keeps the Ray node alive.
    while not stop.is_set():
        if _marker_exists(args.rendezvous_dir, DONE_FILENAME, min_written_at=worker_start):
            _log(f"Worker rank {rank} saw head done-marker; shutting down.")
            break
        time.sleep(POLL_INTERVAL)
    # Final flush of this worker node's Ray session logs before Ray teardown.
    ray_log_sync_stop.set()
    sync_ray_session_logs(args.rendezvous_dir, node_id, f"worker rank {rank} teardown")
    ray_stop()
    return 0


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Bootstrap one cross-node Ray cluster on an iris GPU slice and run "
        "the MarinSkyRL training driver on rank 0. Everything after `--` is the "
        "training command (e.g. `python -m skyrl_train.entrypoints.main_base <hydra args>`).",
    )
    parser.add_argument(
        "--ray-port",
        type=int,
        default=int(os.environ.get("OT_AGENT_IRIS_RAY_PORT", "6379")),
        help="Port the Ray head binds (default 6379).",
    )
    parser.add_argument(
        "--rendezvous-dir",
        default=os.environ.get("OT_AGENT_IRIS_RENDEZVOUS_DIR"),
        help="Shared object-store/dir for the head/worker rendezvous (gs://, s3://, "
        "or a shared path). Defaults to $OT_AGENT_IRIS_RENDEZVOUS_DIR.",
    )
    parser.add_argument(
        "--rendezvous-timeout",
        type=int,
        default=DEFAULT_RENDEZVOUS_TIMEOUT,
        help=f"Seconds workers poll for the head rendezvous (default {DEFAULT_RENDEZVOUS_TIMEOUT}).",
    )
    parser.add_argument(
        "--cluster-join-timeout",
        type=int,
        default=DEFAULT_CLUSTER_JOIN_TIMEOUT,
        help=f"Seconds to wait for all nodes to join the Ray cluster (default {DEFAULT_CLUSTER_JOIN_TIMEOUT}).",
    )
    parser.add_argument(
        "--train-data",
        default=os.environ.get("OT_AGENT_IRIS_TRAIN_DATA", ""),
        help="JSON list of train_data HF dataset(s) to stage (extract to the node-local "
        "task dir) on EVERY node before Ray starts. Required for agentic terminal_bench "
        "rollouts on a multi-node slice with no shared filesystem.",
    )
    parser.add_argument(
        "--prestage-model",
        default=os.environ.get("OT_AGENT_IRIS_PRESTAGE_MODEL", ""),
        help="HF repo-id of the policy model to pre-download into the node-local HF "
        "cache on EVERY node before Ray starts. Set by the launcher when the config "
        "runs HF_HUB_OFFLINE=1, so the FSDP ranks load from a warm node-local cache "
        "instead of each racing HF Hub at init (the 80B init-straggle fix).",
    )
    args, train_argv = parser.parse_known_args()
    # argparse leaves the `--` separator out of train_argv; strip a leading one
    # if the shell passed it through.
    if train_argv and train_argv[0] == "--":
        train_argv = train_argv[1:]
    if not train_argv:
        parser.error("No training command given. Pass it after `--`.")
    return args, train_argv


def _print_env_snapshot() -> None:
    _log("environment snapshot:")
    for key in (
        "IRIS_TASK_ID", "IRIS_NUM_TASKS", "IRIS_ADVERTISE_HOST",
        "RAY_ADDRESS", "SKYRL_HOME", "PYTHONPATH", "HF_HOME",
        "NUM_INFERENCE_ENGINES", "POLICY_NUM_NODES", "TENSOR_PARALLEL_SIZE",
    ):
        print(f"  {key}={os.environ.get(key, '<unset>')}", flush=True)


def main() -> None:
    args, train_argv = parse_args()
    _print_env_snapshot()
    # Pin virtual-hosted S3 addressing for the boto3 path (Ray object-spill IO
    # workers) BEFORE any `ray start`, on head + every worker — CoreWeave R2
    # rejects path-style with PathStyleRequestNotAllowed. Companion to the
    # fsspec FSSPEC_S3_CONFIG_KWARGS + the _fs_and_path rendezvous pin.
    _pin_boto3_s3_addressing_style()
    # Ensure the NCCL flight-recorder dump dir exists on THIS node BEFORE any torch/NCCL
    # init (head + every worker), so a collective-timeout FR dump actually writes instead
    # of silently failing — torch's DebugInfoWriter does not mkdir -p its parent (the
    # 80b-next-cp1 gs1 hang lost its dump this way). See ensure_fr_dump_dir.
    ensure_fr_dump_dir()
    # Stage the task dataset on THIS node before Ray bootstrap (head + every worker).
    # Without this, only rank-0 has the extracted tasks and the rollout workers die
    # with FileNotFoundError on task.toml (see stage_train_data docstring).
    if args.train_data:
        stage_train_data(args.train_data)
    # Pre-download the policy weights into the node-local HF cache BEFORE Ray, so the
    # FSDP ranks load from a warm cache under HF_HUB_OFFLINE=1 instead of each racing
    # HF Hub inside init_model (the init-straggle store->get barrier kill). See stage_model.
    if args.prestage_model:
        stage_model(args.prestage_model)
    rank = _rank()
    if rank == 0:
        exit_code = run_head(args, train_argv)
    else:
        exit_code = run_worker(args)
    if exit_code != 0:
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
