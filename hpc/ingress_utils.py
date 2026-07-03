"""ingress_utils.py — controller-ingress wiring helpers (Stage 4).

Shared by the RL / datagen launchers and the eval listener when
``--ingress-mode controller`` is selected. Under the default ``pinggy`` mode
NONE of this is reached, so the legacy path stays byte-identical (plan G0).

The controller path replaces the pinggy tunnel with a stable public URL fronting
the Iris controller proxy behind our auth-gating sidecar (``hpc/ingress_sidecar``):

    api_base = https://<ingress_host>/proxy/<endpoint_name>/v1

and the sandbox carries ``IRIS_INGRESS_API_KEY`` as a bearer. The installed
OpenAI-compatible agents (qwen-code/codex/hermes/trae/openhands) read the key
from ``OPENAI_API_KEY`` / ``LLM_API_KEY`` beside their base URL (evidence B), so
injecting the key into those env vars is the universal lever — no per-agent code
change. Secrets are referenced by env-var NAME only; the real token is never
hardcoded, and never written to a file (the api_key kwarg uses the
``${IRIS_INGRESS_API_KEY}`` placeholder so it is resolved at agent runtime).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Protocol, Tuple

# The sandbox-facing static bearer token secret (name-only; see secrets.env).
INGRESS_KEY_ENV = "IRIS_INGRESS_API_KEY"

# Env var iris sets in-cluster to the externally-visible host a task should
# advertise its services under (``JobInfo.advertise_host``). The controller
# resolves a registered endpoint by connecting to its ``address``, so the
# upstream (raw vLLM, or the co-located RecordProxy) must be reachable at THIS
# host — a loopback ``127.0.0.1`` would only be reachable from inside the task.
# Off-cluster / in tests the var is unset and we fall back to loopback.
ADVERTISE_HOST_ENV = "IRIS_ADVERTISE_HOST"
DEFAULT_ADVERTISE_HOST = "127.0.0.1"

# The raw vLLM HTTP port the RL/datagen servers bind on the task node.
DEFAULT_VLLM_PORT = 8000

# Env vars the five installed OpenAI-compatible agents read beside their base URL
# (evidence B): OPENAI_API_KEY (qwen/codex/hermes/trae) and LLM_API_KEY (openhands).
_AGENT_KEY_ENV_VARS = ("OPENAI_API_KEY", "LLM_API_KEY")

# Placeholder written into agent kwargs so the real secret never lands in a
# config file; Harbor's resolve_env_vars expands it from the process env.
INGRESS_KEY_PLACEHOLDER = "${" + INGRESS_KEY_ENV + "}"


def controller_endpoint_name(job_name: Optional[str]) -> str:
    """Deterministic iris endpoint name for a job's vLLM (Stage 2 registers this name).

    ``otagent-<sanitized-job-name>`` — unique per job, single path segment. The name
    MUST be DOT-FREE: the controller EndpointProxy decodes ``.`` -> ``/`` in the
    ``/proxy/<name>/`` route (``endpoint_proxy.py``: ``slashed = encoded_name.replace('.','/')``,
    then resolves ``/<slashed>``). So a literal dot in the registered name is looked up
    as a ``/`` path segment and can never match — that is the 404 the first live smoke
    hit despite a successful register. We map dots (and anything outside ``[A-Za-z0-9_-]``)
    to ``-`` so the registered name and the dot-decoded lookup are identical.
    """
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", (job_name or "job")).strip("-_").lower()
    return f"otagent-{slug or 'job'}"


def build_controller_api_base(ingress_host: str, endpoint_name: str) -> str:
    """``https://<ingress_host>/proxy/<endpoint_name>/v1`` (scheme optional on host)."""
    host = (ingress_host or "").rstrip("/")
    if not (host.startswith("http://") or host.startswith("https://")):
        host = f"https://{host}"
    return f"{host}/proxy/{endpoint_name}/v1"


def controller_api_base_for_job(ingress_host: str, job_name: Optional[str]) -> str:
    return build_controller_api_base(ingress_host, controller_endpoint_name(job_name))


def inject_ingress_agent_key(env: Optional[dict] = None) -> bool:
    """Copy ``IRIS_INGRESS_API_KEY`` into the agent-facing key env vars.

    Returns True if the key was present and injected. Reads the secret by env-var
    NAME only; never prints it. No-op (returns False) if the key is unset.
    """
    env = os.environ if env is None else env
    key = env.get(INGRESS_KEY_ENV)
    if not key:
        return False
    for var in _AGENT_KEY_ENV_VARS:
        env[var] = key
    return True


def build_controller_endpoint_meta(ingress_host: str, endpoint_name: str) -> Dict[str, str]:
    """Endpoint metadata dict for controller mode (api_base + placeholder api_key).

    metrics_endpoint is intentionally omitted: the sidecar forwards ONLY the
    ``/v1`` inference surface, so ``/metrics`` is not reachable through it.
    """
    return {
        "api_base": build_controller_api_base(ingress_host, endpoint_name),
        "api_key": INGRESS_KEY_PLACEHOLDER,
    }


# --------------------------------------------------------------------------- #
# Endpoint registration (shared by the controller path and the +literal combo)
# --------------------------------------------------------------------------- #
#
# The controller-ingress api_base ``https://<host>/proxy/<name>/v1`` only
# resolves once the vLLM (or, in the literal combo, the co-located RecordProxy)
# is REGISTERED with the iris controller under ``<name>``. This registration is
# the missing link the ``--ingress-mode controller`` path did not yet wire; the
# helpers below fill it and are shared so the plain and the record_literal combo
# register the SAME name (only the address differs), keeping the api_base fixed.
#
# NOTE ON NAMESPACING: we register through the leased :class:`EndpointClient`
# (``iris.cluster.client.endpoint_client``) directly rather than through
# ``ctx.registry`` — the latter (``NamespacedEndpointRegistry``) auto-prefixes the
# name with the job namespace (``<user>/<root>/<name>``), which would break the
# fixed single-segment ``/proxy/otagent.<slug>/v1`` route the ingress path already
# builds (and that the live smoke verified with the un-prefixed name).
# ``EndpointClient.register(name, address, task_attempt, metadata)`` does NOT
# prefix, so the wire name stays the single ``otagent.<slug>`` segment.
#
# NOTE ON LEASING: iris endpoints are LEASED. ``EndpointClient`` owns the RPC stub
# AND a background ``EndpointLeaseRenewer`` daemon — the lease is what keeps the
# controller serving the endpoint, and a one-shot register with no renewal expires
# within minutes (which 404'd the whole run). So we build a dedicated
# ``EndpointClient`` (its renewer daemon running), register through it, and KEEP IT
# ALIVE for the entire harbor run via the returned :class:`ControllerEndpointRegistration`
# handle, then ``close()`` it (stops renewing + unregisters) on run exit.


class EndpointRegistrar(Protocol):
    """The ``register(name, address, metadata) -> endpoint_id`` shape we drive.

    The in-cluster leased-``EndpointClient`` adapter and unit-test fakes both
    satisfy it, so the registration wiring is testable in-process without a live
    controller. An implementation MAY also expose ``close()`` to stop lease
    renewal and unregister; the registration handle calls it on teardown.
    """

    def register(
        self, name: str, address: str, metadata: Optional[Dict[str, str]] = None
    ) -> str: ...


@dataclass
class ControllerEndpointRegistration:
    """A live controller endpoint registration whose lease is being renewed.

    Holds the real ``endpoint_id`` and a ``close`` callable that stops the
    background lease renewer and unregisters the endpoint. The caller MUST keep
    this handle alive for the whole harbor run (so the daemon keeps renewing) and
    call :meth:`close` on exit; a dropped handle lets the lease lapse and the
    ``/proxy/<name>/v1`` route starts 404-ing mid-run.
    """

    endpoint_id: str
    _close: Callable[[], None]

    def close(self) -> None:
        """Stop lease renewal and unregister the endpoint (best-effort, idempotent)."""
        self._close()


class _LeasedEndpointRegistrar:
    """Registers through a dedicated leased iris :class:`EndpointClient`.

    Owns the ``EndpointClient`` (and thus its background ``EndpointLeaseRenewer``
    daemon), so ``register`` returns the real endpoint_id and keeps the lease
    renewed until ``close``. Registers WITHOUT namespace-prefixing (see module
    note) so the wire name is the single ``otagent.<slug>`` segment the
    ``/proxy/<name>/v1`` api_base uses.
    """

    def __init__(self, client: "EndpointClient", task_attempt: "TaskAttempt") -> None:
        self._client = client
        self._task_attempt = task_attempt

    def register(
        self, name: str, address: str, metadata: Optional[Dict[str, str]] = None
    ) -> str:
        return self._client.register(name, address, self._task_attempt, metadata or {})

    def close(self) -> None:
        # Stops the renewer daemon and best-effort unregisters everything still
        # registered, then disconnects the stub.
        self._client.close()


def controller_upstream_address(port: int, *, env: Optional[dict] = None) -> str:
    """``http://<advertise-host>:<port>`` — the address the controller resolves to.

    ``<advertise-host>`` is ``IRIS_ADVERTISE_HOST`` (the task's externally-visible
    host, set by iris in-cluster) falling back to ``127.0.0.1`` off-cluster. The
    controller connects here to reach the registered upstream, so it must NOT be a
    loopback address on a live cluster.
    """
    env = os.environ if env is None else env
    host = env.get(ADVERTISE_HOST_ENV) or DEFAULT_ADVERTISE_HOST
    return f"http://{host}:{port}"


def _default_endpoint_registrar() -> _LeasedEndpointRegistrar:
    """Build the in-cluster leased-``EndpointClient`` registrar.

    Constructs a dedicated ``EndpointClient`` from the task's
    ``IRIS_CONTROLLER_ADDRESS`` (network-level trust in-cluster, no credentials)
    and the task identity from :func:`iris.cluster.client.job_info.get_job_info`.
    The ``EndpointClient`` starts a background lease renewer on ``register`` so
    the controller keeps serving the endpoint for the whole run.

    Raises loudly (never returns ``None``) when iris is unavailable or we are not
    inside a task — a silent no-op here is exactly what produced the 404-ing run.
    """
    # iris is only importable inside the cluster runtime image; on the
    # controller-ingress path (worker / SLURM node) it is always present.
    from iris.cluster.client.endpoint_client import EndpointClient
    from iris.cluster.client.job_info import get_job_info
    from iris.rpc.compression import IRIS_RPC_COMPRESSIONS
    from iris.rpc.controller_connect import EndpointServiceClientSync

    info = get_job_info()
    if info is None or not info.controller_address:
        raise RuntimeError(
            "controller endpoint registration requires an in-cluster iris task "
            "(IRIS_TASK_ID + IRIS_CONTROLLER_ADDRESS); none found. Registration "
            "cannot proceed — the /proxy/<name>/v1 route would 404."
        )
    stub = EndpointServiceClientSync(
        info.controller_address,
        accept_compression=IRIS_RPC_COMPRESSIONS,
        send_compression=None,
    )
    return _LeasedEndpointRegistrar(EndpointClient(stub), info.task_attempt)


def register_controller_endpoint(
    endpoint_name: str,
    address: str,
    *,
    registrar: Optional[EndpointRegistrar] = None,
    metadata: Optional[Dict[str, str]] = None,
) -> ControllerEndpointRegistration:
    """Register ``address`` under ``endpoint_name`` with the iris controller.

    Registers through a leased ``EndpointClient`` (its background lease renewer
    keeps the controller serving the endpoint) so ``/proxy/<endpoint_name>/v1``
    resolves to ``address`` for the whole run. ``registrar`` is injectable for
    unit tests; in production it is the in-cluster :func:`_default_endpoint_registrar`.

    Returns a :class:`ControllerEndpointRegistration` handle with the REAL
    ``endpoint_id`` and a ``close()`` that stops renewal + unregisters. The caller
    MUST keep the handle alive for the whole run and ``close()`` it on exit.

    Raises if no registrar is available or the register call yields no id — a
    broken registration must fail loud, not silently 404 the run.
    """
    reg = registrar if registrar is not None else _default_endpoint_registrar()
    endpoint_id = reg.register(endpoint_name, address, metadata or {})
    if not endpoint_id:
        raise RuntimeError(
            f"controller endpoint registration for {endpoint_name} -> {address} "
            "returned no endpoint_id; refusing to proceed (would 404 the run)."
        )
    close = getattr(reg, "close", None)
    return ControllerEndpointRegistration(
        endpoint_id=endpoint_id,
        _close=close if callable(close) else (lambda: None),
    )


def controller_registration_plan(
    ingress_host: str,
    job_name: Optional[str],
    *,
    record_literal: bool,
    proxy_port: int,
    vllm_port: int = DEFAULT_VLLM_PORT,
    env: Optional[dict] = None,
) -> Tuple[str, str, str]:
    """Compute ``(endpoint_name, register_address, api_base)`` for controller mode.

    The single decision point the launchers share so the plain and the
    ``record_literal`` combo stay consistent:

      * ``endpoint_name`` — the same ``otagent.<slug>`` either way, so ``api_base``
        is byte-identical whether or not literal capture is on.
      * ``register_address`` — the co-located RecordProxy's ``proxy_port`` when
        ``record_literal`` is set (controller -> RecordProxy -> vLLM, so literal
        tokens are captured on the served path), otherwise raw vLLM's ``vllm_port``.
      * ``api_base`` — ``https://<ingress_host>/proxy/<endpoint_name>/v1``.
    """
    endpoint_name = controller_endpoint_name(job_name)
    port = proxy_port if record_literal else vllm_port
    register_address = controller_upstream_address(port, env=env)
    api_base = build_controller_api_base(ingress_host, endpoint_name)
    return endpoint_name, register_address, api_base
