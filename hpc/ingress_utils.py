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
from typing import Any, Dict, Optional, Protocol, Tuple

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

    ``otagent.<sanitized-job-name>`` — unique per job, single path segment (the
    controller proxy route is ``/proxy/<name>/<sub_path>``, so ``name`` must not
    contain ``/``).
    """
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", (job_name or "job")).strip("-.").lower()
    return f"otagent.{slug or 'job'}"


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
# NOTE ON NAMESPACING: we register through the LOW-LEVEL cluster client
# (``ctx.client._cluster_client.register_endpoint``) rather than
# ``ctx.registry.register`` — the latter (``NamespacedEndpointRegistry``)
# auto-prefixes the name with the job namespace (``<user>/<root>/<name>``),
# which would break the fixed single-segment ``/proxy/otagent.<slug>/v1`` route
# the ingress path already builds (and that the S3 live smoke verified with the
# un-prefixed name). The ``.register(name, address, metadata)`` call shape still
# matches the primitive the design references; we just skip the prefixing.


class EndpointRegistrar(Protocol):
    """The ``register(name, address, metadata) -> endpoint_id`` shape we drive.

    Both the in-cluster cluster-client adapter and unit-test fakes satisfy it,
    so the registration wiring is testable in-process without a live controller.
    """

    def register(
        self, name: str, address: str, metadata: Optional[Dict[str, str]] = None
    ) -> str: ...


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


class _ClusterClientRegistrar:
    """Adapts the iris cluster client to :class:`EndpointRegistrar`.

    Registers WITHOUT namespace-prefixing (see module note) so the wire name is
    the single ``otagent.<slug>`` segment the ``/proxy/<name>/v1`` api_base uses.
    """

    def __init__(self, cluster: Any, task_attempt: Any) -> None:
        self._cluster = cluster
        self._task_attempt = task_attempt

    def register(
        self, name: str, address: str, metadata: Optional[Dict[str, str]] = None
    ) -> str:
        return self._cluster.register_endpoint(
            name=name,
            address=address,
            task_attempt=self._task_attempt,
            metadata=metadata or {},
        )


def _default_endpoint_registrar() -> Optional[EndpointRegistrar]:
    """Best-effort in-cluster registrar from :func:`iris.client.iris_ctx`.

    Returns ``None`` (never raises) when iris is unavailable or we are not running
    inside a task (no controller address / task attempt) — e.g. off-cluster or in
    tests — so callers degrade gracefully.
    """
    try:  # iris is only importable inside the cluster runtime image.
        from iris.client import iris_ctx  # type: ignore
    except Exception:
        return None
    try:
        ctx = iris_ctx()
        cluster = getattr(getattr(ctx, "client", None), "_cluster_client", None)
        task_attempt = getattr(ctx, "task_attempt", None)
        if cluster is None or task_attempt is None:
            return None
        return _ClusterClientRegistrar(cluster, task_attempt)
    except Exception:
        return None


def register_controller_endpoint(
    endpoint_name: str,
    address: str,
    *,
    registrar: Optional[EndpointRegistrar] = None,
    metadata: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """Register ``address`` under ``endpoint_name`` with the iris controller.

    Uses the ``register(name, address, metadata)`` primitive so
    ``/proxy/<endpoint_name>/v1`` resolves to ``address``. ``registrar`` is
    injectable for unit tests; in production it is resolved in-cluster from
    :func:`iris.client.iris_ctx`. Returns the endpoint_id, or ``None`` when no
    registrar is available (off-cluster / iris missing) — the caller logs and
    proceeds (the sidecar/api_base wiring is already set).
    """
    reg = registrar if registrar is not None else _default_endpoint_registrar()
    if reg is None:
        return None
    return reg.register(endpoint_name, address, metadata or {})


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
