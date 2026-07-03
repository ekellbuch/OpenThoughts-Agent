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
from typing import Dict, Optional

# The sandbox-facing static bearer token secret (name-only; see secrets.env).
INGRESS_KEY_ENV = "IRIS_INGRESS_API_KEY"

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
