"""literal_proxy_utils.py — co-located RecordProxy wiring for literal-token capture.

Drives harbor's LITERAL-TOKEN trace machinery (``harbor.literal.proxy.RecordProxy``)
from the OT-Agent launch path. When ``--record_literal`` is set, the RL / datagen
launchers co-locate a :class:`~harbor.literal.proxy.RecordProxy` alongside the
on-cluster vLLM server and route the agent's inference endpoint THROUGH the proxy.
The proxy transparently injects ``return_token_ids=True`` / ``logprobs=True`` into
each forwarded completion and appends the returned prompt/completion token IDs +
logprobs to a ``literal.jsonl`` log, which the ``opencode`` agent (and any agent
with ``SUPPORTS_LITERAL_TRACES``) merges back into its trajectory steps.

Default OFF: :func:`maybe_serve_literal_proxy` is a NULL context manager that
yields the upstream vLLM endpoint UNCHANGED and starts no server, so the launch
path is byte-identical to today (parity gate G0).

Contract (from ``harbor/src/harbor/literal/proxy.py``, read 2026-07-03):
  * ``RecordProxy(upstream_base_url, log_path, timeout=600.0)`` — forwards to
    ``upstream_base_url`` (a vLLM ``.../v1`` base) and records to ``log_path``.
  * ``RecordProxy.app()`` returns a FastAPI app serving ``/v1/chat/completions``
    and ``/v1/completions`` (the surface harbor's installed agents call). We serve
    that app with uvicorn on a co-located port; the agent's api_base points at it.

Nothing here binds a socket unless ``enabled`` is True — the pure helpers and the
disabled path are fully unit-testable without a server (see tests/hpc/test_literal_proxy.py).
"""

from __future__ import annotations

import re
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

# The co-located proxy binds this port by default. vLLM/datagen use 8000, so 8010
# avoids a collision while staying on the loopback interface (the proxy only ever
# fronts the LOCAL vLLM; external reach is via the existing pinggy/controller path).
DEFAULT_LITERAL_PROXY_PORT = 8010
DEFAULT_LITERAL_PROXY_HOST = "127.0.0.1"

# Filename harbor's opencode agent reads back from its logs dir
# (``OpenCode._LITERAL_LOG_FILENAME``); we name the co-located log the same for
# symmetry, though the co-located log lives on the CLUSTER (beside the job logs).
LITERAL_LOG_FILENAME = "literal.jsonl"


def literal_log_path(experiments_dir: str | Path, job_name: str) -> Path:
    """Deterministic path for the co-located RecordProxy's ``literal.jsonl`` log.

    Lives under ``<experiments_dir>/logs/`` beside the other per-job logs.
    """
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", (job_name or "job")).strip("-.") or "job"
    return Path(experiments_dir) / "logs" / f"{slug}_{LITERAL_LOG_FILENAME}"


def literal_proxy_endpoint(
    host: str = DEFAULT_LITERAL_PROXY_HOST, port: int = DEFAULT_LITERAL_PROXY_PORT
) -> str:
    """The OpenAI-compatible base URL the agent should target to hit the proxy."""
    return f"http://{host}:{port}/v1"


@contextmanager
def serve_record_proxy(
    upstream_endpoint: str,
    log_path: str | Path,
    *,
    host: str = DEFAULT_LITERAL_PROXY_HOST,
    port: int = DEFAULT_LITERAL_PROXY_PORT,
    timeout: float = 600.0,
    startup_timeout: float = 30.0,
) -> Iterator[str]:
    """Serve harbor's RecordProxy in front of ``upstream_endpoint`` on ``host:port``.

    Yields the proxy's OpenAI-compatible base URL (``http://host:port/v1``). The
    uvicorn server runs on a daemon thread and is shut down on context exit. The
    upstream need not be live at start time — RecordProxy forwards lazily per
    request, so it is safe to start the proxy BEFORE the vLLM engines are up.

    This binds a real socket and is used ONLY on the flag-ON launch path; unit
    tests exercise the app in-process via ``RecordProxy.app()`` + ASGITransport
    instead of calling this.
    """
    import uvicorn
    from harbor.literal.proxy import RecordProxy

    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    proxy = RecordProxy(upstream_endpoint, log_path, timeout=timeout)
    app = proxy.app()

    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, name="record-proxy", daemon=True)
    thread.start()

    # Wait for uvicorn to flip started=True so the endpoint is live before we
    # hand it to the agent. Fail loud if it never comes up.
    waited = 0.0
    step = 0.1
    while not server.started and thread.is_alive() and waited < startup_timeout:
        thread.join(timeout=step)
        waited += step
    if not server.started:
        server.should_exit = True
        raise RuntimeError(
            f"RecordProxy failed to start on {host}:{port} within {startup_timeout}s"
        )

    print(
        f"[literal-proxy] RecordProxy serving {literal_proxy_endpoint(host, port)} "
        f"-> {upstream_endpoint} (log: {log_path})",
        flush=True,
    )
    try:
        yield literal_proxy_endpoint(host, port)
    finally:
        server.should_exit = True
        thread.join(timeout=10.0)


@contextmanager
def maybe_serve_literal_proxy(
    enabled: bool,
    upstream_endpoint: str,
    *,
    experiments_dir: str | Path,
    job_name: str,
    host: str = DEFAULT_LITERAL_PROXY_HOST,
    port: int = DEFAULT_LITERAL_PROXY_PORT,
) -> Iterator[str]:
    """Flag-gated wrapper around :func:`serve_record_proxy`.

    When ``enabled`` is False (the default), this is a NULL context manager: it
    yields ``upstream_endpoint`` UNCHANGED and starts no server, so the caller's
    endpoint handoff is byte-identical to today. When True, it serves a co-located
    RecordProxy in front of ``upstream_endpoint`` and yields the proxy base URL.
    """
    if not enabled:
        yield upstream_endpoint
        return
    log_path = literal_log_path(experiments_dir, job_name)
    with serve_record_proxy(
        upstream_endpoint, log_path, host=host, port=port
    ) as proxy_endpoint:
        yield proxy_endpoint
