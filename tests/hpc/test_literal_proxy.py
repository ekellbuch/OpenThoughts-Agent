"""Unit tests for the co-located RecordProxy literal-capture wiring.

Covers ``hpc/literal_proxy_utils.py`` (the OT-Agent side of harbor's literal-token
trace machinery):

  * pure helpers (log path, proxy endpoint, default port) — deterministic, no I/O.
  * FLAG-OFF PARITY: ``maybe_serve_literal_proxy(False, ...)`` is a null context
    manager — it yields the upstream endpoint UNCHANGED and NEVER touches
    ``serve_record_proxy`` (no socket, no thread). This is the G0 byte-identical
    proof for the launch-path functions.
  * FLAG-ON gating: ``maybe_serve_literal_proxy(True, ...)`` routes through
    ``serve_record_proxy`` and yields its endpoint (stubbed — no real socket).
  * FLAG-ON surface: harbor's ``RecordProxy.app()`` (what we co-locate) injects the
    literal params and records ``literal.jsonl`` — exercised in-process against a
    STUB vLLM via ``httpx.ASGITransport`` (no GPU, no socket), mirroring the
    sidecar tests' stubbing style.

Run:
    .venv/bin/python -m pytest tests/hpc/test_literal_proxy.py -v
"""

from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.testclient import TestClient

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from hpc import literal_proxy_utils as lp  # noqa: E402


# --------------------------------------------------------------------------- #
# 1. Pure helpers
# --------------------------------------------------------------------------- #
def test_literal_proxy_endpoint_default():
    assert lp.literal_proxy_endpoint() == "http://127.0.0.1:8010/v1"
    assert lp.literal_proxy_endpoint("0.0.0.0", 9001) == "http://0.0.0.0:9001/v1"


def test_default_port_avoids_vllm():
    # vLLM / datagen bind 8000; the co-located proxy must not collide.
    assert lp.DEFAULT_LITERAL_PROXY_PORT != 8000


def test_upstream_origin_strips_v1_base():
    # Harbor's RecordProxy re-appends the full request path (/v1/chat/completions),
    # so the base it receives must be origin-only; a .../v1 base would double the
    # segment (/v1/v1/...) and vLLM 404s. upstream_origin strips path/query.
    assert lp.upstream_origin("http://127.0.0.1:8000/v1") == "http://127.0.0.1:8000"
    assert lp.upstream_origin("http://127.0.0.1:8000/v1/") == "http://127.0.0.1:8000"
    assert lp.upstream_origin("http://10.0.0.5:8000") == "http://10.0.0.5:8000"
    # non-absolute URLs must fail loud, not silently forward a bad base.
    with pytest.raises(ValueError):
        lp.upstream_origin("127.0.0.1:8000/v1")


def test_literal_log_path_sanitizes_and_places_under_logs(tmp_path):
    p = lp.literal_log_path(tmp_path, "rl-qwen3_8b/run.1")
    assert p.parent == tmp_path / "logs"
    assert p.name == "rl-qwen3_8b-run.1_literal.jsonl"
    # empty/None job name -> a stable 'job' slug
    assert lp.literal_log_path(tmp_path, "").name == "job_literal.jsonl"


# --------------------------------------------------------------------------- #
# 2. FLAG-OFF PARITY — null context manager, byte-identical endpoint
# --------------------------------------------------------------------------- #
def test_maybe_serve_disabled_yields_upstream_unchanged(monkeypatch, tmp_path):
    upstream = "http://10.0.0.1:8000/v1"

    # If the disabled path ever tries to serve, fail loudly.
    def _boom(*a, **k):
        raise AssertionError("serve_record_proxy must NOT be called when disabled")

    monkeypatch.setattr(lp, "serve_record_proxy", _boom)

    with lp.maybe_serve_literal_proxy(
        False, upstream, experiments_dir=tmp_path, job_name="job"
    ) as effective:
        # identity: the exact same object flows through unchanged
        assert effective is upstream


# --------------------------------------------------------------------------- #
# 3. FLAG-ON gating — routes through serve_record_proxy (stubbed, no socket)
# --------------------------------------------------------------------------- #
def test_maybe_serve_enabled_routes_through_serve(monkeypatch, tmp_path):
    upstream = "http://10.0.0.1:8000/v1"
    calls: dict = {}

    @contextlib.contextmanager
    def _fake_serve(up, log_path, *, host=lp.DEFAULT_LITERAL_PROXY_HOST,
                    port=lp.DEFAULT_LITERAL_PROXY_PORT, **kw):
        calls["upstream"] = up
        calls["log_path"] = Path(log_path)
        yield lp.literal_proxy_endpoint(host, port)

    monkeypatch.setattr(lp, "serve_record_proxy", _fake_serve)

    with lp.maybe_serve_literal_proxy(
        True, upstream, experiments_dir=tmp_path, job_name="myjob"
    ) as effective:
        assert effective == lp.literal_proxy_endpoint()
    # upstream forwarded verbatim; log path derived under <exp>/logs/
    assert calls["upstream"] == upstream
    assert calls["log_path"] == tmp_path / "logs" / "myjob_literal.jsonl"


# --------------------------------------------------------------------------- #
# 4. FLAG-ON surface — harbor's RecordProxy.app() injects + records, in-process
# --------------------------------------------------------------------------- #
def _make_stub_vllm() -> FastAPI:
    """A stub OpenAI-compatible server that echoes literal token fields."""
    stub = FastAPI()

    @stub.post("/v1/chat/completions")
    async def _chat(request: Request):
        body = await request.json()
        # Record what the proxy forwarded so the test can assert injection.
        return JSONResponse(
            {
                "id": "resp-1",
                "model": body.get("model", "m"),
                "prompt_token_ids": [10, 11, 12],
                "choices": [
                    {
                        "provider_specific_fields": {"token_ids": [13, 14]},
                        "logprobs": {"content": [{"logprob": -0.1}, {"logprob": -0.2}]},
                        "_echoed_request": body,
                    }
                ],
            }
        )

    return stub


def test_record_proxy_app_injects_and_records(tmp_path, monkeypatch):
    from harbor.literal.proxy import RecordProxy

    log_path = tmp_path / "literal.jsonl"
    proxy = RecordProxy("http://vllm.local", log_path, timeout=10.0)

    # Stub the upstream client with an ASGITransport to the fake vLLM (no socket).
    stub_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=_make_stub_vllm()),
        base_url="http://vllm.local",
    )

    async def _fake_get_client():
        return stub_client

    monkeypatch.setattr(proxy, "_get_client", _fake_get_client)

    app = proxy.app()
    with TestClient(app) as client:
        r = client.post(
            "/v1/chat/completions",
            json={"model": "glm-4.6", "messages": []},
        )

    assert r.status_code == 200
    payload = r.json()
    assert payload["id"] == "resp-1"

    # The proxy injected literal params into the forwarded request body.
    echoed = payload["choices"][0]["_echoed_request"]
    assert echoed["return_token_ids"] is True
    assert echoed["logprobs"] is True

    # literal.jsonl recorded the extracted token ids + logprobs.
    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["literal"]["prompt_token_ids"] == [10, 11, 12]
    assert entry["literal"]["completion_token_ids"] == [13, 14]
    assert entry["literal"]["logprobs"] == [-0.1, -0.2]


def test_record_proxy_from_v1_base_does_not_double_v1(tmp_path, monkeypatch):
    """Regression: the launcher's .../v1 base must NOT forward to /v1/v1/... (404).

    Production hands ``serve_record_proxy`` the agent-facing vLLM ``.../v1`` base;
    harbor's RecordProxy re-appends the full request path, so we strip to origin
    first (``upstream_origin``). Without that strip the forwarded URL is
    ``/v1/v1/chat/completions`` and the stub (like real vLLM) 404s — the exact
    live-ingress failure this fixes.
    """
    from harbor.literal.proxy import RecordProxy

    # Build RecordProxy the way serve_record_proxy does: strip the /v1 base to origin.
    proxy = RecordProxy(
        lp.upstream_origin("http://vllm.local/v1"), tmp_path / "l.jsonl", timeout=10.0
    )
    stub_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=_make_stub_vllm()),
        base_url="http://vllm.local",
    )

    async def _fake_get_client():
        return stub_client

    monkeypatch.setattr(proxy, "_get_client", _fake_get_client)

    with TestClient(proxy.app()) as client:
        r = client.post("/v1/chat/completions", json={"model": "m", "messages": []})

    # 200 proves the forwarded path is /v1/chat/completions (single /v1). If the
    # base had kept its /v1, this would forward to /v1/v1/... and 404.
    assert r.status_code == 200
    assert r.json()["id"] == "resp-1"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
