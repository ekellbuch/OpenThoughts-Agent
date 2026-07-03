"""Stage 4 wiring unit tests — controller-mode api_base + key injection.

Proves the controller-ingress wiring lever WITHOUT launching a job (gate (b) of
Stage 4): the shared helpers build the right ``/proxy/<name>/v1`` api_base and
inject the key env, and merge_agent_kwargs carries the api_key placeholder in
controller mode while staying byte-identical (no api_key) on the legacy path.

All tokens are in-process dummies; no real secret is referenced.

Run:
    .venv/bin/python -m pytest tests/hpc/test_ingress_wiring.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.testclient import TestClient

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from hpc.harbor_utils import build_endpoint_meta, merge_agent_kwargs  # noqa: E402
from hpc.ingress_utils import (  # noqa: E402
    ADVERTISE_HOST_ENV,
    DEFAULT_VLLM_PORT,
    INGRESS_KEY_ENV,
    INGRESS_KEY_PLACEHOLDER,
    build_controller_api_base,
    build_controller_endpoint_meta,
    controller_api_base_for_job,
    controller_endpoint_name,
    controller_registration_plan,
    controller_upstream_address,
    inject_ingress_agent_key,
    register_controller_endpoint,
)
from hpc.literal_proxy_utils import DEFAULT_LITERAL_PROXY_PORT  # noqa: E402


def test_controller_endpoint_name_sanitizes():
    assert controller_endpoint_name("rl-qwen3_8b/run.1") == "otagent.rl-qwen3_8b-run.1"
    assert controller_endpoint_name(None) == "otagent.job"
    assert controller_endpoint_name("") == "otagent.job"
    # single path segment (no slashes) — required by the /proxy/<name>/ route
    assert "/" not in controller_endpoint_name("a/b/c")


def test_build_controller_api_base_scheme_handling():
    assert (
        build_controller_api_base("ingress.example", "otagent.job1")
        == "https://ingress.example/proxy/otagent.job1/v1"
    )
    # explicit scheme preserved; trailing slash stripped
    assert (
        build_controller_api_base("http://10.0.0.1:8443/", "ep")
        == "http://10.0.0.1:8443/proxy/ep/v1"
    )


def test_controller_api_base_for_job():
    assert (
        controller_api_base_for_job("ingress.example", "myjob")
        == "https://ingress.example/proxy/otagent.myjob/v1"
    )


def test_inject_ingress_agent_key():
    env = {INGRESS_KEY_ENV: "dummy-key-123"}
    assert inject_ingress_agent_key(env) is True
    assert env["OPENAI_API_KEY"] == "dummy-key-123"
    assert env["LLM_API_KEY"] == "dummy-key-123"
    # no-op when unset
    empty: dict = {}
    assert inject_ingress_agent_key(empty) is False
    assert empty == {}


def test_build_endpoint_meta_api_key_dormant_by_default():
    # legacy path: no api_key arg -> byte-identical dict (no api_key key)
    meta = build_endpoint_meta("http://host:8000/v1")
    assert "api_key" not in meta
    assert meta["api_base"] == "http://host:8000/v1"
    # controller path: api_key carried through
    meta2 = build_endpoint_meta("http://host:8000/v1", api_key=INGRESS_KEY_PLACEHOLDER)
    assert meta2["api_key"] == INGRESS_KEY_PLACEHOLDER


def test_merge_agent_kwargs_carries_api_key_only_in_controller_mode():
    cfg = {"agents": [{"name": "qwen-code", "kwargs": {"model_name": "m"}}]}

    # legacy/pinggy endpoint_meta (no api_key) -> agent_kwargs has NO api_key
    legacy_meta = build_endpoint_meta("http://host:8000/v1")
    merged_legacy, _ = merge_agent_kwargs(cfg, agent_name="qwen-code", endpoint_meta=legacy_meta)
    assert "api_key" not in merged_legacy
    assert merged_legacy["api_base"] == "http://host:8000/v1"

    # controller endpoint_meta -> api_base is /proxy/<name>/v1 + api_key placeholder
    ctrl_meta = build_controller_endpoint_meta("ingress.example", "otagent.myjob")
    merged_ctrl, _ = merge_agent_kwargs(cfg, agent_name="qwen-code", endpoint_meta=ctrl_meta)
    assert merged_ctrl["api_base"] == "https://ingress.example/proxy/otagent.myjob/v1"
    assert merged_ctrl["api_key"] == INGRESS_KEY_PLACEHOLDER
    # metrics endpoint intentionally absent in controller mode (sidecar blocks /metrics)
    assert "metrics_endpoint" not in ctrl_meta


def test_eval_listener_to_env_controller_additive():
    """SbatchParams.to_env(): pinggy default byte-identical; controller adds EVAL_INGRESS_*."""
    from eval.unified_eval_listener import SbatchParams

    # default (pinggy) mode: no EVAL_INGRESS_* keys
    default_env = SbatchParams(pinggy_url="x.a.pinggy.link", pinggy_token="tok").to_env()
    assert not any(k.startswith("EVAL_INGRESS_") for k in default_env)
    assert default_env["EVAL_PINGGY_URL"] == "x.a.pinggy.link"

    # controller mode: EVAL_INGRESS_* emitted
    ctrl_env = SbatchParams(ingress_mode="controller", ingress_host="ingress.example").to_env()
    assert ctrl_env["EVAL_INGRESS_MODE"] == "controller"
    assert ctrl_env["EVAL_INGRESS_HOST"] == "ingress.example"


# --------------------------------------------------------------------------- #
# Endpoint registration helpers (shared by the controller + literal-combo path)
# --------------------------------------------------------------------------- #
class _FakeRegistrar:
    """Records register(name, address, metadata) calls; returns a fixed id."""

    def __init__(self):
        self.calls = []

    def register(self, name, address, metadata=None):
        self.calls.append({"name": name, "address": address, "metadata": metadata})
        return "endpoint-id-xyz"


def test_controller_upstream_address_uses_advertise_host():
    # in-cluster: IRIS_ADVERTISE_HOST wins, on the requested port
    env = {ADVERTISE_HOST_ENV: "10.16.4.7"}
    assert controller_upstream_address(8010, env=env) == "http://10.16.4.7:8010"
    # off-cluster (unset) -> loopback fallback
    assert controller_upstream_address(8000, env={}) == "http://127.0.0.1:8000"


def test_register_controller_endpoint_uses_injected_registrar():
    reg = _FakeRegistrar()
    eid = register_controller_endpoint(
        "otagent.myjob", "http://10.0.0.1:8010", registrar=reg, metadata={"k": "v"}
    )
    assert eid == "endpoint-id-xyz"
    assert reg.calls == [
        {"name": "otagent.myjob", "address": "http://10.0.0.1:8010", "metadata": {"k": "v"}}
    ]


def test_register_controller_endpoint_no_registrar_returns_none(monkeypatch):
    # off-cluster / iris unavailable: best-effort no-op, never raises.
    monkeypatch.setattr(
        "hpc.ingress_utils._default_endpoint_registrar", lambda: None
    )
    assert register_controller_endpoint("otagent.j", "http://h:8000") is None


def test_controller_registration_plan_plain_registers_vllm_port():
    """Plain controller: register raw vLLM:8000; api_base is the /proxy/<name>/v1 URL."""
    name, address, api_base = controller_registration_plan(
        "ingress.example",
        "myjob",
        record_literal=False,
        proxy_port=DEFAULT_LITERAL_PROXY_PORT,
        env={ADVERTISE_HOST_ENV: "10.0.0.5"},
    )
    assert name == "otagent.myjob"
    assert address == f"http://10.0.0.5:{DEFAULT_VLLM_PORT}"
    assert address.endswith(":8000")
    assert api_base == "https://ingress.example/proxy/otagent.myjob/v1"


def test_controller_registration_plan_literal_registers_proxy_port():
    """COMBINED path: the REGISTERED address is the RecordProxy's port, NOT vLLM's.

    api_base + endpoint name are IDENTICAL to the plain path (only the address
    differs), so the controller URL the agent targets is unchanged.
    """
    name, address, api_base = controller_registration_plan(
        "ingress.example",
        "myjob",
        record_literal=True,
        proxy_port=DEFAULT_LITERAL_PROXY_PORT,
        env={ADVERTISE_HOST_ENV: "10.0.0.5"},
    )
    assert name == "otagent.myjob"
    # the proxy's address is registered, never raw vLLM:8000
    assert address == f"http://10.0.0.5:{DEFAULT_LITERAL_PROXY_PORT}"
    assert address.endswith(":8010")
    assert ":8000" not in address
    # api_base unchanged vs the plain path
    assert api_base == "https://ingress.example/proxy/otagent.myjob/v1"


def test_combined_path_registers_proxy_and_sets_api_base_and_key():
    """End-to-end (in-process) proof of the record_literal x controller wiring.

    Mirrors exactly what the launcher does when BOTH flags are set: compute the
    registration plan, register via the (fake) registrar, build the agent kwargs.
    Asserts: registered address is the PROXY's (not vLLM's), api_base is the
    /proxy/<name>/v1 URL, api_key == the IRIS_INGRESS_API_KEY placeholder.
    """
    reg = _FakeRegistrar()
    name, address, api_base = controller_registration_plan(
        "ingress.example",
        "myjob",
        record_literal=True,
        proxy_port=DEFAULT_LITERAL_PROXY_PORT,
        env={ADVERTISE_HOST_ENV: "10.0.0.5"},
    )
    register_controller_endpoint(name, address, registrar=reg)

    # the endpoint registered points at the RecordProxy, not raw vLLM
    assert reg.calls[0]["address"].endswith(f":{DEFAULT_LITERAL_PROXY_PORT}")
    assert not reg.calls[0]["address"].endswith(":8000")

    # the agent sees the controller /proxy/<name>/v1 URL + the ingress key placeholder
    meta = build_controller_endpoint_meta("ingress.example", name)
    merged, _ = merge_agent_kwargs(
        {"agents": [{"name": "qwen-code", "kwargs": {"model_name": "m"}}]},
        agent_name="qwen-code",
        endpoint_meta=meta,
    )
    assert merged["api_base"] == "https://ingress.example/proxy/otagent.myjob/v1"
    assert merged["api_base"] == api_base
    assert merged["api_key"] == INGRESS_KEY_PLACEHOLDER
    assert INGRESS_KEY_PLACEHOLDER == "${" + INGRESS_KEY_ENV + "}"


def test_record_proxy_accepts_unauthenticated_calls():
    """The registered upstream (RecordProxy) must accept NO-bearer in-cluster calls.

    EndpointProxy strips Authorization before forwarding upstream, so the
    RecordProxy — like raw vLLM today — must serve requests with no bearer.
    Exercised in-process via httpx.ASGITransport (no socket/GPU/controller).
    """
    from harbor.literal.proxy import RecordProxy

    stub = FastAPI()

    @stub.post("/v1/chat/completions")
    async def _chat(request: Request):  # noqa: ANN001
        return JSONResponse({"id": "r1", "model": "m", "choices": [{}]})

    import tempfile

    with tempfile.TemporaryDirectory() as td:
        proxy = RecordProxy("http://vllm.local", Path(td) / "literal.jsonl", timeout=10.0)
        stub_client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=stub), base_url="http://vllm.local"
        )

        async def _fake_get_client():
            return stub_client

        proxy._get_client = _fake_get_client  # type: ignore[assignment]
        with TestClient(proxy.app()) as client:
            # NO Authorization header — must still forward + 200.
            r = client.post("/v1/chat/completions", json={"model": "m", "messages": []})
        assert r.status_code == 200
        assert "authorization" not in {k.lower() for k in r.request.headers}


def test_notimplementederror_guard_removed_from_launchers():
    """The record_literal x controller NotImplementedError guard is gone from both launchers."""
    guard = "record_literal + ingress_mode=controller is not supported"
    for fname in ("rl_launch_utils.py", "datagen_launch_utils.py"):
        src = (_REPO_ROOT / "hpc" / fname).read_text()
        assert guard not in src, f"stale NotImplementedError guard still in hpc/{fname}"
        assert "NotImplementedError(\n" not in src or guard not in src
