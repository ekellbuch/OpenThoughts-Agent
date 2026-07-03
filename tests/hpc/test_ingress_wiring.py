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

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from hpc.harbor_utils import build_endpoint_meta, merge_agent_kwargs  # noqa: E402
from hpc.ingress_utils import (  # noqa: E402
    INGRESS_KEY_ENV,
    INGRESS_KEY_PLACEHOLDER,
    build_controller_api_base,
    build_controller_endpoint_meta,
    controller_api_base_for_job,
    controller_endpoint_name,
    inject_ingress_agent_key,
)


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
