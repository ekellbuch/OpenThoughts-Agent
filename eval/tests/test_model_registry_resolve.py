#!/usr/bin/env python
"""Stage-1 unit tests for the shared model-config registry loader/resolver.

Validates load_model_registry's behavior in isolation against a hand-written fixture:
  * group expansion + per-model override (mirrors the legacy loader),
  * the 4-tier precedence exact+variant > exact > group > pattern,
  * shallow-merge variant semantics (variant replaces a named top-level key wholesale),
  * the G3 STRICT-SUPERSET property: a profile with NO matching variant resolves to the
    base entry byte-identical to the legacy exact/group/pattern result.

Run: /path/to/otagent/bin/python eval/tests/test_model_registry_resolve.py
(self-contained; no pytest required — prints PASS/FAIL and exits nonzero on any failure.)
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
os.environ.setdefault("DCFT", str(_REPO_ROOT))

import eval.unified_eval_listener as uel  # noqa: E402


_FIXTURE = """
groups:
  - models:
      - "org/group-a"
      - "org/group-b"
    tensor_parallel_size: 2
    max_model_len: 32768
    extra_args: "--enable-prefix-caching"

models:
  # exact override on top of a group member (override wins)
  "org/group-a":
    trust_remote_code: true
  # plain exact entry with a variant block
  "org/has-variant":
    tensor_parallel_size: 2
    max_model_len: 32768
    trust_remote_code: true
    agent_kwargs:
      - 'extra_body={"chat_template_kwargs":{"enable_thinking":true}}'
    variants:
      gh200:
        tensor_parallel_size: 1
        max_model_len: 65536
  # exact entry, NO variants (superset control)
  "org/no-variant":
    tensor_parallel_size: 2
    conda_env: some-env

patterns:
  - match: "(?i)32[Bb]"
    trust_remote_code: true
    tensor_parallel_size: 4
    extra_args: "--enable-prefix-caching"
  - match: ".*"
    trust_remote_code: true
    tensor_parallel_size: 1
"""


def _load(profile):
    """Fresh load of the fixture under a given hardware_profile (resets the memo globals)."""
    uel._BASELINE_MODEL_CONFIGS = None
    uel._BASELINE_MODEL_PATTERNS = None
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(_FIXTURE)
        path = f.name
    try:
        configs = uel.load_model_registry(path, hardware_profile=profile)
        patterns = uel._BASELINE_MODEL_PATTERNS
    finally:
        os.unlink(path)
    return configs, patterns


_failures = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        _failures.append(f"{name}: {detail}")


def main() -> int:
    print("== Stage-1 registry resolver unit tests ==")

    # --- profile=None (no variant active): base entries, superset property ---
    cfg, pats = _load(None)

    # group expansion: group-b gets the group config verbatim
    check("group expansion: group-b inherits group config",
          cfg["org/group-b"] == {"tensor_parallel_size": 2, "max_model_len": 32768,
                                 "extra_args": "--enable-prefix-caching"},
          repr(cfg.get("org/group-b")))

    # per-model override merges on top of the group (override wins; group fields preserved)
    check("override-on-group: group-a keeps group fields + adds trust_remote_code",
          cfg["org/group-a"] == {"tensor_parallel_size": 2, "max_model_len": 32768,
                                 "extra_args": "--enable-prefix-caching",
                                 "trust_remote_code": True},
          repr(cfg.get("org/group-a")))

    # superset: a model WITH a variant block, but profile=None -> base entry, variants STRIPPED
    check("superset (profile=None): has-variant is the base entry, no `variants` key",
          cfg["org/has-variant"] == {
              "tensor_parallel_size": 2, "max_model_len": 32768, "trust_remote_code": True,
              "agent_kwargs": ['extra_body={"chat_template_kwargs":{"enable_thinking":true}}'],
          } and "variants" not in cfg["org/has-variant"],
          repr(cfg.get("org/has-variant")))

    # no-variant entry unchanged
    check("no-variant entry passes through unchanged",
          cfg["org/no-variant"] == {"tensor_parallel_size": 2, "conda_env": "some-env"},
          repr(cfg.get("org/no-variant")))

    # patterns preserved in order, variants stripped
    check("patterns: order preserved + 2 entries",
          [p.get("match") for p in pats] == ["(?i)32[Bb]", ".*"], repr(pats))
    check("patterns: no `variants` key leaks through",
          all("variants" not in p for p in pats))

    # --- profile=gh200 (variant active): shallow-merge over base ---
    cfg_g, _ = _load("gh200")
    check("variant active: has-variant TP overridden to 1 (variant wins per-field)",
          cfg_g["org/has-variant"]["tensor_parallel_size"] == 1,
          repr(cfg_g["org/has-variant"]))
    check("variant active: has-variant max_model_len overridden to 65536",
          cfg_g["org/has-variant"]["max_model_len"] == 65536,
          repr(cfg_g["org/has-variant"]))
    check("variant active: base intrinsic fields PRESERVED (trust_remote_code, agent_kwargs)",
          cfg_g["org/has-variant"].get("trust_remote_code") is True
          and cfg_g["org/has-variant"].get("agent_kwargs")
          == ['extra_body={"chat_template_kwargs":{"enable_thinking":true}}'],
          repr(cfg_g["org/has-variant"]))
    check("variant active: a model WITHOUT that variant is unchanged (superset)",
          cfg_g["org/no-variant"] == {"tensor_parallel_size": 2, "conda_env": "some-env"},
          repr(cfg_g["org/no-variant"]))

    # --- 4-tier precedence via the real resolvers on the merged dict ---
    # set a cluster shape so get_vllm_env_overrides has gpus_per_node
    uel._CLUSTER_CONFIG = {"hardware": {"gpus_per_node": 8}}
    uel.resolve_base_model_name = lambda m: None  # hermetic
    uel._BASE_MODEL_NAME_CACHE.clear()

    cfg_n, _ = _load(None)
    # exact wins over pattern: has-variant resolves to its exact TP (2), not the 32B pattern
    env_exact = uel.get_vllm_env_overrides("org/has-variant", cfg_n)
    check("precedence: exact entry wins over pattern",
          env_exact["EVAL_VLLM_TENSOR_PARALLEL_SIZE"] == "2", repr(env_exact))
    # pattern tier: an UNLISTED 32B name hits the (?i)32[Bb] pattern (TP 4)
    env_pat = uel.get_vllm_env_overrides("unlisted/Foo-32B", cfg_n)
    check("precedence: unlisted 32B name hits the 32B pattern (TP=4)",
          env_pat["EVAL_VLLM_TENSOR_PARALLEL_SIZE"] == "4", repr(env_pat))
    # catch-all .*: an unlisted small name hits the .* pattern (TP 1)
    env_catch = uel.get_vllm_env_overrides("unlisted/tiny", cfg_n)
    check("precedence: unlisted small name hits the .* catch-all (TP=1)",
          env_catch["EVAL_VLLM_TENSOR_PARALLEL_SIZE"] == "1", repr(env_catch))

    # exact+variant wins over exact: under gh200, has-variant resolves to TP 1
    cfg_gv, _ = _load("gh200")
    env_var = uel.get_vllm_env_overrides("org/has-variant", cfg_gv)
    check("precedence: exact+variant wins (gh200 -> TP=1, max_model_len 65536)",
          env_var["EVAL_VLLM_TENSOR_PARALLEL_SIZE"] == "1"
          and env_var.get("EVAL_VLLM_MAX_MODEL_LEN") == "65536", repr(env_var))

    print(f"\n== {len(_failures)} failure(s) ==" if _failures else "\n== ALL TESTS PASS ==")
    return 1 if _failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
