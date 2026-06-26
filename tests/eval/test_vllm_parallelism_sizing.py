"""Tests for size-aware, node-filling vLLM TP/DP defaults in the eval listener.

Fix (2026-06-26): a <=14B eval served TP=1/DP=1 used only 1 of a 4-GPU node's GPUs
(~25% throughput). The listener now defaults to FILL the node by model size:
<=14B -> TP1/DP(gpus_per_node); >14B -> TP2/DP(gpus_per_node//2). An explicit
baseline-config TP/DP still wins per-field.
"""
import importlib

import pytest

mod = importlib.import_module("eval.unified_eval_listener")


@pytest.fixture
def leonardo(monkeypatch):
    """4-GPU A100 node."""
    monkeypatch.setattr(mod, "_CLUSTER_CONFIG", {"hardware": {"gpus_per_node": 4}})


# ---- infer_size_tp_dp (pure) -------------------------------------------------

def test_infer_small_fills_node():
    assert mod.infer_size_tp_dp("foo-8b", 4) == (1, 4)
    assert mod.infer_size_tp_dp("foo-14b", 4) == (1, 4)   # boundary inclusive


def test_infer_large_tp2():
    assert mod.infer_size_tp_dp("foo-32b", 4) == (2, 2)
    assert mod.infer_size_tp_dp("foo-30b-a3b", 4) == (2, 2)  # MoE: largest token wins


def test_infer_whole_node_one_gpu():
    assert mod.infer_size_tp_dp("foo-8b", 1) == (1, 1)
    assert mod.infer_size_tp_dp("foo-32b", 1) == (2, 1)


# ---- get_vllm_env_overrides (size-default + cfg precedence) -------------------

def test_8b_in_name_fills_4gpu_node(leonardo, monkeypatch):
    monkeypatch.setattr(mod, "resolve_base_model_name", lambda m: None)
    env = mod.get_vllm_env_overrides("laion/run-8b", {})
    assert env["EVAL_VLLM_TENSOR_PARALLEL_SIZE"] == "1"
    assert env["EVAL_VLLM_DATA_PARALLEL_SIZE"] == "4"


def test_a1_finetune_resolves_size_via_base_model(leonardo, monkeypatch):
    # a1-* names carry no size token; the Supabase base model (Qwen3-8B) does.
    monkeypatch.setattr(mod, "resolve_base_model_name", lambda m: "Qwen/Qwen3-8B")
    env = mod.get_vllm_env_overrides("DCAgent/a1-ghactions", {})
    assert env["EVAL_VLLM_TENSOR_PARALLEL_SIZE"] == "1"
    assert env["EVAL_VLLM_DATA_PARALLEL_SIZE"] == "4"


def test_32b_tp2_dp2(leonardo, monkeypatch):
    monkeypatch.setattr(mod, "resolve_base_model_name", lambda m: None)
    env = mod.get_vllm_env_overrides("laion/run-32b", {})
    assert env["EVAL_VLLM_TENSOR_PARALLEL_SIZE"] == "2"
    assert env["EVAL_VLLM_DATA_PARALLEL_SIZE"] == "2"


def test_explicit_cfg_tp_wins_dp_fills_remainder(leonardo, monkeypatch):
    monkeypatch.setattr(mod, "resolve_base_model_name", lambda m: None)
    cfg = {"laion/big": {"tensor_parallel_size": 4}}
    env = mod.get_vllm_env_overrides("laion/big", cfg)
    assert env["EVAL_VLLM_TENSOR_PARALLEL_SIZE"] == "4"
    # dp = 4 // 4 = 1 -> not emitted (single replica)
    assert "EVAL_VLLM_DATA_PARALLEL_SIZE" not in env


def test_explicit_cfg_dp_wins(leonardo, monkeypatch):
    monkeypatch.setattr(mod, "resolve_base_model_name", lambda m: None)
    cfg = {"laion/x-8b": {"data_parallel_size": 2}}
    env = mod.get_vllm_env_overrides("laion/x-8b", cfg)
    assert env["EVAL_VLLM_TENSOR_PARALLEL_SIZE"] == "1"   # size default
    assert env["EVAL_VLLM_DATA_PARALLEL_SIZE"] == "2"     # cfg override


def test_unknown_size_treated_small(leonardo, monkeypatch):
    monkeypatch.setattr(mod, "resolve_base_model_name", lambda m: None)
    env = mod.get_vllm_env_overrides("laion/mystery-run", {})
    assert env["EVAL_VLLM_TENSOR_PARALLEL_SIZE"] == "1"
    assert env["EVAL_VLLM_DATA_PARALLEL_SIZE"] == "4"


def test_tacc_whole_node_one_gpu_unchanged(monkeypatch):
    monkeypatch.setattr(mod, "_CLUSTER_CONFIG", {"hardware": {"gpus_per_node": 1}})
    monkeypatch.setattr(mod, "resolve_base_model_name", lambda m: "Qwen/Qwen3-8B")
    env = mod.get_vllm_env_overrides("DCAgent/a1-x", {})
    assert env["EVAL_VLLM_TENSOR_PARALLEL_SIZE"] == "1"
    assert "EVAL_VLLM_DATA_PARALLEL_SIZE" not in env  # dp=1, single-GPU whole-node
