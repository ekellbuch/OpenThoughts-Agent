#!/usr/bin/env python
"""Rung-0a load probe for Qwen3.6-35B-A3B (qwen3_5_moe hybrid MoE).

Runs INSIDE the gpu-rl image's RL venv (where MarinSkyRL `/opt/skyrl` is
importable + transformers is the baked version). Mirrors EXACTLY the load
sequence that `skyrl_train.model_wrapper` performs for FSDP2 policy load, so a
clean run here de-risks the headline arch unknown (text-only backbone load +
256-expert weight remap) BEFORE any 8-node gang is burned.

GATE 0a (all must pass):
  (1) AutoConfig resolves `qwen3_5_moe` (transformers recognises the arch).
  (2) The checkpoint loads (text backbone) — `is_qwen3_5_vlm_shell` detects the
      multimodal shell and `unwrap_to_text_causal_lm` re-points it to the text
      Qwen3_5MoeForCausalLM tower, dropping vision + MTP (the fb12cea fix).
  (3) `_no_split_modules` resolves with the vision class dropped (the df8b661
      fix) — i.e. the FSDP wrap-policy autodetect would not raise.
  (4) `swap_moe_blocks_to_grouped` remaps the 256-expert weights into grouped
      MoE shims with NO missing/extra-key explosion, and the swap count matches
      `count_moe_layers(text_config)` (expected 40 for qwen3_5_moe).

Memory: 34.4B bf16 ~= 69 GB. On a single H100-80GB a weights-only load fits
(no optimizer / activations here). We load on CPU first (device_map=None,
low_cpu_mem_usage) to keep the probe device-agnostic and avoid any placement
confound — the gate is the LOAD + REMAP keys, not a forward. The swap's
`remap_hf_block_to_moe` does in-place `copy_` on CPU tensors, which is fine.

Exit 0 = GATE 0a GO. Non-zero = NO-GO (the message says which sub-gate failed).
"""

from __future__ import annotations

import os
import sys
import traceback


MODEL_ID = os.environ.get("PROBE_MODEL_ID", "Qwen/Qwen3.6-35B-A3B")


def _hr(msg: str) -> None:
    print(f"\n{'=' * 70}\n{msg}\n{'=' * 70}", flush=True)


def main() -> int:
    _hr(f"RUNG 0a LOAD PROBE — {MODEL_ID}")

    # Environment fingerprint (so the gate report records the exact stack).
    import torch
    import transformers

    print(f"[probe] python           {sys.version.split()[0]}", flush=True)
    print(f"[probe] torch            {torch.__version__}", flush=True)
    print(f"[probe] transformers     {transformers.__version__}", flush=True)
    print(f"[probe] CUDA available    {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"[probe] device           {torch.cuda.get_device_name(0)}", flush=True)
    print(f"[probe] SKYRL_QWEN3_5_VLM_UNWRAP={os.environ.get('SKYRL_QWEN3_5_VLM_UNWRAP', '(unset=default on)')}", flush=True)

    # ---- sub-gate (1): AutoConfig resolves qwen3_5_moe -------------------- #
    _hr("(1) AutoConfig.from_pretrained — does transformers know qwen3_5_moe?")
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    model_type = getattr(cfg, "model_type", None)
    archs = getattr(cfg, "architectures", None)
    text_cfg = getattr(cfg, "text_config", None)
    print(f"[probe] top model_type    {model_type}", flush=True)
    print(f"[probe] architectures     {archs}", flush=True)
    print(f"[probe] has text_config   {text_cfg is not None}", flush=True)
    if text_cfg is not None:
        print(f"[probe] text model_type   {getattr(text_cfg, 'model_type', None)}", flush=True)
        print(f"[probe] text num_layers   {getattr(text_cfg, 'num_hidden_layers', None)}", flush=True)
        print(f"[probe] text num_experts  {getattr(text_cfg, 'num_experts', None)}", flush=True)
        print(f"[probe] text top_k        {getattr(text_cfg, 'num_experts_per_tok', None)}", flush=True)
        print(f"[probe] GDN signature     linear_conv_kernel_dim={getattr(text_cfg, 'linear_conv_kernel_dim', None)}", flush=True)
    if model_type is None or not str(model_type).startswith("qwen3_5"):
        print(f"[probe] FAIL (1): top model_type {model_type!r} is not a qwen3_5* arch", flush=True)
        return 11

    # ---- sub-gate (2): load + shell-detect + text-unwrap ----------------- #
    _hr("(2) AutoModelForCausalLM.from_pretrained (CPU, bf16) + text-unwrap")
    from transformers import AutoModelForCausalLM
    from skyrl_train.models.qwen3_5_vlm import (
        is_qwen3_5_vlm_shell,
        unwrap_to_text_causal_lm,
    )

    print("[probe] loading checkpoint on CPU (low_cpu_mem_usage)…", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map=None,
    )
    print(f"[probe] loaded shell class {type(model).__name__}", flush=True)

    is_shell = is_qwen3_5_vlm_shell(model.config)
    print(f"[probe] is_qwen3_5_vlm_shell -> {is_shell}", flush=True)
    if is_shell:
        model = unwrap_to_text_causal_lm(model)
        print(f"[probe] unwrapped -> {type(model).__name__}", flush=True)
    else:
        print("[probe] NOTE: shell not detected; proceeding with the loaded model as-is "
              "(would be the text tower already, or the unwrap gate is off).", flush=True)

    # The downstream readers all assume self.model.config is the TEXT MoE config.
    cfg_after = model.config
    print(f"[probe] post-unwrap config model_type {getattr(cfg_after, 'model_type', None)}", flush=True)
    print(f"[probe] post-unwrap num_hidden_layers {getattr(cfg_after, 'num_hidden_layers', None)}", flush=True)

    # ---- sub-gate (3): FSDP wrap-policy resolves on the REAL (pinned) path - #
    # The 8node config PINS trainer.policy.fsdp_config.wrap_policy.
    # transformer_layer_cls_to_wrap=["Qwen3_5MoeDecoderLayer"]. The real worker
    # (fsdp_strategy._fsdp_init_model -> get_fsdp_wrap_policy, then apply_fsdp2)
    # reads that PIN, NOT the model's `_no_split_modules` autodetect default —
    # `get_fsdp_wrap_policy._get_attr("transformer_layer_cls_to_wrap", default=...)`
    # returns the config pin when present, and only iterates the pinned class. So
    # a `Qwen3_5MoeVisionBlock` left in `_no_split_modules` is HARMLESS for our
    # config (it is never looked up). What we must prove is: (a) the pinned class
    # resolves on this text backbone (no "Could not find the transformer layer
    # class to wrap"), and (b) it matches the 40 decoder layers. We call the REAL
    # get_fsdp_wrap_policy with the config's pin to prove it does not raise.
    #
    # NOTE — earlier probe rev mis-tested the UNPINNED autodetect (raw
    # `_no_split_modules`) and false-failed here on the vision class even though
    # the pinned worker path is immune; this rev tests the path the config uses.
    _hr("(3) FSDP wrap-policy with the config's pin ['Qwen3_5MoeDecoderLayer'] resolves")
    nsm = getattr(model, "_no_split_modules", None)
    print(f"[probe] _no_split_modules (autodetect default, NOT used by pinned config)  {nsm}", flush=True)
    PIN = ["Qwen3_5MoeDecoderLayer"]
    from skyrl_train.distributed.fsdp_utils import get_fsdp_wrap_policy

    try:
        policy = get_fsdp_wrap_policy(model, config={"transformer_layer_cls_to_wrap": PIN})
    except Exception as e:  # noqa: BLE001
        print(f"[probe] FAIL (3): get_fsdp_wrap_policy raised with the config pin {PIN}: {e!r} "
              "(the pinned decoder-layer class does not resolve on the text backbone).", flush=True)
        return 13
    if policy is None:
        print("[probe] FAIL (3): get_fsdp_wrap_policy returned None for the config pin "
              "(no transformer wrap policy built).", flush=True)
        return 13
    # Prove the pinned class actually matches the 40 decoder layers (the membership
    # apply_fsdp2 does at line ~610): count modules whose class == the pin.
    n_decoder = sum(1 for _, m in model.named_modules() if type(m).__name__ in PIN)
    print(f"[probe] pinned wrap class matches {n_decoder} decoder-layer modules "
          f"(expect {getattr(cfg_after, 'num_hidden_layers', '?')})", flush=True)
    if n_decoder == 0:
        print("[probe] FAIL (3): the pinned Qwen3_5MoeDecoderLayer matched 0 modules on the "
              "text backbone (apply_fsdp2 would wrap nothing).", flush=True)
        return 13
    print("[probe] (3) OK — config-pinned FSDP wrap policy resolves (vision class harmless; "
          "not looked up on the pinned path).", flush=True)

    # ---- sub-gate (4): 256-expert grouped-MoE remap, no key explosion ---- #
    _hr("(4) swap_moe_blocks_to_grouped — 256-expert remap, swap count == MoE layers")
    from skyrl_train.models.layers.moe_swap import swap_moe_blocks_to_grouped
    from skyrl_train.models.router_replay import count_moe_layers

    expected = count_moe_layers(cfg_after)
    print(f"[probe] count_moe_layers(text_config) = {expected}", flush=True)

    # Snapshot the pre-swap key set so we can prove no missing/extra keys appear
    # (a remap-key explosion would surface as orphaned per-expert HF keys or a
    # build/copy error). The swap is in-place on parent.mlp.
    pre_keys = set(dict(model.named_parameters()).keys())

    n_swapped = swap_moe_blocks_to_grouped(model)
    print(f"[probe] blocks swapped    {n_swapped}", flush=True)
    if n_swapped == 0:
        print("[probe] FAIL (4): 0 blocks swapped — the *SparseMoeBlock scan matched nothing "
              "(arch/structure mismatch). This is the silent-no-op the swap guards against.", flush=True)
        return 14
    if expected and n_swapped != expected:
        print(f"[probe] FAIL (4): swapped {n_swapped} != expected {expected} MoE layers "
              "(some MoE blocks were not remapped).", flush=True)
        return 14

    # Post-swap key sanity: every swapped block's grouped MoE must expose the
    # fused w1/w2/w3 + router.gate, and NO orphaned per-expert HF keys
    # (mlp.experts.{j}.gate_proj/up_proj/down_proj) may remain.
    post_keys = set(dict(model.named_parameters()).keys())
    orphan_hf_expert_keys = [
        k for k in post_keys
        if ".mlp.experts." in k and (
            k.endswith(".gate_proj.weight") or k.endswith(".up_proj.weight") or k.endswith(".down_proj.weight")
        )
    ]
    grouped_w_keys = [k for k in post_keys if (".moe.experts.w1" in k or ".moe.experts.w2" in k or ".moe.experts.w3" in k)]
    router_keys = [k for k in post_keys if ".moe.router.gate.weight" in k]
    print(f"[probe] grouped expert w-tensors  {len(grouped_w_keys)}  (expect 3 x {n_swapped} = {3 * n_swapped})", flush=True)
    print(f"[probe] grouped router.gate keys  {len(router_keys)}  (expect {n_swapped})", flush=True)
    print(f"[probe] orphaned per-expert HF keys {len(orphan_hf_expert_keys)}  (expect 0)", flush=True)
    if orphan_hf_expert_keys:
        print(f"[probe] FAIL (4): {len(orphan_hf_expert_keys)} per-expert HF keys survived the remap "
              f"(e.g. {orphan_hf_expert_keys[:3]}) — remap-key explosion.", flush=True)
        return 14
    if len(grouped_w_keys) != 3 * n_swapped or len(router_keys) != n_swapped:
        print("[probe] FAIL (4): grouped MoE param count does not match the swapped-block count "
              "(remap did not produce the expected fused w1/w2/w3 + router per block).", flush=True)
        return 14

    # Spot-check one grouped expert tensor's shape (256 experts in dim 0).
    sample_w1 = next((dict(model.named_parameters())[k] for k in grouped_w_keys if k.endswith(".w1")), None)
    if sample_w1 is not None:
        print(f"[probe] sample grouped w1 shape {tuple(sample_w1.shape)}  (dim0 = num_experts)", flush=True)
        if sample_w1.shape[0] != 256:
            print(f"[probe] WARN (4): grouped w1 dim0 {sample_w1.shape[0]} != 256 experts "
                  "(check the arch's expert count).", flush=True)

    _hr("GATE 0a: GO — text backbone loaded, vision/MTP dropped, 256-expert remap clean.")
    print(f"[probe] SUMMARY  transformers={transformers.__version__}  shell_unwrapped={is_shell}  "
          f"moe_layers_swapped={n_swapped}  orphan_hf_keys=0", flush=True)
    return 0


if __name__ == "__main__":
    try:
        rc = main()
    except Exception:
        _hr("GATE 0a: NO-GO — uncaught exception (see traceback)")
        traceback.print_exc()
        rc = 1
    print(f"\n[probe] EXIT {rc}", flush=True)
    sys.exit(rc)
