"""Regression test: the eval Iris launcher delivers thinking via the LIVE
nested ``extra_body`` agent-kwarg, carried by the preset's generic
``agent_kwargs`` list — never the dead bare ``enable_thinking=`` and never a
dedicated ``--enable-thinking`` flag (both removed).

Thinking is a chat-template kwarg — client-specified, server-applied (vLLM
reads ``request.chat_template_kwargs`` and forwards it to
``apply_chat_template``). The only delivery that actually reaches the request
body through terminus-2 is the nested form
``extra_body={"chat_template_kwargs":{"enable_thinking":true}}`` (terminus-2's
``extra_body`` param folds it into every LLM call). A BARE
``enable_thinking=true`` agent-kwarg has no terminus-2 parameter and is silently
discarded.

This pins ``EvalIrisLauncher._apply_preset`` so it can never regress to the dead
bare kwarg, and pins the on-disk catalog so a preset can never silently lose its
thinking agent-kwarg.
"""

import argparse
import sys

import pytest

from eval.cloud.launch_eval_iris import EvalIrisLauncher
from eval.presets import load_presets

NESTED_THINKING_KWARG = (
    'extra_body={"chat_template_kwargs":{"enable_thinking":true}}'
)


def _make_args(**overrides) -> argparse.Namespace:
    base = dict(
        preset="tb2",
        dataset=None,
        dataset_path=None,
        n_concurrent=None,
        agent_kwarg=[],
    )
    base.update(overrides)
    return argparse.Namespace(**base)


@pytest.fixture
def launcher(monkeypatch):
    # Avoid the full IrisLauncher.__init__ (cluster/Ray side effects); we only
    # exercise the pure _apply_preset logic.
    inst = object.__new__(EvalIrisLauncher)
    # _cli_has inspects sys.argv; make it look like no CLI overrides were given
    # so preset values are applied.
    monkeypatch.setattr(sys, "argv", ["launch_eval_iris.py", "--preset", "tb2"])
    return inst


def test_apply_preset_emits_nested_extra_body_thinking_kwarg(launcher, monkeypatch):
    monkeypatch.setattr(
        "eval.cloud.launch_eval_iris.load_presets",
        lambda: {"tb2": {"agent_kwargs": [NESTED_THINKING_KWARG]}},
    )
    args = _make_args()

    launcher._apply_preset(args)

    assert NESTED_THINKING_KWARG in args.agent_kwarg, args.agent_kwarg
    # The dead bare form must NEVER be emitted.
    assert not any(
        kw.split("=", 1)[0] == "enable_thinking" for kw in args.agent_kwarg
    ), f"bare enable_thinking kwarg leaked: {args.agent_kwarg}"


def test_apply_preset_no_agent_kwargs_emits_nothing(launcher, monkeypatch):
    monkeypatch.setattr(
        "eval.cloud.launch_eval_iris.load_presets",
        lambda: {"tb2": {}},
    )
    args = _make_args()

    launcher._apply_preset(args)

    assert not any("extra_body" in kw for kw in args.agent_kwarg)
    assert not any("enable_thinking" in kw for kw in args.agent_kwarg)


def test_apply_preset_respects_caller_supplied_extra_body(launcher, monkeypatch):
    """A user-supplied extra_body kwarg suppresses the preset's same-key kwarg
    (we must not clobber a deliberate caller override)."""
    monkeypatch.setattr(
        "eval.cloud.launch_eval_iris.load_presets",
        lambda: {"tb2": {"agent_kwargs": [NESTED_THINKING_KWARG]}},
    )
    user_kwarg = 'extra_body={"chat_template_kwargs":{"enable_thinking":false}}'
    args = _make_args(agent_kwarg=[user_kwarg])

    launcher._apply_preset(args)

    # Only the user's extra_body remains; the preset did not append a second one.
    assert args.agent_kwarg == [user_kwarg]


@pytest.mark.parametrize("preset_name", ["swebench", "tb2", "aider", "v2"])
def test_catalog_presets_carry_thinking_agent_kwarg(preset_name):
    """The standard eval presets must keep delivering thinking via the live
    nested form in their agent_kwargs — and must NOT carry the removed
    enable_thinking key."""
    preset = load_presets()[preset_name]
    assert "enable_thinking" not in preset, (
        f"{preset_name}: stale enable_thinking key — removed in favor of agent_kwargs"
    )
    assert NESTED_THINKING_KWARG in preset.get("agent_kwargs", []), (
        f"{preset_name}: lost its thinking agent-kwarg: {preset.get('agent_kwargs')}"
    )
