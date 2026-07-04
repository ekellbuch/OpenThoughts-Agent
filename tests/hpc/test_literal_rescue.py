"""Tests for the rescue/cleanup FAVOR-literals default (Stage 0/1).

Covers ``scripts/harbor/make_and_upload_trace_dataset.resolve_literal_inclusion``
(auto-detect + favor-when-present + opt-out + force-require) and the Stage-1
correlator guards (strip stale trajectory token_ids; the fail-loud 0-bind
condition).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.harbor.make_and_upload_trace_dataset import resolve_literal_inclusion  # noqa: E402


def _job_with_literal(tmp_path: Path, name: str = "job") -> Path:
    """A job dir with a trial + a sibling logs/<slug>_literal.jsonl (durable layout)."""
    job = tmp_path / name
    (job / "trial-A" / "agent").mkdir(parents=True)
    (job / "trial-A" / "agent" / "trajectory.json").write_text(json.dumps({"steps": []}))
    (job / "logs").mkdir()
    lit = job / "logs" / f"{name}_literal.jsonl"
    lit.write_text('{"status_code":200,"request":{"messages":[]},"literal":{"completion_token_ids":[1]}}\n')
    return job


def _job_without_literal(tmp_path: Path) -> Path:
    job = tmp_path / "plain"
    (job / "trial-A" / "agent").mkdir(parents=True)
    (job / "trial-A" / "agent" / "trajectory.json").write_text(json.dumps({"steps": []}))
    return job


# --------------------------------------------------------------------------- #
# Stage 0 — resolve_literal_inclusion
# --------------------------------------------------------------------------- #
def test_parity_no_literal_no_flags_is_text_only(tmp_path):
    # GLOBAL INVARIANT: a job with no literal.jsonl and no flags -> text-only.
    job = _job_without_literal(tmp_path)
    assert resolve_literal_inclusion(
        str(job), literal_log=None, include_literal_tokens=False, no_literal_tokens=False
    ) == (False, None)


def test_favor_literal_present_default_on(tmp_path):
    # FAVOR: a discoverable sibling logs/*literal.jsonl -> literals ON, no flag needed.
    job = _job_with_literal(tmp_path)
    include, uri = resolve_literal_inclusion(
        str(job), literal_log=None, include_literal_tokens=False, no_literal_tokens=False
    )
    assert include is True
    assert uri is not None and uri.endswith("job_literal.jsonl")


def test_opt_out_forces_text_only_even_with_literal(tmp_path):
    job = _job_with_literal(tmp_path)
    assert resolve_literal_inclusion(
        str(job), literal_log=None, include_literal_tokens=False, no_literal_tokens=True
    ) == (False, None)


def test_explicit_literal_log_wins(tmp_path):
    job = _job_without_literal(tmp_path)
    explicit = "gs://bucket/x/logs/y_literal.jsonl"
    assert resolve_literal_inclusion(
        str(job), literal_log=explicit, include_literal_tokens=False, no_literal_tokens=False
    ) == (True, explicit)


def test_require_missing_fails_loud(tmp_path):
    # --include_literal_tokens (REQUIRE) with no literal.jsonl -> refuse (SystemExit),
    # never silently downgrade to text-only.
    job = _job_without_literal(tmp_path)
    with pytest.raises(SystemExit):
        resolve_literal_inclusion(
            str(job), literal_log=None, include_literal_tokens=True, no_literal_tokens=False
        )


def test_no_literal_wins_over_require(tmp_path):
    # --no_literal_tokens is an explicit text-only override; it short-circuits before
    # the require check so it never raises.
    job = _job_without_literal(tmp_path)
    assert resolve_literal_inclusion(
        str(job), literal_log=None, include_literal_tokens=True, no_literal_tokens=True
    ) == (False, None)


# --------------------------------------------------------------------------- #
# Stage 2 — post-upload literal-column verify helper
# --------------------------------------------------------------------------- #
def test_count_populated_literal_rows():
    import pyarrow as pa
    from scripts.harbor.make_and_upload_trace_dataset import count_populated_literal_rows

    # 3 rows: two with non-empty prompt_token_ids, one empty.
    t = pa.table({"prompt_token_ids": [[[1, 2, 3]], [], [[4]]], "conversations": [[], [], []]})
    assert count_populated_literal_rows(t) == 2
    # column absent -> 0 (a text-only dataset), never raises.
    t2 = pa.table({"conversations": [[], []]})
    assert count_populated_literal_rows(t2) == 0
