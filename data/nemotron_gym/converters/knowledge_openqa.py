"""Convert nvidia/Nemotron-RL-knowledge-openqa."""

from __future__ import annotations

from ..adapter import (
    HarborTask,
    STANDARD_TEST_SH,
    render_dockerfile,
    render_metadata,
    sanitize_text,
    task_id_for,
)
from ..verifiers import NORMALIZED_TEXT_VERIFIER_PY
from . import register
from ._common import extract_prompt


_BASE_IMAGE = "python:3.11-slim-bookworm"
_INSTRUCTION_HEADER = (
    "You are answering an open-ended question. Write your concise final answer "
    "to `/app/answer.txt`. The verifier normalizes whitespace, case, and "
    "punctuation, and accepts an answer that contains any of the expected "
    "reference answers as a substring (post-normalization).\n\n"
    "---\n\n"
)


@register("nvidia/Nemotron-RL-knowledge-openqa")
def convert_openqa(row: dict, row_idx: int) -> HarborTask | None:
    prompt = extract_prompt(row)
    raw_expected = row.get("expected_answers")
    if isinstance(row.get("expected_answer"), str):
        raw_expected = [row["expected_answer"]]
    if not isinstance(raw_expected, list) or not raw_expected:
        return None
    expected: list[str] = []
    for i, v in enumerate(raw_expected):
        if isinstance(v, str) and v.strip():
            expected.append(sanitize_text(v, field_name=f"expected_answers[{i}]", max_len=8 * 1024))
    if not expected:
        return None
    uuid = row.get("uuid") if isinstance(row.get("uuid"), str) else None
    task_id = task_id_for(
        "openqa",
        (uuid or prompt[:128]) + "|" + "||".join(expected[:3]),
    )
    return HarborTask(
        task_id=task_id,
        instruction_md=_INSTRUCTION_HEADER + prompt,
        dockerfile=render_dockerfile(base=_BASE_IMAGE),
        test_sh=STANDARD_TEST_SH,
        verifier_py=NORMALIZED_TEXT_VERIFIER_PY,
        verifier_data={"expected_answers": expected},
        metadata=render_metadata(
            source_dataset="nvidia/Nemotron-RL-knowledge-openqa",
            source_uuid=uuid,
            extra={"row_index": row_idx, "family": "openqa"},
        ),
    )
