"""Convert nvidia/Nemotron-RL-Identity-Following-v1 → Harbor LLM-judge task.

Actual schema: `responses_create_params` (user prompt), `principle` (rubric),
`agent_ref`, `dataset`. The principle is a multi-criterion rubric used by an
LLM judge to score persona-consistency.
"""

from __future__ import annotations

from ..adapter import (
    HarborTask,
    STANDARD_TEST_SH,
    render_dockerfile,
    render_metadata,
    sanitize_text,
    task_id_for,
)
from ..verifiers import LLM_JUDGE_VERIFIER_PY
from . import register
from ._common import extract_prompt


_BASE_IMAGE = "python:3.11-slim-bookworm"
_INSTRUCTION_HEADER = (
    "Respond to the user request below as appropriate to your identity. "
    "Write your final response to `/app/response.txt`. An LLM judge will score "
    "your response against an identity-consistency rubric.\n\n"
    "---\n\n"
)
_MAX_PRINCIPLE_BYTES = 16 * 1024


@register("nvidia/Nemotron-RL-Identity-Following-v1")
def convert_identity_following(row: dict, row_idx: int) -> HarborTask | None:
    prompt = extract_prompt(row)
    principle = row.get("principle")
    if not isinstance(principle, str) or not principle.strip():
        return None
    principle = sanitize_text(principle, field_name="principle", max_len=_MAX_PRINCIPLE_BYTES)
    upstream_dataset = row.get("dataset") if isinstance(row.get("dataset"), str) else None
    task_id = task_id_for(
        "identity",
        prompt[:128] + "|" + principle[:128] + "|" + str(row_idx),
    )
    return HarborTask(
        task_id=task_id,
        instruction_md=_INSTRUCTION_HEADER + prompt,
        dockerfile=render_dockerfile(
            base=_BASE_IMAGE,
            pip_packages=("litellm==1.51.3",),
        ),
        test_sh=STANDARD_TEST_SH,
        verifier_py=LLM_JUDGE_VERIFIER_PY,
        verifier_data={
            "instruction": prompt,
            "principle": principle,
            "judge_prompt_template": (
                "You are scoring identity-following behavior.\n\n"
                "User request:\n{instruction}\n\n"
                "Candidate response:\n{response}\n\n"
                "Evaluation rubric:\n{principle}\n\n"
                "Apply each criterion in the rubric. Score 0.0 (fails all "
                "criteria) to 1.0 (satisfies all criteria).\n"
                "End with \\boxed{{<score>}} on the last line."
            ),
            "score_threshold": 0.5,
        },
        metadata=render_metadata(
            source_dataset="nvidia/Nemotron-RL-Identity-Following-v1",
            source_uuid=None,
            extra={
                "row_index": row_idx,
                "family": "llm_judge_identity",
                "judge": "litellm:default(openai/gpt-4o-mini)",
                "upstream_dataset": upstream_dataset,
            },
        ),
    )
