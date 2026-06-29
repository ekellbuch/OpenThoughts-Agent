"""Single source of truth for infrastructure-error classification.

`INFRA_ERROR_TYPES` is the set of exception types that represent INFRASTRUCTURE
failures (Daytona/sandbox/environment/verification-wrapper) rather than genuine
agent/task failures. Harbor's resume filters retry these, and the eval listener's
disk-based resume scanner counts them.

This module is the canonical definition: both `eval/unified_eval_listener.py`
(the resume scanner) and `database/unified_db/utils.py` (the DB write-point that
persists the count onto `sandbox_jobs.stats`) import from here. Do NOT duplicate
the set elsewhere.

NOTE: `VerificationNotCompletedError` is the OUTER wrapper every VNC trial stores
when its true cause is an infra failure (e.g. EnvironmentStartTimeoutError); it is
intentionally classified as infra.

This module is dependency-free (stdlib only) so it can be imported from anywhere.
"""

from typing import Any, Dict, Mapping, Tuple

# Infrastructure errors that harbor's resume filters will retry.
# MUST stay in sync with the sbatch resume branch's --filter-error-type list
# (eval/tacc/eval_harbor.sbatch).
INFRA_ERROR_TYPES = {
    "DaytonaError",
    "DaytonaAuthenticationError",
    "DaytonaAuthorizationError",
    "DaytonaNotFoundError",
    "EnvironmentStartTimeoutError",
    "DaytonaRateLimitError",
    "CancelledError",
    "SandboxBuildFailedError",
    "AgentEnvironmentTimeoutError",
    "VerificationNotCompletedError",
}


def compute_infra_error_stats(stats: Mapping[str, Any]) -> Tuple[int, Dict[str, int]]:
    """Compute the infrastructure-error count + per-type breakdown from a Harbor
    `stats` blob.

    Reads `stats.evals.<key>.exception_stats` (a `{error_type: [trial_ids]}` map),
    keeps only types in `INFRA_ERROR_TYPES`, and sums their counts. The count for a
    type is `len(ids)` when `ids` is a list, else 1 (defensive).

    Args:
        stats: The Harbor `stats` dict (i.e. `result["stats"]`).

    Returns:
        (n_infra_errors, infra_error_breakdown) where breakdown is
        `{error_type: count}` containing only infra types with a non-zero count.
        Types not present / not infra are omitted from the breakdown.
    """
    n_infra = 0
    breakdown: Dict[str, int] = {}
    if not isinstance(stats, Mapping):
        return 0, {}
    evals = stats.get("evals")
    if not isinstance(evals, Mapping):
        return 0, {}
    for eval_data in evals.values():
        if not isinstance(eval_data, Mapping):
            continue
        exception_stats = eval_data.get("exception_stats")
        if not isinstance(exception_stats, Mapping):
            continue
        for exc_type, ids in exception_stats.items():
            if exc_type not in INFRA_ERROR_TYPES:
                continue
            n = len(ids) if isinstance(ids, list) else 1
            n_infra += n
            breakdown[exc_type] = breakdown.get(exc_type, 0) + n
    return n_infra, breakdown
