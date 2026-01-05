from __future__ import annotations

from pathlib import Path


def sync_eval_outputs(cluster_name: str | None, remote_path: str, local_dir: str) -> None:
    """Placeholder sync hook for pulling eval artifacts back to the caller's machine.

    This shim keeps all sync logic in one place so we can swap in a real
    implementation (e.g., sky storage mounts, GCS buckets, etc.) without touching
    the launcher. For now we simply ensure the destination exists and emit
    instructions for the operator.
    """

    target = Path(local_dir).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)

    hint = remote_path
    if cluster_name:
        hint = f"{cluster_name}:{remote_path}"

    print(
        "[cloud-sync] Eval outputs were left in "
        f"'{remote_path}'. Syncing is not automated yet.\n"
        f"Copy them into '{target}' using your preferred method "
        f"(e.g., `sky storage cp {hint} {target}` or `sky exec`)."
    )
