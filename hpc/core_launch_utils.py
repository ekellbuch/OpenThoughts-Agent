"""Shared launch utilities that are lightweight enough for cross-module imports."""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Optional, Union

PathInput = Union[str, PathLike[str], Path, None]
_VALID_TRACE_BACKENDS = {"vllm", "ray", "vllm_local", "none"}


def cleanup_endpoint_file(path_like: PathInput, *, descriptor: str = "endpoint file") -> None:
    """Remove a stale endpoint JSON if it exists."""

    if not path_like:
        return
    try:
        candidate = Path(path_like).expanduser()
    except Exception:
        return
    if not candidate.exists():
        return
    try:
        candidate.unlink()
        print(f"Removed {descriptor}: {candidate}")
    except OSError as exc:
        print(f"Warning: failed to remove {descriptor} {candidate}: {exc}")


def validate_trace_backend(
    backend_value: Optional[str],
    *,
    allow_vllm: bool,
    job_type: str,
) -> str:
    """Normalize and validate the requested trace backend."""

    backend = (backend_value or "vllm").strip().lower()
    if backend not in _VALID_TRACE_BACKENDS:
        raise ValueError(
            f"Unsupported trace backend '{backend_value}'. "
            f"Valid options: {sorted(_VALID_TRACE_BACKENDS)}"
        )
    if backend == "vllm" and not allow_vllm:
        raise RuntimeError(
            f"trace_backend=vllm is not supported for {job_type} jobs. "
            "Use a Ray-backed backend or disable trace generation."
        )
    return backend


__all__ = [
    "cleanup_endpoint_file",
    "validate_trace_backend",
]
