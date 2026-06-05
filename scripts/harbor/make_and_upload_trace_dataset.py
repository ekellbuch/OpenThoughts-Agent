#!/usr/bin/env python3
"""
Export traces from a Harbor job directory into a Hugging Face dataset and upload it.

This is a SINGLE streaming path. It enumerates trial directories
NON-RECURSIVELY-stat'd (via Harbor's pruning ``iter_trial_dirs`` os.walk —
never ``root.rglob("*")``), processes trials in chunks, writes each chunk as a
parquet shard to a temp dir, and uploads the shards to the HF dataset repo via
``HfApi.upload_folder``. Peak RAM is bounded to roughly one chunk, NOT the whole
dataset — this avoids the ~150 GB git-LFS commit balloon that OOM-killed prior
trace uploads.

The UPLOAD mechanism is ALWAYS this streaming path — it does NOT depend on
whether the dataset is registered in Supabase. Registration is a separate,
ORTHOGONAL post-upload step controlled by ``--skip_register``:

- No flag (default)  -> stream-upload, THEN register the finished HF repo in
  Supabase (datagen path: the trace dataset IS the artifact).
- ``--skip_register``  -> stream-upload only, no Supabase registration (RL path:
  the trace dataset rides along, but the RL *model* is registered separately).

Crucially, ``--skip_register`` can no longer change the upload mechanism or
cause the old git-LFS OOM — it only toggles the Supabase registration step.

Examples:

  # Datagen: export, upload (streaming), AND register in Supabase
  python -m scripts.harbor.make_and_upload_trace_dataset \
    --job_dir /path/to/jobs/codecontests_glm46 \
    --repo_id DCAgent/code_contests-GLM-4.6-traces \
    --episodes last \
    --filter success

  # RL cleanup: upload only (streaming), do NOT register
  python -m scripts.harbor.make_and_upload_trace_dataset \
    --job_dir "$EXPERIMENTS_DIR/<job>/<job>" \
    --repo_id penfever/<job> \
    --episodes last \
    --skip_register

Notes:
- Requires Harbor to be installed/importable and a completed Harbor job dir.
- Auth to HF via HF_TOKEN env var or `huggingface-cli login`.
- Datasets are created PUBLIC by default. Pass --private for private repos.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


# ---------------------------------------------------------------------------
# Harbor monkeypatches (preserved from the previous implementation). These make
# Harbor's per-trial collectors robust against malformed agent dirs, surrogate
# characters, and inline subagent merging. They are installed before any
# collection runs.
# ---------------------------------------------------------------------------


def _install_safe_episode_guard() -> None:
    """Patch Harbor's episode discovery to skip invalid agent directories."""

    try:
        from harbor.utils import traces_utils  # type: ignore
    except Exception:
        return

    original_find = getattr(traces_utils, "find_episode_dirs", None)
    if original_find is None:
        return
    if getattr(original_find, "__dcagent_safe__", False):
        return

    def safe_find_episode_dirs(trial_dir: Path):
        episodes_root = Path(trial_dir) / "agent"
        if episodes_root.exists() and not episodes_root.is_dir():
            print(
                f"[trace-export] Skipping trial {trial_dir} because {episodes_root} is not a directory."
            )
            return []
        try:
            return original_find(trial_dir)
        except NotADirectoryError as exc:
            print(f"[trace-export] Skipping trial {trial_dir}: {exc}")
            return []

    safe_find_episode_dirs.__dcagent_safe__ = True  # type: ignore[attr-defined]
    traces_utils.find_episode_dirs = safe_find_episode_dirs


def _install_dataset_sanitizer() -> None:
    """Patch Harbor's rows_to_dataset to sanitize surrogate characters before HF conversion."""
    try:
        from harbor.utils import traces_utils  # type: ignore
        from scripts.harbor.run_and_export_traces import _strip_surrogates  # type: ignore
    except Exception:
        return

    original_rows_to_dataset = getattr(traces_utils, "rows_to_dataset", None)
    if original_rows_to_dataset is None:
        return
    if getattr(original_rows_to_dataset, "__dcagent_surrogate_sanitized__", False):
        return

    def safe_rows_to_dataset(rows, *args, **kwargs):
        cleaned_rows = []
        for row in rows:
            if isinstance(row, dict):
                cleaned_rows.append({k: _strip_surrogates(v) for k, v in row.items()})
            else:
                cleaned_rows.append(row)
        return original_rows_to_dataset(cleaned_rows, *args, **kwargs)

    safe_rows_to_dataset.__dcagent_surrogate_sanitized__ = True  # type: ignore[attr-defined]
    traces_utils.rows_to_dataset = safe_rows_to_dataset


def _install_inline_subagent_merger() -> None:
    """
    Patch Harbor's conversation extraction so subagent trajectories are injected into
    the main agent conversations instead of being exported as standalone rows.
    """

    try:
        from harbor.utils import traces_utils  # type: ignore
    except Exception:
        return

    if getattr(traces_utils, "__dcagent_inline_subagents__", False):
        return

    subagent_extractor = getattr(traces_utils, "_extract_complete_subagent_conversation", None)
    if subagent_extractor is None:
        return

    def _infer_subagent_label(path_fragment: str) -> str:
        name = Path(path_fragment).name
        if name.startswith("trajectory.") and name.endswith(".json"):
            return name[len("trajectory.") : -len(".json")]
        return name

    def _append_conversation_turn(
        messages: List[Dict[str, str]], role: str, content: str
    ) -> None:
        """Append a turn, merging with the previous one if the role matches."""
        if not content:
            return
        role_key = "assistant" if role == "assistant" else "user"
        if messages and messages[-1]["role"] == role_key:
            messages[-1]["content"] = f"{messages[-1]['content']}\n\n{content}"
        else:
            messages.append({"role": role_key, "content": content})

    def _format_subagent_conversation(
        conv_dict: Dict[str, Any], label: str, summary: Optional[str]
    ) -> List[Dict[str, str]]:
        conversations = conv_dict.get("conversations")
        if not isinstance(conversations, list) or not conversations:
            return []

        prefix = f"[subagent:{label}]"
        header_parts = [prefix]
        if summary:
            header_parts.append(summary)
        formatted: List[Dict[str, str]] = []
        formatted.append(
            {
                "role": "user",
                "content": " ".join(part for part in header_parts if part).strip(),
            }
        )

        for message in conversations:
            if not isinstance(message, dict):
                continue
            role = "assistant" if message.get("role") == "assistant" else "user"
            content = message.get("content") or ""
            if not content:
                continue
            tagged_content = f"{prefix} {role.upper()}: {content}"
            formatted.append({"role": role, "content": tagged_content})

        return [msg for msg in formatted if msg.get("content")]

    def _observation_to_text(observation: Any) -> Optional[str]:
        if not isinstance(observation, dict):
            return None
        results = observation.get("results")
        if not isinstance(results, list):
            return None
        observation_contents: List[str] = []
        for result in results:
            if isinstance(result, dict) and "content" in result:
                observation_contents.append(result["content"])
        if observation_contents:
            return "\n".join(observation_contents)
        return None

    def _append_subagent_transcripts(
        messages: List[Dict[str, str]],
        step: Dict[str, Any],
        trajectory_dir: Path,
        cache: Dict[str, Optional[Dict[str, Any]]],
        run_metadata: Dict[str, Any],
    ) -> None:
        observation = step.get("observation")
        if not isinstance(observation, dict):
            return
        results = observation.get("results")
        if not isinstance(results, list):
            return

        for result in results:
            if not isinstance(result, dict):
                continue
            refs = result.get("subagent_trajectory_ref")
            if not isinstance(refs, list):
                continue
            for ref in refs:
                if not isinstance(ref, dict):
                    continue
                rel_path = ref.get("trajectory_path")
                if not rel_path:
                    continue
                subagent_path = (trajectory_dir / rel_path).resolve()
                cache_key = str(subagent_path)
                subagent_conv = cache.get(cache_key)
                if subagent_conv is None:
                    if not subagent_path.exists():
                        cache[cache_key] = None
                        continue
                    try:
                        subagent_conv = subagent_extractor(subagent_path, run_metadata)  # type: ignore[misc]
                    except Exception:
                        subagent_conv = None
                    cache[cache_key] = subagent_conv

                if not subagent_conv:
                    continue

                label = _infer_subagent_label(rel_path)
                extra = ref.get("extra")
                summary = extra.get("summary") if isinstance(extra, dict) else None
                formatted_messages = _format_subagent_conversation(
                    subagent_conv, label, summary
                )
                for formatted in formatted_messages:
                    _append_conversation_turn(
                        messages, formatted["role"], formatted["content"]
                    )

    def _extract_episode_with_subagents(
        steps: List[Dict[str, Any]],
        episode_num: int,
        run_metadata: Dict[str, Any],
        trajectory_dir: Path,
        cache: Dict[str, Optional[Dict[str, Any]]],
    ) -> Optional[Dict[str, Any]]:
        conv: Dict[str, Any] = {
            "conversations": [],
            "agent": run_metadata["agent_name"],
            "model": run_metadata["model_name"],
            "model_provider": run_metadata["model_provider"],
            "date": run_metadata["start_time"],
            "task": None,
            "episode": f"episode-{episode_num}",
            "run_id": run_metadata["run_id"],
            "trial_name": None,
        }

        agent_steps = [idx for idx, step in enumerate(steps) if step.get("source") == "agent"]

        for idx, step in enumerate(steps):
            source = step.get("source")
            message = step.get("message", "")

            if source in {"system", "user"}:
                _append_conversation_turn(conv["conversations"], "user", message)
            elif source == "agent":
                content_parts: List[str] = []
                reasoning_content = step.get("reasoning_content")
                if reasoning_content:
                    content_parts.append(f"<think>{reasoning_content}</think>")
                if message:
                    # Fix orphaned </think> tags: when the generation prompt includes <think>
                    # but vLLM only returns generated tokens, the opening tag is missing.
                    if '</think>' in message and '<think>' not in message:
                        message = '<think>' + message
                    content_parts.append(message)
                tool_calls = step.get("tool_calls")
                if isinstance(tool_calls, list):
                    for call in tool_calls:
                        if not isinstance(call, dict):
                            continue
                        tool_call_obj = {
                            "name": call.get("function_name"),
                            "arguments": call.get("arguments", {}),
                        }
                        tool_call_json = json.dumps(tool_call_obj, ensure_ascii=False)
                        content_parts.append(f"<tool_call>\n{tool_call_json}\n</tool_call>")
                assistant_content = "\n".join(content_parts) if content_parts else ""
                _append_conversation_turn(
                    conv["conversations"], "assistant", assistant_content
                )

                is_last_agent_step = agent_steps and (idx == agent_steps[-1])
                if not is_last_agent_step:
                    observation_text = _observation_to_text(step.get("observation"))
                    if observation_text:
                        _append_conversation_turn(
                            conv["conversations"], "user", observation_text
                        )

            _append_subagent_transcripts(conv["conversations"], step, trajectory_dir, cache, run_metadata)

        return conv

    def patched_extract_conversations_from_trajectory(
        trajectory_file: Path, run_metadata: Dict[str, Any], embed_tools_in_conversation: bool = True,
    ) -> List[Dict[str, Any]]:
        try:
            trajectory_data = json.loads(trajectory_file.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[traces] Skipping trajectory {trajectory_file}: invalid JSON ({exc})")
            return []

        steps = trajectory_data.get("steps", [])
        agent_info = trajectory_data.get("agent", {})
        trajectory_agent_name = agent_info.get("name") or run_metadata["agent_name"]
        trajectory_model_name = agent_info.get("model_name") or run_metadata["model_name"]
        trajectory_run_metadata = {
            **run_metadata,
            "agent_name": trajectory_agent_name,
            "model_name": trajectory_model_name,
        }

        agent_step_indices: List[int] = []
        for idx, step in enumerate(steps):
            if step.get("source") == "agent" and not step.get("is_copied_context"):
                agent_step_indices.append(idx)

        if not agent_step_indices:
            return []

        trajectory_dir = trajectory_file.parent
        subagent_cache: Dict[str, Optional[Dict[str, Any]]] = {}
        conversations: List[Dict[str, Any]] = []
        for episode_num, agent_step_idx in enumerate(agent_step_indices):
            conv = _extract_episode_with_subagents(
                steps[: agent_step_idx + 1],
                episode_num,
                trajectory_run_metadata,
                trajectory_dir,
                subagent_cache,
            )
            if conv and conv.get("conversations"):
                conversations.append(conv)
        return conversations

    patched_extract_conversations_from_trajectory.__dcagent_inline_subagents__ = True  # type: ignore[attr-defined]
    traces_utils.extract_conversations_from_trajectory = patched_extract_conversations_from_trajectory


def _install_harbor_patches() -> None:
    """Install all Harbor monkeypatches used by the streaming exporter."""
    _install_safe_episode_guard()
    _install_dataset_sanitizer()
    _install_inline_subagent_merger()


# ---------------------------------------------------------------------------
# Streaming export helpers
# ---------------------------------------------------------------------------


def _import_traces_utils():
    """Resolve Harbor's traces_utils module (the per-trial collectors live here)."""
    try:
        from harbor.utils import traces_utils  # type: ignore
        return traces_utils
    except Exception as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "Harbor is not available. Install it (pip install -e ../harbor) "
            f"or ensure it's on PYTHONPATH. Import error: {exc}"
        )


def _finalize_chunk(ds):
    """Apply the same sanitize/format cleanup the legacy path applied, per-chunk.

    Mirrors ``scripts.harbor.run_and_export_traces._finalize_trace_dataset`` and
    ``data.commons.clean_empty_structs`` but is applied to a single bounded
    chunk so peak memory stays small.
    """
    try:
        from scripts.harbor.run_and_export_traces import _finalize_trace_dataset  # type: ignore
        ds = _finalize_trace_dataset(ds)
    except Exception:
        pass
    return ds


def _iter_trial_dirs_nonrecursive(traces_utils, root: Path) -> Iterator[Path]:
    """Yield trial dirs using Harbor's PRUNING os.walk enumeration.

    Harbor's ``iter_trial_dirs`` uses ``os.walk`` with in-place pruning of
    ``dirnames`` once a trial dir is found — it NEVER descends into a trial's
    ~20-30 inner files and NEVER does ``root.rglob("*")``. This keeps the cost
    at O(directory skeleton) instead of O(every file), which is what GPFS-hangs
    on large runs.
    """
    yield from traces_utils.iter_trial_dirs(root, recursive=True)


def _collect_trial_rows(
    traces_utils,
    trial_dir: Path,
    *,
    episodes: str,
    success_filter: Optional[str],
    include_instruction: bool,
    include_verifier_output: bool,
    verbose: bool,
) -> List[Dict[str, Any]]:
    """Collect conversation rows for a single trial. Returns [] on skip/error."""
    from harbor.models.agent.name import AgentName  # type: ignore
    from harbor.agents.factory import AgentFactory  # type: ignore

    try:
        run_meta = traces_utils.load_run_metadata(trial_dir)
    except (ValueError, FileNotFoundError) as exc:
        if verbose:
            print(f"[traces] Skipping {trial_dir.name}: {exc}")
        return []

    agent_name = run_meta["agent_name"]
    agent_enum = AgentName(agent_name)
    agent_class = AgentFactory._AGENT_MAP.get(agent_enum)
    if agent_class is None or not agent_class.SUPPORTS_ATIF:
        raise NotImplementedError(
            f"{agent_name} does not support Harbor's trajectory format (ATIF), cannot export traces"
        )

    if success_filter in ("success", "failure"):
        succ = traces_utils._trial_is_success(trial_dir)
        if succ is None:
            if verbose:
                print(f"[traces] Trial {trial_dir.name}: missing result.json; skipping due to filter")
            return []
        if success_filter == "success" and not succ:
            return []
        if success_filter == "failure" and succ:
            return []

    try:
        return traces_utils.collect_conversations_from_trial(
            trial_dir,
            run_meta=run_meta,
            episodes=episodes,
            verbose=verbose,
            include_instruction=include_instruction,
            include_verifier_output=include_verifier_output,
            embed_tools_in_conversation=True,
        )
    except (UnicodeDecodeError, ValueError, OSError) as e:
        if verbose:
            print(f"[traces] Trial {trial_dir.name}: skipping due to decode/read error: {e}")
        return []


def _flush_chunk_to_parquet(rows: List[Dict[str, Any]], to_sharegpt: bool, traces_utils, shard_path: Path) -> int:
    """Convert a chunk of rows to a (finalized) Dataset and write one parquet shard.

    Returns the number of rows written. The intermediate Dataset is dropped as
    soon as the parquet file is on disk, so peak RAM is bounded to one chunk.
    """
    if not rows:
        return 0
    ds = traces_utils.rows_to_dataset(rows)
    if to_sharegpt:
        ds = traces_utils.convert_openai_to_sharegpt(ds, "conversations", "conversations_sharegpt")
    ds = _finalize_chunk(ds)
    n = len(ds)
    ds.to_parquet(str(shard_path))
    return n


def stream_export_and_upload(
    *,
    job_dir: Path,
    repo_id: str,
    episodes: str,
    success_filter: Optional[str],
    to_sharegpt: bool,
    private: bool,
    include_instruction: bool = True,
    include_verifier_output: bool = True,
    chunk_size: int = 200,
    verbose: bool = False,
) -> int:
    """Single streaming path: enumerate trials, write parquet shards, upload via HfApi.

    Memory is bounded to roughly one ``chunk_size`` worth of rows at any time:
    we never hold the full row set in RAM, never build one giant in-memory
    Dataset, and never git-clone / git-LFS. Returns the total row count.
    """
    from huggingface_hub import HfApi  # type: ignore

    traces_utils = _import_traces_utils()
    _install_harbor_patches()

    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    print(f"[trace-export] Repo {repo_id} ensured (private={private}).")

    tmp_root = Path(tempfile.mkdtemp(prefix="trace_shards_"))
    shards_dir = tmp_root / "data"
    shards_dir.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    shard_idx = 0
    chunk: List[Dict[str, Any]] = []

    def _flush() -> None:
        nonlocal chunk, shard_idx, total_rows
        if not chunk:
            return
        shard_path = shards_dir / f"train-{shard_idx:05d}.parquet"
        written = _flush_chunk_to_parquet(chunk, to_sharegpt, traces_utils, shard_path)
        if written:
            print(f"[trace-export] Wrote shard {shard_path.name} ({written} rows; running total {total_rows + written}).")
            total_rows += written
            shard_idx += 1
        else:
            # Nothing materialized (e.g. all rows finalized away); drop empty file.
            if shard_path.exists():
                shard_path.unlink()
        chunk = []

    try:
        n_trials = 0
        for trial_dir in _iter_trial_dirs_nonrecursive(traces_utils, job_dir):
            n_trials += 1
            rows = _collect_trial_rows(
                traces_utils,
                trial_dir,
                episodes=episodes,
                success_filter=success_filter,
                include_instruction=include_instruction,
                include_verifier_output=include_verifier_output,
                verbose=verbose,
            )
            if rows:
                chunk.extend(rows)
            if len(chunk) >= chunk_size:
                _flush()
        _flush()  # final partial chunk

        print(f"[trace-export] Enumerated {n_trials} trial dirs; collected {total_rows} rows into {shard_idx} shard(s).")

        if shard_idx == 0:
            # Still write an empty dataset so downstream consumers see a valid
            # (zero-row) repo rather than nothing — mirrors prior behavior.
            empty_ds = traces_utils.rows_to_dataset([])
            empty_path = shards_dir / "train-00000.parquet"
            empty_ds.to_parquet(str(empty_path))
            print("[trace-export] No rows collected; wrote an empty parquet shard.")

        print(f"[trace-export] Uploading parquet shard(s) from {shards_dir} to {repo_id} via HfApi...")
        api.upload_folder(
            folder_path=str(shards_dir),
            path_in_repo="data",
            repo_id=repo_id,
            repo_type="dataset",
        )
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    return total_rows


def register_trace_dataset_in_supabase(repo_id: str, dataset_type: str = "SFT") -> None:
    """Register an already-uploaded trace dataset repo in Supabase.

    This is the ORTHOGONAL post-upload step. It runs against the FINISHED HF
    repo (``register_hf_dataset`` reads the dataset's metadata back from the HF
    API), so it is completely decoupled from the streaming upload mechanism —
    it cannot affect peak memory and cannot trigger the old git-LFS OOM. It is
    invoked by ``main()`` only when ``--skip_register`` is NOT passed.

    This reuses the same registration logic the legacy
    ``data.commons.upload_traces_to_hf`` default branch used, but called as a
    clean separate step rather than entangled with the upload.
    """
    try:
        from database.unified_db.utils import register_hf_dataset  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "Could not import register_hf_dataset for Supabase registration. "
            f"Import error: {exc}"
        )
    register_hf_dataset(repo_name=repo_id, dataset_type=dataset_type)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Make and upload a trace dataset from a Harbor job directory (single streaming path)",
    )
    p.add_argument("--job_dir", required=True, help="Path to Harbor job directory")
    p.add_argument("--repo_id", required=True, help="Target HF dataset repo (org/name)")
    p.add_argument(
        "--episodes",
        choices=["all", "last"],
        default="last",
        help="Which episodes to export per trial (default: last)",
    )
    p.add_argument(
        "--filter",
        choices=["success", "failure", "none"],
        default="none",
        help="Filter exported episodes (default: none)",
    )
    p.add_argument(
        "--to_sharegpt",
        action="store_true",
        help="Export in ShareGPT-style format where applicable",
    )
    p.add_argument(
        "--dataset_type",
        default="SFT",
        help="Dataset type label ('SFT' or 'RL') used by the Supabase registration step. Default: SFT",
    )
    p.add_argument(
        "--skip_register",
        action="store_true",
        help="Upload only; do NOT register the dataset in Supabase. This flag "
        "controls ONLY registration — the (always-streaming, memory-safe) upload "
        "path is identical either way. RL cleanup passes this (the model is "
        "registered separately); datagen omits it (the dataset is the artifact).",
    )
    p.add_argument(
        "--private",
        action="store_true",
        help="Create the HF dataset repo as private (default: public)",
    )
    p.add_argument(
        "--chunk_size",
        type=int,
        default=200,
        help="Trials' rows per parquet shard (bounds peak memory). Default: 200",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose per-trial logging",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    job_dir = Path(args.job_dir).expanduser().resolve()
    if not job_dir.exists() or not job_dir.is_dir():
        raise SystemExit(f"job_dir does not exist or is not a directory: {job_dir}")

    success_filter = None if args.filter == "none" else args.filter

    # Step 1 — the single, always-streaming, memory-safe upload mechanism.
    print(f"[trace-export] Streaming export from: {job_dir}")
    total = stream_export_and_upload(
        job_dir=job_dir,
        repo_id=args.repo_id,
        episodes=args.episodes,
        success_filter=success_filter,
        to_sharegpt=bool(args.to_sharegpt),
        private=bool(args.private),
        chunk_size=max(1, int(args.chunk_size)),
        verbose=bool(args.verbose),
    )

    print(
        f"[trace-export] Upload complete ({total} rows): "
        f"https://huggingface.co/datasets/{args.repo_id}"
    )

    # Step 2 — ORTHOGONAL post-upload Supabase registration, gated only by
    # --skip_register. This runs against the finished HF repo and cannot affect
    # the upload's memory profile.
    if args.skip_register:
        print(
            "[trace-export] --skip_register set: upload only, NOT registering "
            "in Supabase."
        )
    else:
        print(
            f"[trace-export] Registering {args.repo_id} in Supabase "
            f"(dataset_type={args.dataset_type})..."
        )
        register_trace_dataset_in_supabase(args.repo_id, dataset_type=args.dataset_type)
        print(f"[trace-export] Supabase registration complete for {args.repo_id}.")


if __name__ == "__main__":
    main()
