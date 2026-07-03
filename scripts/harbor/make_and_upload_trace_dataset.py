#!/usr/bin/env python3
"""
Export traces from a Harbor job directory into a Hugging Face dataset and upload it.

This is a SINGLE streaming path. It enumerates trial directories
NON-RECURSIVELY-stat'd (via Harbor's pruning ``iter_trial_dirs`` os.walk —
never ``root.rglob("*")``), processes trials in chunks, writes each chunk as a
parquet shard, uploads that shard to the HF dataset repo IMMEDIATELY via an
additive ``HfApi.upload_file`` commit, then DROPS the shard from disk and frees
the in-process Arrow/datasets buffers. Peak RAM is bounded to roughly one chunk,
NOT the whole dataset — this avoids the ~150 GB balloon that OOM-killed prior
trace uploads.

Why per-shard incremental upload (not one ``upload_folder`` at the end)?
- The previous implementation accumulated EVERY parquet shard in a local temp
  dir and only uploaded them in ONE ``upload_folder`` commit at the very end.
  That had two failure modes: (1) an OOM/crash before the final upload left
  ZERO shards on the hub (the whole run was lost — exactly what happened on
  explore-tis-untrunc), and (2) the per-chunk ``Dataset.from_list`` + ``.map``
  + ``.to_parquet`` cycle leaks memory-mapped Arrow tables and HF ``datasets``
  cache files that are never released, so RSS still climbed monotonically
  across hundreds of chunks even though the live row buffer was bounded.
- The fix pushes each shard as its own additive commit and immediately frees
  the chunk (``del`` the Dataset, ``gc.collect()``, release the pyarrow memory
  pool, and discard the per-chunk HF datasets cache dir). A mid-run failure now
  leaves all already-pushed shards intact + resumable, and peak RSS tracks
  ``chunk_size`` × row-size flat across shards.

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
import gc
import json
import os
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
        include_literal_tokens: bool = False,
    ) -> List[Dict[str, Any]]:
        # `include_literal_tokens` accepted for signature-compat with harbor's
        # extract_conversations_from_trajectory (export_traces always forwards it since the
        # trace-literal merge). This inline-subagent-merger reimplements extraction and does NOT
        # emit literal token columns — that plumbing belongs to the opencode literal-traces runtime
        # (not yet wired). For non-literal uploads (terminus-2 datagen) it is a no-op; accepting it
        # here unblocks all trace uploads, which currently crash with `unexpected keyword argument`.
        _ = include_literal_tokens
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


def _install_harbor_patches(include_literal_tokens: bool = False) -> None:
    """Install the Harbor monkeypatches used by the streaming exporter.

    The inline-subagent-merger patch reimplements ``extract_conversations_from_trajectory``
    and does NOT emit the literal token columns. So when ``include_literal_tokens``
    is requested we SKIP that patch and let Harbor's native (literal-aware)
    extraction run, which lifts ``step.metrics.{prompt_token_ids,completion_token_ids,
    logprobs}`` into parallel columns. Default (no literal) installs all patches, so
    the terminus-2 / non-literal upload path is byte-identical.
    """
    _install_safe_episode_guard()
    _install_dataset_sanitizer()
    if not include_literal_tokens:
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
    include_literal_tokens: bool = False,
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
            include_literal_tokens=include_literal_tokens,
        )
    except (UnicodeDecodeError, ValueError, OSError) as e:
        if verbose:
            print(f"[traces] Trial {trial_dir.name}: skipping due to decode/read error: {e}")
        return []


def _release_arrow_memory() -> None:
    """Force-free Arrow + Python buffers so per-chunk RSS does not accumulate.

    HF ``datasets`` (``Dataset.from_list`` + ``.map`` + ``.to_parquet``) leaves
    memory-mapped Arrow tables and writer buffers alive that the pyarrow memory
    pool does NOT return to the OS on its own. Across hundreds of chunks this
    is the dominant monotonic-RSS leak even when the live row buffer is bounded.
    We explicitly run a GC pass and release the default pyarrow memory pool's
    unused blocks after every chunk.
    """
    gc.collect()
    try:
        import pyarrow as pa  # type: ignore

        pool = pa.default_memory_pool()
        # release_unused() hands freed-but-cached arena blocks back to the OS;
        # available on pyarrow's system/jemalloc/mimalloc pools.
        release = getattr(pool, "release_unused", None)
        if callable(release):
            release()
    except Exception:
        pass


def _flush_chunk_to_parquet(
    rows: List[Dict[str, Any]],
    to_sharegpt: bool,
    traces_utils,
    shard_path: Path,
    cache_dir: Path,
) -> int:
    """Convert a chunk of rows to a (finalized) Dataset and write one parquet shard.

    Returns the number of rows written. The intermediate Dataset is dropped and
    the per-chunk HF datasets cache dir is wiped as soon as the parquet file is
    on disk, so peak RAM (and disk-cache) is bounded to one chunk. ``cache_dir``
    isolates this chunk's ``.map`` cache files so they can be deleted eagerly
    instead of piling up under the shared ``~/.cache/huggingface`` tree.
    """
    if not rows:
        return 0
    import datasets as _hf_datasets  # type: ignore

    # Pin every intermediate Dataset's on-disk artifacts to this chunk's
    # throwaway cache dir so they are removed with it (the global HF datasets
    # cache otherwise grows one Arrow file per chunk and never shrinks).
    prev_cache = _hf_datasets.config.HF_DATASETS_CACHE
    try:
        _hf_datasets.config.HF_DATASETS_CACHE = cache_dir
        ds = traces_utils.rows_to_dataset(rows)
        if to_sharegpt:
            ds = traces_utils.convert_openai_to_sharegpt(ds, "conversations", "conversations_sharegpt")
        ds = _finalize_chunk(ds)
        n = len(ds)
        ds.to_parquet(str(shard_path))
        del ds
    finally:
        _hf_datasets.config.HF_DATASETS_CACHE = prev_cache
        shutil.rmtree(cache_dir, ignore_errors=True)
        _release_arrow_memory()
    return n


def _upload_shard(api, repo_id: str, shard_path: Path) -> None:
    """Push ONE shard as its own additive commit, then remove it locally.

    Each shard lands independently — a mid-run failure leaves the
    already-pushed shards intact and resumable rather than losing the whole
    run (the old end-only ``upload_folder`` lost everything on any crash).
    """
    path_in_repo = f"data/{shard_path.name}"
    # Test seam: when set, copy the shard to a local dir instead of hitting the
    # Hub (lets the subprocess path be exercised offline without real creds).
    fake_dir = os.environ.get("TRACE_EXPORT_FAKE_UPLOAD_DIR")
    if fake_dir:
        dest = Path(fake_dir) / shard_path.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(shard_path, dest)
    else:
        api.upload_file(
            path_or_fileobj=str(shard_path),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Add {path_in_repo}",
        )
    print(f"[trace-export] Uploaded {path_in_repo} to {repo_id}.")
    shard_path.unlink(missing_ok=True)


def _process_one_shard_in_subprocess(
    *,
    trial_dirs: List[str],
    shard_idx: int,
    repo_id: str,
    episodes: str,
    success_filter: Optional[str],
    to_sharegpt: bool,
    include_instruction: bool,
    include_verifier_output: bool,
    tmp_root: str,
    verbose: bool,
    include_literal_tokens: bool = False,
) -> int:
    """Worker body: collect this batch's rows, write + upload ONE shard, return row count.

    This runs in a FRESH process (multiprocessing ``spawn``). The collection
    path (Harbor's ``collect_conversations_from_trial`` + the ``datasets``
    ``from_list``/``.map``/``to_parquet`` cycle) accumulates per-row native
    memory that ``gc.collect()`` + ``pyarrow.release_unused()`` do NOT reliably
    return to the OS — empirically ~1 GB per 200-row shard, climbing linearly
    until OOM on a 24k-row run. Doing each shard in its own process makes the
    OS reclaim ALL of that on process exit, so parent RSS stays flat at roughly
    one chunk regardless of how leaky the per-chunk libraries are. The parent
    only ever holds a list of trial-dir path strings.
    """
    from huggingface_hub import HfApi  # type: ignore

    traces_utils = _import_traces_utils()
    _install_harbor_patches(include_literal_tokens=include_literal_tokens)

    chunk: List[Dict[str, Any]] = []
    for td in trial_dirs:
        rows = _collect_trial_rows(
            traces_utils,
            Path(td),
            episodes=episodes,
            success_filter=success_filter,
            include_instruction=include_instruction,
            include_verifier_output=include_verifier_output,
            verbose=verbose,
            include_literal_tokens=include_literal_tokens,
        )
        if rows:
            chunk.extend(rows)

    if not chunk:
        return 0

    shard_path = Path(tmp_root) / f"train-{shard_idx:05d}.parquet"
    cache_dir = Path(tmp_root) / f"_cache-{shard_idx:05d}"
    written = _flush_chunk_to_parquet(
        chunk, to_sharegpt, traces_utils, shard_path, cache_dir
    )
    if not written:
        shard_path.unlink(missing_ok=True)
        return 0

    api = HfApi()
    _upload_shard(api, repo_id, shard_path)
    return written


def _run_shard_subprocess(kwargs: Dict[str, Any]) -> int:
    """Run one shard in a spawned subprocess and return its row count.

    On any subprocess crash (including OOM-kill of the child, which no longer
    takes down the parent) we raise so the caller can decide. Because shards
    are additive commits, a crash leaves prior shards intact.
    """
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=1, maxtasksperchild=1) as pool:
        return pool.apply(_shard_worker_entry, (kwargs,))


def _shard_worker_entry(kwargs: Dict[str, Any]) -> int:
    """Top-level (picklable) entry the spawned worker runs."""
    return _process_one_shard_in_subprocess(**kwargs)


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
    include_literal_tokens: bool = False,
) -> int:
    """Single streaming path: enumerate trials, write parquet shards, upload via HfApi.

    Memory is bounded to roughly one ``chunk_size`` worth of TRIALS at any time.
    The parent process only enumerates trial directories (cheap, pruned os.walk)
    and batches their path strings; each batch's collect + parquet-write +
    upload happens in a FRESH spawned subprocess that exits afterward, so the OS
    reclaims the per-chunk native-memory leak (Harbor collection + ``datasets``
    Arrow buffers) that gc/pyarrow-release alone do NOT free. Peak parent RSS is
    flat across shards. Each shard is an additive hub commit, so a mid-run
    failure leaves prior shards intact + resumable. Returns the total row count.
    """
    from huggingface_hub import HfApi  # type: ignore

    traces_utils = _import_traces_utils()

    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    print(f"[trace-export] Repo {repo_id} ensured (private={private}).")

    # Working root: each shard is written here by its subprocess, uploaded, then
    # deleted. We never keep more than one shard's worth of parquet on disk.
    tmp_root = Path(tempfile.mkdtemp(prefix="trace_shards_"))

    total_rows = 0
    shard_idx = 0

    def _dispatch(batch: List[str]) -> None:
        nonlocal total_rows, shard_idx
        if not batch:
            return
        written = _run_shard_subprocess(
            dict(
                trial_dirs=batch,
                shard_idx=shard_idx,
                repo_id=repo_id,
                episodes=episodes,
                success_filter=success_filter,
                to_sharegpt=to_sharegpt,
                include_instruction=include_instruction,
                include_verifier_output=include_verifier_output,
                tmp_root=str(tmp_root),
                verbose=verbose,
                include_literal_tokens=include_literal_tokens,
            )
        )
        if written:
            total_rows += written
            shard_idx += 1
            print(f"[trace-export] Shard {shard_idx - 1:05d} done ({written} rows; running total {total_rows}).")

    try:
        n_trials = 0
        batch: List[str] = []
        for trial_dir in _iter_trial_dirs_nonrecursive(traces_utils, job_dir):
            n_trials += 1
            batch.append(str(trial_dir))
            # Batch by TRIAL count — the driver of per-shard memory. (rows/trial
            # varies, but a fixed trial-batch keeps each subprocess bounded.)
            if len(batch) >= chunk_size:
                _dispatch(batch)
                batch = []
        _dispatch(batch)  # final partial batch

        print(f"[trace-export] Enumerated {n_trials} trial dirs; collected {total_rows} rows into {shard_idx} shard(s).")

        if shard_idx == 0:
            # Still write+push an empty dataset so downstream consumers see a
            # valid (zero-row) repo rather than nothing — mirrors prior behavior.
            empty_path = tmp_root / "train-00000.parquet"
            empty_ds = traces_utils.rows_to_dataset([])
            empty_ds.to_parquet(str(empty_path))
            del empty_ds
            _release_arrow_memory()
            _upload_shard(api, repo_id, empty_path)
            print("[trace-export] No rows collected; uploaded an empty parquet shard.")
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
        help="Trials per parquet shard / per subprocess (bounds peak memory). Default: 200",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose per-trial logging",
    )
    p.add_argument(
        "--include_literal_tokens",
        action="store_true",
        help="Emit per-turn prompt_token_ids / completion_token_ids / logprobs columns "
        "from step.metrics (for agents with SUPPORTS_LITERAL_TRACES, e.g. opencode run "
        "behind the RecordProxy via --record_literal). When set, Harbor's native "
        "literal-aware extraction is used (the inline-subagent-merger patch, which drops "
        "token columns, is skipped). Default off = current text-only behavior. Pass this "
        "when the job was generated with hpc.launch --record_literal.",
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
        include_literal_tokens=bool(args.include_literal_tokens),
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
