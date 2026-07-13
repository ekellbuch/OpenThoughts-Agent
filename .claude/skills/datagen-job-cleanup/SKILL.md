---
name: datagen-job-cleanup
description: >-
  Post-run cleanup for a datagen (trace-generation) job on an HPC cluster (Jupiter/Leonardo/Perlmutter):
  get the generated traces onto HF (penfever org) and free disk. There is NO model checkpoint — the
  artifact is the trace dataset. Covers the TIMEOUT-strands-traces gotcha (uploads silently never run),
  the ONE-level trace_jobs nesting (vs RL's double-nest), the real-vs-failed sanity check (avg_turns ≈ 1.0
  = dead run, don't upload), the otagent-env uploader, the non-empty HF verify, and safe disk cleanup
  (one-off task dirs vs shared canonical tasks; leave Daytona snapshots). Use when a datagen/trace job
  finishes (COMPLETED or TIMEOUT) and its traces need uploading + verifying, or when consolidating a
  chunked datagen run. Distinct from RL/SFT cleanup (those publish a model checkpoint).
---

# datagen-job-cleanup

After a datagen (trace-generation) job terminates, follow these steps to get the traces onto HF and free
disk. Unlike RL/SFT there is no model checkpoint — the artifact is the trace dataset.

## 0. Recognize the common case: TIMEOUT stranded the traces
Datagen jobs run to a wall-clock `--time_limit`; when they TIMEOUT, Harbor's terminal trace-upload step is
killed mid-run, leaving completed trials on disk but **no upload**. (Seen repeatedly: MiniMax chunks
525868/9, GLM-5.1 525611/528568.) A clean COMPLETED exit usually uploads automatically; a TIMEOUT almost
never does. Either way, do NOT assume the upload happened — verify (step 4).
```bash
sacct -j <jobid> -X --format=JobID,JobName%55,State,Elapsed,End -n
```

## 1. Locate the trace_jobs dir (ONE-level nesting for datagen)
Datagen writes to `<run_dir>/trace_jobs/<inner_run_name>/<task>__<id>/` — a SINGLE level of nesting under
`trace_jobs/`, NOT the double-nest `<run>/<run>/trace_jobs` of the RL path (two subagents tripped on this).
The `--job_dir` you pass to the uploader must be the dir that directly CONTAINS the `<task>__<id>` trial
dirs, i.e. `trace_jobs/<inner_run_name>`. Find it:
```bash
# Direct-ssh launches land under experiments/; --experiments_dir launches under ot-baf.
RUN=/e/scratch/jureap59/feuer1/OpenThoughts-Agent/experiments/<job_name>   # or /e/data1/.../ot-baf/<job_name>
INNER=$(ls -d $RUN/trace_jobs/*/ 2>/dev/null | head -1); echo "$INNER"
```

## 2. Sanity-check the trials are REAL before uploading
A served `/v1/models` healthcheck does NOT mean generation worked. Compute the average turn count and the
exception rate; if avg turns ≈ 1.0, the run is near-total failure (e.g. GLM-5.1 528568: endpoint healthy +
973 result.json, but every trial was a 1-turn `InternalServerError` from a dead EngineCore — do NOT
"upload" that). A real run has multi-step trajectories (turns > 1) and a tolerable exception rate (tezos
datagen runs ~20-25% AgentTimeout is normal).
```bash
# trajectory.json is a dict with a "steps" list; turns ≈ len(steps).
$OTAGENT_PY - <<'PY'
import json, glob, os, statistics
inner = os.environ["INNER"]
dirs = glob.glob(os.path.join(inner, "*__*/"))
turns, exc = [], 0
for d in dirs:
    tj = os.path.join(d, "agent", "trajectory.json")
    if os.path.exists(tj):
        t = json.load(open(tj))
        turns.append(len(t.get("steps", [])) if isinstance(t, dict) else len(t))
    r = os.path.join(d, "result.json")
    if os.path.exists(r):
        if (json.load(open(r)).get("exception_info") or {}).get("exception_type"): exc += 1
print(f"trials={len(dirs)} avg_turns={statistics.mean(turns):.2f} exceptions={exc}" if turns else "no trajectories")
PY
```
If avg_turns ≈ 1.0 → the run failed; do NOT upload. Diagnose instead (read a trial's `exception.txt` + the
`_vllm.log` for the engine-side error) and write an agent_log; the traces are not worth keeping.

## 3. Upload the traces to HF penfever org
From the `otagent` conda env — the uploader needs `google.cloud.storage` + matplotlib, which `envs/rl` lacks:
```bash
# otagent env, source secrets first. $INNER = the dir DIRECTLY containing the <task>__<id> trial dirs
# (for a gs:// rescue that is /tmp/<job>_traces/<job>, the INNER subdir — see the gs:// trap below).
python -m scripts.harbor.make_and_upload_trace_dataset \
  --job_dir "$INNER" \
  --repo_id penfever/<descriptive-name> \
  --episodes last \
  --served_model <the exact served model ref> \   # required for decodable literals
  --include_literal_tokens \                       # REQUIRE literals (fail loud if 0 bind)
  --single_commit                                  # ONE HF commit — avoids the 128-commit/hr 429 on big sets
```
Default to the `penfever/` org and `--episodes last`. For a chunked launch, upload each chunk's trace_jobs
to its own `_chunk{i}` repo. Public by default (per `feedback_hf_public_default`).

**Literals are AUTO-INCLUDED when present (no flag needed).** The uploader now FAVORS the durable
`literal.jsonl`: it auto-discovers the sibling `<experiments_dir>/logs/*_literal.jsonl` (searching `--job_dir`
and a few parents) and, when found, correlates the RecordProxy records into the trajectory step metrics so the
exported dataset carries the trainable `prompt_token_ids` / `completion_token_ids` / `logprobs` columns.
Requirements + knobs:
- **`--job_dir` must sit inside the experiments-dir tree** so the parent-walk reaches `…/logs/`. `$INNER` (the
  trial-containing dir) qualifies as long as the local run dir still has its sibling `logs/` (local runs do).
- **⚠ gs:// iris rescue — the INNER `<job>/` is the `--job_dir`, NOT the outer rescue root (2026-07-12 trap).**
  You rsync the **OUTER** `gs://…/ot-agent/<job>/` → `/tmp/<job>_traces` so `logs/*_literal.jsonl` rides along
  (per monitor-cron-sweep-iris §4a). But the trial dirs sit one level down at `/tmp/<job>_traces/<job>/<trial>`,
  so the `--job_dir` you PASS must be **`/tmp/<job>_traces/<job>`** (the inner subdir that DIRECTLY contains the
  `<task>__<id>` trial dirs). Passing the outer `/tmp/<job>_traces` still finds the trials by recursion so TEXT
  rows upload, but the literal correlation keys off the trial path relative to `--job_dir` → the extra `<job>/`
  prefix makes **0 trials bind → silent text-only dataset** (with `--include_literal_tokens` it then fails loud
  and leaves the repo empty/partial). Always point `--job_dir` at the inner `<job>/` for a gs:// rescue. (Also:
  `gsutil rsync` needs the local dest to EXIST — `mkdir -p /tmp/<job>_traces` before the rsync.)
- **⚠ Big datasets → add `--single_commit` (HF 128-commits/hour cap).** The uploader commits one shard per
  commit (~1 commit / `chunk_size`=200 rows). A multi-thousand-row set + any re-attempt blows past HF's **128
  repo-commits/hour** limit → `429 Too Many Requests` mid-push → a **partial** dataset (only the shards that
  committed before the 429). `--single_commit` pushes the whole dataset in ONE commit → never trips the limit.
  Use it for any real campaign arm. (If you already 429'd, the repo may be partial or empty; wait ~1h for the
  window to reset, then re-run with `--single_commit`.)
- **Correlator memory (2026-07-12 fix) — one upload at a time on a RAM-limited host.** The literal correlator
  (`scripts/harbor/literal_correlator.py`) now **streams** each `*_literal.jsonl` line-by-line (was
  `read_text().splitlines()`, which held the whole multi-GB file as a string + a full line-list before parsing).
  But the correlation still holds ALL parsed records (token ids) in memory for chain-reconstruction, so a big
  arm (e.g. a 1.5 GB literal file) is inherently RAM-heavy — run **ONE upload at a time**, never two concurrent
  (that flooded a 36 GB Mac). If even one-at-a-time OOMs, offload to a bigger box or drop `messages` from records
  after `reconstruct_chains`.
- A job with **no** `literal.jsonl` exports text-only, byte-identical to before (parity).
- `--no_literal_tokens` forces text-only even when a `literal.jsonl` is present.
- `--include_literal_tokens` now means REQUIRE: it fails loud if literals are expected but none are found (use
  it in a cron to guarantee a `--record_literal` job never silently ships literal-less).
- The uploader FAILS LOUD if a `literal.jsonl` is present but 0 trials bind (a regression, not a valid dataset).
- **Pass `--served_model` on any literal upload.** The token-id columns are only decodable with the EXACT
  tokenizer the engine served with (a same-family tokenizer decodes word tokens to garbage), so the uploader
  stamps the model ref into `tokenizer_provenance.json` + the dataset-card README. For the opencode-131k
  campaign that is `--served_model Qwen/Qwen3.5-122B-A10B-FP8` (or the gs:// mirror
  `gs://marin-models-us/ot-agent/models/Qwen/Qwen3.5-122B-A10B-FP8/` — the storage root resolves via
  `marin_prefix()`; see `.claude/ops/iris/coreweave_gpu_ops.md` §rendezvous, don't hardcode the region bucket).
  Omitting it still uploads the columns
  but only stamps the engine-reported served-name (a warning prints) — always pass the real ref so consumers
  can pull the tokenizer.
- **Schema-pin (fixed OT-Agent `7c978b78`):** the exporter now pins the literal token columns to an explicit
  nested type per shard. Before this fix, a chunk whose leading rows had no literals dropped the token lists of
  every other row in the chunk → under-populated literal yield + whole shards with no token columns (and
  `load_dataset` `CastError`); `--chunk_size` larger made it worse. Datasets uploaded before `7c978b78` are
  degraded — **re-rescue them to recover full yield.** Full reference (decoding, tokenizer provenance,
  re-rescue procedure, literals→SFT): **`.claude/projects/harbor/harbor.md`** (§ Literal-token trace datasets).

## 4. Verify the HF dataset is non-empty
The repo may exist as a 0-row shell (a prior failed/partial upload, or Harbor pre-creating it); an existing
repo is NOT proof of success. Confirm row count:

> **⚠ Re-uploading over a repo that had a prior TEXT-ONLY upload (2026-07-12).** A leftover
> `train-00000-of-00001.parquet` from the earlier text-only push has a **no-token schema** that can MASK the
> new token-bearing shards → `count_populated_literal_rows` reads **0** even though the re-upload landed. If a
> post-upload verify shows 0 populated literals but the uploader logged a good `Literal yield`, **delete the
> stale text-only parquet(s) from the repo** (`huggingface_hub.HfApi().delete_file`) and re-verify. Cleaner:
> re-upload to a **fresh repo id** to avoid stale-shard masking entirely.
```bash
curl -s -H "Authorization: Bearer $HF_TOKEN" \
  "https://huggingface.co/api/datasets/penfever/<descriptive-name>" \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print('files:', len(d.get('siblings',[])), 'lastMod:', d.get('lastModified'))"
```
The uploader's own "Generating train split: N examples" line is the ground-truth count — N should match the
real (non-1-turn) trial count from step 2. Zero files / 0 rows = the upload did not land; re-run step 3.

**Also verify the literal columns landed (jobs run with `--record_literal`).** The uploader prints a
`[trace-export] Literal yield: X/Y trials …` line — X should be > 0. Post-upload, confirm the columns are
populated in the pushed dataset (>0 non-empty `prompt_token_ids` rows):
```python
import datasets
from scripts.harbor.make_and_upload_trace_dataset import count_populated_literal_rows
ds = datasets.load_dataset("penfever/<descriptive-name>", split="train")
print("rows w/ literals:", count_populated_literal_rows(ds.data.table))   # must be > 0 for a --record_literal job
```
0 populated rows on a `--record_literal` job = the literals dropped — re-run step 3 with the correct
`--job_dir` (must reach the sibling `logs/`), or pass `--literal_log <gs://…/logs/<slug>_literal.jsonl>`.

## 5. Clean up disk (only after step 4 confirms the upload)
- **Remove the run/experiments dir** (trace_jobs is the bulk — tens of GB for a full tezos run):
  ```bash
  rm -rf $RUN
  ```
- **Remove the task directory IF it was a one-off / temp set** created just for this run — e.g. the symlink
  subsets under `…/ot-baf/tmp_tasks/<name>` made for a chunking smoke test, or a
  `scripts.datagen.extract_tasks_from_parquet` output you won't reuse. Do NOT delete the shared canonical
  task dirs under `/e/data1/datasets/playground/ot/tasks/<benchmark>` — those are reused across runs.
  ```bash
  rm -rf /e/data1/datasets/playground/ot-baf/tmp_tasks/<one-off-name>   # only if one-off
  ```
- **Daytona snapshots**: leave them — snapshot lifecycle is managed manually (snapshots are keyed by
  task-environment hash and shared across every run using the same tasks, so per-run deletion is unsafe).

---

## Operating notes (folded from memory 2026-06-14)

- **Auto-advance the MiniMax-M2.7 131k queue without asking between datasets** (96-row queue; cycle is mechanical). Tracker = `experiments/active/datagen/minimax-m2.7-tt/tracker.md`. Per-dataset cycle: extract tasks → launch chunks (`chunk_size 500`, `--chunk_array_max 5`, single-node, verifier-ON) → wait ALL chunks COMPLETE (validate via sacct, not empty squeue) → consolidate `_chunk{N}` via `join_hf_repos.py` + **verify row count == sum, each chunk near-full, no run_id gaps, realness (avg_turns>1)** → DELETE chunk repos (only after consolidated verified) → clean local dirs → launch next row. Target repo: `penfever/<source-basename>-minimax-m27-131k-traces`. **Only hard auto-skip = oversized-extraction rows** (≫10k rows busting the inode budget, e.g. knowledge-mcqa 616k) → ask. Snapshots: do NOT reclaim per-dataset (shared envs); track cumulative tally vs the org cap (see daytona doc). The 3h cron is the natural driver.
- **LLM-as-judge verifier rows (`openai/gpt-4o-mini`) can be silently reward-dead** (all 0.0) if `OPENAI_API_KEY` is invalid/revoked → judge 401 → reward defaults 0.0 while traces are genuine (`avg_turns` healthy). The tell at consolidation: `verifier_output` has `judge API error … 401` on ~all rows. **So check the REWARD DISTRIBUTION at consolidation, not just avg_turns.** FIXED 2026-06-11 (wired the valid `OPENAI_OUMI_KEY` into `OPENAI_API_KEY` on all 3 hosts; verified row #25 = 90% solve, 0 auth-401s). Rows generated BEFORE the fix (e.g. #22) stay reward-dead unless re-scored. User precedent: accept reward-dead rows as turns-only SFT data + advance (record the caveat in the tracker row). Test a key: `curl …/v1/chat/completions -H "Authorization: Bearer $OPENAI_API_KEY" …` → expect 200.
