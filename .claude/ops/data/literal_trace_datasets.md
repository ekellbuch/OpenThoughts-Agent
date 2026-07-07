# Literal-token trace datasets (opencode datagen)

Canonical reference for the `--record_literal` opencode trace datasets and their
downstream SFT use. Applies to the opencode-131k campaign
(`penfever/<task>-qwen3.5-122b-131k-opencode-traces`) and any future literal datagen.

## What the literal columns are

`make_and_upload_trace_dataset` on a `--record_literal` job emits three parallel columns
alongside the text `conversations`:

- `prompt_token_ids`, `completion_token_ids` — **list-of-lists**, one inner list per agent
  step (turn). The verbatim tokens the serving engine emitted (RecordProxy capture, correlated
  by `literal_correlator.py`).
- `logprobs` — list-of-lists of floats, same shape as `completion_token_ids`.

A row is "literal-populated" when `prompt_token_ids` is non-empty
(`count_populated_literal_rows` checks exactly this).

## Decoding the token IDs → text

The IDs only decode with the **EXACT tokenizer the engine served with**. For the 131k
campaign that is **`Qwen/Qwen3.5-122B-A10B-FP8`** (a `qwen3_5_moe`, `Qwen2Tokenizer`, **vocab
248044**, specials at 248044–248076). A stock Qwen3 tokenizer (~152k vocab) shares only the
low-id digit/whitespace/structure tokens, so word tokens decode to **garbage** — verified.

- GCS mirror (guaranteed source; the HF repo may be gated):
  `gs://marin-models-us/ot-agent/models/Qwen/Qwen3.5-122B-A10B-FP8/` — pull just
  `tokenizer.json`/`tokenizer_config.json`/`vocab.json`/`merges.txt` (~22 MB, no weights).
- Which tokenizer produced a given dataset is recorded in its **`tokenizer_provenance.json`**
  (+ a README decode recipe), written by `make_and_upload_trace_dataset` when you pass
  `--served_model`. **Always pass `--served_model Qwen/Qwen3.5-122B-A10B-FP8` on any literal
  upload/rescue** — omitting it still uploads the columns but stamps only the engine-reported
  served-name (a warning prints).
- Decode per-turn (list-of-lists) with `skip_special_tokens=False` to keep
  `<|im_end|>`/`<tool_call>`/`<think>` markers.

## Export schema-pin bug — FIXED (OT-Agent `7c978b78`), but old datasets are still degraded

**Symptom:** a job whose literal correlation enriched N trials landed far fewer literal rows
in parquet; whole shards came out with **no token columns at all**, and `load_dataset` fails
with a schema `CastError` from the heterogeneous shards.

**Root cause:** `make_and_upload_trace_dataset` wrote the token columns with pyarrow/HF type
inference driven by each **chunk's leading rows**. A chunk whose first rows had no literals
inferred a null/empty type and **silently dropped the token lists of every other row in that
chunk**. `--chunk_size` larger makes it WORSE (bigger chunks → more rows lost per null-leading
chunk), not better.

**Fix:** `_pin_literal_token_columns` rebuilds the three token columns from the source rows
with an explicit `Sequence(Sequence(int64/float64))` type as the last step before writing each
shard (the chunk pipeline preserves row order, so `ds[i]` aligns to `rows[i]`). Every
literal-job shard now carries the columns with a stable schema; text-only exports are
unchanged (gated on `include_literal_tokens`).

**ACTION for datasets uploaded before `7c978b78`:** they are under-populated (whole shards
missing literals) — **re-rescue them with the fixed exporter to recover full yield.** Observed:

| arm | pre-fix literal | post-fix literal | status |
|---|---|---|---|
| #6 exp_rpt_stack-junit-v6 | 205 / 858 | **660 / 858** | re-rescued 2026-07-06 (`7c978b78`) |
| #1 inferredbugs | 5441 / 5827 (shard 00005 empty) | **5633 / 5827** | re-rescued 2026-07-06 (`8c588783`, 3-file union) |
| #2 code-contests | 4240 / 4914 (shards 00001,00019 empty) | **4616 / 4914** | re-rescued 2026-07-06 (`8c588783`, 2-file union) |
| #3 nemotron-code-oracle | 5351 / 5688 (tail shard 00028 empty) | **5441 / 5688** | re-rescued 2026-07-06 (`8c588783`, 2-file union) |

Re-rescue = rsync the OUTER `gs://marin-models-{us,eu}/ot-agent/<job>/` (so sibling
`logs/*_literal.jsonl` rides along), clear the target repo's partial `data/` (keep README +
`tokenizer_provenance.json`), then re-run the uploader with `--served_model` from the
otagent env. Verify with `count_populated_literal_rows` that literal count ≈ the correlation
yield, not the old partial.

## Why in-job auto-upload is unreliable → rescue is MANDATORY, not optional

Two independent reasons the worker's end-of-job HF upload cannot be trusted for these
preemptible (v5p + `--max-retries`) datagen jobs — so **every terminal job must be rescued
from banked GCS**, and "SUCCEEDED" or "repo exists" is never proof of trainable data:

1. **Text-only when it does run** — the pinned `:tpu` worker image predates the schema-pin
   fix, so its in-job export lands the `conversations` text but drops the literal token
   columns (see above). Observed on #4/#5/#7/#8/#9/#12.
2. **Fails outright on preempt-resumed runs (SYSTEMIC, harbor bug).** The end-of-job
   `harbor jobs start … --export-push` step **re-invokes `harbor jobs start`** against the
   already-populated gs:// job dir. On a preempt-resumed run the on-disk config no longer
   byte-matches the relaunch config, so `Job.create` (`harbor/job.py:263 _maybe_init_existing_job`)
   raises `FileExistsError: Job directory <gs://…/<job>/<job>> already exists and cannot be
   resumed with a different config` **before export runs** → the job goes FAILED (state 5)
   with no upload. Deterministic (not HF 429 / not OOM); recurs on any preempted job.
   Observed on #10 (recovered 4081/4095) and #11 (339/381). The banked trials + the durable
   `logs/*_literal.jsonl` are intact in GCS, so the rescue recovers full yield regardless.

Consequence: `FAILED at export-push` ≈ `landed text-only` — both mean "rescue from GCS." The
3-hourly ops cron auto-rescues every terminal job for exactly this reason. (Candidate real
fixes, for later: rebuild the `:tpu` worker with the schema-pin fix so #1 goes away; and make
harbor's `--export-push` not re-enter `Job.create` / tolerate the existing dir so #2 goes
away — until then rescue is the reliable path.)

## Literal traces → SFT

`scripts/harbor/literal_traces_to_sft.py` converts a literal trace dataset into an SFT
dataset whose **assistant turns are decoded verbatim from the literal completion tokens**
(real `<think>` + native tool calls). It emits `conversations` (ShareGPT) + a
reasoning-preserving `text` string, and **auto-resolves the tokenizer from the source's
`tokenizer_provenance.json`** (override with `--tokenizer`). Rows without literals, or whose
assistant-turn count ≠ literal step count, are dropped. `--validate N` dry-runs. First output:
`laion/nemotron-code-oracle-qwen3.5-122b-opencode-sft` (5351 rows).

**LAION upload gotcha:** the `laion` org's PRIVATE storage quota is full — push SFT datasets
there **public** (public storage isn't quota-capped). A private push succeeds but 403s on
readback (`Private repository storage limit reached`).
