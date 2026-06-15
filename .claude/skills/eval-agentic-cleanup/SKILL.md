---
name: eval-agentic-cleanup
description: >-
  Recover a finished agentic eval whose automatic HF-trace-upload or Supabase DB-registration failed (path
  mismatch, missing result.json, bogus model name): run manual_db_eval_push.py against the trace_jobs dir,
  fix the vLLM-numeric-ID → real-HF-model-name registration (with cross-user FK safety), verify in Supabase,
  then free disk. Use when an eval COMPLETED but didn't upload/register, when a sweep finds an eval with a
  technical DB-registration failure, or to re-register/correct an eval's model. Distinct from the model-
  publishing cleanups (rl-job-cleanup / sft-job-cleanup) and datagen-job-cleanup — this is the EVAL path.
---

# eval-agentic-cleanup

Normal evals auto-upload their traces + auto-register in Supabase on completion. Use this when that
**didn't happen or went wrong** (path mismatch, missing `result.json`, or a bogus/numeric model name).

## 0. First — should this eval be uploaded at all?
**EXEMPT (do NOT upload/register):** grid / throughput / OOM-test **measurement** runs targeting
`DCAgent2/*` — upload only the **production winners**, not the calibration sweeps. If it's a measurement
run, stop here. (Per the cron sweep directive.)

## 1. Manual upload + DB register — `manual_db_eval_push.py`
Pass the **`trace_jobs/<RUN_TAG>`** path (where Harbor writes the `<task>__<id>` trial dirs), **NOT**
`eval_jobs/<RUN_TAG>` (that only has `meta.env`). The script auto-resolves nested trial subdirs and
auto-detects agent/model/benchmark from job metadata.
```bash
source ~/secrets.env   # needs SUPABASE_URL, SUPABASE_ANON_KEY, HF_TOKEN
cd <OpenThoughts-Agent>
python scripts/database/manual_db_eval_push.py --job-dir trace_jobs/<RUN_TAG> --verbose
#   --hf-repo DCAgent2/<RUN_TAG>-traces   # explicit HF repo
#   --skip-hf                             # DB only (traces already uploaded)
#   --forced-update                       # overwrite existing records
```

## 2. ⚠️ CRITICAL — verify + fix the model name (with cross-user FK safety)
The script auto-detects the model from trial `result.json` → `agent_info.model_info.name`. For vLLM-served
models that field is the **vLLM served-model name** (a numeric ID like `1774950145766573`), NOT the HF repo
— so it can register a **bogus numeric `models` row**. Get the real name from the eval config:
```bash
python3 -c "import json; d=json.load(open('experiments/<RUN_TAG>/configs/<RUN_TAG>_eval_config.json')); print(d['model_hf_name'])"
```
Then check what got registered:
```python
c.table("sandbox_jobs").select("model_id,username").eq("id", "<JOB_ID>").execute()
c.table("models").select("name").eq("id", "<MODEL_ID>").execute()
```
If it's a numeric ID, repoint to the real model — **but FIRST the cross-user FK safety pre-check**
(`feedback_supabase_filter_username`): you are about to UPDATE `sandbox_jobs` / `sandbox_trial_model_usage`
and DELETE a `models` row. **Only touch rows you OWN.** If the `sandbox_jobs` row (or any FK'd
`sandbox_trial_model_usage` row) belongs to ANOTHER user, **STOP** and surface it — do NOT repoint or delete
it. (Mutating another user's eval rows is exactly what broke `zhuang1`'s eval jobs on 2026-05-26.)
```python
import os; me = os.environ.get("USER")
job = c.table("sandbox_jobs").select("id,username,model_id").eq("id", "<JOB_ID>").execute().data[0]
assert job["username"] == me, f"FK-SAFETY STOP: job owned by {job['username']}, not {me} — do not mutate"
correct = c.table("models").select("id").eq("name", "laion/<real-model-name>").execute()
c.table("sandbox_jobs").update({"model_id": correct.data[0]["id"]}).eq("id", "<JOB_ID>").eq("username", me).execute()
c.table("sandbox_trial_model_usage").update({"model_id": correct.data[0]["id"]}).eq("model_id", "<BOGUS_ID>").execute()  # only if all FK'd rows are yours
c.table("models").delete().eq("id", "<BOGUS_ID>").execute()  # only if no other-user row FKs it
```

## 3. Verify it landed
Confirm `sandbox_jobs.<JOB_ID>.model_id` now points to the real `laion/<name>` row and the trial scores are
attached. If `--skip-hf` wasn't used, confirm the trace dataset (`DCAgent2/<RUN_TAG>-traces`) is non-empty on HF.

## 4. Clean up disk (only after upload + register verified)
Remove the local eval run dir once the traces are on HF + the DB row is correct (the `trace_jobs` tree is
the bulk). Detach a large GPFS `rm` (nohup/tmux); never `du`/`find` to size it first. Do NOT delete shared
canonical task dirs.

---
Sibling cleanups: **`rl-job-cleanup`** (RL model), **`sft-job-cleanup`** (SFT model), **`datagen-job-cleanup`** (trace dataset). Launching evals → **`eval-agentic-launch`**. Per-cluster particulars → `.claude/ops/<cluster>/`.

---

## Operating notes (folded from memory 2026-06-14)

- **`notes/ot-agent/task_repos/rl_to_check.md` is a QUEUE file, not documentation** — bare HF repo URLs, one per line, consumed line-by-line by the smoke-test runner (processed entries move to `rl_checked.md`, same flat format). "Update rl_to_check.md with the fixes" = **append the newly-uploaded repo URLs**, one per line. Do NOT add markdown tables / sections / writeups (breaks the parser). Fix notes belong in the chat response or a dedicated `rl_fixes_<date>.md`. Same caution for all `task_repos/*.md` (flat URL/path lists feeding tooling).
