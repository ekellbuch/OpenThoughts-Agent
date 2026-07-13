---
name: crud-purge-below-gate-evals
description: >-
  Guardrailed DELETE of auto-registered eval `sandbox_jobs` rows that DID score but FAILED the
  harvest gate — partial evals (valid-complete <90% or non-benign infra-error >10%). These are the
  "DE-REGISTER candidate" rows the flawed_summ harvest defers.
  DISTINCT from `crud-purge-stale-eval-placeholders` (that removes NEVER-populated Pending/Started
  rows; this removes rows that populated stats/metrics but are below-gate). Use when asked to remove
  "partial / below-gate / <90%-completed-without-errors" evals owned by us. The MANDATORY parts:
  the authoritative gate utility (`scripts/database/eval_guardrail.py` — NEVER hand-roll the BENIGN
  set / error count), the cross-user FK-safety pre-check, the grandchild→child→job cascade, and
  REPORTING BACK the exact purged rows (§5) so the supervisor knows which state docs (e.g. flawed_summ
  STATE.md) to reconcile.
---

# crud-purge-below-gate-evals

A **guardrailed DELETE** that removes auto-registered eval `sandbox_jobs` rows which **scored but
failed the harvest gate** — the incomplete/contaminated evals a re-eval campaign flags as
"DE-REGISTER candidate DEFERRED." These rows DID populate `stats`/`metrics` (unlike the stale
placeholders that never ran), so they pass the placeholder purge's filter and need a **completion +
error-quality gate** instead.

> **This is a DELETE on the shared `sandbox_jobs` table. The AgentTimeout-benign gate (§1), the
> cross-user FK-safety pre-check (§2), and the grandchild→child→job cascade (§3) are ALL mandatory.**
> DRY-RUN with boundary controls first, review, then delete, then re-read.
>
> **⚠ THE #1 TRAP — a naïve `n_errors < 10%` filter deletes almost everything.** `stats` error
> counts include **`AgentTimeoutError`**, which is a *legitimate reward-0 model outcome* (a weak
> model exhausting its per-task time budget), NOT an infra failure. In one real run the literal
> formula flagged **286 of 293 rows (97.6%)**; the correct benign-aware gate flagged **32**. You
> MUST treat AgentTimeout (and the other benign exceptions in §1) as non-errors.

## 0. Connect (run LOCALLY)

Same as `crud-otagent-supabase` / the placeholder purge — Mac, `otagent` env, service-role key:
```bash
cd /Users/benjaminfeuer/Documents
set -a; source "${DC_AGENT_SECRET_ENV:?set DC_AGENT_SECRET_ENV to the secrets file first}"; set +a
```
`SUPABASE_SERVICE_ROLE_KEY` bypasses RLS → §2 is mandatory. Schema DDL: `OpenThoughts-Agent/schema/`.
**PAGINATE** (`sandbox_jobs` ~9000 rows > the 1000-row default).

## 1. The gate — what makes an eval "below-gate" (removable)

**The COUNTS come from ONE authoritative utility — do NOT hand-roll them here.**
`OpenThoughts-Agent/scripts/database/eval_guardrail.py` is the single Python port of the leaderboard's
guardrail (`OT-Agent-Leaderboard server/storage.ts:416-449` — the "Errors: k" badge). Hand-rolling the
BENIGN set / error-count is exactly how this skill's old inline gate DRIFTED (a wrong BENIGN member, an
invented AgentTimeout axis, a per-eval sum that under-counted). Use the utility:

```python
import sys; sys.path.insert(0, "OpenThoughts-Agent/scripts/database")
from eval_guardrail import guardrail_counts, passes_gate
```
- `guardrail_counts(stats, planned)` → the authoritative counts: `invalid_error_count`, `is_high_errors`,
  `is_incomplete`, `attempted_n_trials`, `planned_n_trials`.
- `passes_gate(stats, planned, max_invalid_errors=…, min_complete_frac=…)` → `(passed, reason)`.

The THRESHOLDS are a POLICY decision — read the campaign's `POLICY.md §"The gate"` each run in case they
moved. For flawed_summ (`planned = sandbox_jobs.n_trials` = 300 swe/dev_set_v2, 267 = 89×3 tb2) the gate is
**non-benign ≤ 10%** and **valid-complete ≥ 90%**, applied to the utility's counts:

```python
def gate(job):
    """(below_gate: bool|None, reason: str). planned = job['n_trials']. Counting delegated to eval_guardrail."""
    planned = job.get("n_trials") or 0
    if not planned:
        return (None, None)                          # can't score → SURFACE, do not delete
    passed, reason = passes_gate(
        job.get("stats"), planned,
        max_invalid_errors=round(0.10 * planned),    # campaign "non-benign <= 10%"
        min_complete_frac=0.90,                       # campaign "valid-complete >= 90%"
    )
    return (not passed, reason)                       # below_gate == not passed
```

Row is **BELOW-GATE (DELETE candidate)** iff `gate(job)[0]` is True; **GOOD (KEEP)** iff False; **None** =
unscoreable → surface, never delete.

> **⚠ The old inline gate had an `AgentTimeout ≥ 80%` "contamination" axis. It is NOT in the leaderboard's
> canonical guardrail and it false-flagged valid low-scoring evals — it is DROPPED here. If the campaign
> wants an AgentTimeout-rate discriminator back, add it as an explicit POLICY axis over `guardrail_counts`,
> not as a hand-rolled reimplementation.**

**Scope of candidates:** rows with `username IN OURS` that actually HIT the DB with results —
`job_status='Finished'` OR non-null `stats`/`metrics`. INCLUDE `Started` rows only if they recorded
partial trials (0% valid-complete → below-gate) or are explicitly tracker-listed. EXCLUDE pure
never-populated `Pending`/`Started` placeholders — those are `crud-purge-stale-eval-placeholders`.
Time-box to "recent" (default `created_at ≤ 7d`) OR any id the tracker explicitly lists as a
de-register candidate.

## 2. Cross-user FK safety (MANDATORY pre-check)

`OURS = {"bfeuer00", "penfever", "feuer1", "benjaminfeuer"}` (our eval/re-eval owners). **Only delete
rows you own.** Rows owned by other users (`zhuang1`, `richard.zhuang`, …) are **REPORTED, never
deleted** — a re-eval campaign's cross-user baselines live in that table and breaking them corrupts
someone else's tables. Scope the match AND the delete by `username IN OURS`.

## 3. The cascade — `sandbox_jobs.id` IS FK'd (REQUIRED)

Same chain as the placeholder purge — a plain `sandbox_jobs` delete FK-fails Postgres `23503`:
```
sandbox_trial_model_usage.trial_id → sandbox_trials.id → sandbox_jobs.id
```
These below-gate evals ran real trials, so expect **many** child rows (one real run deleted 32 jobs →
~6,900 `sandbox_trials` + their grandchildren). Delete grandchild → child → job. Children carry no
`username`; ownership is transitive from the job you own (§2).

## 4. Procedure — DRY-RUN + boundary controls, review, delete, re-read

```python
OURS = {"bfeuer00","penfever","feuer1","benjaminfeuer"}
def page(t, cols):
    out=[]; off=0
    while True:
        d=c.table(t).select(cols).range(off,off+999).execute().data; out+=d
        if len(d)<1000: break
        off+=1000
    return out

jobs = [j for j in page("sandbox_jobs","id,username,job_status,created_at,n_trials,stats,metrics,model_id,benchmark_id")
        if j["username"] in OURS]
cand = [j for j in jobs if (j["job_status"]=="Finished" or j["stats"] is not None or j["metrics"] is not None)]
# apply recency OR tracker-listed id set:
TRACKER_IDS = {...}   # the campaign tracker's "DE-REGISTER candidate DEFERRED" ids
recent = lambda j: age_h(j["created_at"]) <= 24*7
scored = [(j,)+gate(j) for j in cand if recent(j) or j["id"] in TRACKER_IDS]   # (job, below, reason)
below  = [j for (j,b,r) in scored if b is True]
unscoreable = [j for (j,b,r) in scored if b is None]     # surface, do NOT delete

# --- BOUNDARY CONTROLS: prove the gate before trusting it ---
# pick rows just ABOVE the line and assert they are KEPT (not in `below`). Use guardrail_counts() to show
# each near-boundary row's authoritative counts (invalid_error_count vs round(0.10*planned);
# attempted/planned vs 0.90) and confirm the KEEP/DELETE by hand.

print(f"candidates={len(scored)}  below-gate={len(below)}  unscoreable={len(unscoreable)}")
from collections import Counter; print("below by user:", Counter(j['username'] for j in below))
# per-row table: id | user | model | benchmark | status | invalid_err/planned | attempted/planned | age | reason
# SANITY: if below-gate is a huge fraction of candidates (e.g. >150 or >~50%), STOP — the counts look wrong.
# The counting is the utility's (eval_guardrail); a large below-gate set means a bad THRESHOLD or scope, not
# a hand-rolled BENIGN mistake (that class of bug is now impossible — the utility owns the BENIGN set).
```

Then, **only after reviewing the dry-run + boundary controls**:
```python
DELETE = False    # flip to True after review
if DELETE:
    for j in below:                                  # own rows only (§2 already filtered)
        tids=[t["id"] for t in c.table("sandbox_trials").select("id").eq("job_id", j["id"]).execute().data]
        for i in range(0,len(tids),200):
            ch=tids[i:i+200]
            if ch: c.table("sandbox_trial_model_usage").delete().in_("trial_id", ch).execute()   # grandchild
        c.table("sandbox_trials").delete().eq("job_id", j["id"]).execute()                        # child
        c.table("sandbox_jobs").delete().eq("id", j["id"]).eq("username", j["username"]).execute()# job (yours)
    left = c.table("sandbox_jobs").select("id").in_("id",[j["id"] for j in below]).execute().data
    assert not left, f"{len(left)} targets survived"
    # re-read the KEEP-controls to confirm they are STILL PRESENT (didn't over-delete)
```

## 5. Report back — so the supervisor can reconcile the state docs

After the delete (or after the dry-run, if you were asked to stop before deleting), **REPORT the exact
purge outcome back to the supervisor** so they know whether a campaign state document (e.g. flawed_summ
`STATE.md`) needs updating. A purge that isn't reported back silently desyncs the tracker from the DB.
Return, as explicit lists:
- **PURGED (deleted):** for each row — `sandbox_jobs.id`, `username`, model, benchmark, and the gate
  fractions (vc% / nb% / at%) that failed it, plus the cascaded child/grandchild counts.
- **REPORTED-not-deleted (cross-user, §2):** the other-user below-gate rows you surfaced but did NOT
  touch (id, username, model, benchmark).
- **UNSCOREABLE / surfaced:** any row you could not score and left alone.

The supervisor uses this list to flip/remove the matching `STATE.md` rows and correct the counts.

## Guardrails

- **REPORT BACK the purge outcome (§5).** Always return the purged `sandbox_jobs.id`s (+ model /
  benchmark / gate-fractions), the cross-user rows reported-not-deleted, and any unscoreable rows — the
  supervisor needs this to reconcile the campaign state doc (e.g. flawed_summ `STATE.md`) with the DB.
- **AgentTimeout is BENIGN.** Never count `AgentTimeoutError` (or the §1 BENIGN set) as an error. If
  your below-gate set is a large fraction of candidates, you almost certainly got this wrong — STOP.
- **Boundary controls before deleting.** Confirm near-90% KEEP rows are classified KEEP and survive
  the delete (re-read them after). This is the guard against a mis-tuned threshold.
- **Cross-user FK safety is mandatory.** Own rows only (`OURS`); report other users' below-gate rows,
  never delete them.
- **Cascade required.** grandchild → child → job; a plain job delete FK-fails `23503`.
- **Never delete a `models` row** — this removes only the eval `sandbox_jobs` + its trial cascade.
- **DRY-RUN → review → delete → re-read.** Leave `DELETE=False` until the dry-run + boundary controls
  are reviewed; after deleting, confirm targets gone AND controls survive.
- **STOP + surface** any row you can't score (`n_trials` missing / stats shape unrecognized) rather
  than guess-deleting. Reads are free; deletes are dangerous (service-role bypasses RLS).
- **A REGISTERED below-gate row is DE-REGISTERED, never deferred (POLICY / `eval-agentic-cleanup` HARD
  GATE: "if the auto-pipeline ALREADY registered it, DE-REGISTER it").** "Defer" applies to the RE-EVAL
  timing (when to re-fire), NOT to leaving a contaminating registered row in the DB — an already-registered
  below-gate row shows on the leaderboard as `Errors: k` / a partial NOW, so remove it now; the pending /
  in-flight re-fire re-registers a clean row later. Do NOT skip de-registering a registered below-gate row
  because it "has a live re-fire" (that was a real bug — 8 high-error tb2/v2 rows sat registered because the
  purge deferred them). The only genuine defer: an UNregistered below-gate leg that never hit the DB needs
  no delete — it just re-fires. **A leaderboard HOLE (no result) is the CORRECT outcome when the only result
  is invalid — NEVER condition de-registration on a clean sibling / valid alternative existing.** (Cross-user
  FK safety §2 is a separate, ownership rule: still only delete OURS, never a zhuang row.)
- **⚠ A row's DB `stats` can be OVERWRITTEN by a later re-fire of an already-registered row (the
  `--force-reeval` duplicate trap).** So a below-gate `stats` reading may reflect a spurious *later* resume,
  not the clean eval the row was REGISTERED from. Before de-registering a row that carries a valid registered
  score, check whether its current `stats` came from a post-registration re-fire — if so, the registration
  may be legitimate and the fix is to stop re-firing registered rows, not to delete. (DB `stats` and the raw
  `result.json` DO agree on valid-complete — verified 2026-07-12; the risk is stale/overwritten stats, not a
  lossy field.)

## Related

- **`crud-purge-stale-eval-placeholders`** — the sibling DELETE for NEVER-populated `Pending`/`Started`
  placeholders (>36h, null metrics+stats). Use THAT when rows never scored; use THIS when they scored
  but failed the gate.
- **`crud-otagent-supabase`** — the general read/aggregate/write skill (`get_metric` shape helper,
  ID/OOD scoring, registration). The cross-user FK-safety rule there applies to all deletes.
- **`eval-agentic-cleanup`** + the campaign's policy (e.g. flawed_summ `POLICY.md` §"The gate" +
  `STATE.md`) — where the gate/discriminator convention and the "DE-REGISTER candidate" list are maintained.
