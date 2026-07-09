---
name: crud-purge-below-gate-evals
description: >-
  Guardrailed DELETE of auto-registered eval `sandbox_jobs` rows that DID score but FAILED the
  harvest gate — partial/contaminated evals (valid-complete <90%, or infra-error >10%, or
  AgentTimeout ≥80%). These are the "DE-REGISTER candidate" rows the flawed_summ harvest defers.
  DISTINCT from `crud-purge-stale-eval-placeholders` (that removes NEVER-populated Pending/Started
  rows; this removes rows that populated stats/metrics but are below-gate). Use when asked to remove
  "partial / below-gate / <90%-completed-without-errors" evals owned by us. The MANDATORY parts:
  the AgentTimeout-is-BENIGN gate (a literal n_errors formula is catastrophically wrong), the
  cross-user FK-safety pre-check, and the grandchild→child→job cascade.
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

**Source of truth = the campaign tracker's convention** (e.g. flawed_summ
`~/Documents/experiments/active/flawed_summ_evals/reeval_tracker.md`, the "valid-complete /
non-benign / AgentTimeout benign" defs). Read it each run in case thresholds moved. The canonical
gate:

For each candidate row, with **`planned = sandbox_jobs.n_trials`** (the PLANNED count — 300 for
swe/dev_set_v2, 267 = 89×3 for terminal_bench_2):

| axis | formula | FAIL when |
|---|---|---|
| **valid-complete%** | `Σ stats.evals.*.n_trials` (trials that recorded a valid reward, incl. AgentTimeout→0) / `planned` | **< 90%** |
| **non-benign%** | `Σ non-benign exception_stats` / `planned` | **> 10%** |
| **AgentTimeout%** | `AgentTimeoutError` / `planned` | **≥ 80%** (contamination discriminator) |

A row is **BELOW-GATE (DELETE)** iff it fails **ANY** axis. A row is **GOOD (KEEP)** iff it passes
**ALL** (valid-complete ≥90% AND non-benign ≤10% AND AgentTimeout <80%).

**BENIGN exceptions (do NOT count toward non-benign):**
```python
BENIGN = {"AgentTimeoutError", "ContextLengthExceededError", "SummarizationTimeoutError",
          "SummarizationError", "BadRequestError", "NonZeroAgentExitCodeError", "VerifierRuntimeError"}
# NON-BENIGN (infra) = everything else: SandboxBuildFailedError, DaytonaAuthenticationError,
#   DaytonaValidationError, VerificationNotCompleted, VerifierTimeoutError, ...
```

**⚠ stats has TWO shapes — read `exception_stats`, not a bare `n_errors`:**
- **Shape A:** top-level `stats.n_errors` == the infra-error count (exception_stats sums to it).
- **Shape B:** `stats.evals.<key>.n_completed_trials` / `n_errored_trials`, where **`n_errored_trials`
  INCLUDES AgentTimeout** — same field name, different semantics.
So compute non-benign from the **`exception_stats` breakdown** (per-exception-type counts), summing
only the non-BENIGN types. Never trust a single `n_errors` integer.

```python
def gate(job):
    """Return (below_gate: bool, vc, nb, at) fractions. planned = job['n_trials']."""
    planned = job.get("n_trials") or 0
    if not planned: return (None, None, None, None)          # can't score → surface, don't delete
    evals = (job.get("stats") or {}).get("evals") or {}
    recorded = exc = 0
    excby = {}                                                # exception_type -> count
    for k, ev in evals.items():
        recorded += ev.get("n_trials") or ev.get("n_completed_trials") or 0
        for etype, n in (ev.get("exception_stats") or {}).items():
            excby[etype] = excby.get(etype, 0) + n
    at   = excby.get("AgentTimeoutError", 0)
    nb   = sum(n for e, n in excby.items() if e not in BENIGN)
    vc_f = recorded / planned
    nb_f = nb / planned
    at_f = at / planned
    below = (vc_f < 0.90) or (nb_f > 0.10) or (at_f >= 0.80)
    return (below, vc_f, nb_f, at_f)
```

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
scored = [(j,)+gate(j) for j in cand if recent(j) or j["id"] in TRACKER_IDS]
below  = [j for (j,b,vc,nb,at) in scored if b is True]
unscoreable = [j for (j,b,vc,nb,at) in scored if b is None]     # surface, do NOT delete

# --- BOUNDARY CONTROLS: prove the gate before trusting it ---
# pick rows just ABOVE the line and assert they are KEPT (not in `below`):
#   e.g. 91.0% valid / 9.0% nb  -> KEEP ;  90.7% / 9.3% -> KEEP ;  73% / 19.3% -> DELETE
# print a couple of near-boundary rows with their vc/nb/at and confirm the classification by hand.

print(f"candidates={len(scored)}  below-gate={len(below)}  unscoreable={len(unscoreable)}")
from collections import Counter; print("below by user:", Counter(j['username'] for j in below))
# per-row table: id | user | model | benchmark | status | vc% | nb% | at% | age
# SANITY: if below-gate is a huge fraction of candidates (e.g. >150 or >~50%), STOP — you probably
# counted AgentTimeout as an error. Re-check §1 BENIGN handling before ANY delete.
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

## Guardrails

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
- **Honor the campaign's defer pattern only if asked to.** A re-eval campaign may DEFER de-registering
  a below-gate row until its clean re-fire lands; purge those only on an explicit request.

## Related

- **`crud-purge-stale-eval-placeholders`** — the sibling DELETE for NEVER-populated `Pending`/`Started`
  placeholders (>36h, null metrics+stats). Use THAT when rows never scored; use THIS when they scored
  but failed the gate.
- **`crud-otagent-supabase`** — the general read/aggregate/write skill (`get_metric` shape helper,
  ID/OOD scoring, registration). The cross-user FK-safety rule there applies to all deletes.
- **`eval-agentic-cleanup`** + the campaign tracker (e.g. flawed_summ `reeval_tracker.md`) — where the
  gate/discriminator convention and the "DE-REGISTER candidate" list are maintained.
