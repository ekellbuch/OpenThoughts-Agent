# 2026-06-19 — #232 de-risk: CORRECTED verdict (NOT blocked on #237)

> **This log was rewritten.** The original claimed "#232 blocked on #237 via the count-607744
> AllGather." That was a wrong pattern-match. Corrected analysis below.

## What was run
- `925720` = de-risk floor config: **32k / CP1 / R3-off**, policy TP1, vLLM engine TP2,
  env gsm8k, model **base Qwen3-30B-A3B**. (User's "find the working floor" strategy.)
- `925438` = parallel disable-custom-AR / R3-on probe → `CANCELLED+` (superseded).

## Original (WRONG) conclusion
Saw `AllGather … count 607744 … opCount 0` at gs0/59min and concluded #232 == #237's deadlock.

## Why that was wrong (evidence)
Comparison against the **healthy** #217 coder job `913908` (same infra) overturned it:
1. **count-607744 is BENIGN** — fires **1719×** in `913908`, which is at **79/80**. It's a normal
   recurring rollout collective, not a deadlock signature.
2. **925720 was NOT frozen** — max opCount reached **3.7e8** with live NCCL traffic. The
   "opCount 0" reading was only the init-header traces; collectives were progressing.
3. The `EngineDeadError`/`NodeDiedError` in the log are timestamped **14:22:55, AFTER the 14:14
   scancel** — teardown artifacts, not root cause.
4. The de-risk ran **R3-OFF**; #237 is specifically an **R3-capture** AllGather bug — so this
   config can't have exercised the #237 path at all.

## CORRECTED conclusion
`913908` (#217) and `925720` (#232 de-risk) are infrastructurally identical (TP2/CP1/32k/4-engine
/gsm8k). The ONE difference is the model:
- #217 = **Qwen3-Coder-30B-A3B-Instruct** → emits EOS, short completions → rollouts finish fast → 79/80.
- #232 de-risk = **base Qwen3-30B-A3B** → on gsm8k almost certainly generates degenerate /
  non-terminating completions that hit the 4096-token cap every sample → a single rollout batch
  takes pathologically long → step 1 never lands in ~1h.

**=> #232 de-risk's gs0 was a generation/model problem (base, non-instruct model), NOT the #237
infra deadlock.** Killing 925720 was still correct (it produced nothing), but #232 is NOT blocked
on #237.

## Re-derisk recommendation
Re-run the floor config with an **instruct-tuned** 30B-A3B (as #217 uses), or confirm the intended
#232 production model actually generates sanely on the target env, before reading any infra signal.

## #237 status (separate bug)
#237 = R3 routed-experts-capture AllGather **rank-asymmetry** deadlock at long-ctx/higher-TP.
Commits: `269277979`/`e7025e03c` (Stage 0 instrumentation, GATE GREEN), `3de064d3` (Stage 1 FIX B:
rank-symmetric R3 capture epilogue), `36a23bf6a` (Stage 3 token-id parity dump), `690e4dce1`
(worker_response_mq chunk sizing for R3 long-ctx). Current branch `feuer/r3-rank-symmetry-norope`
carries a rope-patch revert (`07e9fbca3`). **Blocker (inferred from git, no notes doc): FIX B is
committed but unverified / incompletely effective, entangled with a rope-scaling regression — the
Stage 3 parity validation has not closed.**
