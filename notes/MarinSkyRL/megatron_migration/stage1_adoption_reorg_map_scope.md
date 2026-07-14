# Stage 1 — Adoption-strategy decision + reorg divergence map ★ (critical path)

- **Date:** 2026-07-13 · **Status:** `scoped GO` (discovery; no fix)
- **Companion:** `megatron_migration_plan.md`

## Why this is the right next step
Everything downstream depends on WHERE the megatron backend lands relative to our pre-reorg fork. The megatron backend only exists in upstream's **post-reorg `skyrl/backends/skyrl_train/…`** tree; our SoT is on **pre-reorg `skyrl-train/skyrl_train/…`**. Choosing the adoption path (and quantifying the reorg cost) must precede any wiring, or we build on the wrong tree.

## Change-set
**Read-only / decision doc; no source touched.** Produce a divergence map: for each of `worker.py`, `model_wrapper.py`, `dispatch.py` (R3), `trainer.py`/`fully_async_trainer.py`, `dataset/preprocess.py`, and the config module, record pre-reorg path ↔ post-reorg path + degree of drift. Evaluate the three options:
1. **Rebase MarinSkyRL onto upstream's post-reorg `skyrl/`** and re-apply our features on top (clean future upstreaming; large one-time reorg + feature-reapply cost — the same gap blocking our OPD PRs).
2. **Port the megatron backend INTO our pre-reorg fork** (`skyrl/backends/skyrl_train/megatron/*` copied under `skyrl-train/skyrl_train/`; keeps our layout; carries the megatron code as a long-lived divergence, harder to track upstream).
3. **Move to upstream `main` + re-apply our fork features** (maximal upstream alignment; the biggest single-shot merge, riskiest to in-flight FSDP2 arms).

Weigh each against: feature-reapply surface (R3/TIS/seqnorm/fully-async/agentic), future upstreamability, blast radius on the running FSDP2 arms, and the vLLM coupling (Stage 3).

## Validation gate (GO / NO-GO)
- **GO:** one option chosen with a written migration path + an estimated feature-reapply cost (which of R3/TIS/seqnorm/fully-async/agentic move cleanly vs need rewrite) + a blast-radius statement for the FSDP2 arms.
- **NO-GO:** all three options exceed an acceptable divergence/maintenance budget → recommend staying on FSDP2 and attacking its pain points directly (the plan's fallback).
- **Cost:** CPU (read + reasoning).

## Composes with / depends on
Informed by Stage 0 (if 80B is infeasible, adoption calculus changes). Feeds Stage 4 (wiring targets the chosen tree) and Stage 3 (vLLM coupling).
