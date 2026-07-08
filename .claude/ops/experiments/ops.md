# Experiments workspace — `/Users/benjaminfeuer/Documents/experiments`

The local per-experiment workspace on this Mac. **Each experiment / experiment-series gets its own
subdirectory here**, and each subdirectory typically holds **its own tracker(s)** — the source-of-truth
state for that experiment (queue, status table, per-row results, skipped items, plots, reports).

> **active/ vs complete/ split (2026-06):** experiments are now bucketed one level down —
> in-flight series live under **`active/`** (e.g. `active/delphi/`,
> `active/ablation_exploration_in_rl/`) and finished series under **`complete/`**
> (e.g. `complete/a1/`, `complete/a3/`, `complete/gsm8k_grid_leonardo/`, …). So an experiment's
> subdir is `experiments/active/<name>/` or `experiments/complete/<name>/`. New / running work goes
> under `active/`; move a series to `complete/` once it concludes.

> **FLAT datagen layout (2026-07-08):** datagen experiments are **NO LONGER nested under an
> `active/datagen/` parent** — each is its own FLAT sibling directory named descriptively
> `<model>-<ctx>-datagen-<taskset>-<cluster>`, e.g. `active/qwen3.5-122b-131k-datagen-opencode-iris/`,
> `active/minimax-m2.7-datagen-terminus2-jupiter/`. (Datagen is managed by a **different agent** — the
> supervisor does not drive it; see the datagen descope in `supervisor-init` / `monitor-restore-iris`.)
> Durable, reusable knowledge from a datagen experiment's docs does NOT live in the tracker: **ops/cluster
> facts → `.claude/ops/<cluster>/`, launch/monitor how-to → the `datagen-launch-*` skill, dated
> history → `~/Documents/agent_logs/`**; the experiment dir keeps only live campaign status.

> Distinct from other local dirs: this is **per-experiment working state + trackers**.
> `notes/` is the broader knowledge base; `agent_logs/` is dated failure/remediation logs; the cluster-side
> `experiments/` dirs (under each repo checkout on Jupiter/Leonardo) hold the actual run artifacts
> (`logs/`, `configs/`, `sbatch/`, `checkpoints/`). This Mac dir is where the human-readable trackers live.

## Convention
- **One subdirectory per experiment or series**, named for the experiment (e.g. `a3/`, `ablation_exploration_in_rl/`, `gsm8k_grid_leonardo/`, `iris_capacity/`, `delphi/`, `chat_templating/`, `cluster_timing_comparison/`, `flawed_summ_evals/`). Datagen experiments follow the FLAT `<model>-<ctx>-datagen-<taskset>-<cluster>` naming above (no `datagen/` parent).
  - `flawed_summ_evals/` → the **SummarizationTimeoutError-deflated re-eval campaign (a1-`<benchmark>` models)**: `reeval_tracker.md` (source of truth — has the per-sweep blocks + the "🚦 CAMPAIGN DRIVER" section) + `affected_evals.md` (the deflated-eval universe). Driver = harvest terminal legs + refill the next Section-A ⏳ rows to **the in-flight target `reeval_tracker.md` records** (its "🚦 CAMPAIGN DRIVER" / latest-sweep "Target =" line — the number is a property of THIS series and changes by directive; the monitor skills must NOT hardcode it, they look it up there each sweep). Subject to the HARD Daytona snapshot cap of 60 — clean stale sandboxes, never raise the cap; the binding watch is concurrent SANDBOXES, not snapshots. Referenced by `monitor-cron-sweep` / `monitor-restore`.
- **Trackers live inside the subdir**, usually `*.md` — and a series often has several: a queue/plan, a status/results tracker, a skipped-list, a report, plus subfolders for plots/per-run dirs. Examples seen in the tree:
  - `a3/` → `a3_rl_tracker.md` (status), `a3_rl_experiments.md` (launch log), `a3_skipped_datasets.md`, `reward_plots/`, a PDF report.
  - `ablation_exploration_in_rl/` → `HERO_LEARNED_BEHAVIORS.md`, `COMPARISON_SWEBENCH_PINNED.md` + per-run subdirs (`hero_rl_run/`, `explore-tis-*-8B/`, `shaped-45-8B/`, …).
  - `gsm8k_grid_leonardo/` → `grid.md` (plan), `accuracy_grid.md` (results), `grid_experiment_log.md`.
  - `iris_capacity/` → `iris_capacity_analysis.md` + interim/batch trackers.
- **Tracker naming is not rigid** — `*_tracker.md` / `grid.md` / `notes.md` / `DESIGN.md` / `*_log.md` all appear. When working an experiment, **read the subdir's `*.md` files first** to find its tracker; treat the one the user points at (or the most status-like) as source of truth.

## How to use it
- **Starting a new experiment:** create `experiments/active/<name>/` and a tracker inside it; record the queue/plan and update status as runs land. (Some series keep their canonical tracker elsewhere — follow the pointer the experiment itself gives. Datagen trackers live in their own FLAT `active/<model>-...-datagen-...-<cluster>/` dir, per `.claude/projects/daytona` / the `datagen-launch-*` skills.)
- **During a cron sweep / cleanup:** when a run for an experiment completes or changes state, update that experiment's tracker here (status table, results, reward plots) in addition to the global experiment log (`notes/claude/claude_experiments.md`).
- **This is local working state** — not git-tracked in the OT-Agent repo; don't pull large artifacts/checkpoints here (keep those on cluster scratch). Trackers + small plots/reports only.

## Migrating an experiment to `complete/`

An experiment lives under `active/` only while it is genuinely in flight. **When a series concludes** — all
runs terminal, results captured, the operator says it's closed, or it's otherwise done — migrate it. This is
a real state transition with side effects, not just a `mv`; do all of it in one pass so a future session
doesn't re-drive a closed campaign:

1. **Move the dir:** `git mv`/`mv experiments/active/<name>/ experiments/complete/<name>/` (it's not OT-Agent-git-tracked, so a plain `mv` is fine). Fix any tracker cross-references that hardcode the `active/` path.
2. **Retire every autonomous rule that fed it.** This is the load-bearing step. If the experiment was driven by a keep-N auto-launch, a refill loop, a cron step, or a monitor keep-full target, **remove that rule from the live cron AND from the canonical skill** (`monitor-restore` / `monitor-restore-iris`, and any `monitor-cron-sweep` reference) in the SAME pass. A `complete/` experiment is CLOSED: its autonomous rules are RETIRED — do not keep-N, refill, or re-drive it. Leaving the rule live is how a concluded campaign silently re-launches after a compaction.
3. **Drop a dated `CLOSED.md`** in the moved dir stating: the conclusion date, the final result/verdict (or where it's recorded), and **exactly which autonomous rules were retired** (cron step, keep-N line, refill loop) so the retirement is auditable.
4. **Update the global log** (`notes/claude/claude_experiments.md`) to reflect the closure.

Examples: the 32k `qwen3.5-122b-tt` datagen **keep-2** line was closed + moved to `complete/` on 2026-07-07 with its keep-2 auto-launch retired; the `a3/` series concluded and lives under `complete/a3/`.

**Reverse rarely:** only move `complete/ → active/` if the operator explicitly re-opens a series — and then re-establish its autonomous rule deliberately, don't assume the old one is still wired.
